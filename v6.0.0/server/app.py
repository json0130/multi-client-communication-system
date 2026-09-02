"""
app.py
======
Entry point. Wires every layer together and starts the Flask server.

Start the server:
    python3 app.py

What happens at startup:
  1. Config loads (reads .env)
  2. Scenario profiles load and validate (fails fast on an invalid profile)
  3. RBAC filter + grant store are created and shared across all robots
  4. RobotRegistry is created
  5. WebSocketGateway is created (connects TO robots when told to)
  6. DelegationHandler is created
  7. HTTP Blueprint is registered on the Flask app
  8. DemoOrchestrator is created and wired to WebSocketGateway
  9. Demo blueprint registered (POST /demo/start, /demo/stop, etc.)
 10. Cleanup task starts (removes idle robot instances every 5 min)
 11. Flask starts listening for web UI requests on SERVER_PORT (default 5000)

Robots are NOT connected automatically at startup.
Use the web UI or POST /robots/<id>/connect to connect a robot.
"""

import os
import signal
import sys
import logging

from flask import Flask

from core.config import cfg
from core.profiles import ProfileRegistry
from core.rbac import BatchingAuditSink, GrantStore, RBACFilter
from decision import BatchingDecisionSink, DecisionRecorder
from robot.robot_registry import RobotRegistry
from gateway.websocket_gateway import WebSocketGateway
from gateway.delegation_handler import DelegationHandler
from gateway.http_gateway import create_http_gateway
from gateway.persona_gateway import create_persona_gateway
from gateway.demo_gateway import create_demo_gateway
from gateway.project_gateway import create_project_gateway
from gateway.kg_gateway import create_kg_gateway
from demo.demo_orchestrator import DemoOrchestrator
from demo.demo_script import DEMO_STEPS

# ── Fix OpenMP duplicate lib issue on some Linux setups ───────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Scenario profiles live alongside the server, one YAML per deployment.
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def build_rbac() -> tuple[ProfileRegistry, RBACFilter, GrantStore]:
    """
    Load scenario profiles and build the shared RBAC objects.

    Profile validation is deliberately allowed to raise: an invalid access level
    or a scenario with no Manager must stop the boot, not surface as a silent
    deny at the first retrieval.

    The audit sink is best-effort — if the database is unreachable, decisions go
    unrecorded but retrieval keeps working.
    """
    profiles = ProfileRegistry.from_directory(PROFILE_DIR)

    # Reconcile declared access levels into the DB. The level is data, so this
    # is the only place the profile touches robot configuration.
    if len(profiles):
        try:
            from data import robot_repo
            synced = profiles.sync_to_db(robot_repo.set_access_level)
            print(f"[App] Reconciled access level for {synced} robot(s).")
        except Exception as e:
            print(f"[App] Could not reconcile access levels (continuing): {e}")

    try:
        from data.rbac_audit_repo import write_events
        audit = BatchingAuditSink(writer=write_events)
    except Exception as e:
        from core.rbac import NullAuditSink
        print(f"[App] RBAC audit sink unavailable, decisions will not be logged: {e}")
        audit = NullAuditSink()

    return profiles, RBACFilter(audit_sink=audit), GrantStore()


def build_decision_recorder() -> DecisionRecorder:
    """
    Build the sink that records demo decisions and supervisor corrections.

    Best-effort for the same reason as the RBAC audit sink, and more so: this one
    runs in front of visitors. If the database is unreachable the demo proceeds
    with decisions unrecorded rather than failing.
    """
    try:
        from data.demo_decision_repo import write_corrections, write_events
        return DecisionRecorder(BatchingDecisionSink(
            decision_writer=write_events,
            correction_writer=write_corrections,
        ))
    except Exception as e:
        print(f"[App] Decision sink unavailable, demo decisions will not be logged: {e}")
        return DecisionRecorder()


# How long a routing snapshot of the graph is reused before being refetched.
# Routing happens per visitor turn; refetching the whole graph each time would
# put a database round-trip on the path a question takes. Corrections land at
# human speed, so a few seconds of staleness costs nothing.
_KG_TTL_SEC = 10.0
_kg_cache: dict = {"at": 0.0, "router": None}


def build_kg_router():
    """A KGRouter over the current graph, or None if the graph is unusable.

    Returns None — meaning "fall back to the baseline" — when the vocabulary is
    unseeded or the database is unreachable. Routing must never depend on the
    graph being healthy.
    """
    import time as _time
    if _time.time() - _kg_cache["at"] < _KG_TTL_SEC:
        return _kg_cache["router"]
    router = None
    try:
        from data import demo_kg_repo as repo
        from decision.kg import RobotTopicEdge
        from decision.kg_policy import KGRouter
        topics = repo.all_topics()
        if topics:
            edges = [RobotTopicEdge.from_row(r) for r in repo.graph()]
            links = [(l["topic_a"], l["topic_b"], float(l["weight"]))
                     for l in repo.all_links()]
            router = KGRouter(edges, links, topics)
    except Exception as e:
        print(f"[App] KG router unavailable, routing falls back to receiver: {e}")
    _kg_cache.update(at=_time.time(), router=router)
    return router


def apply_kg_observations(observations) -> None:
    """Persist outcome observations emitted when a Q&A window closes cleanly.

    Best-effort, like every other write on a demo's critical path: a graph that
    cannot be updated must not interrupt a tour. Also invalidates the routing
    snapshot, so the next question routes against what was just learned rather
    than a cache up to _KG_TTL_SEC stale.
    """
    try:
        from data import demo_kg_repo as repo
        from decision.kg_feedback import apply
        apply(observations, repo)
        _kg_cache["at"] = 0.0
    except Exception as e:
        print(f"[App] Could not record KG outcomes: {e}")


def record_duration(kind: str, row: dict) -> None:
    """Persist one timing row. Best-effort, like every write on a demo's path.

    Scripted steps and Q&A windows go to SEPARATE tables and are never averaged
    together — see data/migrations/008_demo_durations.sql for why mixing them
    makes every step estimate worse as more data arrives.
    """
    try:
        from data import demo_duration_repo as repo
        if kind == "step":
            repo.write_step_durations([row])
        else:
            repo.write_qa_durations([row])
    except Exception as e:
        print(f"[App] Could not record {kind} duration: {e}")


def create_app() -> tuple[Flask, WebSocketGateway, RobotRegistry]:
    """
    Build and return the configured Flask app plus the gateway objects.
    Separated from main() so tests can import create_app() directly.
    """
    # ── RBAC (before the registry — instances are built with these) ───────────
    profiles, rbac, grants = build_rbac()

    # ── Core objects (order matters — registry first) ─────────────────────────
    registry   = RobotRegistry(rbac=rbac, grants=grants, profiles=profiles)
    recorder   = build_decision_recorder()
    ws_gateway = WebSocketGateway(registry, recorder=recorder,
                                  kg_router_factory=build_kg_router,
                                  kg_observer=apply_kg_observations)

    # ── Flask app ─────────────────────────────────────────────────────────────
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Suppress Werkzeug HTTP access logs (auto-refresh polling is noisy)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # Register HTTP routes
    blueprint = create_http_gateway(registry, ws_gateway)
    app.register_blueprint(blueprint)

    # Register persona routes
    persona_blueprint = create_persona_gateway(ws_gateway)
    app.register_blueprint(persona_blueprint)

    # ── Demo orchestrator ─────────────────────────────────────────────────────
    # The recorder is shared with the gateway so an operator's "Move On" is
    # recorded as a correction *of* the decision the gateway just logged, rather
    # than as an unattached row.
    orchestrator = DemoOrchestrator(
        ws_gateway,
        recorder=recorder,
        session_context=ws_gateway.session_context,
        duration_sink=record_duration,
    )
    orchestrator.load_script(DEMO_STEPS)
    ws_gateway.set_demo_orchestrator(orchestrator)

    demo_blueprint = create_demo_gateway(orchestrator)
    app.register_blueprint(demo_blueprint)

    project_blueprint = create_project_gateway()
    app.register_blueprint(project_blueprint)

    # Robot→topic competence graph (dashboard tab + /kg/observe).
    app.register_blueprint(create_kg_gateway(registry, ws_gateway))

    return app, ws_gateway, registry


def main():
    app, ws_gateway, registry = create_app()

    # ── Start background cleanup task ─────────────────────────────────────────
    registry.start_cleanup_task()

    # ── Graceful shutdown on Ctrl+C / SIGTERM ────────────────────────────────
    def _shutdown(sig, frame):
        print("\n[App] Shutting down...")
        registry.stop_cleanup_task()
        ws_gateway.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start Flask ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Robot Management Server")
    print("=" * 55)
    print(f"  Host     : {cfg.server.host}")
    print(f"  Port     : {cfg.server.port}")
    print(f"  Ollama   : {cfg.llm.ollama_host}:{cfg.llm.ollama_port}")
    print(f"  DB       : {cfg.db.url[:40]}...")
    print()
    print("  Endpoints:")
    print(f"    GET  /                          server status")
    print(f"    GET  /robots                    list all robots")
    print(f"    POST /robots/register           register a robot")
    print(f"    PUT  /robots/<id>/role          update role + tags")
    print(f"    POST /robots/<id>/connect       open WS to robot")
    print(f"    POST /robots/<id>/disconnect    close WS to robot")
    print(f"    GET  /robots/<id>/health        module health")
    print(f"    POST /robots/<id>/chat          send a chat message")
    print()
    print("  Demo:")
    print(f"    POST /demo/start                start demo from step 1")
    print(f"    POST /demo/stop                 stop and reset")
    print(f"    POST /demo/pause                pause at current step")
    print(f"    POST /demo/resume               resume")
    print(f"    POST /demo/next                 skip to next step (recovery)")
    print(f"    GET  /demo/status               current state + step info")
    print()
    print("  Knowledge graph:")
    print(f"    GET  /kg/graph                  robot->topic competence")
    print(f"    POST /kg/seed                   build the topic vocabulary")
    print(f"    POST /kg/observe                fold in one observation")
    print()
    print(f"  Demo script: {len(DEMO_STEPS)} steps loaded")
    print()
    print("  Waiting for requests... (Ctrl+C to stop)")
    print("=" * 55)

    app.run(
        host=cfg.server.host,
        port=cfg.server.port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
