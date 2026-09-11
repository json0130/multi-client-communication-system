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

from typing import Optional

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
from gateway.flow_gateway import create_flow_gateway
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


# How long a graph/duration snapshot is reused before being refetched. Routing
# and revision-planning both happen per visitor turn; refetching from Supabase
# on every turn would put a database round-trip on the path a question takes.
# Corrections and duration writes land at human speed, so a few seconds of
# staleness costs nothing either reads.
_SNAPSHOT_TTL_SEC = 10.0
_kg_cache: dict = {"at": 0.0, "topics": [], "edges": [], "links": []}
_duration_cache: dict = {"at": 0.0, "durations": {}}

# build_flow_plan needs the orchestrator's standing visitor_profile, but it is
# constructed as a module-level function referenced by name (passed into
# WebSocketGateway BEFORE the orchestrator exists — the orchestrator wraps the
# gateway, so the dependency runs the other way). Same fix as the caches above:
# a mutable cell set once create_app() has built the orchestrator.
_orchestrator_ref: dict = {"o": None}

# Same shape, for the scenario profile: build_flow_plan is referenced by name
# before create_app() has loaded the profiles.
_profiles_ref: dict = {"p": None}


def _kg_snapshot():
    """(topics, edges, links) from the current competence graph.

    Shared by routing (build_kg_router) and revision planning (build_flow_plan)
    so both act on the same view of the graph within one TTL window, and so
    adding a second reader did not mean a second set of Supabase calls.
    """
    import time as _time
    if _time.time() - _kg_cache["at"] < _SNAPSHOT_TTL_SEC:
        return _kg_cache["topics"], _kg_cache["edges"], _kg_cache["links"]
    topics, edges, links = [], [], []
    try:
        from data import demo_kg_repo as repo
        from decision.kg import RobotTopicEdge
        topics = repo.all_topics()
        if topics:
            edges = [RobotTopicEdge.from_row(r) for r in repo.graph()]
            links = [(l["topic_a"], l["topic_b"], float(l["weight"]))
                     for l in repo.all_links()]
    except Exception as e:
        print(f"[App] KG snapshot unavailable: {e}")
    _kg_cache.update(at=_time.time(), topics=topics, edges=edges, links=links)
    return topics, edges, links


def build_kg_router():
    """A KGRouter over the current graph, or None if the graph is unusable.

    Returns None — meaning "fall back to the baseline" — when the vocabulary is
    unseeded or the database is unreachable. Routing must never depend on the
    graph being healthy.
    """
    from decision.kg_policy import KGRouter
    topics, edges, links = _kg_snapshot()
    if not topics:
        return None
    # explore=False, EXPLICITLY. This is the live server, and exploration
    # deliberately routes to the less-observed robot when the graph cannot
    # separate two candidates — which is right for a rollout campaign and
    # wrong in front of visitors. It was defaulting to True here, so every
    # real question was an exploration step: decision/kg_infer.py::route
    # states the rule ("the rollout harness turns it on, a live demo turns it
    # off") and this call was the one place not honouring it. Found because
    # a controlled experiment routed the same question to different robots in
    # different conditions.
    return KGRouter(edges, links, topics, explore=False)


def apply_kg_observations(observations) -> None:
    """Persist outcome observations emitted when a Q&A window closes cleanly.

    Best-effort, like every other write on a demo's critical path: a graph that
    cannot be updated must not interrupt a tour. Also invalidates the graph
    snapshot, so the next question routes — and the next revision plans —
    against what was just learned rather than a cache up to
    _SNAPSHOT_TTL_SEC stale.
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
        # A new step timing changes what the planner should predict next turn;
        # invalidate rather than wait out the TTL mid-demo.
        _duration_cache["at"] = 0.0
    except Exception as e:
        print(f"[App] Could not record {kind} duration: {e}")


def _step_durations() -> dict:
    """{step_id: mean_sec} from every run logged so far.

    Read fresh at most once per _SNAPSHOT_TTL_SEC — the same reasoning as the KG
    snapshot. Early in a campaign this is mostly empty and FlowGraph.estimate()
    falls back to DEFAULT_STEP_SEC per step; measured_coverage on the planner's
    result says how much of an estimate is actually earned versus guessed.
    """
    import time as _time
    if _time.time() - _duration_cache["at"] < _SNAPSHOT_TTL_SEC:
        return _duration_cache["durations"]
    durations = {}
    try:
        from data.demo_duration_repo import step_stats
        from decision.flow import MIN_RUNS_TO_TRUST
        rows = step_stats()
        # demo_step_duration_stats groups by (step_id, block_robot_id), and
        # one step_id can legitimately appear under several blocks: with the
        # robots in a different order, transition_to_silbot_01 belongs to
        # whichever block precedes Silbot. FlowGraph.estimate looks steps up
        # by step_id ALONE, so those rows have to be combined — a dict
        # comprehension over the rows silently kept whichever Supabase
        # happened to return last, which is a different number run to run.
        #
        # Combined as a run-weighted mean, so ten observations under one
        # block outweigh three under another rather than counting equally,
        # and the trust threshold is applied to the COMBINED count: three
        # runs is three runs whether or not the block ordering varied.
        agg: dict = {}
        for r in rows:
            if r.get("mean_sec") is None:
                continue
            n = int(r.get("runs") or 0)
            if n <= 0:
                continue
            total_n, total_sec = agg.get(r["step_id"], (0, 0.0))
            agg[r["step_id"]] = (total_n + n, total_sec + n * float(r["mean_sec"]))

        durations = {sid: secs / n for sid, (n, secs) in agg.items()
                     if n >= MIN_RUNS_TO_TRUST}
        untrusted = sum(1 for n, _ in agg.values() if n < MIN_RUNS_TO_TRUST)
        if untrusted:
            print(f"[App] {len(durations)} step timing(s) trusted, {untrusted} below "
                  f"{MIN_RUNS_TO_TRUST} runs — planning those from defaults")
    except Exception as e:
        print(f"[App] step duration stats unavailable, planning from defaults: {e}")
    _duration_cache.update(at=_time.time(), durations=durations)
    return durations


_style_cache: dict = {"at": 0.0, "fits": {}}


def lookup_style_fit(robot_id: str, style: str):
    """One robot's record with one audience, or None when nothing is known.

    Cached on the same TTL as the graph and duration snapshots and for the
    same reason: this sits on the path every generated step takes, and a
    Supabase round-trip per utterance would put database latency into the
    pause before a robot speaks. Ratings arrive at human speed, so a few
    seconds of staleness costs nothing.

    Returns None rather than raising on any failure — generation then falls
    back to the plain style directive, which is what it did before style fit
    existed.
    """
    import time as _time
    if _time.time() - _style_cache["at"] >= _SNAPSHOT_TTL_SEC:
        try:
            from data import demo_style_repo
            _style_cache.update(at=_time.time(), fits=demo_style_repo.snapshot())
        except Exception as e:
            print(f"[App] style fit unavailable, framing from the directive alone: {e}")
            _style_cache.update(at=_time.time(), fits={})
    return _style_cache["fits"].get((robot_id, style))


def lookup_subject(robot_id: str) -> str:
    """What this robot researches, as a phrase for the guide's introduction.

    Built from the DECLARED scope in the competence graph — the same rows
    routing narrows on — so the subject the guide announces and the subject
    questions are routed on are one fact, not two that can drift. A live run
    had Pepper introduce Silbot as working on understanding emotions because
    the script told it to describe "the research area" without ever saying
    what that area was.

    Falls back to "" on any failure, which returns the instruction to its
    previous wording rather than blocking the demo.
    """
    # The scenario profile's role FIRST. It is a concise area label —
    # "Conversational AI researcher" — which is what an introduction wants.
    # Joining the declared topics instead made the guide recite the
    # vocabulary: "conversational memory, knowledge graphs, and large
    # language models to enhance long-term interaction and retrieval
    # augmented generation" was one real generation. Accurate, unusable.
    try:
        prof = _profiles_ref["p"]
        if prof is not None:
            entry = prof.find_robot(robot_id) if hasattr(prof, "find_robot") else None
            if entry is None:
                for sc in getattr(prof, "_by_scenario", {}).values():
                    entry = sc.get(robot_id)
                    if entry is not None:
                        break
            if entry is not None and getattr(entry, "role", ""):
                return str(entry.role)
    except Exception as e:
        print(f"[App] profile role unavailable for {robot_id}: {e}")

    # Fall back to the declared topics, capped — better a short list than a
    # recitation, and better either than nothing.
    try:
        topics, edges, _links = _kg_snapshot()
        labels = {t["id"]: t.get("label", t["id"]) for t in topics}
        declared = sorted(labels.get(e.topic_id, e.topic_id) for e in edges
                          if e.robot_id == robot_id and e.specialised)
        return ", ".join(declared[:2])
    except Exception as e:
        print(f"[App] subject lookup failed for {robot_id}: {e}")
        return ""


def build_flow_plan(obs) -> Optional[dict]:
    """
    PLAN_REVISE's real implementation — decision.planner fed real data.

    Returns None — "not applicable", never an error — in the two cases where
    there is nothing to plan against: no time budget was set for this run (the
    operator never opted into clock-driven revision, so there is no target to
    fit), or the remaining script has no project blocks left to act on (e.g.
    only the closing remains). HeuristicPolicy degrades to its own thin
    fallback ladder in either case, and on any exception raised here.

    A visitor's stated interest, if this turn resolved to one, feeds
    block_importance's visitor-derived layer — the same topic resolution
    QA_ROUTE uses, read from the same graph snapshot QA_ROUTE reads, so a
    stated interest shapes both which robot answers and what survives a cut.
    """
    if not obs.time_budget_sec or not obs.remaining_steps:
        return None

    from decision.flow import FlowGraph
    from decision.planner import block_importance, plan_for_budget, resolve_emphasis

    graph = FlowGraph.from_script(obs.remaining_steps)
    if not graph.blocks:
        return None

    remaining_budget = max(0.0, obs.time_budget_sec - obs.elapsed_sec)
    durations = _step_durations()
    topics, edges, links = _kg_snapshot()

    # EXPLICIT trigger-type ordering, made a named sequence rather than
    # implicit in what happened to be computed here:
    #   1. a freshly stated interest THIS TURN
    #   2. the pre-demo visitor profile's standing interest
    #   3. neither — the lab's own per-project priorities apply alone, read
    #      from the scenario profile's `importance:` key
    utterance_topics = None
    if topics and obs.user_utterance:
        from decision.kg_policy import KGRouter
        tid = KGRouter([], [], topics).resolve_topic(obs.user_utterance)
        if tid:
            utterance_topics = [tid]

    profile = _orchestrator_ref["o"].visitor_profile if _orchestrator_ref["o"] else None
    profile_topics = list(profile.topics) if profile and profile.topics else None

    visitor_topics, emphasis_source = resolve_emphasis(utterance_topics, profile_topics)
    if visitor_topics:
        print(f"[App] PLAN_REVISE emphasis source: {emphasis_source} ({visitor_topics})")

    # The lab's own per-project priorities, from the scenario profile. This
    # was defaults={}, so every block scored DEFAULT_IMPORTANCE, the sort
    # fell through to its (importance, robot_id) tie-break, and which project
    # a visitor lost under time pressure was decided by robot_id spelling.
    # Still {} if the profile declares none — tools/check_cold_start.py
    # reports that as INERT rather than letting it pass silently.
    defaults = {}
    try:
        defaults = _profiles_ref["p"].importance_defaults() if _profiles_ref["p"] else {}
    except Exception as e:
        print(f"[App] per-project importance unavailable: {e}")

    importance = block_importance(graph, defaults=defaults,
                                  visitor_topics=visitor_topics or None,
                                  kg_edges=edges, kg_links=links)

    # How long a window of each block ACTUALLY runs, from visitor-ended
    # windows only. demo_duration_repo enforces that exclusion in one place
    # (_visitor_ended) because setting a window's length from observed window
    # lengths is the same shape as the loop already found and fixed once: a
    # window cut to 5s recorded 5s, which pulled the estimate down, which cut
    # more windows. Passing a mean over everything here would rebuild it.
    measured_qa = {}
    try:
        from data.demo_duration_repo import qa_median_by_block
        measured_qa = qa_median_by_block()
    except Exception as e:
        print(f"[App] per-block Q&A medians unavailable: {e}")

    # allocate=True: the same total Q&A time, split by measured length and
    # this run's importance, BEFORE the ladder cuts anything. A visitor who
    # came for the emotion work gets a longer Navel round paid for by shorter
    # others, which costs them nothing they came for — where cutting always
    # costs somebody something.
    return plan_for_budget(graph, remaining_budget, durations=durations,
                           importance=importance, measured_qa=measured_qa,
                           allocate=True)


def create_app() -> tuple[Flask, WebSocketGateway, RobotRegistry]:
    """
    Build and return the configured Flask app plus the gateway objects.
    Separated from main() so tests can import create_app() directly.
    """
    # ── RBAC (before the registry — instances are built with these) ───────────
    profiles, rbac, grants = build_rbac()
    _profiles_ref["p"] = profiles

    # ── Core objects (order matters — registry first) ─────────────────────────
    registry   = RobotRegistry(rbac=rbac, grants=grants, profiles=profiles)
    recorder   = build_decision_recorder()
    ws_gateway = WebSocketGateway(registry, recorder=recorder,
                                  kg_router_factory=build_kg_router,
                                  kg_observer=apply_kg_observations,
                                  flow_planner=build_flow_plan)
    try:
        ws_gateway.presence.set_separate_stations(profiles.separate_stations())
    except Exception as e:
        print(f"[App] station layout unavailable: {e}")

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
        style_fit=lookup_style_fit,
        subject_lookup=lookup_subject,
    )
    # Unreviewed project facts are usable — that is how they get tested — but
    # they must never be demoed without someone knowing. A robot states them
    # hedged (see decision/grounding.py), and this is the other half: the
    # operator is told before the tour, not after a visitor repeats one.
    try:
        from data.demo_facts_repo import unverified_count
        pending = unverified_count()
        if pending:
            print(f"[App] WARNING: {pending} topic fact(s) are UNVERIFIED. Robots "
                  f"will hedge them rather than state them as results. Review with: "
                  f"python3 tools/seed_topic_facts.py --report")
    except Exception:
        pass

    _orchestrator_ref["o"] = orchestrator
    orchestrator.load_script(DEMO_STEPS)
    ws_gateway.set_demo_orchestrator(orchestrator)

    demo_blueprint = create_demo_gateway(orchestrator, ws_gateway)
    app.register_blueprint(demo_blueprint)

    project_blueprint = create_project_gateway()
    app.register_blueprint(project_blueprint)

    # Robot→topic competence graph (dashboard tab + /kg/observe).
    app.register_blueprint(create_kg_gateway(registry, ws_gateway))

    # Flow graph + planner preview (dashboard tab + /flow/plan).
    app.register_blueprint(create_flow_gateway(orchestrator))

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
