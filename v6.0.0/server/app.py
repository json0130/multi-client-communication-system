"""
app.py
======
Entry point. Wires every layer together and starts the Flask server.

Start the server:
    python3 app.py

What happens at startup:
  1. Config loads (reads .env)
  2. RobotRegistry is created
  3. WebSocketGateway is created (connects TO robots when told to)
  4. DelegationHandler is created
  5. HTTP Blueprint is registered on the Flask app
  6. DemoOrchestrator is created and wired to WebSocketGateway
  7. Demo blueprint registered (POST /demo/start, /demo/stop, etc.)
  8. Cleanup task starts (removes idle robot instances every 5 min)
  9. Flask starts listening for web UI requests on SERVER_PORT (default 5000)

Robots are NOT connected automatically at startup.
Use the web UI or POST /robots/<id>/connect to connect a robot.
"""

import os
import signal
import sys
import logging

from flask import Flask

from core.config import cfg
from robot.robot_registry import RobotRegistry
from gateway.websocket_gateway import WebSocketGateway
from gateway.delegation_handler import DelegationHandler
from gateway.http_gateway import create_http_gateway
from gateway.persona_gateway import create_persona_gateway
from gateway.demo_gateway import create_demo_gateway
from demo.demo_orchestrator import DemoOrchestrator
from demo.demo_script import DEMO_STEPS

# ── Fix OpenMP duplicate lib issue on some Linux setups ───────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def create_app() -> tuple[Flask, WebSocketGateway, RobotRegistry]:
    """
    Build and return the configured Flask app plus the gateway objects.
    Separated from main() so tests can import create_app() directly.
    """
    # ── Core objects (order matters — registry first) ─────────────────────────
    registry   = RobotRegistry()
    ws_gateway = WebSocketGateway(registry)

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
    orchestrator = DemoOrchestrator(ws_gateway)
    orchestrator.load_script(DEMO_STEPS)
    ws_gateway.set_demo_orchestrator(orchestrator)

    demo_blueprint = create_demo_gateway(orchestrator)
    app.register_blueprint(demo_blueprint)

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
