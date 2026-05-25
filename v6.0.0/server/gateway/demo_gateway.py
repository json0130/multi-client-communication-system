"""
gateway/demo_gateway.py
========================
HTTP endpoints for controlling the CARES lab demo state machine.

Endpoints:
    POST /demo/start    — Load script and start from step 0
    POST /demo/stop     — Stop and reset to IDLE
    POST /demo/pause    — Pause at current step
    POST /demo/resume   — Resume from current step
    POST /demo/next     — Manually advance (skips ACK wait — use to recover from stuck step)
    GET  /demo/status   — Current state, step info, progress
"""

from flask import Blueprint, jsonify, request


def create_demo_gateway(orchestrator) -> Blueprint:
    """
    Factory — pass the DemoOrchestrator instance.
    Called from app.py after the orchestrator is created.
    """
    bp = Blueprint("demo", __name__)

    @bp.route("/demo/start", methods=["POST"])
    def start():
        data      = request.get_json(silent=True) or {}
        robot_ids = data.get("robot_ids") or []
        orchestrator.start(robot_ids=robot_ids if robot_ids else None)
        return jsonify({"message": "Demo started.", **orchestrator.get_status()})

    @bp.route("/demo/stop", methods=["POST"])
    def stop():
        orchestrator.stop()
        return jsonify({"message": "Demo stopped.", **orchestrator.get_status()})

    @bp.route("/demo/pause", methods=["POST"])
    def pause():
        orchestrator.pause()
        return jsonify({"message": "Demo paused.", **orchestrator.get_status()})

    @bp.route("/demo/resume", methods=["POST"])
    def resume():
        orchestrator.resume()
        return jsonify({"message": "Demo resumed.", **orchestrator.get_status()})

    @bp.route("/demo/next", methods=["POST"])
    def next_step():
        orchestrator.manual_next()
        return jsonify({"message": "Advanced to next step.", **orchestrator.get_status()})

    @bp.route("/demo/status", methods=["GET"])
    def status():
        return jsonify(orchestrator.get_status())

    @bp.route("/demo/qa", methods=["POST"])
    def qa_start():
        data = request.get_json(silent=True) or {}
        orchestrator.qa_interrupt(message=data.get("message", ""))
        return jsonify({"message": "Q&A window opened.", **orchestrator.get_status()})

    @bp.route("/demo/qa_end", methods=["POST"])
    def qa_stop():
        orchestrator.qa_end()
        return jsonify({"message": "Q&A window closed.", **orchestrator.get_status()})

    # CORS pre-flight (matches the pattern used by the existing gateways)
    @bp.route("/demo/<path:path>", methods=["OPTIONS"])
    @bp.route("/demo", methods=["OPTIONS"])
    def options(path=""):
        from flask import request, make_response
        response = make_response()
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    return bp
