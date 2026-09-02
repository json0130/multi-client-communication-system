"""
gateway/demo_gateway.py
========================
HTTP endpoints for controlling the CARES lab demo state machine.

Endpoints:
    POST /demo/start    — Load script and start from step 0 (optional time_budget_sec)
    POST /demo/stop     — Stop and reset to IDLE
    POST /demo/pause    — Pause at current step
    POST /demo/resume   — Resume from current step
    POST /demo/next     — Manually advance (skips ACK wait — use to recover from stuck step)
    POST /demo/revise   — Change the remaining script (skip / reorder / compress / …)
    GET  /demo/status   — Current state, step info, progress, run clock

Every control route passes source="operator". That is not cosmetic: it is what
separates a human deciding the demo should move on from the system deciding it,
and only the human's version is recorded as a supervisor correction. A route that
forgets it silently drops a training label. See decision/ and
demo/demo_orchestrator.py::_record_correction.

`reason` is optional everywhere it is accepted. An operator running a live demo
will not stop to fill in a form, and an un-annotated click is still a label — the
timestamp and step say what was wrong with the timing.
"""

from flask import Blueprint, jsonify, request

from decision.models import PlanOp


def _reason(data: dict) -> str:
    """Operator's note, if they left one. Truncated — this is a label, not a log."""
    return str((data or {}).get("reason") or "")[:500]


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

        # Without a budget the tour has nothing to run late against, so
        # clock-driven plan revision stays inert. Passing one opts the run in.
        budget = data.get("time_budget_sec")
        try:
            budget = float(budget) if budget not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify({"error": "time_budget_sec must be a number of seconds."}), 400
        if budget is not None and budget <= 0:
            return jsonify({"error": "time_budget_sec must be positive."}), 400

        orchestrator.start(
            robot_ids=robot_ids if robot_ids else None,
            time_budget_sec=budget,
        )
        return jsonify({"message": "Demo started.", **orchestrator.get_status()})

    @bp.route("/demo/stop", methods=["POST"])
    def stop():
        orchestrator.stop()
        return jsonify({"message": "Demo stopped.", **orchestrator.get_status()})

    @bp.route("/demo/pause", methods=["POST"])
    def pause():
        data = request.get_json(silent=True) or {}
        orchestrator.pause(source="operator", reason=_reason(data))
        return jsonify({"message": "Demo paused.", **orchestrator.get_status()})

    @bp.route("/demo/resume", methods=["POST"])
    def resume():
        data = request.get_json(silent=True) or {}
        orchestrator.resume(source="operator", reason=_reason(data))
        return jsonify({"message": "Demo resumed.", **orchestrator.get_status()})

    @bp.route("/demo/next", methods=["POST"])
    def next_step():
        data = request.get_json(silent=True) or {}
        orchestrator.manual_next(source="operator", reason=_reason(data))
        return jsonify({"message": "Advanced to next step.", **orchestrator.get_status()})

    @bp.route("/demo/status", methods=["GET"])
    def status():
        return jsonify(orchestrator.get_status())

    @bp.route("/demo/qa", methods=["POST"])
    def qa_start():
        data = request.get_json(silent=True) or {}
        orchestrator.qa_interrupt(
            message=data.get("message", ""),
            source="operator",
            reason=_reason(data),
        )
        return jsonify({"message": "Q&A window opened.", **orchestrator.get_status()})

    @bp.route("/demo/qa_end", methods=["POST"])
    def qa_stop():
        data = request.get_json(silent=True) or {}
        orchestrator.qa_end(source="operator", reason=_reason(data))
        return jsonify({"message": "Q&A window closed.", **orchestrator.get_status()})

    @bp.route("/demo/revise", methods=["POST"])
    def revise():
        """
        Change the remaining tour.

        Body: {"ops": [{"kind": "skip", "robot_id": "navel_001"}, ...],
               "reason": "visitors short on time"}

        Ops that do not apply — naming a robot whose part is already over, for
        instance — come back under "ignored" with a reason rather than as an
        error. Half a revision applying is the correct outcome when a visitor
        asks for something that is partly already true.
        """
        data = request.get_json(silent=True) or {}
        raw_ops = data.get("ops")
        if not isinstance(raw_ops, list) or not raw_ops:
            return jsonify({"error": "ops must be a non-empty list."}), 400

        try:
            ops = [PlanOp.from_payload(o) for o in raw_ops]
        except (ValueError, AttributeError, TypeError) as e:
            return jsonify({"error": str(e)}), 400

        result = orchestrator.revise_script(
            ops, source="operator", reason=_reason(data)
        )
        return jsonify({
            "message": f"Applied {len(result['applied'])} of {len(ops)} op(s).",
            "revision": result,
            **orchestrator.get_status(),
        })

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
