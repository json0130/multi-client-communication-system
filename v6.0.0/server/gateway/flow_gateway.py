"""
gateway/flow_gateway.py
========================
HTTP endpoints for inspecting the flow graph and previewing the planner.

    GET  /flow/status    the remaining tour as a FlowGraph, no cuts applied
    POST /flow/plan       what the planner would do to fit a stated budget

Both are READ-ONLY — neither calls revise_script(). This is a preview surface:
it lets an operator or a developer ask "what would the planner do here?"
without a visitor having to say "we're running out of time" during a live
tour. Use POST /demo/revise, or a real PLAN_REVISE trigger, to actually change
a running demo.

Works from whatever is CURRENTLY loaded on the orchestrator: before a demo
starts that is the pre-loaded script end to end; once one is running it is
genuinely the steps still ahead of the play head, matching the invariant
revise_script() enforces on the write side.
"""

from flask import Blueprint, jsonify, request

from decision.flow import DEFAULT_QA_BUDGET_SEC, FlowGraph, StepRef
from decision.planner import block_importance, plan_for_budget


def _remaining_steps(orchestrator) -> tuple:
    """StepRef tuple for whatever is ahead of the play head right now.

    Before a demo starts, nothing has "happened" yet, so the whole script
    counts as remaining. Once one is running, only the steps after the current
    index do — the same boundary revise_script() enforces on writes.
    """
    status = orchestrator.get_status()
    idx = int(status.get("step_idx") or 0)
    steps = status.get("steps") or []
    live = status.get("state") not in ("idle", "completed", "error")
    start = idx + 1 if live else 0
    return tuple(
        StepRef(step_id=s.get("step_id") or "", robot_id=s.get("robot_id") or "",
               role=s.get("role") or "", qa_window=bool(s.get("qa_window")),
               block_robot_id=s.get("block_robot_id"))
        for s in steps[start:]
    )


def _duration_snapshot() -> dict:
    try:
        from data.demo_duration_repo import step_stats
        return {r["step_id"]: float(r["mean_sec"])
               for r in step_stats() if r.get("mean_sec") is not None}
    except Exception as e:
        print(f"[flow_gateway] duration stats unavailable: {e}")
        return {}


def _kg_snapshot() -> tuple:
    try:
        from data import demo_kg_repo as repo
        from decision.kg import RobotTopicEdge
        topics = repo.all_topics()
        if not topics:
            return [], [], []
        edges = [RobotTopicEdge.from_row(r) for r in repo.graph()]
        links = [(l["topic_a"], l["topic_b"], float(l["weight"])) for l in repo.all_links()]
        return topics, edges, links
    except Exception as e:
        print(f"[flow_gateway] KG snapshot unavailable: {e}")
        return [], [], []


def create_flow_gateway(orchestrator) -> Blueprint:
    """Factory — pass the DemoOrchestrator instance, same pattern as demo_gateway."""
    bp = Blueprint("flow", __name__)

    @bp.route("/flow/status", methods=["GET"])
    def status():
        durations = _duration_snapshot()
        raw = orchestrator.get_status()
        graph = FlowGraph.from_script(_remaining_steps(orchestrator))

        def step_row(s):
            return {"step_id": s.step_id, "role": s.role,
                   "compressible": s.compressible, "qa_window": s.qa_window}

        return jsonify({
            # Full step detail for opening/closing too, not just counts — the
            # flowchart needs real boxes for these, not a number.
            "opening": [step_row(s) for s in graph.opening],
            "closing": [step_row(s) for s in graph.closing],
            "blocks": [
                {
                    "robot_id": b.robot_id,
                    "steps": [step_row(s) for s in b.steps],
                    "scripted_sec": round(b.scripted_seconds(durations), 1),
                    "compression_saving_sec": round(b.compression_saving(durations), 1),
                    "qa_windows": len(b.qa_steps),
                }
                for b in graph.blocks
            ],
            "opening_steps": len(graph.opening),
            "closing_steps": len(graph.closing),
            "fixed_sec": round(graph.fixed_seconds(durations), 1),
            "estimate_no_cuts": graph.estimate(durations, qa_budget=DEFAULT_QA_BUDGET_SEC),
            # An estimate built entirely from DEFAULT_STEP_SEC is arithmetic, not
            # a prediction — surfaced so the UI can say so.
            "measured_coverage": round(graph.measured_coverage(durations), 3),
            # What's actually running right now, so the UI can highlight it.
            "current_step_id": raw.get("step_id"),
            "current_state": raw.get("state"),
            # The WHOLE tour, completed steps included — for the flowchart.
            # `blocks`/`opening`/`closing` above are deliberately remaining-only
            # (the planner's view: only what a cut could still touch); a
            # flowchart showing "here is the tour and here is where we are"
            # needs the full sequence, so it is a separate field rather than
            # overloading the planner's.
            "full_steps": [
                {"step_id": st.get("step_id"), "robot_id": st.get("robot_id"),
                 "role": st.get("role") or "", "block_robot_id": st.get("block_robot_id"),
                 "qa_window": bool(st.get("qa_window")),
                 "is_current": i == int(raw.get("step_idx") or 0),
                 "is_completed": i < int(raw.get("step_idx") or 0)}
                for i, st in enumerate(raw.get("steps") or [])
            ],
        })

    @bp.route("/flow/plan", methods=["POST"])
    def plan():
        """
        Body: {budget_sec, interest?: str, importance?: {robot_id: 0..1}}

        `interest` resolves through the same topic matcher QA_ROUTE uses, off
        the same graph snapshot — so previewing "what would the planner do for
        someone interested in X" here exercises the identical resolution path a
        live visitor turn would.
        """
        data = request.get_json(silent=True) or {}
        try:
            budget_sec = float(data.get("budget_sec"))
        except (TypeError, ValueError):
            return jsonify({"error": "budget_sec must be a number of seconds."}), 400
        if budget_sec <= 0:
            return jsonify({"error": "budget_sec must be positive."}), 400

        graph = FlowGraph.from_script(_remaining_steps(orchestrator))
        if not graph.blocks:
            return jsonify({"error": "No project blocks in the remaining script "
                                     "— load or start a demo first."}), 400

        durations = _duration_snapshot()
        topics, edges, links = _kg_snapshot()
        visitor_topics = None
        interest = (data.get("interest") or "").strip()
        if interest and topics:
            from decision.kg_policy import KGRouter
            tid = KGRouter([], [], topics).resolve_topic(interest)
            visitor_topics = [tid] if tid else None

        importance = block_importance(
            graph, defaults=data.get("importance") or {},
            visitor_topics=visitor_topics, kg_edges=edges, kg_links=links)

        result = plan_for_budget(graph, budget_sec, durations=durations,
                                 importance=importance)
        return jsonify({
            "feasible": result["feasible"],
            "fits_already": result["fits_already"],
            "ops": [o.payload() for o in result["ops"]],
            "estimate": result["estimate"],
            "trace": result["trace"],
            "measured_coverage": result["measured_coverage"],
            "importance": importance,
            "resolved_topic": visitor_topics[0] if visitor_topics else None,
            "interest_resolved": bool(visitor_topics),
        })

    @bp.route("/flow/<path:path>", methods=["OPTIONS"])
    @bp.route("/flow", methods=["OPTIONS"])
    def options(path=""):
        from flask import make_response
        r = make_response()
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r

    return bp
