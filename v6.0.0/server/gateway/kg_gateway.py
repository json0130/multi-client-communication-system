"""
gateway/kg_gateway.py
=====================
HTTP endpoints for the robot→topic competence graph.

    GET  /kg/graph          all edges, with confidence/clamped precomputed
    GET  /kg/graph/<robot>  one robot's edges
    GET  /kg/topics         the vocabulary and its topic-topic links
    GET  /kg/summary        size of the graph and the human/outcome split
    POST /kg/seed           build the vocabulary from robot roles + projects
    POST /kg/observe        fold one observation into an edge

Thin, like the other gateways: validate, call the repo, return JSON. The update
arithmetic is in decision/kg.py and is not touched here.
"""

from flask import Blueprint, jsonify, request

from data import demo_kg_repo as repo
from decision.kg import Evidence


def _resolve_topic(utterance: str):
    """Best-matching topic for a question, or None. Shares KGRouter's resolver so
    the topic a correction is filed against is the same one routing used."""
    try:
        from decision.kg_policy import KGRouter
        topics = repo.all_topics()
        if not topics:
            return None
        return KGRouter([], [], topics).resolve_topic(utterance)
    except Exception as e:
        print(f"[kg_gateway] topic resolution failed: {e}")
        return None


def create_kg_gateway(registry=None, ws_gateway=None) -> Blueprint:
    bp = Blueprint("kg", __name__)

    @bp.route("/kg/graph", methods=["GET"])
    @bp.route("/kg/graph/<robot_id>", methods=["GET"])
    def graph(robot_id: str = None):
        return jsonify({"edges": repo.graph(robot_id)})

    @bp.route("/kg/topics", methods=["GET"])
    def topics():
        return jsonify({"topics": repo.all_topics(), "links": repo.all_links()})

    @bp.route("/kg/summary", methods=["GET"])
    def summary():
        return jsonify(repo.summary())

    @bp.route("/kg/observe", methods=["POST"])
    def observe():
        """
        Body: {robot_id, topic_id, target: 0..1, kind: supervisor|outcome}

        `target` is what this observation claims the weight should be — 1.0 for
        "this robot should have taken that question", 0.0 for "it should not".
        The edge moves partway toward it, never all the way.
        """
        data = request.get_json(silent=True) or {}
        robot_id = (data.get("robot_id") or "").strip()
        topic_id = (data.get("topic_id") or "").strip()
        if not robot_id or not topic_id:
            return jsonify({"error": "robot_id and topic_id are required."}), 400
        try:
            target = float(data.get("target"))
        except (TypeError, ValueError):
            return jsonify({"error": "target must be a number in [0,1]."}), 400
        try:
            kind = Evidence(str(data.get("kind") or "supervisor").lower())
        except ValueError:
            return jsonify({"error": "kind must be 'supervisor' or 'outcome'."}), 400

        edge = repo.apply_observation(robot_id, topic_id, target, kind)
        if edge is None:
            return jsonify({"error": "Could not write the edge — see server log."}), 500
        return jsonify({
            "robot_id": edge.robot_id, "topic_id": edge.topic_id,
            "weight": round(edge.weight, 4),
            "n_supervisor": edge.n_supervisor, "n_outcome": edge.n_outcome,
            "n_obs": edge.n_obs,
            "confidence": round(edge.confidence, 4),
            "clamped": round(edge.clamped, 4),
        })

    @bp.route("/kg/reroute", methods=["POST"])
    def reroute():
        """
        An operator sent a question to a different robot. THIS CLOSES THE LOOP.

        Body: {utterance | topic_id, chosen_robot_id, displaced_robot_id?,
               session_id?, step_id?}

        Writes to two places on purpose. demo_correction_log keeps the audit
        trail — what a person did, when, and to which decision. demo_robot_topic
        is what actually learns. Before this endpoint existed only the first
        happened, so a campaign of rollouts changed no weight at all.

        Resolving the topic is required and is allowed to fail: an observation
        filed against the wrong topic is worse than none, because afterwards it
        is indistinguishable from a real one.
        """
        from decision.kg_feedback import apply, from_reroute

        data = request.get_json(silent=True) or {}
        chosen = (data.get("chosen_robot_id") or "").strip()
        if not chosen:
            return jsonify({"error": "chosen_robot_id is required."}), 400

        topic_id = (data.get("topic_id") or "").strip() or None
        utterance = (data.get("utterance") or "").strip()
        if not topic_id and utterance:
            topic_id = _resolve_topic(utterance)
        if not topic_id:
            return jsonify({
                "applied": [], "topic_id": None,
                "message": "No topic resolved — correction logged, graph unchanged.",
            })

        # Who DECLARES this topic as their subject. Changes what a
        # displacement means: routing away from a specialist to someone
        # outside its scope is a statement about the subject or about
        # availability, not a judgement on how well the specialist handles
        # its own area — see kg_feedback.from_reroute.
        try:
            specialists = {
                e["robot_id"] for e in repo.graph()
                if e["topic_id"] == topic_id and e.get("specialised")
            }
        except Exception as e:
            print(f"[kg_gateway] could not read declared scope: {e}")
            specialists = set()

        observations = from_reroute(topic_id, chosen,
                                    (data.get("displaced_robot_id") or "").strip() or None,
                                    specialists=specialists)
        edges = apply(observations, repo)

        # Suppress this segment's outcome observations. Without it a corrected
        # window would ALSO emit weak positives at close, counting one event
        # twice and inflating n_obs — the exact thing the correction/outcome
        # split exists to keep apart.
        if ws_gateway is not None:
            try:
                ws_gateway.note_routing_correction()
            except Exception as e:
                print(f"[kg_gateway] could not mark the segment corrected: {e}")
        return jsonify({
            "topic_id": topic_id,
            "applied": [{"robot_id": e.robot_id, "weight": round(e.weight, 4),
                         "n_obs": e.n_obs, "clamped": round(e.clamped, 4)}
                        for e in edges],
            "requested": [o.as_row() for o in observations],
        })

    @bp.route("/kg/style", methods=["GET", "POST"])
    def style_fit():
        """
        How well a robot pitched to this audience.

        GET  -> every (robot, style) row with confidence/clamped precomputed.
        POST {robot_id, style, target: 0..1}
             target 1.0 = well pitched for that audience, 0.0 = not.

        SUPERVISOR JUDGEMENTS ONLY. There is deliberately no automatic path
        into this table: a Q&A window closing cleanly means nobody objected
        to the ROUTING and says nothing about how the answer was pitched, so
        crediting style fit for it would manufacture a judgement nobody made.
        It moves when a person says the pitch was good or bad, or not at all.

        Never affects routing — see decision/style_fit.py. It changes the
        framing directive the robot is briefed with at generation time.
        """
        from data import demo_style_repo

        if request.method == "GET":
            return jsonify({"fits": demo_style_repo.all_fits()})

        data = request.get_json(silent=True) or {}
        robot_id = (data.get("robot_id") or "").strip()
        style = (data.get("style") or "").strip().lower()
        if not robot_id or not style:
            return jsonify({"error": "robot_id and style are required."}), 400

        from decision.visitor_profile import STYLE_FRAMING
        if style not in STYLE_FRAMING:
            return jsonify({
                "error": f"style must be one of: {', '.join(STYLE_FRAMING)}",
            }), 400
        try:
            target = float(data.get("target"))
        except (TypeError, ValueError):
            return jsonify({"error": "target must be a number in [0,1]."}), 400

        fit = demo_style_repo.record(robot_id, style, target)
        if fit is None:
            return jsonify({"error": "Could not write — see server log."}), 500
        return jsonify({
            "robot_id": fit.robot_id, "style": fit.style,
            "weight": round(fit.weight, 4), "n_supervisor": fit.n_supervisor,
            "confidence": round(fit.confidence, 4),
            "clamped": round(fit.clamped, 4),
            # What this actually changes: whether the robot gets the extra
            # corrective note on top of the ordinary directive.
            "reinforced": fit.needs_reinforcement,
        })

    @bp.route("/kg/seed", methods=["POST"])
    def seed():
        """
        Build the topic vocabulary and its similarity links.

        Separate endpoint rather than something that happens at boot: seeding
        writes to the graph, and a demo server restarting should not silently
        redefine the vocabulary the learned weights are attached to.
        """
        from tools.seed_kg import seed_from_db
        try:
            return jsonify(seed_from_db(dry_run=bool(
                (request.get_json(silent=True) or {}).get("dry_run"))))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route("/kg/<path:path>", methods=["OPTIONS"])
    @bp.route("/kg", methods=["OPTIONS"])
    def options(path=""):
        from flask import make_response
        r = make_response()
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r

    return bp
