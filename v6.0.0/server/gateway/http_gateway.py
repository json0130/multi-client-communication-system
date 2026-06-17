"""
gateway/http_gateway.py
========================
All Flask HTTP routes.

Two categories:
  1. Management routes  — used by the web UI (register robot, assign role/tags,
                          connect/disconnect, view status)
  2. Robot routes       — used to trigger actions on a connected robot
                          (send a chat message server-side, get health)

The gateway is intentionally thin:
  - Validate input
  - Call registry or ws_gateway
  - Return JSON

No business logic lives here.
"""

from __future__ import annotations
import re
import time
from flask import Flask, request, jsonify, Blueprint

from robot.robot_registry import RobotRegistry
from gateway.websocket_gateway import WebSocketGateway, _looks_like_question
from data import robot_repo


def create_http_gateway(
    registry: RobotRegistry,
    ws_gateway: WebSocketGateway,
) -> Blueprint:
    """
    Returns a Flask Blueprint with all routes attached.
    Registered onto the Flask app in app.py.
    """
    bp = Blueprint("api", __name__)

    # ── Health ────────────────────────────────────────────────────────────────

    @bp.route("/", methods=["GET"])
    def root():
        return jsonify({
            "service": "Robot Management Server",
            "status": "running",
            "connected_robots": ws_gateway.get_connected_ids(),
            "timestamp": time.time(),
        })

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "timestamp": time.time()})

    # ── Robot management (web UI) ─────────────────────────────────────────────

    @bp.route("/robots", methods=["GET"])
    def list_robots():
        """List all robots registered in the DB with their connection status."""
        connected = set(ws_gateway.get_connected_ids())
        robots = robot_repo.get_all_active_robots()

        # Also get all robots (active or not) — get_all_active_robots only
        # returns is_active=True, so we query all here for the management view
        from data.connection import get_client
        try:
            resp = get_client().table("robots").select("*").execute()
            all_rows = resp.data or []
        except Exception:
            all_rows = []

        return jsonify({
            "robots": [
                {
                    "client_id": r["client_id"],
                    "robot_name": r.get("robot_name"),
                    "robot_role": r.get("robot_role"),
                    "allowed_tags": r.get("allowed_tags", []),
                    "modules": r.get("modules", []),
                    "ip_address": r.get("ip_address"),
                    "ws_port": r.get("ws_port"),
                    "ws_connected": r["client_id"] in connected,
                    "is_active": r.get("is_active", False),
                }
                for r in all_rows
            ]
        })

    @bp.route("/robots/register", methods=["POST"])
    def register_robot():
        """
        Register a new robot or update an existing one.
        Called from the web UI when setting up a robot for the first time.

        Body:
        {
            "client_id": "chatbox_01",
            "robot_name": "ChatBox",
            "robot_role": "You are a friendly front-desk assistant.",
            "allowed_tags": ["[WAVE]", "[HAPPY]", "[DEFAULT]"],
            "modules": ["gpt", "speech", "emotion", "rag"],
            "ip_address": "192.168.1.50",
            "ws_port": 8765
        }
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        required = ["client_id", "robot_name"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        robot = robot_repo.upsert_robot(
            client_id=data["client_id"],
            robot_name=data["robot_name"],
            robot_role=data.get("robot_role", "You are a helpful robot."),
            allowed_tags=data.get("allowed_tags", ["[DEFAULT]"]),
            modules=data.get("modules", ["gpt"]),
            ip_address=data.get("ip_address"),
            ws_port=data.get("ws_port"),
        )

        if not robot:
            return jsonify({"error": "Failed to register robot"}), 500

        return jsonify({
            "success": True,
            "client_id": robot.client_id,
            "robot_name": robot.robot_name,
            "message": f"Robot '{robot.robot_name}' registered successfully.",
        })

    @bp.route("/robots/<client_id>", methods=["PUT", "DELETE"])
    def manage_robot(client_id: str):
        """
        PUT    — update name, IP, port, or modules.
        DELETE — disconnect then permanently delete from the database.
        """
        if request.method == "DELETE":
            ws_gateway.disconnect_robot(client_id)
            ok = robot_repo.delete_robot(client_id)
            if not ok:
                return jsonify({"error": "Delete failed"}), 500
            return jsonify({"success": True, "message": f"'{client_id}' deleted."})

        # PUT
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        allowed = {"robot_name", "ip_address", "ws_port", "modules"}
        fields = {k: v for k, v in data.items() if k in allowed}
        if not fields:
            return jsonify({"error": "No valid fields provided"}), 400
        ok = robot_repo.update_robot(client_id, **fields)
        if not ok:
            return jsonify({"error": "Update failed"}), 500
        return jsonify({"success": True, "message": f"'{client_id}' updated."})

    @bp.route("/robots/<client_id>/role", methods=["PUT"])
    def update_role(client_id: str):
        """
        Update a robot's role and/or allowed tags from the web UI.
        Changes take effect on the robot's NEXT chat message (live refresh).

        Body: { "robot_role": "...", "allowed_tags": ["[DEFAULT]", "[WAVE]"] }
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        ok = robot_repo.update_role_and_tags(
            client_id,
            robot_role=data.get("robot_role"),
            allowed_tags=data.get("allowed_tags"),
        )
        if not ok:
            return jsonify({"error": "Update failed"}), 500

        return jsonify({
            "success": True,
            "message": f"Role/tags updated for '{client_id}'. "
                       "Takes effect on next chat.",
        })

    # ── Connection management ─────────────────────────────────────────────────

    @bp.route("/robots/<client_id>/connect", methods=["POST"])
    def connect_robot(client_id: str):
        """
        Tell the server to open a WebSocket connection to this robot.
        The robot must already be registered with an ip_address and ws_port.
        """
        ok = ws_gateway.connect_robot(client_id)
        if not ok:
            return jsonify({
                "error": f"Could not connect to '{client_id}'. "
                         "Check ip_address and ws_port are set in the DB."
            }), 400

        return jsonify({
            "success": True,
            "message": f"Connection initiated to '{client_id}'.",
        })

    @bp.route("/robots/<client_id>/disconnect", methods=["POST"])
    def disconnect_robot(client_id: str):
        """Close the server's WebSocket connection to a robot."""
        ws_gateway.disconnect_robot(client_id)
        return jsonify({
            "success": True,
            "message": f"'{client_id}' disconnected.",
        })

    # ── Robot actions ─────────────────────────────────────────────────────────

    @bp.route("/robots/<client_id>/health", methods=["GET"])
    def robot_health(client_id: str):
        """Get health/status of a connected robot's modules."""
        instance = registry.get(client_id)
        if not instance:
            return jsonify({
                "error": f"'{client_id}' is not connected.",
                "connected_robots": ws_gateway.get_connected_ids(),
            }), 404
        return jsonify(instance.get_health())

    @bp.route("/robots/<client_id>/chat", methods=["POST"])
    def robot_chat(client_id: str):
        """Send a chat message to a robot from the server dashboard."""
        instance = registry.get(client_id)
        if not instance:
            return jsonify({"error": f"'{client_id}' is not connected."}), 404

        data = request.get_json()
        message = (data or {}).get("message", "").strip()
        if not message:
            return jsonify({"error": "message field required"}), 400

        # Stop any in-progress TTS immediately — user talking = robot listens
        if not any(p in message.lower() for p in ws_gateway._QA_ADVANCE_PHRASES):
            ws_gateway.send_to_robot(client_id, {"event": "tts_stop"})
            # Also pause the demo if it was running
            if ws_gateway._demo_orchestrator:
                status = ws_gateway._demo_orchestrator.get_status()
                if status["state"] in ("running", "waiting_ack"):
                    ws_gateway._demo_orchestrator.qa_interrupt()

        ws_gateway.check_qa_advance_from_user(message)

        # Q&A intent classification — skip classifier for obvious questions
        if ws_gateway._demo_orchestrator:
            if ws_gateway._demo_orchestrator.get_status()["state"] == "qa_window":
                if not any(p in message.lower() for p in ws_gateway._QA_ADVANCE_PHRASES):
                    if _looks_like_question(message):
                        print(f"[QA Router] '{message[:60]}' → question detected → full chat")
                    else:
                        print(f"[QA Router] '{message[:60]}' → calling LLM classifier...")
                        intent = instance.classify_qa_intent(message)
                        if intent == "done":
                            print(f"[QA Router] Classifier → 'done' → advancing demo")
                            ws_gateway._demo_orchestrator.qa_end()
                            ws_gateway.send_to_robot(client_id, {
                                "event": "demo_step",
                                "step_id": "_qa_classifier_done",
                                "text": "[DEFAULT] Great! Let's continue with the demonstration then!",
                                "require_ack": False,
                            })
                            return jsonify({
                                "client_id": client_id,
                                "response": "Resuming demonstration.",
                                "emotion_tag": "",
                                "clean_text": "Resuming demonstration.",
                                "is_delegation": False,
                                "delegation_target": None,
                            })
                        else:
                            print(f"[QA Router] Classifier → 'continue' → full chat")

        def _on_sentence(clean_text, emotion_tag):
            if '```' in clean_text:  # Skip delegation JSON blocks — never speak raw JSON
                return
            ws_gateway.send_to_robot(client_id, {
                "event": "chat_sentence",
                "text": clean_text,
                "emotion_tag": emotion_tag,
            })

        result = instance.process_chat_stream(message, _on_sentence)

        # Send "more questions?" if still in QA window after response
        if ws_gateway._demo_orchestrator:
            if ws_gateway._demo_orchestrator.get_status()["state"] == "qa_window":
                ws_gateway.send_to_robot(client_id, {
                    "event": "demo_step",
                    "step_id": "_qa_more_questions",
                    "text": "[DEFAULT] Do you have any other questions, or shall we continue the demonstration?",
                    "require_ack": False,
                })

        # Handle delegation — run synchronously so the browser gets the target's answer
        delegation_result = None
        if result.is_delegation and result.delegation_target:
            from gateway.delegation_handler import DelegationHandler
            handler = DelegationHandler(registry, ws_gateway)
            target_id_del, task_del = handler._extract(result.response)
            if target_id_del and task_del:
                delegation_result = handler.execute_sync(client_id, target_id_del, task_del)

        # Strip internal ```json delegation block from what the browser displays
        clean_for_browser = re.sub(r'```(?:json)?\s*[\s\S]*?```', '', result.clean_text or '').strip()

        return jsonify({
            "client_id":         client_id,
            "response":          result.response,
            "emotion_tag":       result.emotion_tag,
            "clean_text":        clean_for_browser or result.clean_text,
            "is_delegation":     result.is_delegation,
            "delegation_target": result.delegation_target,
            "delegation_result": delegation_result,
        })

    # ── CORS ──────────────────────────────────────────────────────────────────

    @bp.after_request
    def add_cors(response):
        from core.config import cfg
        response.headers["Access-Control-Allow-Origin"] = cfg.server.cors_origins
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        return response

    @bp.route("/<path:path>", methods=["OPTIONS"])
    def options_handler(path):
        return jsonify({}), 200

    return bp