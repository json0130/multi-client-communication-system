"""
gateway/persona_gateway.py
===========================
HTTP routes for persona management.
Registered as a separate Blueprint and mounted in app.py.

Endpoints:
  GET    /personas                  list all personas
  GET    /personas/<id>             get one persona
  POST   /personas                  create a persona
  PUT    /personas/<id>             update a persona
  DELETE /personas/<id>             delete a persona
  POST   /robots/<id>/persona       assign persona to robot
                                    → updates DB
                                    → pushes live update to robot via WS
"""

from __future__ import annotations
import time
from flask import Blueprint, request, jsonify

from data import persona_repo, robot_repo
from core.config import cfg


def create_persona_gateway(ws_gateway) -> Blueprint:
    bp = Blueprint("personas", __name__)

    # ── Persona CRUD ──────────────────────────────────────────────────────────

    @bp.route("/personas", methods=["GET"])
    def list_personas():
        personas = persona_repo.get_all()
        return jsonify({
            "personas": [_persona_dict(p) for p in personas]
        })

    @bp.route("/personas/<persona_id>", methods=["GET"])
    def get_persona(persona_id: str):
        p = persona_repo.get_by_id(persona_id)
        if not p:
            return jsonify({"error": "Persona not found"}), 404
        return jsonify(_persona_dict(p))

    @bp.route("/personas", methods=["POST"])
    def create_persona():
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        if not data.get("name"):
            return jsonify({"error": "name is required"}), 400

        persona = persona_repo.create(
            name=data["name"],
            description=data.get("description", ""),
            robot_role=data.get("robot_role", "You are a helpful robot."),
            allowed_tags=data.get("allowed_tags", ["[DEFAULT]"]),
            modules=data.get("modules", ["gpt"]),
            voice_config=data.get("voice_config", {}),
            capabilities=data.get("capabilities", {}),
            personality=data.get("personality", {
                "O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5
            }),
            is_default=data.get("is_default", False),
        )

        if not persona:
            return jsonify({"error": "Failed to create persona"}), 500

        return jsonify({
            "success": True,
            "message": f"Persona '{persona.name}' created.",
            "persona": _persona_dict(persona),
        })

    @bp.route("/personas/<persona_id>", methods=["PUT"])
    def update_persona(persona_id: str):
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        # Only update fields that were provided
        allowed_fields = {
            "name", "description", "robot_role", "allowed_tags",
            "modules", "voice_config", "capabilities", "personality", "is_default",
        }
        updates = {k: v for k, v in data.items() if k in allowed_fields}
        if not updates:
            return jsonify({"error": "No valid fields to update"}), 400

        persona = persona_repo.update(persona_id, updates)
        if not persona:
            return jsonify({"error": "Persona not found or update failed"}), 404

        return jsonify({
            "success": True,
            "message": f"Persona '{persona.name}' updated.",
            "persona": _persona_dict(persona),
        })

    @bp.route("/personas/<persona_id>", methods=["DELETE"])
    def delete_persona(persona_id: str):
        p = persona_repo.get_by_id(persona_id)
        if not p:
            return jsonify({"error": "Persona not found"}), 404
        if p.is_default:
            return jsonify({"error": "Cannot delete the default persona"}), 400

        ok = persona_repo.delete(persona_id)
        if not ok:
            return jsonify({"error": "Delete failed"}), 500

        return jsonify({"success": True, "message": f"Persona '{p.name}' deleted."})

    # ── Assign persona to robot ───────────────────────────────────────────────

    @bp.route("/robots/<client_id>/persona", methods=["POST"])
    def assign_persona(client_id: str):
        """
        Assign a persona to a robot.

        1. Validates robot + persona exist
        2. Updates robots.persona_id in Supabase
        3. Updates robots.robot_role + allowed_tags (for existing queries)
        4. Pushes live persona_update event to robot via WebSocket
           so it takes effect immediately without reboot

        Body: { "persona_id": "<uuid>" }
        """
        data = request.get_json()
        persona_id = (data or {}).get("persona_id")
        if not persona_id:
            return jsonify({"error": "persona_id required"}), 400

        # Validate robot exists
        robot = robot_repo.get_robot(client_id)
        if not robot:
            return jsonify({"error": f"Robot '{client_id}' not found"}), 404

        # Validate persona exists
        persona = persona_repo.get_by_id(persona_id)
        if not persona:
            return jsonify({"error": f"Persona '{persona_id}' not found"}), 404

        # 1. Update Supabase — store persona_id + sync role/tags/modules
        from data.connection import get_client as db
        try:
            db().table("robots").update({
                "persona_id":  persona_id,
                "robot_role":  persona.robot_role,
                "allowed_tags": persona.allowed_tags,
                "modules":     persona.modules,
            }).eq("client_id", client_id).execute()
        except Exception as e:
            return jsonify({"error": f"DB update failed: {e}"}), 500

        # 2. Push live update to robot via WebSocket (if connected)
        ws_connected = False
        if ws_gateway and client_id in ws_gateway.get_connected_ids():
            ws_gateway.send_to_robot(client_id, {
                "event": "persona_update",
                "persona_id":   persona_id,
                "persona_name": persona.name,
                "robot_role":   persona.robot_role,
                "allowed_tags": persona.allowed_tags,
                "modules":      persona.modules,
                "voice_config": persona.voice_config,
                "capabilities": persona.capabilities,
                "personality":  persona.personality,
            })
            ws_connected = True

        return jsonify({
            "success": True,
            "message": (
                f"Persona '{persona.name}' assigned to '{client_id}'."
                + (" Live update sent." if ws_connected else
                   " Robot offline — will apply on next connect.")
            ),
            "client_id":    client_id,
            "persona_id":   persona_id,
            "persona_name": persona.name,
            "live_updated": ws_connected,
        })

    # ── CORS ─────────────────────────────────────────────────────────────────

    @bp.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        return response

    @bp.route("/personas", methods=["OPTIONS"])
    @bp.route("/personas/<path:path>", methods=["OPTIONS"])
    @bp.route("/robots/<path:path>/persona", methods=["OPTIONS"])
    def options_handler(path=""):
        return jsonify({}), 200

    return bp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _persona_dict(p) -> dict:
    return {
        "id":           p.id,
        "name":         p.name,
        "description":  p.description,
        "robot_role":   p.robot_role,
        "allowed_tags": p.allowed_tags,
        "modules":      p.modules,
        "voice_config": p.voice_config,
        "capabilities": p.capabilities,
        "personality":  p.personality,
        "is_default":   p.is_default,
        "created_at":   p.created_at,
    }