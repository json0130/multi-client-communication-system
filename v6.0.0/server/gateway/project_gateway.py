"""
gateway/project_gateway.py
===========================
HTTP routes for project management and RDAC (Robot Database Access Control).

Endpoints:
  GET    /projects                    list all projects
  POST   /projects                    create a project
  GET    /projects/<id>               get one project
  PUT    /projects/<id>               update a project
  DELETE /projects/<id>               delete a project
  GET    /projects/for/<robot_id>     RDAC-filtered list for a robot
  POST   /projects/<id>/access        grant robot access  { robot_id }
  DELETE /projects/<id>/access/<rid>  revoke robot access
"""

from __future__ import annotations
from flask import Blueprint, request, jsonify

from data import project_repo


def create_project_gateway() -> Blueprint:
    bp = Blueprint("projects", __name__)

    # ── Project CRUD ──────────────────────────────────────────────────────────

    @bp.route("/projects", methods=["GET"])
    def list_projects():
        return jsonify({"projects": [_dict(p) for p in project_repo.get_all()]})

    @bp.route("/projects/for/<robot_id>", methods=["GET"])
    def projects_for_robot(robot_id: str):
        """RDAC-filtered — returns only projects the robot has access to."""
        projects = project_repo.get_for_robot(robot_id)
        return jsonify({"robot_id": robot_id, "projects": [_dict(p) for p in projects]})

    @bp.route("/projects/<project_id>", methods=["GET"])
    def get_project(project_id: str):
        p = project_repo.get_by_id(project_id)
        if not p:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(_dict(p))

    @bp.route("/projects", methods=["POST"])
    def create_project():
        data = request.get_json()
        if not data or not data.get("name"):
            return jsonify({"error": "name is required"}), 400

        p = project_repo.create(
            name=data["name"],
            description=data.get("description", ""),
            researcher=data.get("researcher", ""),
            robot_id=data.get("robot_id", ""),
            keywords=data.get("keywords", []),
            details=data.get("details", ""),
        )
        if not p:
            return jsonify({"error": "Failed to create project"}), 500

        return jsonify({"success": True, "project": _dict(p)}), 201

    @bp.route("/projects/<project_id>", methods=["PUT"])
    def update_project(project_id: str):
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        allowed = {"name", "description", "researcher", "robot_id", "keywords", "details"}
        updates = {k: v for k, v in data.items() if k in allowed}
        if not updates:
            return jsonify({"error": "No valid fields to update"}), 400

        p = project_repo.update(project_id, updates)
        if not p:
            return jsonify({"error": "Project not found or update failed"}), 404

        return jsonify({"success": True, "project": _dict(p)})

    @bp.route("/projects/<project_id>", methods=["DELETE"])
    def delete_project(project_id: str):
        if not project_repo.get_by_id(project_id):
            return jsonify({"error": "Project not found"}), 404
        ok = project_repo.delete(project_id)
        if not ok:
            return jsonify({"error": "Delete failed"}), 500
        return jsonify({"success": True, "message": f"Project '{project_id}' deleted."})

    # ── RDAC management ───────────────────────────────────────────────────────

    @bp.route("/projects/<project_id>/access", methods=["POST"])
    def grant_access(project_id: str):
        data = request.get_json()
        robot_id = (data or {}).get("robot_id")
        if not robot_id:
            return jsonify({"error": "robot_id required"}), 400

        ok = project_repo.grant_access(robot_id, project_id)
        if not ok:
            return jsonify({"error": "Grant failed"}), 500
        return jsonify({"success": True, "message": f"Granted '{robot_id}' access to '{project_id}'."})

    @bp.route("/projects/<project_id>/access/<robot_id>", methods=["DELETE"])
    def revoke_access(project_id: str, robot_id: str):
        ok = project_repo.revoke_access(robot_id, project_id)
        if not ok:
            return jsonify({"error": "Revoke failed"}), 500
        return jsonify({"success": True, "message": f"Revoked '{robot_id}' access to '{project_id}'."})

    # ── CORS ──────────────────────────────────────────────────────────────────

    @bp.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        return response

    @bp.route("/projects", methods=["OPTIONS"])
    @bp.route("/projects/<path:path>", methods=["OPTIONS"])
    def options_handler(path=""):
        return jsonify({}), 200

    return bp


# ── Helper ────────────────────────────────────────────────────────────────────

def _dict(p) -> dict:
    return {
        "id":          p.id,
        "name":        p.name,
        "description": p.description,
        "researcher":  p.researcher,
        "robot_id":    p.robot_id,
        "keywords":    p.keywords,
        "details":     p.details,
        "created_at":  p.created_at,
    }
