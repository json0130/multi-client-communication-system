"""
tests/test_write_token.py
=========================
With API_TOKEN set, every request that changes something needs the header;
reads stay open. Without it, nothing is enforced (existing setups keep working).
"""

from __future__ import annotations

from flask import Flask, jsonify

import app as server_app


def _client(token):
    app = Flask(__name__)
    server_app.install_write_token(app, token)

    @app.route("/thing", methods=["GET", "POST", "PUT", "DELETE"])
    def thing():
        return jsonify({"ok": True})

    return app.test_client()


def test_a_write_without_the_token_is_refused():
    c = _client("s3cret")
    for method in (c.post, c.put, c.delete):
        assert method("/thing").status_code == 401


def test_a_write_with_the_wrong_token_is_refused():
    c = _client("s3cret")
    assert c.post("/thing", headers={"X-API-Token": "guess"}).status_code == 401


def test_a_write_with_the_token_goes_through():
    c = _client("s3cret")
    assert c.post("/thing", headers={"X-API-Token": "s3cret"}).status_code == 200


def test_reads_stay_open():
    assert _client("s3cret").get("/thing").status_code == 200


def test_no_token_configured_enforces_nothing():
    assert _client("").post("/thing").status_code == 200
