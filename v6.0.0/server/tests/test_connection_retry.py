"""
tests/test_connection_retry.py
===============================
data.connection._RetryingTransport — regression for real data loss.

A live demo run had a Supabase write fail with "Server disconnected without
sending a response" (httpx.RemoteProtocolError from a stale pooled
connection) and the observation was gone for good — the caller's best-effort
error handling swallowed it, correctly for demo continuity, but with nothing
downstream to notice. This proves the transport retries exactly once on that
specific failure and lets a second failure propagate rather than looping
forever or masking a real, sustained outage.
"""

from __future__ import annotations

import httpx
import pytest

from data.connection import _RetryingTransport


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.supabase.co/rest/v1/demo_robot_topic",
                          json={"robot_id": "chatbox_01", "topic_id": "topic:llm"})


class TestRetryOnDroppedConnection:
    def test_retries_once_and_returns_the_second_attempt(self, monkeypatch):
        calls = {"n": 0}
        ok_response = httpx.Response(200, request=_request())

        def fake_handle(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response")
            return ok_response

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
        transport = _RetryingTransport()

        result = transport.handle_request(_request())

        assert result is ok_response
        assert calls["n"] == 2

    def test_a_second_consecutive_failure_propagates(self, monkeypatch):
        # A real, sustained outage must not be swallowed forever — one retry
        # is for a stale connection, not a substitute for surfacing a real
        # failure to the caller's existing best-effort handling.
        calls = {"n": 0}

        def fake_handle(self, request):
            calls["n"] += 1
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
        transport = _RetryingTransport()

        with pytest.raises(httpx.RemoteProtocolError):
            transport.handle_request(_request())

        assert calls["n"] == 2

    def test_a_clean_success_does_not_retry(self, monkeypatch):
        calls = {"n": 0}
        ok_response = httpx.Response(200, request=_request())

        def fake_handle(self, request):
            calls["n"] += 1
            return ok_response

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
        transport = _RetryingTransport()

        result = transport.handle_request(_request())

        assert result is ok_response
        assert calls["n"] == 1

    def test_a_different_error_is_not_retried(self, monkeypatch):
        # Only the specific "connection was already dead" failure is worth a
        # blind retry. A different transport error (bad TLS, DNS failure) is
        # not fixed by trying again immediately and should surface as-is.
        calls = {"n": 0}

        def fake_handle(self, request):
            calls["n"] += 1
            raise httpx.ConnectError("Name or service not known")

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
        transport = _RetryingTransport()

        with pytest.raises(httpx.ConnectError):
            transport.handle_request(_request())

        assert calls["n"] == 1


class TestRequestBodyIsSafeToResend:
    """The retry resends the same Request object — this pins the assumption
    that makes that safe: postgrest's JSON bodies are fully-buffered bytes,
    not a one-shot stream, so reading them twice yields the same content."""

    def test_json_request_stream_is_repeatable(self):
        req = _request()
        first = b"".join(req.stream)
        second = b"".join(req.stream)
        assert first == second
        assert b"chatbox_01" in first
