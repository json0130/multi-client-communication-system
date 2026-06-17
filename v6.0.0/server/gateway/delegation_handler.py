"""
gateway/delegation_handler.py
==============================
Detects a delegation JSON block inside an LLM response and executes it.

Flow:
  1. Robot A's LLM returns a response containing a ```json block
  2. DelegationHandler extracts the target_robot_id and task
  3. Sends the task to Robot B via the WebSocket gateway
  4. Robot B processes it in execution mode and its response is
     sent back to Robot B's physical robot via WebSocket

This runs in a background thread so Robot A's HTTP response
returns immediately without waiting for Robot B to finish.
"""

from __future__ import annotations
import json
import re
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from robot.robot_registry import RobotRegistry
    from gateway.websocket_gateway import WebSocketGateway


class DelegationHandler:

    def __init__(self, registry: "RobotRegistry", ws_gateway: "WebSocketGateway"):
        self._registry = registry
        self._ws = ws_gateway

    # ── Public API ────────────────────────────────────────────────────────────

    def handle(self, source_id: str, response_text: str) -> bool:
        """
        Check response_text for a delegation block.
        If found, execute the delegation in a background thread.
        Returns True if a delegation was detected and launched.
        """
        target_id, task = self._extract(response_text)
        if not target_id or not task:
            return False

        print(f"[Delegation] {source_id} → {target_id}: '{task}'")
        threading.Thread(
            target=self._execute,
            args=(source_id, target_id, task),
            daemon=True,
        ).start()
        return True

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Pull target_robot_id and task out of a ```json block.
        Returns (None, None) if no valid block found.
        """
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            return None, None
        try:
            raw = match.group(1).strip()
            # Fix double-brace hallucination {{ }} → { }
            if raw.startswith("{{") and raw.endswith("}}"):
                raw = "{" + raw[2:-2] + "}"
            data = json.loads(raw)
            target = data.get("target_robot_id")
            task = data.get("task")
            if target and task:
                return target, task
        except json.JSONDecodeError as e:
            print(f"[Delegation] JSON parse error: {e}")
        return None, None

    def execute_sync(self, source_id: str, target_id: str, task: str) -> "dict | None":
        """
        Run delegation synchronously in the calling thread.
        Sends verbal handoff to source robot, then sends result to target robot's WebSocket.
        Returns {"robot_name": str, "clean_text": str} or None on failure.
        """
        target = self._registry.get(target_id)
        if not target:
            print(f"[Delegation] Target '{target_id}' not connected — cannot delegate.")
            return None
        try:
            robot_name = target.robot_name or target_id

            # Verbal handoff: source robot (Pepper) addresses the target out loud
            verbal_address = f"{robot_name}, {task}"
            self._ws.send_to_robot(source_id, {
                "event": "chat_sentence",
                "text": verbal_address,
                "emotion_tag": "[DEFAULT]",
            })

            result = target.process_chat(task, is_delegated=True)
            print(f"[Delegation] {target_id} response: {result.response}")
            self._ws.send_to_robot(target_id, {
                "event": "chat_response",
                "response": result.response,
                "emotion_tag": result.emotion_tag,
                "clean_text": result.clean_text,
            })
            return {
                "robot_name": robot_name,
                "clean_text": result.clean_text or result.response,
            }
        except Exception as e:
            print(f"[Delegation] Execution error for {target_id}: {e}")
            return None

    def _execute(self, source_id: str, target_id: str, task: str):
        """
        Run in background thread (used by WebSocket gateway path).
        1. Get target robot instance
        2. Process the task in execution mode
        3. Push the response to the target robot via WebSocket
        """
        self.execute_sync(source_id, target_id, task)