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

Context Serialization (RBAC)
----------------------------
A hand-off is also where the paper's Context Serialization happens. Before the
target executes, the source robot retrieves context *as itself* — under its own
access level — and issues a short-lived DelegationGrant naming those exact
snippet IDs. The snippets travel with the task and appear only in the target's
temporary execution prompt.

The target's standing access level never changes. The grant is scoped to one
task, expires on a timer, and is revoked when the task completes. Nothing
granted is written into the target's own memory.
"""

from __future__ import annotations
import json
import re
import threading
import uuid
from typing import Optional, Sequence, TYPE_CHECKING

from core.rbac import DelegationGrant, MemoryRecord, new_grant

if TYPE_CHECKING:
    from robot.robot_registry import RobotRegistry
    from gateway.websocket_gateway import WebSocketGateway


# How many context snippets a Manager may serialize into one hand-off.
MAX_DELEGATED_SNIPPETS = 3


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

        Serializes the source robot's context into a short-lived grant so the
        target inherits conversational state without its access level widening.
        """
        target = self._registry.get(target_id)
        if not target:
            print(f"[Delegation] Target '{target_id}' not connected — cannot delegate.")
            return None

        task_id = str(uuid.uuid4())
        snippets: list[MemoryRecord] = []
        try:
            robot_name = target.robot_name or target_id

            # ── Context Serialization ─────────────────────────────────────────
            snippets = self._serialize_context(source_id, target_id, task, task_id)

            # Verbal handoff: source robot (Pepper) addresses the target out
            # loud. Deliberately generic rather than echoing `task` verbatim —
            # task is LLM-generated and can come out stilted ("Navel, Please
            # explain more about your research project."); a fixed natural
            # phrase reads the same way every time. Navel still gets the real
            # `task` text below for what it actually answers.
            self._ws.send_to_robot(source_id, {
                "event": "chat_sentence",
                "text": f"{robot_name}, can you tell us more about that?",
                "emotion_tag": "[DEFAULT]",
            })

            result = target.process_chat(
                task,
                is_delegated=True,
                delegated_context=snippets,
                task_id=task_id,
            )
            print(f"[Delegation] {target_id} response: {result.response}")
            self._ws.send_to_robot(target_id, {
                "event": "chat_response",
                "response": result.response,
                "emotion_tag": result.emotion_tag,
                "clean_text": result.clean_text,
            })
            self._prompt_for_more_questions(target_id)
            return {
                "robot_name": robot_name,
                "clean_text": result.clean_text or result.response,
            }
        except Exception as e:
            print(f"[Delegation] Execution error for {target_id}: {e}")
            return None
        finally:
            # Revoke on task completion, success or failure. Grants also carry
            # their own expiry, so a crash between here and there still closes.
            self._revoke(task_id)

    def _prompt_for_more_questions(self, target_id: str) -> None:
        """
        Ask "any other questions?" from whoever actually just answered, once
        their delegated reply has actually been sent — not from the source
        robot right after it kicks the delegation off.

        execute_sync() runs synchronously for one caller (http_gateway.py)
        and in a background thread for the other (websocket_gateway.py, via
        handle()/_execute()) — a caller-side "resend if still in qa_window"
        check fires BEFORE this method's caller even starts in the async
        case, and in the sync case it used to run earlier in the function,
        before delegation had even been dispatched. A real run had exactly
        this: Pepper got asked "any other questions?" again while Navel's
        answer was still being generated, sent to the wrong robot, before
        the visitor had even heard what they asked for. Living here instead
        means it fires at the one moment that is actually correct regardless
        of which caller invoked it.
        """
        orch = getattr(self._ws, "_demo_orchestrator", None)
        if orch is None or orch.get_status().get("state") != "qa_window":
            return
        self._ws.send_to_robot(target_id, {
            "event": "demo_step",
            "step_id": "_qa_more_questions",
            "text": "[DEFAULT] Do you have any other questions, or shall we continue the demonstration?",
            "require_ack": False,
        })

    # ── Context Serialization ─────────────────────────────────────────────────

    def _serialize_context(
        self, source_id: str, target_id: str, task: str, task_id: str
    ) -> list[MemoryRecord]:
        """
        Retrieve context as the *source* robot, then grant those exact snippets
        to the target for this task only.

        Returns the snippet records to travel with the hand-off. The target
        re-validates them against the grant before using them, so the payload
        alone confers nothing.
        """
        source = self._registry.get(source_id)
        if not source:
            return []
        try:
            cleared = source._get_rag_context(task)[:MAX_DELEGATED_SNIPPETS]
            if not cleared:
                return []

            records = [c.record for c in cleared]
            grant = new_grant(
                snippet_ids=[r.record_id for r in records],
                granted_to=target_id,
                granted_by=source_id,
                task_id=task_id,
                session_id=source.identity.session_id,
            )
            self._grant_store().issue(grant)
            print(
                f"[Delegation] Serialized {len(records)} snippet(s) to {target_id} "
                f"(grant {grant.grant_id[:8]}, task {task_id[:8]})"
            )
            return records
        except Exception as e:
            # Context serialization is an enhancement, not a precondition — a
            # failure here must not abort the hand-off itself.
            print(f"[Delegation] Context serialization failed (continuing): {e}")
            return []

    def _grant_store(self):
        return self._registry.grants

    def _revoke(self, task_id: str) -> None:
        try:
            n = self._grant_store().revoke_task(task_id)
            if n:
                print(f"[Delegation] Revoked {n} grant(s) for task {task_id[:8]}")
        except Exception as e:
            print(f"[Delegation] Grant revoke error (ignored): {e}")

    def _execute(self, source_id: str, target_id: str, task: str):
        """
        Run in background thread (used by WebSocket gateway path).
        1. Get target robot instance
        2. Process the task in execution mode
        3. Push the response to the target robot via WebSocket
        """
        self.execute_sync(source_id, target_id, task)