"""
robot/robot_instance.py
========================
One RobotInstance per connected robot.

Responsibilities:
  - Hold references to whichever modules are enabled for this robot
  - Coordinate modules to handle chat, speech, and image frame requests
  - Refresh role/tags/access level from the DB before each chat (so web UI and
    database changes apply live)
  - Log every exchange to Supabase via chat_log_repo

Does NOT:
  - Build prompts (that's prompt_builder)
  - Query Supabase directly (that's the data repos)
  - Know about Flask or WebSockets (that's the gateway layer)
  - Decide access (that's core.rbac.policy; this class only supplies identity)

RBAC
----
Every retrieval this instance performs is filtered by core.rbac before it can
reach prompt assembly. The instance's access level is refreshed from the
database on each chat alongside role and tags, so access hierarchies stay
adjustable at the database level without reconfiguring individual robot nodes.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence, TYPE_CHECKING

from data import robot_repo, user_repo, chat_log_repo
from core.rbac import (
    AccessLevel,
    ClearedRecord,
    DelegationGrant,
    GrantStore,
    MemoryRecord,
    RBACFilter,
    RobotIdentity,
    Visibility,
)
from robot.prompt_builder import build_delegation_prompt, build_execution_prompt

if TYPE_CHECKING:
    from modules.llm.llm_module import LLMModule
    from modules.speech.speech_module import SpeechModule
    from modules.emotion.emotion_module import EmotionModule
    from modules.rag.rag_module import RagModule


@dataclass
class ChatResult:
    response: str           # full raw LLM response (includes [TAG])
    emotion_tag: str        # e.g. "WAVE"
    clean_text: str         # response with tag stripped (for TTS)
    user_emotion: str       # emotion detected from camera (if available)
    is_delegation: bool     # True if response contains a delegation JSON block
    delegation_target: Optional[str] = None   # target robot_id if delegating


@dataclass
class SpeechResult:
    transcription: str
    confidence: float
    chat: Optional[ChatResult] = None   # set if GPT module is also enabled


class RobotInstance:

    def __init__(
        self,
        client_id: str,
        robot_name: str,
        user_id: int,
        enabled_modules: set[str],
        rbac: Optional[RBACFilter] = None,
        grants: Optional[GrantStore] = None,
        access_level: object = AccessLevel.LOCAL,
        scenario_id: Optional[str] = None,
        session_id: Optional[str] = None,
        default_visibility: object = Visibility.LOCAL,
    ):
        self.client_id = client_id
        self.robot_name = robot_name
        self.user_id = user_id
        self.enabled_modules = enabled_modules

        # Module slots — populated by robot_registry during initialization
        self.llm: Optional[LLMModule] = None
        self.speech: Optional[SpeechModule] = None
        self.emotion: Optional[EmotionModule] = None
        self.rag: Optional[RagModule] = None

        # Conversation history (kept in memory per session).
        # This is the temporal buffer. It is per-instance and never shared across
        # robots, so it needs no RBAC filtering — the only path by which context
        # crosses robot instances is a delegation grant (see process_chat).
        self._history: list[dict] = []
        self._max_history = 14  # 7 back-and-forth turns

        # Cached role/tags/access level — refreshed from DB before each chat
        self._robot_role = "You are a helpful robot."
        self._allowed_tags: list[str] = ["[DEFAULT]"]

        # ── RBAC ──────────────────────────────────────────────────────────────
        # Fail closed: with no filter supplied, nothing is retrievable.
        self._rbac = rbac or RBACFilter()
        self._grants = grants or GrantStore()
        self._access_level = access_level
        self._scenario_id = scenario_id
        self._session_id = session_id or f"{client_id}:{int(time.time())}"
        self._default_visibility = default_visibility

        self.created_at = time.time()
        self.last_active = time.time()

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def identity(self) -> RobotIdentity:
        """
        The robot's Social Identity as far as access control is concerned.
        Rebuilt on each access so a DB-level change to access_level takes effect
        without reconnecting the robot.
        """
        return RobotIdentity(
            robot_id=self.client_id,
            scenario_id=self._scenario_id,
            session_id=self._session_id,
            access_level=self._access_level,
            role=self._robot_role,
        )

    @property
    def access_level(self) -> object:
        return self._access_level

    @property
    def rbac(self) -> RBACFilter:
        return self._rbac

    @property
    def grants(self) -> GrantStore:
        return self._grants

    # ── Chat ─────────────────────────────────────────────────────────────────

    def process_chat(
        self,
        message: str,
        is_delegated: bool = False,
        delegated_context: Optional[Sequence[MemoryRecord]] = None,
        task_id: Optional[str] = None,
    ) -> ChatResult:
        """
        Handle an incoming text message.
        is_delegated=True means this came from a peer robot, not a user.

        delegated_context carries snippets serialized by the delegating Manager.
        They are re-validated here against this robot's own grants — a snippet
        not covered by an active grant is dropped, so handing over context is
        never enough on its own to make it readable.
        """
        self.last_active = time.time()
        self._refresh_role_from_db()

        if not self.llm or not self.llm.is_available():
            return ChatResult(
                response="[DEFAULT] LLM module not available.",
                emotion_tag="DEFAULT",
                clean_text="LLM module not available.",
                user_emotion=self._current_user_emotion(),
                is_delegation=False,
            )

        # Build prompt
        if is_delegated:
            system, user_msg = build_execution_prompt(
                self.robot_name, self._robot_role,
                self._allowed_tags, message,
                self._clear_delegated_context(delegated_context, task_id),
            )
        else:
            rag_context = self._get_rag_context(message)
            active_peers = self._get_active_peers()
            system, user_msg = build_delegation_prompt(
                self.robot_name, self._robot_role, self._allowed_tags,
                message, active_peers, rag_context
            )

        # Generate response
        llm_resp = self.llm.generate_with_history(
            system, self._history, user_msg
        )

        # Update conversation history
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": llm_resp.text})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Persist to DB and update RAG
        self._persist(message, llm_resp.text)

        # Check for delegation JSON in response
        is_deleg, target_id = self._parse_delegation(llm_resp.text)

        return ChatResult(
            response=llm_resp.text,
            emotion_tag=llm_resp.emotion_tag,
            clean_text=llm_resp.clean_text,
            user_emotion=self._current_user_emotion(),
            is_delegation=is_deleg,
            delegation_target=target_id,
        )

    def process_chat_stream(
        self,
        message: str,
        on_sentence,
        is_delegated: bool = False,
        delegated_context: Optional[Sequence[MemoryRecord]] = None,
        task_id: Optional[str] = None,
    ) -> ChatResult:
        """
        Like process_chat() but fires on_sentence(clean_text, emotion_tag) for each
        sentence as the LLM generates it. Returns the final ChatResult after streaming.
        on_sentence is called synchronously — the caller thread is held during generation.
        """
        self.last_active = time.time()
        self._refresh_role_from_db()

        if not self.llm or not self.llm.is_available():
            clean = "LLM module not available."
            on_sentence(clean, "DEFAULT")
            return ChatResult(
                response="[DEFAULT] LLM module not available.",
                emotion_tag="DEFAULT",
                clean_text=clean,
                user_emotion=self._current_user_emotion(),
                is_delegation=False,
            )

        if is_delegated:
            system, user_msg = build_execution_prompt(
                self.robot_name, self._robot_role, self._allowed_tags, message,
                self._clear_delegated_context(delegated_context, task_id),
            )
        else:
            rag_context = self._get_rag_context(message)
            active_peers = self._get_active_peers()
            system, user_msg = build_delegation_prompt(
                self.robot_name, self._robot_role, self._allowed_tags,
                message, active_peers, rag_context,
            )

        from modules.llm.llm_provider import parse_response
        full_text = ""
        first = True
        for sentence in self.llm.stream_with_history(system, self._history, user_msg):
            full_text += (" " if full_text else "") + sentence
            if first:
                parsed = parse_response(sentence)
                on_sentence(parsed.clean_text, parsed.emotion_tag)
                first = False
            else:
                on_sentence(sentence, "")

        full_text = full_text.strip()

        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": full_text})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        self._persist(message, full_text)

        final = parse_response(full_text)
        is_deleg, target_id = self._parse_delegation(full_text)
        return ChatResult(
            response=final.text,
            emotion_tag=final.emotion_tag,
            clean_text=final.clean_text,
            user_emotion=self._current_user_emotion(),
            is_delegation=is_deleg,
            delegation_target=target_id,
        )

    def classify_qa_intent(self, user_message: str) -> str:
        """
        Fast binary classifier: should the demo resume, or does the visitor have more questions?
        Returns 'done' (resume demo) or 'continue' (more Q&A).
        Uses a single-turn generate() call — no conversation history needed.
        Safe default is 'continue' so real questions are never skipped.
        """
        if not self.llm or not self.llm.is_available():
            return "continue"
        system = (
            "You are an intent classifier for a robot demo guide. "
            "The guide robot just asked the visitor: "
            "'Do you have any other questions, or shall we continue the demonstration?' "
            "Classify the visitor's reply with ONLY one word:\n"
            "  done     — visitor is satisfied and ready to continue the demo\n"
            "  continue — visitor has more questions or is still engaged\n"
            "Reply with exactly one word. No punctuation, no explanation."
        )
        try:
            resp = self.llm.generate(system, user_message[:200])
            first_word = resp.text.strip().lower().split()[0] if resp.text.strip() else "continue"
            decision = "done" if first_word == "done" else "continue"
            print(f"[QA Classifier] '{user_message[:60]}' → LLM first word: '{first_word}' → {decision}")
            return decision
        except Exception as e:
            print(f"[QA Classifier] LLM error ({e}) → defaulting to 'continue'")
            return "continue"

    # ── Demo speech generation ────────────────────────────────────────────────

    def generate_demo_speech(self, instruction: str) -> "ChatResult":
        """
        Generate speech for a demo step from an instruction/prompt.
        Unlike process_chat(), this uses a demo-appropriate system prompt:
          - No delegation logic
          - No rigid 1-2 sentence cap (length is set by the instruction itself)
          - Conversation history is maintained so context builds across steps
        Falls back to a ChatResult wrapping the raw instruction on error.
        """
        self.last_active = time.time()
        self._refresh_role_from_db()

        if not self.llm or not self.llm.is_available():
            return ChatResult(
                response=instruction,
                emotion_tag="",
                clean_text=instruction,
                user_emotion=self._current_user_emotion(),
                is_delegation=False,
            )

        tags_str    = ", ".join(self._allowed_tags) if self._allowed_tags else "[DEFAULT]"
        example_tag = self._allowed_tags[0] if self._allowed_tags else "[DEFAULT]"

        system_prompt = (
            f"You are {self.robot_name}. Your role: {self._robot_role}\n\n"
            f"DEMO SPEECH RULES:\n"
            f"1. The VERY FIRST character of your response MUST be '['.\n"
            f"2. Use EXACTLY ONE emotion tag chosen from: {tags_str}\n"
            f"3. Speak naturally as yourself — the instruction you receive tells you\n"
            f"   what to say, the emotional tone, and how long to speak.\n"
            f"4. Do NOT include any extra commentary, JSON, or meta-text.\n\n"
            f"CORRECT:   {example_tag} Welcome to the CARES lab! I am {self.robot_name}, your guide today.\n"
            f"INCORRECT: Welcome! {example_tag} I am {self.robot_name}.   <- tag must be first"
        )

        llm_resp = self.llm.generate_with_history(system_prompt, self._history, instruction)

        # Maintain history so each step builds on prior context
        self._history.append({"role": "user",      "content": instruction})
        self._history.append({"role": "assistant",  "content": llm_resp.text})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return ChatResult(
            response=llm_resp.text,
            emotion_tag=llm_resp.emotion_tag,
            clean_text=llm_resp.clean_text,
            user_emotion=self._current_user_emotion(),
            is_delegation=False,
        )

    # ── Speech ────────────────────────────────────────────────────────────────

    def process_speech(self, audio_b64: str) -> SpeechResult:
        """Transcribe audio, then optionally run through chat pipeline."""
        self.last_active = time.time()

        if not self.speech or not self.speech.is_available():
            raise RuntimeError("Speech module not available for this robot.")

        from modules.speech.speech_module import SpeechResult as SR
        result: SR = self.speech.transcribe_b64(audio_b64)

        if not result.success:
            raise RuntimeError(f"Transcription failed: {result.error}")

        chat_result = None
        if self.llm and self.llm.is_available() and result.transcription.strip():
            chat_result = self.process_chat(result.transcription)

        return SpeechResult(
            transcription=result.transcription,
            confidence=result.confidence,
            chat=chat_result,
        )

    # ── Emotion / image frame ─────────────────────────────────────────────────

    def process_frame(self, frame_b64: str) -> dict:
        """Run a camera frame through the emotion module."""
        self.last_active = time.time()

        if not self.emotion or not self.emotion.is_available():
            raise RuntimeError("Emotion module not available for this robot.")

        result = self.emotion.process_frame_b64(frame_b64)
        return {
            "emotion": result.emotion,
            "confidence": result.confidence,
            "changed": result.changed,
            "distribution": result.distribution,
            "status": result.status,
        }

    # ── Health ────────────────────────────────────────────────────────────────

    def get_health(self) -> dict:
        status = {
            "client_id": self.client_id,
            "robot_name": self.robot_name,
            "enabled_modules": list(self.enabled_modules),
            "last_active": self.last_active,
            "modules": {},
        }
        for name, mod in [
            ("llm", self.llm),
            ("speech", self.speech),
            ("emotion", self.emotion),
            ("rag", self.rag),
        ]:
            if mod:
                status["modules"][name] = mod.get_status()
        return status

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _refresh_role_from_db(self):
        """
        Pull latest role, tags and access level from Supabase so web UI and
        database changes apply immediately.

        Access level is re-read here rather than cached at connect time because
        the paper's design calls for hierarchies adjustable at the database level
        without reconfiguring individual robot nodes.
        """
        robot = robot_repo.get_robot(self.client_id)
        if robot:
            self._robot_role = robot.robot_role
            self._allowed_tags = robot.allowed_tags
            if getattr(robot, "access_level", None):
                self._access_level = robot.access_level
            if getattr(robot, "scenario_id", None):
                self._scenario_id = robot.scenario_id

    def _get_rag_context(self, message: str) -> list[ClearedRecord]:
        """
        Retrieve episodic context, RBAC-filtered.

        Returns ClearedRecord, not raw strings — prompt assembly refuses
        anything without a clearance stamp.
        """
        if self.rag and self.rag.is_available():
            try:
                return self.rag.search(
                    message,
                    requester=self.identity,
                    rbac=self._rbac,
                    grants=self._grants.active_for(self.client_id),
                    top_k=5,
                )
            except Exception as e:
                print(f"[RobotInstance] RAG search error: {e}")
        return []

    def _clear_delegated_context(
        self,
        delegated_context: Optional[Sequence[MemoryRecord]],
        task_id: Optional[str],
    ) -> list[ClearedRecord]:
        """
        Validate snippets serialized by a delegating Manager against this robot's
        own active grants.

        The hand-off carries the snippet text; the grant carries the authority.
        A record that arrives without a matching active grant is denied here, so
        a Worker cannot be handed context it was never granted, and the snippets
        expire with the grant.
        """
        if not delegated_context:
            return []
        active = self._grants.active_for(self.client_id, task_id=task_id)
        if not active:
            return []
        return self._rbac.filter_records(
            self.identity, delegated_context, active, store="delegation"
        )

    def _get_active_peers(self) -> list[dict]:
        """Fetch all other currently active robots from DB."""
        peers = robot_repo.get_all_active_robots(exclude_id=self.client_id)
        return [
            {
                "client_id": r.client_id,
                "robot_name": r.robot_name,
                "robot_role": r.robot_role,
            }
            for r in peers
        ]

    def _current_user_emotion(self) -> str:
        if self.emotion and self.emotion.is_available():
            em, _ = self.emotion.get_current()
            return em
        return "unknown"

    def _persist(self, message: str, response: str):
        """
        Save to chat_logs and update RAG index, stamping provenance and
        visibility so the record can be reasoned about later.

        Note what is *not* here: delegated context. Only the robot's own
        message and response are written, so a snippet granted to this robot for
        one task never enters its own memory.
        """
        visibility = self._visibility_value()
        try:
            chat_log_repo.insert(
                self.user_id, message, response,
                source_robot_id=self.client_id,
                scenario_id=self._scenario_id,
                session_id=self._session_id,
                visibility=visibility,
                subject_user_id=self.user_id,
            )
        except Exception as e:
            print(f"[RobotInstance] DB log error: {e}")
        if self.rag and self.rag.is_available():
            try:
                self.rag.add(
                    message,
                    subject_user_id=self.user_id,
                    visibility=visibility,
                )
            except Exception as e:
                print(f"[RobotInstance] RAG add error: {e}")

    def _visibility_value(self) -> str:
        v = self._default_visibility
        return v.value if isinstance(v, Visibility) else str(v)

    def _parse_delegation(self, response_text: str) -> tuple[bool, Optional[str]]:
        """
        Check if the LLM response contains a delegation JSON block.
        Returns (is_delegation, target_robot_id).
        """
        import re
        import json
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, response_text, re.DOTALL)
        if not match:
            return False, None
        try:
            data = json.loads(match.group(1).strip())
            target = data.get("target_robot_id")
            if target:
                return True, target
        except json.JSONDecodeError:
            pass
        return False, None