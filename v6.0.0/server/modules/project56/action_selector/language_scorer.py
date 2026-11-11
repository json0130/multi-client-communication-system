"""
modules/action_selector/language_scorer.py
==========================================
LLM-backed dialogue-relevance scorer.

What it does:
  - Receives generated dialogue and a fixed set of candidate action tags
  - Asks the existing server LLM to score each candidate from 0.0 to 1.0
  - Parses plain or fenced JSON returned by the model
  - Clamps scores to the documented range
  - Ignores tags that were not supplied by the caller

The scorer reuses the LLMModule already attached to the robot instance. It
does not create another provider or rely on robot-side computing resources.
The legacy Method A mode prefers DEFAULT when scoring fails. Method B instead
returns zero scores so its selector can preserve no-action as None.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType

from modules.project56.action_selector.action_catalog import (
    V2_ACTION_DESCRIPTIONS,
    V2_1_ACTION_DESCRIPTIONS,
    get_action_descriptions,
)
from modules.project56.action_selector.models import ActionContext
from modules.project56.action_selector.diagnostics import LanguageScoringTrace
from modules.project56.robot_profile import ActionProfile


class LanguageScorer:
    """Assign dialogue-relevance scores to candidate action tags."""

    def __init__(
        self,
        llm,
        action_profiles: Mapping[str, ActionProfile] | None = None,
        scoring_mode: str = "legacy",
        action_description_set: str = "v1",
        scorer_context: str = "none",
    ):
        """
        llm is an existing LLMModule.

        It is injected instead of constructed here, so this module does not
        create another provider or duplicate configuration.
        """
        self._llm = llm
        self._action_profiles = action_profiles
        if scoring_mode not in {"legacy", "choice"}:
            raise ValueError("scoring_mode must be 'legacy' or 'choice'")
        self._scoring_mode = scoring_mode
        if action_description_set not in {"v1", "v2", "v2.1"}:
            raise ValueError("action_description_set must be 'v1', 'v2', or 'v2.1'")
        self._action_description_set = action_description_set
        if scorer_context not in {"none", "situation", "situation_history"}:
            raise ValueError(
                "scorer_context must be 'none', 'situation', or 'situation_history'"
            )
        self._scorer_context = scorer_context

    def score_actions(
        self,
        tags: list[str],
        context: ActionContext,
        *,
        no_action_on_failure: bool = False,
        temperature: float | None = None,
        diagnostic_callback: Callable[[LanguageScoringTrace], None] | None = None,
    ) -> dict[str, float]:
        """
        Return one normalised relevance score for each supplied tag.

        Missing candidates in the model response receive 0.0. Additional keys
        returned by the model are ignored so it cannot introduce unsupported
        robot actions.
        """
        normalised_tags = [self._normalise(tag) for tag in tags]
        scoring_started = time.perf_counter()

        candidate_actions = self._get_action_descriptions(
            robot_id=context.robot_id,
            tags=normalised_tags,
            use_profile_default=no_action_on_failure,
        )

        if not self._llm or not self._llm.is_available():
            scores = self._fallback_scores(
                normalised_tags,
                no_action_on_failure=no_action_on_failure,
            )
            self._emit_diagnostic(
                diagnostic_callback,
                normalised_tags,
                candidate_actions,
                context,
                scores,
                raw_response=None,
                fallback_reason="provider_unavailable",
                scorer_prompt=None,
                scoring_mode=self._scoring_mode,
                raw_responses=(),
                action_description_set=self._action_description_set,
                scorer_latency_seconds=time.perf_counter() - scoring_started,
            )
            return scores

        legacy_system_prompt = """
You score how appropriately robot actions accompany generated dialogue.

Each candidate includes a tag and a description of what the action means.

For every candidate, return a relevance score from 0.0 to 1.0.

Rules:
- Score how naturally the action accompanies the generated dialogue.
- Give an expressive action a high score when it clearly matches.
- DEFAULT means performing no expressive action.
- Do not give DEFAULT a high score merely because it is generally safe.
- Use DEFAULT only when no expressive candidate clearly fits.
- Do not assess physical availability; it is evaluated separately.
- Only return scores for the supplied candidate actions.

Return exactly one valid JSON object mapping each tag to its score.
Do not return Markdown, explanations, or additional keys.

Example:
{"DEFAULT": 0.4, "WAVE": 0.9, "HAPPY": 0.7}
""".strip()

        method_b_system_prompt = """
You score how appropriately robot actions accompany generated dialogue.

Each candidate includes a tag and a description of what the action means.

For every candidate, return a relevance score from 0.0 to 1.0.

Rules:
- Score how naturally the action accompanies the generated dialogue.
- Give an expressive action a high score when it clearly matches.
- Do not assess physical availability; it is evaluated separately.
- Only return scores for the supplied candidate actions.
- The caller handles the valid outcome where no gesture is selected.

Return exactly one valid JSON object mapping each tag to its score.
Do not return Markdown, explanations, or additional keys.

Example:
{"POINT": 0.8, "AGREE": 0.2, "HAPPY": 0.1}
""".strip()

        system_prompt = (
            method_b_system_prompt
            if no_action_on_failure
            else legacy_system_prompt
        )

        user_prompt = self._build_user_prompt(context, candidate_actions)

        if self._scoring_mode == "choice":
            # Start from the exact legacy prompt selected for this call so
            # candidate semantics/descriptions differ only in output contract.
            choice_system_prompt = system_prompt.replace(
                "For every candidate, return a relevance score from 0.0 to 1.0.",
                "Choose exactly one candidate tag and give its confidence from 0.0 to 1.0.",
            ).replace(
                "Only return scores for the supplied candidate actions.",
                "Return exactly one JSON object with exactly these keys: "
                '{"tag": "<one listed candidate tag>", "confidence": <0.0 to 1.0>}.',
            ).replace(
                "Return exactly one valid JSON object mapping each tag to its score.\n"
                "Do not return Markdown, explanations, or additional keys.\n\n"
                'Example:\n{"POINT": 0.8, "AGREE": 0.2, "HAPPY": 0.1}',
                "Do not return Markdown, explanations, or additional keys.",
            )
            scores, raw_response, fallback_reason, raw_responses, scorer_prompt = (
                self._score_choice(
                    choice_system_prompt, user_prompt, normalised_tags,
                    temperature=temperature,
                )
            )
        else:
            # This legacy call and prompt construction intentionally remain
            # unchanged when the new mode is not enabled.
            response = self._generate(system_prompt, user_prompt, temperature)
            scoring_text = response.clean_text or response.text
            raw_response = response.text
            try:
                parsed = self._parse_json(scoring_text)
                scores = {
                    tag: self._clamp(float(parsed.get(tag, 0.0)))
                    for tag in normalised_tags
                }
                fallback_reason = None
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                scores = self._fallback_scores(
                    normalised_tags,
                    no_action_on_failure=no_action_on_failure,
                )
                fallback_reason = "invalid_or_malformed_response"
            raw_responses = (raw_response,)
            scorer_prompt = user_prompt

        self._emit_diagnostic(
            diagnostic_callback,
            normalised_tags,
            candidate_actions,
            context,
            scores,
            raw_response=raw_response,
            fallback_reason=fallback_reason,
            scorer_prompt=scorer_prompt,
            scoring_mode=self._scoring_mode,
            raw_responses=raw_responses,
            action_description_set=self._action_description_set,
            scorer_latency_seconds=time.perf_counter() - scoring_started,
        )
        return scores

    @staticmethod
    def _emit_diagnostic(
        callback: Callable[[LanguageScoringTrace], None] | None,
        tags: list[str],
        candidate_actions: dict[str, str],
        context: ActionContext,
        scores: dict[str, float],
        *,
        raw_response: str | None,
        fallback_reason: str | None,
        scorer_prompt: str | None,
        scoring_mode: str,
        raw_responses: tuple[str, ...],
        action_description_set: str,
        scorer_latency_seconds: float,
    ) -> None:
        if callback is None:
            return
        callback(LanguageScoringTrace(
            semantic_candidates=tuple(tags),
            candidate_descriptions=MappingProxyType(dict(candidate_actions)),
            robot_id=context.robot_id,
            user_message=context.user_message,
            dialogue=context.dialogue,
            user_emotion=context.user_emotion,
            semantic_scores=MappingProxyType(dict(scores)),
            raw_response=raw_response,
            fallback_reason=fallback_reason,
            scorer_prompt=scorer_prompt,
            scoring_mode=scoring_mode,
            raw_responses=raw_responses,
            action_description_set=action_description_set,
            scorer_latency_seconds=scorer_latency_seconds,
        ))

    def _generate(self, system_prompt: str, user_prompt: str, temperature: float | None):
        """Keep omission of a None override compatible with current callers."""
        return (
            self._llm.generate(system_prompt, user_prompt)
            if temperature is None
            else self._llm.generate(system_prompt, user_prompt, temperature=temperature)
        )

    def _build_user_prompt(
        self,
        context: ActionContext,
        candidate_actions: dict[str, str],
    ) -> str:
        """Build scoring context; the default retains the historic prompt exactly."""
        candidates = json.dumps(candidate_actions, indent=2)
        if self._scorer_context == "none":
            return (
                f"User message:\n{context.user_message}\n\n"
                f"Generated robot dialogue:\n{context.dialogue}\n\n"
                f"Detected user emotion:\n{context.user_emotion}\n\n"
                f"Candidate actions and descriptions:\n{candidates}"
            )
        situation = (
            "Situation: You are Navel, a social robot giving a guided tour of a "
            "robotics laboratory. You are standing in the lab with visitors. Robots, "
            "equipment, rooms and areas you mention may be physically nearby and "
            "visible to the visitors."
        )
        history = ""
        if self._scorer_context == "situation_history":
            earlier = "\n".join(context.previous_dialogue_segments)
            history = (
                f"Earlier in this response:\n"
                f"{earlier or '(this is the first sentence of the response)'}\n\n"
            )
        return (
            f"User message:\n{context.user_message}\n\n"
            f"{situation}\n\n"
            f"{history}"
            f"Current sentence being scored:\n{context.dialogue}\n\n"
            f"Detected user emotion:\n{context.user_emotion}\n\n"
            f"Candidate actions and descriptions:\n{candidates}"
        )

    def _score_choice(self, system_prompt, user_prompt, tags, *, temperature):
        prompts = [user_prompt]
        raw_responses: list[str] = []
        for attempt, prompt in enumerate(prompts):
            response = self._generate(system_prompt, prompt, temperature)
            raw = response.text
            raw_responses.append(raw)
            try:
                parsed = self._parse_json(response.clean_text or raw)
                if set(parsed) != {"tag", "confidence"}:
                    raise ValueError("choice response must contain exactly tag and confidence")
                tag = self._normalise(parsed["tag"])
                confidence = parsed["confidence"]
                if (tag not in tags or isinstance(confidence, bool)
                        or not isinstance(confidence, (int, float))
                        or not 0.0 <= confidence <= 1.0):
                    raise ValueError("invalid choice tag or confidence")
                return ({candidate: float(confidence) if candidate == tag else 0.0 for candidate in tags},
                        raw, None, tuple(raw_responses), prompt)
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                if attempt == 0:
                    prompts.append(
                        user_prompt + "\n\nCorrection: return exactly one JSON object with "
                        'a listed "tag" and numeric "confidence" from 0.0 to 1.0.'
                    )
        fallback_tag = "DEFAULT" if "DEFAULT" in tags else None
        scores = {tag: 1.0 if tag == fallback_tag else 0.0 for tag in tags}
        reason = ("choice_invalid_after_retry_default"
                  if fallback_tag else "choice_invalid_after_retry_no_action")
        return scores, raw_responses[-1], reason, tuple(raw_responses), prompts[-1]

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Accept plain JSON and tolerate an accidental fenced JSON response.
        """
        text = text.strip()

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )

        if fenced:
            text = fenced.group(1)

        return json.loads(text)

    @staticmethod
    def _fallback_scores(
        tags: list[str],
        *,
        no_action_on_failure: bool = False,
    ) -> dict[str, float]:
        """
        Preserve legacy DEFAULT fallback unless Method B requested no-action.
        """
        if no_action_on_failure:
            return {tag: 0.0 for tag in tags}
        return {
            tag: 1.0 if tag == "DEFAULT" else 0.0
            for tag in tags
        }

    @staticmethod
    def _normalise(tag: str) -> str:
        """Convert a configured tag to its canonical unbracketed form."""
        return tag.strip().strip("[]").upper()

    @staticmethod
    def _clamp(value: float) -> float:
        """Restrict a model-produced value to the range 0.0 through 1.0."""
        return max(0.0, min(1.0, value))

    def _get_action_descriptions(
        self,
        robot_id: str,
        tags: list[str],
        *,
        use_profile_default: bool = False,
    ) -> dict[str, str]:
        """Use profile metadata while preserving Method A's legacy DEFAULT."""
        legacy = get_action_descriptions(robot_id=robot_id, tags=tags)
        if self._action_description_set in {"v2", "v2.1"}:
            descriptions = (
                V2_1_ACTION_DESCRIPTIONS
                if self._action_description_set == "v2.1"
                else V2_ACTION_DESCRIPTIONS
            )
            return {
                tag: descriptions.get(tag, legacy[tag])
                for tag in tags
            }
        if self._action_profiles is None:
            return legacy

        return {
            tag: (
                self._action_profiles[tag].description
                if (
                    tag in self._action_profiles
                    and (tag != "DEFAULT" or use_profile_default)
                )
                else legacy[tag]
            )
            for tag in tags
        }
