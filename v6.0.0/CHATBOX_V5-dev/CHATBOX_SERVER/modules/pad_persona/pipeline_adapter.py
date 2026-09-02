"""
PAD pipeline adapter — wire-ready, no existing files modified.

Provides two interchangeable classes:
  PADPipelineAdapter  — live PAD-driven affect state
  NullPADAdapter      — neutral-fallback, identical interface

The host pipeline can swap between them in one line.
"""

from .config import PADWeights, CHATBOX_PERSONA, ELLEBOT_PERSONA
from .affect_stream import AffectStream
from .pad_engine import PADEngine
from .prompt_builder import build_system_prompt

# ---------------------------------------------------------------------------
# Emotion-label → (valence, arousal) mapping
# Based on Russell (1980) circumplex model of affect.
# Used until a dedicated V-A regression head is trained on the camera stream.
# ---------------------------------------------------------------------------
EMOTION_VA: dict[str, tuple[float, float]] = {
    "happy":    ( 0.8,  0.6),
    "neutral":  ( 0.0,  0.0),
    "sad":      (-0.7, -0.4),
    "angry":    (-0.6,  0.7),
    "fear":     (-0.5,  0.8),
    "disgust":  (-0.6,  0.3),
    "surprise": ( 0.1,  0.8),
}

_PERSONA_REGISTRY = {
    "chatbox": (CHATBOX_PERSONA, "ChatBox"),
    "ellebot": (ELLEBOT_PERSONA, "ElleBot"),
}

_NEUTRAL_GESTURE_PARAMS: dict[str, float] = {
    "amplitude":  0.5,
    "tempo":      0.5,
    "posture":    0.5,
    "idle_freq":  0.5,
    "expression": 0.5,
}

_NEUTRAL_DESCRIPTORS: dict[str, str] = {
    "pleasure":  "neutral",
    "arousal":   "moderate",
    "dominance": "neutral",
}


# ---------------------------------------------------------------------------
# Live adapter
# ---------------------------------------------------------------------------

class PADPipelineAdapter:
    """
    Thin stateful adapter between the PAD persona engine and the pipeline.

    One instance per connected robot client.  Holds one PADEngine and one
    AffectStream; call process_turn() once per conversation turn.
    """

    def __init__(
        self,
        robot_id: str,
        affect_stream: AffectStream | None = None,
        weights: PADWeights | None = None,
    ):
        entry = _PERSONA_REGISTRY.get(robot_id.lower())
        if entry is None:
            raise ValueError(
                f"Unknown robot_id '{robot_id}'. "
                f"Available: {list(_PERSONA_REGISTRY)}"
            )
        persona, self._display_name = entry
        self._engine = PADEngine(persona, weights or PADWeights())
        self._stream = affect_stream or AffectStream()

    # ------------------------------------------------------------------
    # Primary turn method
    # ------------------------------------------------------------------

    def process_turn(
        self,
        valence: float,
        arousal: float,
        relationship_tier: str,
        memory_context: str = "",
        rapport: float = 0.0,
        trust: float = 0.0,
        interaction_count: int = 0,
    ) -> dict:
        """Run the full PAD update for one conversation turn.

        Args:
            valence:           Current user affect, [-1, 1].
                               Obtain via emotion_label_to_va() or a V-A model.
            arousal:           Current user arousal, [-1, 1].
            relationship_tier: "close" | "family" | "known" | "visitor" | "unknown"
            memory_context:    RAG-retrieved snippets; empty string if none.
            rapport:           KG rapport score [0, 1] for prompt injection.
            trust:             KG trust score [0, 1] for prompt injection.
            interaction_count: Number of prior KG interactions for prompt injection.

        Returns a dict with keys:
            system_prompt  (str | None) — pass to LLM; None → use existing default
            gesture_params (dict)       — merge into hardware command
            pad_state      (tuple)      — (P, A, D) floats for logging
            descriptors    (dict)       — {"pleasure", "arousal", "dominance"} strings
        """
        dP, dA = self._stream.update(valence, arousal)
        pad    = self._engine.update((dP, dA), relationship_tier)

        descriptors    = self._engine.to_language_descriptors()
        gesture_params = self._engine.to_gesture_params()

        system_prompt = build_system_prompt(
            persona_name=self._display_name,
            descriptors=descriptors,
            relationship_tier=relationship_tier,
            memory_context=memory_context,
            rapport=rapport,
            trust=trust,
            interaction_count=interaction_count,
        )

        return {
            "system_prompt":  system_prompt,
            "gesture_params": gesture_params,
            "pad_state":      pad,
            "descriptors":    descriptors,
        }

    # ------------------------------------------------------------------
    # Hardware command enrichment
    # ------------------------------------------------------------------

    def enrich_hardware_command(self, tag: str, gesture_params: dict) -> dict:
        """Merge PAD gesture parameters into a hardware command dict.

        The "command" key is the plain ASCII string already accepted by the
        current ESP32 firmware (matching validExpressions[]).
        "gesture_params" carries PAD-derived modifiers; current firmware ignores
        them — they become actionable when the firmware is extended to accept JSON.

        Args:
            tag:           Action tag string, e.g. "GREETING" or "SAD".
            gesture_params: Output of PADEngine.to_gesture_params().

        Returns:
            {
              "command":       str,   # lowercase, ready for arduino_output.send_command()
              "gesture_params": dict, # PAD modifiers for future firmware use
              "raw_tag":       str,   # uppercase original tag, for logging
            }
        """
        return {
            "command":        tag.lower(),
            "gesture_params": gesture_params,
            "raw_tag":        tag.upper(),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def emotion_label_to_va(emotion_label: str) -> tuple[float, float]:
        """Convert an EmotionProcessor label to (valence, arousal).

        Falls back to (0.0, 0.0) for unknown labels.
        """
        return EMOTION_VA.get(emotion_label.lower(), (0.0, 0.0))


# ---------------------------------------------------------------------------
# Null fallback
# ---------------------------------------------------------------------------

class NullPADAdapter:
    """Drop-in replacement for PADPipelineAdapter that returns neutral defaults.

    Swap in one line to disable the PAD module without touching any other code:
        pad_adapter = NullPADAdapter()
    """

    def process_turn(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        relationship_tier: str = "unknown",
        memory_context: str = "",
    ) -> dict:
        return {
            "system_prompt":  None,  # signals host to use its own hardcoded prompt
            "gesture_params": dict(_NEUTRAL_GESTURE_PARAMS),
            "pad_state":      (0.0, 0.0, 0.0),
            "descriptors":    dict(_NEUTRAL_DESCRIPTORS),
        }

    def enrich_hardware_command(self, tag: str, gesture_params: dict) -> dict:
        return {
            "command":        tag.lower(),
            "gesture_params": gesture_params,
            "raw_tag":        tag.upper(),
        }

    @staticmethod
    def emotion_label_to_va(emotion_label: str) -> tuple[float, float]:
        return EMOTION_VA.get(emotion_label.lower(), (0.0, 0.0))


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    adapter = PADPipelineAdapter("chatbox")

    turns = [
        # (emotion_label, relationship_tier, memory_context)
        ("happy",    "close",   "Child mentioned they love trains and dinosaurs."),
        ("happy",    "close",   ""),
        ("neutral",  "known",   ""),
        ("sad",      "unknown", ""),
        ("surprise", "known",   "They just told me about their new dog."),
    ]

    print("=" * 70)
    print("PAD Pipeline Adapter — 5-turn CHATBOX demo")
    print("=" * 70)

    for i, (emotion_label, tier, memory) in enumerate(turns, 1):
        v, a = PADPipelineAdapter.emotion_label_to_va(emotion_label)
        result = adapter.process_turn(v, a, tier, memory)

        p, ar, d = result["pad_state"]
        desc = result["descriptors"]
        gp   = result["gesture_params"]

        print(f"\n--- Turn {i} | emotion={emotion_label!r:10s} tier={tier!r} ---")
        print(f"  VA input      valence={v:+.1f}  arousal={a:+.1f}")
        print(f"  PAD state     P={p:+.3f}  A={ar:+.3f}  D={d:+.3f}")
        print(f"  Descriptors   {desc['pleasure']} / {desc['arousal']} / {desc['dominance']}")
        print(f"  Gesture       amplitude={gp['amplitude']:.3f}  tempo={gp['tempo']:.3f}"
              f"  posture={gp['posture']:.3f}  expression={gp['expression']:.3f}")
        print(f"  Hw command    {adapter.enrich_hardware_command('GREETING', gp)}")

        # Show first 2 lines of system prompt only
        prompt_lines = result["system_prompt"].splitlines()
        print(f"  Prompt[0:2]   {prompt_lines[0][:80]}")
        if len(prompt_lines) > 2:
            print(f"               {prompt_lines[2][:80]}")

    print("\n" + "=" * 70)
    print("NullPADAdapter demo (one turn)")
    print("=" * 70)
    null = NullPADAdapter()
    nr = null.process_turn(0.5, 0.3, "known", "")
    print(f"  system_prompt  → {nr['system_prompt']!r}  (None = use existing default)")
    print(f"  pad_state      → {nr['pad_state']}")
    print(f"  gesture_params → {nr['gesture_params']}")
