from .config import PersonaConfig, PADWeights
from .ocean_to_pad import ocean_to_baseline_pad
from .relationship import tier_to_pad_offset


class PADEngine:
    """Stateful PAD (Pleasure-Arousal-Dominance) affect model for a robot persona.

    Call update() once per conversation turn to get the current PAD state,
    then call to_language_descriptors() / to_gesture_params() to translate it
    into prompt text and servo/expression parameters.
    """

    def __init__(self, persona: PersonaConfig, weights: PADWeights = PADWeights()):
        self.persona = persona
        self.weights = weights
        self.baseline_pad: tuple[float, float, float] = ocean_to_baseline_pad(persona.ocean)
        self.current_pad: tuple[float, float, float] = self.baseline_pad

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        affect_offset: tuple[float, float],
        relationship_tier: str,
    ) -> tuple[float, float, float]:
        """Blend baseline + affect offset + relationship offset, then decay.

        Args:
            affect_offset: (dP, dA) from AffectStream.update()
            relationship_tier: one of "close", "family", "known", "visitor", "unknown"

        Returns:
            Updated (P, A, D) clamped to [-1, 1].
        """
        bp, ba, bd = self.baseline_pad
        cp, ca, cd = self.current_pad
        dP_aff, dA_aff = affect_offset
        dP_rel, dA_rel, dD_rel = tier_to_pad_offset(relationship_tier)

        w_u = self.weights.w_user
        w_r = self.weights.w_rel
        alpha = self.weights.alpha_decay

        # Blend live offsets into current state
        new_p = cp + w_u * dP_aff + w_r * dP_rel
        new_a = ca + w_u * dA_aff + w_r * dA_rel
        new_d = cd                 + w_r * dD_rel

        # Decay toward baseline
        new_p = new_p + alpha * (bp - new_p)
        new_a = new_a + alpha * (ba - new_a)
        new_d = new_d + alpha * (bd - new_d)

        clamp = lambda v: max(-1.0, min(1.0, v))
        self.current_pad = (clamp(new_p), clamp(new_a), clamp(new_d))
        return self.current_pad

    # ------------------------------------------------------------------
    # Output translators
    # ------------------------------------------------------------------

    def to_language_descriptors(self) -> dict[str, str]:
        """Bucket current PAD into human-readable word descriptors."""
        p, a, d = self.current_pad

        if p >= 0.2:
            pleasure_word = "warm"
        elif p <= -0.2:
            pleasure_word = "subdued"
        else:
            pleasure_word = "neutral"

        if a >= 0.2:
            arousal_word = "lively"
        elif a <= -0.2:
            arousal_word = "calm"
        else:
            arousal_word = "moderate"

        if d >= 0.2:
            dominance_word = "confident"
        elif d <= -0.2:
            dominance_word = "reserved"
        else:
            dominance_word = "neutral"

        return {
            "pleasure":   pleasure_word,
            "arousal":    arousal_word,
            "dominance":  dominance_word,
        }

    def to_gesture_params(self) -> dict[str, float]:
        """Map current PAD to normalised gesture/expression parameters in [0, 1]."""
        p, a, d = self.current_pad

        # Shift each dimension from [-1, 1] → [0, 1] for downstream use
        p01 = (p + 1.0) / 2.0
        a01 = (a + 1.0) / 2.0
        d01 = (d + 1.0) / 2.0

        return {
            "amplitude":  round(0.4 * a01 + 0.3 * d01 + 0.3 * p01, 3),
            "tempo":      round(0.6 * a01 + 0.4 * d01, 3),
            "posture":    round(d01, 3),
            "idle_freq":  round(0.5 * a01 + 0.2 * p01 + 0.3 * d01, 3),
            "expression": round(p01, 3),
        }
