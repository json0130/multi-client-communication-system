"""
modules/emotion/emotion_tracker.py
===================================
Sliding-window emotion smoother.
Keeps the last N detections and returns the most stable (frequent + confident) emotion.
Pure logic — no CV, no torch. Easy to unit-test in isolation.
"""

from __future__ import annotations
from collections import deque
import numpy as np


class EmotionTracker:

    LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def __init__(self, window_size: int = 5):
        self._window = window_size
        self._emotions: deque[str] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._counts: dict[str, int] = {}

        self.stable_emotion = "neutral"
        self.stable_confidence = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, emotion: str, confidence: float, min_confidence: float = 25.0):
        """
        Push a new detection into the window.
        Detections below min_confidence are silently ignored.
        """
        if confidence < min_confidence:
            return

        # Drop the oldest entry from the count map
        if len(self._emotions) == self._window:
            oldest = self._emotions[0]
            self._counts[oldest] = max(0, self._counts.get(oldest, 1) - 1)
            if self._counts[oldest] == 0:
                del self._counts[oldest]

        self._emotions.append(emotion)
        self._confidences.append(confidence)
        self._counts[emotion] = self._counts.get(emotion, 0) + 1

    def get_stable(
        self,
        change_threshold: float = 20.0,
        update_threshold: float = 0.1,
    ) -> tuple[str, float, bool]:
        """
        Return (emotion, confidence, changed).
        changed=True means the stable emotion just updated — use this to
        decide whether to notify the robot layer.
        Needs at least 3 detections before making a decision.
        """
        if len(self._emotions) < 3 or not self._counts:
            return self.stable_emotion, self.stable_confidence, False

        # Most frequent emotion in the window
        dominant = max(self._counts, key=self._counts.get)

        # Weighted average confidence for the dominant emotion
        matched_confs = []
        matched_weights = []
        for i, (em, cf) in enumerate(zip(self._emotions, self._confidences)):
            if em == dominant:
                matched_confs.append(cf)
                matched_weights.append((i + 1) / len(self._emotions))

        if not matched_confs:
            return self.stable_emotion, self.stable_confidence, False

        weighted_conf = float(np.average(matched_confs, weights=matched_weights))
        emotion_changed = dominant != self.stable_emotion
        conf_shifted = abs(weighted_conf - self.stable_confidence) > update_threshold

        if (emotion_changed and weighted_conf > change_threshold) or conf_shifted:
            self.stable_emotion = dominant
            self.stable_confidence = weighted_conf
            return dominant, weighted_conf, True

        return self.stable_emotion, self.stable_confidence, False

    def get_distribution(self) -> dict[str, float]:
        """Return each emotion as a percentage of the current window."""
        total = sum(self._counts.values())
        if total == 0:
            return {}
        return {em: (cnt / total) * 100 for em, cnt in self._counts.items()}

    def reset(self):
        self._emotions.clear()
        self._confidences.clear()
        self._counts.clear()
        self.stable_emotion = "neutral"
        self.stable_confidence = 0.0