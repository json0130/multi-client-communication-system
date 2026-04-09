"""
modules/emotion/emotion_module.py
==================================
Emotion detection module using:
  - OpenCV Haar cascade for face detection
  - EfficientNet B0 for 7-class emotion classification

Implements BaseModule so robot_instance.py can attach/detach it cleanly.

Requires: pip install torch torchvision opencv-python pillow
"""

from __future__ import annotations
import os
import base64
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from modules.base import BaseModule
from modules.emotion.emotion_tracker import EmotionTracker
from core.config import cfg


@dataclass
class EmotionResult:
    emotion: str            # e.g. "happy"
    confidence: float       # 0.0 – 100.0
    changed: bool           # True if emotion just changed
    distribution: dict      # {"happy": 60.0, "neutral": 40.0, ...}
    status: str             # "success" | "no_faces" | "throttled" | error string


class EmotionModule(BaseModule):

    LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def __init__(self):
        self._model = None
        self._transform = None
        self._face_cascade = None
        self._tracker = EmotionTracker(cfg.emotion.emotion_window_size)
        self._lock = threading.Lock()

        self._available = False
        self._model_loaded = False
        self._face_loaded = False

        # Throttle: skip frames to reduce CPU/GPU load
        self._frame_counter = 0
        self._last_process_time = 0.0
        self._skip_ratio = cfg.emotion.frame_skip_ratio
        self._interval = cfg.emotion.processing_interval_sec

        # Thresholds from config
        self._conf_threshold = cfg.emotion.confidence_threshold
        self._change_threshold = cfg.emotion.emotion_change_threshold
        self._update_threshold = cfg.emotion.emotion_update_threshold

    # ── BaseModule interface ───────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Load face cascade and EfficientNet model."""
        face_ok = self._load_face_cascade()
        model_ok = self._load_model()
        self._available = face_ok and model_ok

        if face_ok and not model_ok:
            print("[EmotionModule] Face detection loaded but model failed. "
                  "Check EMOTION_MODEL_PATH in .env")
        if not face_ok:
            print("[EmotionModule] Face cascade failed to load.")

        return self._available

    def is_available(self) -> bool:
        return self._available

    def get_status(self) -> dict:
        return {
            "module": "emotion",
            "available": self._available,
            "model_loaded": self._model_loaded,
            "face_cascade_loaded": self._face_loaded,
            "current_emotion": self._tracker.stable_emotion,
            "current_confidence": round(self._tracker.stable_confidence, 1),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame_b64(self, frame_b64: str) -> EmotionResult:
        """
        Decode a base64 JPEG/PNG frame and run emotion detection.
        Main entry point called by robot_instance.py.
        """
        if not self._available:
            return EmotionResult(
                self._tracker.stable_emotion, self._tracker.stable_confidence,
                False, {}, "module_not_initialised"
            )

        frame = self._decode_frame(frame_b64)
        if frame is None:
            return EmotionResult(
                self._tracker.stable_emotion, self._tracker.stable_confidence,
                False, {}, "decode_failed"
            )

        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray) -> EmotionResult:
        """
        Run emotion detection on a decoded OpenCV BGR frame.
        Applies frame-skip throttling — not every frame triggers inference.
        """
        if not self._available:
            return EmotionResult("neutral", 0.0, False, {}, "not_available")

        self._frame_counter += 1
        now = time.time()

        # Throttle: only process every N frames AND every X seconds
        should_process = (
            self._frame_counter % self._skip_ratio == 0
            and now - self._last_process_time >= self._interval
        )
        if not should_process:
            em, cf = self._tracker.stable_emotion, self._tracker.stable_confidence
            return EmotionResult(em, cf, False, self._tracker.get_distribution(),
                                 "throttled")

        self._last_process_time = now

        with self._lock:
            try:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40)
                )

                if len(faces) == 0:
                    em, cf = self._tracker.stable_emotion, self._tracker.stable_confidence
                    return EmotionResult(em, cf, False,
                                        self._tracker.get_distribution(), "no_faces")

                # Use the largest face only
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                emotion, confidence = self._classify_face(frame, x, y, w, h)

                self._tracker.add(emotion, confidence, self._conf_threshold)
                stable_em, stable_cf, changed = self._tracker.get_stable(
                    self._change_threshold, self._update_threshold
                )

                return EmotionResult(
                    stable_em, round(stable_cf, 1), changed,
                    self._tracker.get_distribution(), "success"
                )

            except Exception as e:
                print(f"[EmotionModule] Frame processing error: {e}")
                return EmotionResult(
                    self._tracker.stable_emotion, self._tracker.stable_confidence,
                    False, {}, f"error: {e}"
                )

    def get_current(self) -> tuple[str, float]:
        """Return the current stable (emotion, confidence) without processing a frame."""
        return self._tracker.stable_emotion, self._tracker.stable_confidence

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_face_cascade(self) -> bool:
        try:
            import cv2
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                return False
            self._face_cascade = cascade
            self._face_loaded = True
            print("[EmotionModule] Face cascade loaded.")
            return True
        except Exception as e:
            print(f"[EmotionModule] Face cascade error: {e}")
            return False

    def _load_model(self) -> bool:
        model_path = cfg.emotion.model_path
        if not os.path.exists(model_path):
            print(f"[EmotionModule] Model file not found: {model_path}")
            print("  Set EMOTION_MODEL_PATH in your .env to point to the .pth file.")
            return False

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Build same architecture used during training
            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[1].in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, len(self.LABELS)),
            )

            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                state = (checkpoint.get("model_state_dict")
                         or checkpoint.get("state_dict")
                         or checkpoint)
            else:
                state = checkpoint
            model.load_state_dict(state)
            model.eval()
            model.to(device)

            self._model = model
            self._device = device
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            self._model_loaded = True
            print(f"[EmotionModule] EfficientNet B0 loaded on {device}.")
            return True

        except Exception as e:
            print(f"[EmotionModule] Model load error: {e}")
            return False

    def _classify_face(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int
    ) -> tuple[str, float]:
        """Run the face crop through EfficientNet. Returns (emotion, confidence)."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import cv2

        face_bgr = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)

        tensor = self._transform(pil_face).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(tensor)
            probs = F.softmax(output, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item()) * 100

        return self.LABELS[idx], confidence

    def _decode_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        """Decode a base64 image string to an OpenCV BGR frame."""
        try:
            import cv2
            if len(frame_b64) > 600_000:
                print("[EmotionModule] Frame too large, skipping.")
                return None
            raw = base64.b64decode(frame_b64)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 800 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            return frame
        except Exception as e:
            print(f"[EmotionModule] Frame decode error: {e}")
            return None