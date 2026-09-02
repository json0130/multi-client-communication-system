"""
Pluggable emotion detection backends for the webcam KG loop.

Usage:
    det = EmotionDetector.create('hsemotion')          # default, ONNX, 191 fps CPU
    det = EmotionDetector.create('hsemotion-b2')       # larger B2 variant, ~more accurate
    det = EmotionDetector.create('efficientnet')       # original HQRAF model via PyTorch

    emotion, confidence = det.detect(face_crop_bgr)
    emotion, confidence = det.detect(face_crop_bgr, smooth=True)  # 5-frame window

All backends return labels from STANDARD_LABELS and confidence in [0, 100].

Backends
--------
hsemotion      EfficientNet-B0 ONNX — AffectNet+VGAF trained, ~5 ms/frame CPU.
               8 classes; 'Contempt' mapped → 'disgust' for VA compatibility.

hsemotion-b2   EfficientNet-B2 ONNX — same dataset, higher capacity, ~12 ms/frame.

efficientnet   Original HQRAF EfficientNet-B0 via PyTorch + Haar cascade face detect.
               Requires EmotionProcessor from modules/emotion_processor.py.
               Forces device='cpu' to avoid RTX 5060 sm_120 / torch 2.2.2 conflict.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

# ── Label normalisation ───────────────────────────────────────────────────────

STANDARD_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Russell (1980) circumplex — mirrors kg_bridge._EMOTION_VA + contempt/disgust alias
_VA_TABLE: dict[str, tuple[float, float]] = {
    "angry":    (-0.6,  0.7),
    "disgust":  (-0.6,  0.3),
    "fear":     (-0.5,  0.8),
    "happy":    ( 0.8,  0.6),
    "neutral":  ( 0.0,  0.0),
    "sad":      (-0.7, -0.4),
    "surprise": ( 0.1,  0.8),
}

_NORM = {
    # hsemotion (capitalised)
    'Anger':     'angry',
    'Contempt':  'disgust',   # contempt ≈ disgust in Russell VA space
    'Disgust':   'disgust',
    'Fear':      'fear',
    'Happiness': 'happy',
    'Neutral':   'neutral',
    'Sadness':   'sad',
    'Surprise':  'surprise',
    # lowercase / misc
    'anger':     'angry',
    'contempt':  'disgust',
    'disgust':   'disgust',
    'fear':      'fear',
    'happy':     'happy',
    'happiness': 'happy',
    'neutral':   'neutral',
    'sad':       'sad',
    'sadness':   'sad',
    'surprise':  'surprise',
}

def _norm(label: str) -> str:
    return _NORM.get(label, 'neutral')


# ── Simple sliding-window smoother ────────────────────────────────────────────

class _Smoother:
    def __init__(self, window: int = 5):
        self._hist: deque = deque(maxlen=window)

    def update(self, label: str, conf: float) -> tuple[str, float]:
        self._hist.append((label, conf))
        counts: dict[str, float] = {}
        for lbl, c in self._hist:
            counts[lbl] = counts.get(lbl, 0.0) + c
        best = max(counts, key=counts.__getitem__)
        avg_conf = counts[best] / sum(1 for l, _ in self._hist if l == best)
        return best, avg_conf


# ── Abstract base ─────────────────────────────────────────────────────────────

class EmotionDetector:
    """
    Detect emotion from a BGR face crop.

    Subclass and override _infer(face_bgr) → (label, confidence_0_to_100).
    Call detect() for the public interface with optional smoothing.
    """

    name: str = "base"

    def __init__(self, smooth_window: int = 5):
        self._smoother = _Smoother(smooth_window)

    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        """Return (label, confidence_0_to_100, valence, arousal)."""
        raise NotImplementedError

    def detect(
        self,
        frame_bgr:  np.ndarray,
        box:        Optional[tuple] = None,
        smooth:     bool = True,
    ) -> tuple[str, float, float, float]:
        """
        Detect emotion.

        Args:
            frame_bgr : Full BGR webcam frame (or a face crop if box is None).
            box       : (x1, y1, x2, y2) face bounding box in frame_bgr.
                        If None, treats frame_bgr as the face crop directly.
            smooth    : Apply 5-frame sliding-window smoothing (label + conf only).

        Returns:
            (emotion_label, confidence_0_to_100, valence, arousal)
            v/a are from the latest raw frame; label/conf may be window-smoothed.
        """
        if box is not None:
            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                return 'neutral', 0.0, 0.0, 0.0
        else:
            face = frame_bgr

        try:
            label, conf, v, a = self._infer(face)
        except Exception:
            return 'neutral', 0.0, 0.0, 0.0

        label = _norm(label)
        if smooth:
            label, conf = self._smoother.update(label, conf)
        return label, conf, v, a

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def create(backend: str = 'hsemotion', **kwargs) -> 'EmotionDetector':
        """
        Factory method.

        backend options:
            'hsemotion'    — EfficientNet-B0 ONNX (AffectNet, fast)
            'hsemotion-b2' — EfficientNet-B2 ONNX (AffectNet, accurate)
            'efficientnet' — original HQRAF EfficientNet via PyTorch
        """
        b = backend.lower()
        if b == 'hsemotion':
            return HSEmotionDetector(model='enet_b0_8_best_vgaf', **kwargs)
        if b in ('hsemotion-b2', 'hsemotion_b2'):
            return HSEmotionDetector(model='enet_b2_8', **kwargs)
        if b == 'efficientnet':
            return EfficientNetDetector(**kwargs)
        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Choose: 'hsemotion', 'hsemotion-b2', 'efficientnet'"
        )

    @staticmethod
    def available_backends() -> list[str]:
        backends = ['efficientnet']
        try:
            import hsemotion_onnx  # noqa: F401
            backends += ['hsemotion', 'hsemotion-b2']
        except ImportError:
            pass
        return backends


# ─────────────────────────────────────────────────────────────────────────────
# HSEmotion ONNX backend
# ─────────────────────────────────────────────────────────────────────────────

class HSEmotionDetector(EmotionDetector):
    """
    HSE EfficientNet ONNX emotion detector.

    Trained on AffectNet (450k images, 8 classes) — one of the largest
    publicly available emotion datasets. ONNX runtime means zero CUDA/PyTorch
    version dependency: runs at 5 ms/frame on CPU via onnxruntime.

    Models (cached to ~/.hsemotion/ on first use):
        enet_b0_8_best_vgaf  — B0, AffectNet+VGAF, 8 classes  [default, fastest]
        enet_b2_8            — B2, AffectNet, 8 classes        [more accurate, ~12 ms]
        enet_b2_7            — B2, AffectNet, 7 classes        [no Contempt class]
    """

    name = "hsemotion"

    def __init__(self, model: str = 'enet_b0_8_best_vgaf', smooth_window: int = 5):
        super().__init__(smooth_window)
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError:
            raise ImportError("pip install hsemotion-onnx")

        self._model_name = model
        print(f"[EmotionDetector] Loading HSEmotion ONNX model: {model} …")
        self._recognizer = HSEmotionRecognizer(model_name=model)
        self._labels     = self._recognizer.idx_to_class
        print(f"[EmotionDetector] HSEmotion ready  labels={list(self._labels.values())}")

    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        label, scores = self._recognizer.predict_emotions(face_rgb, logits=False)
        conf = float(np.max(scores)) * 100.0
        # Weighted-blend V/A from full softmax distribution (all 8 class probs)
        v, a = 0.0, 0.0
        for i, score in enumerate(scores):
            lbl = _norm(self._labels[i])
            lv, la = _VA_TABLE.get(lbl, (0.0, 0.0))
            v += score * lv
            a += score * la
        return label, conf, float(v), float(a)

    @property
    def all_labels(self) -> list[str]:
        return [_norm(v) for v in self._labels.values()]


# ─────────────────────────────────────────────────────────────────────────────
# Original EfficientNet (HQRAF) via Modules/emotion_processor.py
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetDetector(EmotionDetector):
    """
    Wraps the existing EmotionProcessor (EfficientNet-B0, HQRAF dataset).

    Kept as a comparison baseline.  Runs on CPU to avoid RTX 5060 sm_120
    incompatibility with torch 2.2.2.
    """

    name = "efficientnet"

    def __init__(self, smooth_window: int = 5):
        super().__init__(smooth_window)

        # Add CHATBOX_SERVER root to path if needed
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            from modules.emotion_processor import EmotionProcessor
        except ImportError:
            raise ImportError("modules/emotion_processor.py not found")

        print("[EmotionDetector] Loading EfficientNet (original HQRAF) …")
        self._proc = EmotionProcessor(device="cpu")
        ok, total  = self._proc.initialize()
        print(f"[EmotionDetector] EfficientNet ready ({ok}/{total} components)")

        # Keep Haar cascade for face detection inside the crop
        self._haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        emotion, conf, _ = self._proc.process_emotion_detection_realtime(face_bgr)
        v, a = _VA_TABLE.get(_norm(emotion), (0.0, 0.0))
        return emotion, float(conf), v, a


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, time as _time

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="hsemotion",
                   choices=["hsemotion", "hsemotion-b2", "efficientnet"])
    p.add_argument("--camera",  type=int, default=0)
    args = p.parse_args()

    print(f"\nAvailable backends: {EmotionDetector.available_backends()}")
    det = EmotionDetector.create(args.backend)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera"); sys.exit(1)

    print(f"\nRunning {args.backend} live — press Q to quit\n")
    t0 = _time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        emotion, conf = det.detect(frame, smooth=True)

        fps = frames / max(0.001, _time.time() - t0)
        label = f"{emotion} ({conf:.0f}%)  [{fps:.1f} fps]"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imshow(f"emotion: {args.backend}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
