"""
Pluggable demographic (region / heritage + age) estimation for the webcam KG loop.

This gives the companion robot *cultural awareness*: a soft, appearance-based
guess of a person's regional heritage (e.g. East Asian, European) and age band,
so replies can be culturally inclusive and age-appropriate.

    det = DemographicsDetector.create('fairface')          # ONNX, ~15 ms/frame CPU
    res = det.detect(frame_bgr, box=(x1, y1, x2, y2))      # DemographicsResult
    res.region, res.region_conf   # 'East Asian', 0.88
    res.age, res.age_conf         # '20-29', 0.61
    res.gender, res.gender_conf   # 'female', 0.9   (optional, off by default in the UI)

Design notes
------------
* Estimates are **uncertain, non-identifying appearance guesses** — never facts
  about a person. Downstream code (the prompt) is expected to treat region as a
  gentle hint for *inclusivity*, NOT to stereotype or assume language/origin.
* Demographics are stable per person, so `detect()` runs a longer-window vote
  (`_StableVote`) than the emotion smoother and only needs to fire every few
  frames. Once a region estimate is held with confidence over `lock_after`
  consistent votes it is "locked" and cheap to keep returning.

Backends
--------
fairface   FairFace ResNet-34 (ONNX) — the reference model for balanced-across-
           race age/gender/ethnicity estimation. One forward pass returns
           7 race + 2 gender + 9 age logits. ~15 ms/frame on CPU via onnxruntime.
           Weights (~85 MB) are cached to ~/.fairface/ on first use.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ── FairFace label taxonomy ───────────────────────────────────────────────────
# Output vector is 18-d: [0:7]=race, [7:9]=gender, [9:18]=age  (official ordering
# from dchen236/FairFace res34_fair_align_multi_7). We rename the 7 race classes
# to friendlier *region / heritage* labels for the companion context.

_FAIRFACE_RACE = [
    "White", "Black", "Latino_Hispanic",
    "East Asian", "Southeast Asian", "Indian", "Middle Eastern",
]
# race label → friendly region shown to the user / fed to the prompt
_REGION = {
    "White":            "European",
    "Black":            "African",
    "Latino_Hispanic":  "Latino/Hispanic",
    "East Asian":       "East Asian",
    "Southeast Asian":  "Southeast Asian",
    "Indian":           "South Asian",
    "Middle Eastern":   "Middle Eastern",
}
_GENDER = ["male", "female"]
_AGE = ["0-2", "3-9", "10-19", "20-29", "30-39",
        "40-49", "50-59", "60-69", "70+"]

# Coarse age *stage* derived from the FairFace bin — what the prompt actually
# uses (a robot doesn't need the exact decade, just child/teen/adult/senior).
def _age_stage(age_band: str) -> str:
    return {
        "0-2":   "young child",  "3-9":   "child",     "10-19": "teenager",
        "20-29": "young adult",  "30-39": "adult",     "40-49": "adult",
        "50-59": "older adult",  "60-69": "senior",    "70+":   "senior",
    }.get(age_band, "adult")


@dataclass
class DemographicsResult:
    """One demographic estimate. All confidences are 0–1 softmax probabilities."""
    region:       str   = "unknown"     # friendly heritage label, e.g. 'East Asian'
    region_conf:  float = 0.0
    age:          str   = "unknown"     # FairFace band, e.g. '20-29'
    age_conf:     float = 0.0
    age_stage:    str   = "adult"       # coarse stage: child/teen/adult/senior
    gender:       str   = "unknown"
    gender_conf:  float = 0.0
    locked:       bool  = False         # region held stable over the vote window

    @property
    def ok(self) -> bool:
        return self.region != "unknown"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# ── Stable per-person voter ───────────────────────────────────────────────────

class _StableVote:
    """
    Confidence-weighted majority vote over a sliding window.

    Demographics don't change frame-to-frame, so we accumulate votes and report
    the running winner. `locked` becomes True once the same winner has held for
    `lock_after` filled windows — a signal that the estimate is trustworthy and
    that callers may stop re-inferring so often.
    """

    def __init__(self, window: int = 30, lock_after: int = 8):
        self._hist: deque = deque(maxlen=window)
        self._lock_after = lock_after
        self._streak = 0
        self._last_winner: Optional[str] = None

    def update(self, label: str, conf: float) -> tuple[str, float, bool]:
        self._hist.append((label, conf))
        tally: dict[str, float] = {}
        for lbl, c in self._hist:
            tally[lbl] = tally.get(lbl, 0.0) + c
        winner = max(tally, key=tally.__getitem__)
        n = sum(1 for l, _ in self._hist if l == winner)
        avg = tally[winner] / max(1, n)

        if winner == self._last_winner:
            self._streak += 1
        else:
            self._streak = 0
            self._last_winner = winner
        locked = self._streak >= self._lock_after
        return winner, avg, locked


# ── Abstract base ─────────────────────────────────────────────────────────────

class DemographicsDetector:
    """
    Estimate region / heritage + age from a BGR face crop.

    Subclass and override `_infer(face_bgr) -> DemographicsResult` (raw, no vote).
    Call `detect()` for the public interface with per-person stabilisation.
    """

    name: str = "base"

    def __init__(self, vote_window: int = 30, lock_after: int = 8):
        self._region_vote = _StableVote(vote_window, lock_after)
        self._age_vote    = _StableVote(vote_window, lock_after)

    def _infer(self, face_bgr: np.ndarray) -> DemographicsResult:
        raise NotImplementedError

    def detect(
        self,
        frame_bgr: np.ndarray,
        box:       Optional[tuple] = None,
        smooth:    bool = True,
    ) -> DemographicsResult:
        """
        Args:
            frame_bgr : full BGR frame, or a face crop when box is None.
            box       : (x1, y1, x2, y2) face box in frame_bgr.
            smooth    : stabilise across frames with the per-person vote window.
        """
        if box is not None:
            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            # FairFace was trained on padded, aligned crops — add ~25% margin so
            # the forehead/chin/ears are included (improves both age and region).
            h, w = frame_bgr.shape[:2]
            mx = int((x2 - x1) * 0.25)
            my = int((y2 - y1) * 0.25)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(w, x2 + mx), min(h, y2 + my)
            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                return DemographicsResult()
        else:
            face = frame_bgr

        try:
            res = self._infer(face)
        except Exception:
            return DemographicsResult()

        if smooth and res.ok:
            res.region, res.region_conf, r_lock = self._region_vote.update(
                res.region, res.region_conf)
            res.age, res.age_conf, _ = self._age_vote.update(res.age, res.age_conf)
            res.age_stage = _age_stage(res.age)
            res.locked = r_lock
        return res

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def create(backend: str = "fairface", **kwargs) -> "DemographicsDetector":
        b = backend.lower()
        if b == "fairface":
            return FairFaceDetector(**kwargs)
        raise ValueError(f"Unknown backend {backend!r}. Choose: 'fairface'")

    @staticmethod
    def available_backends() -> list[str]:
        backends: list[str] = []
        try:
            import onnxruntime  # noqa: F401
            backends.append("fairface")
        except ImportError:
            pass
        return backends


# ─────────────────────────────────────────────────────────────────────────────
# FairFace ONNX backend
# ─────────────────────────────────────────────────────────────────────────────

_FAIRFACE_URL   = "https://huggingface.co/garavv/fairface-onnx/resolve/main/fairface.onnx"
_FAIRFACE_CACHE = os.path.join(os.path.expanduser("~"), ".fairface", "fairface.onnx")
# ImageNet normalisation — FairFace's torchvision preprocessing.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FairFaceDetector(DemographicsDetector):
    """
    FairFace ResNet-34 (ONNX) region / age / gender estimator.

    One forward pass returns 18 logits: 7 race + 2 gender + 9 age. Balanced
    across race by construction (the FairFace dataset was built to reduce the
    racial bias of earlier face-attribute sets). Runs on CPU via onnxruntime —
    no torch/CUDA dependency for this path.
    """

    name = "fairface"

    def __init__(self, vote_window: int = 30, lock_after: int = 8,
                 model_path: Optional[str] = None):
        super().__init__(vote_window, lock_after)
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime")

        path = model_path or _FAIRFACE_CACHE
        if not os.path.exists(path):
            path = self._download(path)

        print(f"[Demographics] Loading FairFace ONNX model: {path} …")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self._sess = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        print("[Demographics] FairFace ready  "
              f"regions={sorted(set(_REGION.values()))}")

    @staticmethod
    def _download(dest: str) -> str:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[Demographics] downloading FairFace weights (~85 MB) → {dest}")
        try:
            urllib.request.urlretrieve(_FAIRFACE_URL, dest)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"FairFace model download failed ({exc}). "
                f"Manually place fairface.onnx at {dest} (from {_FAIRFACE_URL})."
            ) from exc
        return dest

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.transpose(x, (2, 0, 1))          # HWC → CHW
        return x[np.newaxis, :].astype(np.float32)

    def _infer(self, face_bgr: np.ndarray) -> DemographicsResult:
        x = self._preprocess(face_bgr)
        out = self._sess.run(None, {self._in_name: x})[0][0]   # (18,)

        race_p   = _softmax(out[0:7])
        gender_p = _softmax(out[7:9])
        age_p    = _softmax(out[9:18])

        ri = int(np.argmax(race_p))
        gi = int(np.argmax(gender_p))
        ai = int(np.argmax(age_p))

        band = _AGE[ai]
        return DemographicsResult(
            region      = _REGION[_FAIRFACE_RACE[ri]],
            region_conf = float(race_p[ri]),
            age         = band,
            age_conf    = float(age_p[ai]),
            age_stage   = _age_stage(band),
            gender      = _GENDER[gi],
            gender_conf = float(gender_p[gi]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time as _time

    p = argparse.ArgumentParser(description="Live FairFace region/age demo")
    p.add_argument("--backend", default="fairface", choices=["fairface"])
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--image",   default=None, help="run on a single image instead")
    args = p.parse_args()

    print(f"\nAvailable backends: {DemographicsDetector.available_backends()}")
    det = DemographicsDetector.create(args.backend)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"cannot read {args.image}"); sys.exit(1)
        r = det.detect(img, smooth=False)
        print(f"region={r.region} ({r.region_conf:.2f})  "
              f"age={r.age}/{r.age_stage} ({r.age_conf:.2f})  "
              f"gender={r.gender} ({r.gender_conf:.2f})")
        sys.exit(0)

    # Use Haar to get a rough box so the demo runs standalone.
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera"); sys.exit(1)

    print(f"\nRunning {args.backend} live — press Q to quit\n")
    t0, frames = _time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.15, 5, minSize=(80, 80))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            r = det.detect(frame, box=(x, y, x + w, y + h), smooth=True)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
            lock = "*" if r.locked else ""
            cv2.putText(frame, f"{r.region} {r.region_conf:.0%}{lock}",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"{r.age} ({r.age_stage})",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        fps = frames / max(0.001, _time.time() - t0)
        cv2.putText(frame, f"{fps:.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("demographics: fairface", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
