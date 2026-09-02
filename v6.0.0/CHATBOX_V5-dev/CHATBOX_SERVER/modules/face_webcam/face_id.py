"""
Face enrollment and recognition using facenet-pytorch.

Detector modes (how faces are *located* in the frame):
    "opencv"  — a single OpenCV Haar `detectMultiScale` pass finds every face box.
                Each box is cropped and embedded with InceptionResnetV1 for identity,
                and the *same* box is handed to the emotion detector. One detection
                feeds both pipelines (no separate MTCNN/Haar per model).  [default]
    "mtcnn"   — facenet-pytorch MTCNN detects + landmark-aligns each face before the
                embedding. Higher recognition accuracy, but MTCNN is the locator.

Embeddings are always InceptionResnetV1 (VGGFace2, 512-dim). If facenet-pytorch is
not installed, both modes fall back to a coarse Haar pixel embedding.
Runs on CPU — avoids CUDA version conflicts with the system torch install.

NOTE: enrollment and identification must use the SAME detector mode — a face
enrolled under MTCNN alignment will match poorly against an OpenCV-cropped probe
(and vice-versa). Re-enroll people after switching modes.

Usage:
    fi = FaceIdentifier(detector="opencv")
    fi.enroll("alice", bgr_frame)          # call multiple times to average embeddings
    fi.enroll("bob",   bgr_frame)
    person_id, sim, box = fi.identify(bgr_frame)   # box is (x1,y1,x2,y2) or None
    fi.save("faces.npz")
    fi.load("faces.npz")
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# ── Load facenet-pytorch models (CPU only — RTX 5060 needs sm_120, see notes) ─

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False

# Fallback: OpenCV Haar cascade for rough pixel-based identity
_HAAR_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# Profile cascade — detects a head turned to one side. The frontal cascade finds
# NOTHING on a turned head, so without this an off-angle face produces no box at
# all and cannot be recognised (or emotion-read) no matter how good the gallery is.
_HAAR_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"

# Crop margin per detection source. Profile boxes are framed tighter and shifted
# toward the visible half of the head, so they need more padding than frontal ones
# or the far cheek gets clipped and similarity suffers.
_MARGIN = {"frontal": 0.20, "profile-l": 0.30, "profile-r": 0.30}
# Frontal wins ties during dedupe: its crop suits both facenet and the emotion model.
_SRC_RANK = {"frontal": 0, "profile-l": 1, "profile-r": 1}

# Default detection downscale — must match what the live loop passes to identify_all,
# so enrolment and identification crop the face identically.
_DET_SCALE = 0.5

# ── Prototype gallery tuning ─────────────────────────────────────────────────
ANCHOR, ADAPTIVE = 0.0, 1.0   # prototype origin (stored in _meta column 1)
K_ANCHOR = 12                 # max deliberately-enrolled views per person
K_ADAPT  = 6                  # max self-learned views per person
W_CAP    = 50.0               # max weight of one prototype (keeps it adapting slowly)

# Guided enrollment poses. Angle robustness comes from COVERAGE, not frame count:
# 30 frontal frames collapse into a single view, five poses give five.
DEFAULT_POSES = [
    ("Look STRAIGHT at the camera", "front"),
    ("Turn your head LEFT  (~30 deg)", "left"),
    ("Turn your head RIGHT (~30 deg)", "right"),
    ("Lift your CHIN UP", "up"),
    ("Tuck your CHIN DOWN", "down"),
]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _area(box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _center_inside(inner, outer) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2.0, (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _dedupe_boxes(cands: list, iou_thresh: float = 0.35) -> list:
    """Collapse overlapping detections of the SAME head to one (box, src).

    Uses IoU *plus* a containment test: a frontal and a profile box on one head are
    typically offset rather than nested and land around IoU 0.25-0.35, which slips
    past a pure-IoU check and leaves a phantom second 'unknown' face — which would
    also silently disable adaptive capture (it requires exactly one face).
    """
    ordered = sorted(cands, key=lambda c: (_SRC_RANK.get(c[1], 9), -_area(c[0])))
    kept: list = []
    for box, src in ordered:
        if any(_iou(box, k) >= iou_thresh
               or _center_inside(box, k) or _center_inside(k, box)
               for k, _ in kept):
            continue
        kept.append((box, src))
    return kept


class FaceIdentifier:
    """
    Named-face enrollment and identification.

    Each enrolled person is represented as an averaged 512-dim L2-normalised
    embedding from InceptionResnetV1 (VGGFace2 pretrained).  Identification
    uses cosine similarity; faces below `threshold` are reported as unknown.

    Enrollment is additive — calling enroll() multiple times for the same name
    updates the running average, so you can refine a profile over many frames.
    """

    UNKNOWN = "__unknown__"

    def __init__(
        self,
        threshold: float = 0.75,
        detector:  str   = "opencv",
        *,
        multi_pose:       bool  = True,
        retain_threshold: float = 0.62,
        switch_margin:    float = 0.08,
        id_margin:        float = 0.05,
        tau_dup:          float = 0.92,
        k_anchor:         int   = K_ANCHOR,
        k_adapt:          int   = K_ADAPT,
    ) -> None:
        """
        Args:
            threshold:        Cosine similarity needed to ACQUIRE an identity [0,1].
            detector:         "opencv" (Haar, shared with emotion) or "mtcnn".
            multi_pose:       Also run the profile cascades (opencv mode) so turned
                              heads are detected at all.
            retain_threshold: Lower bar to KEEP the identity the caller already holds
                              (hysteresis). Must be <= threshold.
            switch_margin:    How far a rival must beat the held identity to take over.
            id_margin:        Runner-up margin; only applied with >= 2 people enrolled.
            tau_dup:          Similarity at/above which a new view is treated as a
                              duplicate of an existing prototype and merged into it.
            k_anchor/k_adapt: Max enrolled / self-learned prototypes per person.
        """
        self.threshold = threshold
        self.detector  = detector.lower()
        if self.detector not in ("opencv", "mtcnn"):
            raise ValueError(f"detector must be 'opencv' or 'mtcnn', got {detector!r}")

        self.multi_pose       = multi_pose
        self.retain_threshold = min(retain_threshold, threshold)
        self.switch_margin    = switch_margin
        self.id_margin        = id_margin
        self.tau_dup          = tau_dup
        self.k_anchor         = k_anchor
        self.k_adapt          = k_adapt

        # Per-person prototype GALLERY: several views per person instead of one
        # averaged vector. Averaging across poses produced a centroid that matched
        # no pose well — the root cause of "only works at one angle".
        #   _protos[name] : (K, 512) float32, each row L2-normalised
        #   _meta[name]   : (K, 4)   float32 — weight, origin, created_ts, last_hit_ts
        #   _counts[name] : total frames ever contributed (unchanged semantics)
        self._protos: dict[str, np.ndarray] = {}
        self._meta:   dict[str, np.ndarray] = {}
        self._counts: dict[str, int]        = {}

        self._device = torch.device("cpu")
        self._mtcnn:     Optional[MTCNN]               = None
        self._mtcnn_all: Optional[MTCNN]               = None  # keep_all=True for multi-face
        self._resnet:    Optional[InceptionResnetV1]   = None
        self._haar:         Optional[cv2.CascadeClassifier] = None
        self._haar_profile: Optional[cv2.CascadeClassifier] = None

        self._init_backend()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        # Haar is always loaded when present: it is the locator for "opencv" mode
        # and the fallback when facenet-pytorch is unavailable.
        if os.path.exists(_HAAR_PATH):
            self._haar = cv2.CascadeClassifier(_HAAR_PATH)
        # Profile cascade is optional — degrade to frontal-only if it is missing.
        if os.path.exists(_HAAR_PROFILE):
            self._haar_profile = cv2.CascadeClassifier(_HAAR_PROFILE)
        elif self.multi_pose:
            print("[FaceID] profile cascade unavailable — frontal detection only "
                  "(turned heads will not be detected)")

        if _FACENET_AVAILABLE:
            self._resnet = InceptionResnetV1(pretrained="vggface2").eval()
            # MTCNN is only needed when it is the chosen locator — skip it in
            # "opencv" mode so we don't load/run a second detector.
            if self.detector == "mtcnn":
                self._mtcnn = MTCNN(
                    image_size=160, margin=20,
                    keep_all=False,       # single-face path (enroll, identify)
                    device=self._device,
                    post_process=True,
                )
                self._mtcnn_all = MTCNN(
                    image_size=160, margin=20,
                    keep_all=True,        # multi-face path (identify_all)
                    device=self._device,
                    post_process=True,
                )
        elif self._haar is not None:
            print("[FaceID] WARNING — facenet-pytorch not installed; using pixel fallback")
        else:
            print("[FaceID] ERROR — neither facenet-pytorch nor Haar cascade available")

    @property
    def backend(self) -> str:
        if self._resnet is not None:
            # facenet embeddings, located by either Haar (opencv) or MTCNN.
            return f"facenet+{self.detector}"
        if self._haar is not None:
            return "haar-pixel"
        return "none"

    # ── Internal embedding helpers ────────────────────────────────────────────

    def _embed_facenet(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """Return (embedding_1d, box_xyxy) or (None, None) if no face found."""
        img_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # detect() returns boxes [[x1,y1,x2,y2],...] and probs
        boxes, probs = self._mtcnn.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return None, None

        # Pick the highest-confidence face
        best_idx = int(np.argmax(probs))
        box = tuple(int(v) for v in boxes[best_idx])  # (x1, y1, x2, y2)

        # forward() returns aligned, normalised tensor for the same face
        face_tensor = self._mtcnn(img_rgb)
        if face_tensor is None:
            return None, None

        with torch.no_grad():
            emb = self._resnet(face_tensor.unsqueeze(0))   # [1, 512]
        # L2-normalise so cosine_sim == dot product
        emb_np = emb.cpu().numpy()[0]
        emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)
        return emb_np, box

    def _embed_haar(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """Pixel-level fallback embedding via resized face crop (grayscale 32×32)."""
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.15,
                                            minNeighbors=4, minSize=(50, 50))
        if len(faces) == 0:
            return None, None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        crop  = cv2.resize(gray[y:y+h, x:x+w], (32, 32)).astype(np.float32)
        emb   = crop.flatten() / (np.linalg.norm(crop) + 1e-8)
        box   = (x, y, x + w, y + h)
        return emb, box

    # ── OpenCV (Haar) locator — shared single detection ───────────────────────

    def _detect_boxes_tagged(
        self, frame_bgr: np.ndarray, scale: float = 1.0, max_faces: int = 4,
        multi_pose: Optional[bool] = None,
    ) -> list[tuple]:
        """Haar pass(es) → [((x1,y1,x2,y2), src), ...] largest first.

        `src` is 'frontal', 'profile-l' or 'profile-r' and tells the caller which crop
        margin to use. With multi_pose on we run three cheap cascades — frontal,
        profile, and profile on a horizontally FLIPPED image (a left-facing cascade
        finds right-facing heads once the image is mirrored) — then dedupe.

        Detection runs on a `scale`-downsized grayscale image for speed; boxes are
        mapped back to original-frame coordinates.
        """
        if self._haar is None:
            return []
        if multi_pose is None:
            multi_pose = self.multi_pose

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, (int(gray.shape[1] * scale),
                                     int(gray.shape[0] * scale)))

        def _xywh(f) -> tuple:
            x, y, w, h = f
            return (int(x), int(y), int(x + w), int(y + h))

        cands = [(_xywh(f), "frontal")
                 for f in self._haar.detectMultiScale(
                     gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))]

        if multi_pose and self._haar_profile is not None:
            # minNeighbors 4 (not 5): the profile cascade is weaker; 5 loses too much.
            for f in self._haar_profile.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40)):
                cands.append((_xywh(f), "profile-l"))
            W = gray.shape[1]
            for (x, y, w, h) in self._haar_profile.detectMultiScale(
                    cv2.flip(gray, 1), scaleFactor=1.15, minNeighbors=4,
                    minSize=(40, 40)):
                cands.append(((int(W - x - w), int(y), int(W - x), int(y + h)),
                              "profile-r"))   # mirror x back to original coords

        kept = _dedupe_boxes(cands)
        kept.sort(key=lambda c: _area(c[0]), reverse=True)
        kept = kept[:max_faces]

        inv = 1.0 / scale
        return [(tuple(int(v * inv) for v in box), src) for box, src in kept]

    def _detect_boxes_opencv(
        self, frame_bgr: np.ndarray, scale: float = 1.0, max_faces: int = 4,
    ) -> list[tuple]:
        """Bare boxes, no source tags — thin wrapper over _detect_boxes_tagged."""
        return [b for b, _src in self._detect_boxes_tagged(
            frame_bgr, scale=scale, max_faces=max_faces)]

    def _crop_with_margin(
        self, frame_bgr: np.ndarray, box: tuple, margin: Optional[float] = None,
        src: str = "frontal",
    ) -> np.ndarray:
        """Crop `box` from the frame with a relative margin (approximates MTCNN's
        margin so Haar-located crops embed closer to MTCNN-aligned ones).

        The margin defaults to a per-source value (_MARGIN): profile boxes crop
        tighter around the visible half of the head and need more padding.
        """
        if margin is None:
            margin = _MARGIN.get(src, 0.20)
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        return frame_bgr[y1:y2, x1:x2]

    def _embed_crop_facenet(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Embed an already-cropped face with InceptionResnetV1.

        Applies facenet's fixed image standardization ((x-127.5)/128) — the same
        preprocessing MTCNN(post_process=True) uses — so opencv-located crops and
        mtcnn-aligned faces land in a comparable embedding space.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(rgb, (160, 160)).astype(np.float32)
        face = (face - 127.5) / 128.0
        tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)  # [1,3,160,160]
        with torch.no_grad():
            emb = self._resnet(tensor)
        emb_np = emb.cpu().numpy()[0]
        return emb_np / (np.linalg.norm(emb_np) + 1e-8)

    def embed_box(self, frame_bgr: np.ndarray, box: tuple,
                  src: str = "frontal") -> Optional[np.ndarray]:
        """Embed one already-located face box. Public so callers that already have a
        box (enrolment, adaptive capture) never re-run detection."""
        return self._embed_crop_facenet(
            self._crop_with_margin(frame_bgr, box, src=src))

    def _embed_opencv(
        self, frame_bgr: np.ndarray, scale: float = _DET_SCALE,
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """Single-face path for opencv mode: largest Haar box → facenet embedding.

        Detects at the SAME scale identify_all uses — a different scale quantises the
        Haar scale-space differently, so enrolment crops would be a few pixels off
        from identification crops and every match would pay a small similarity tax.
        """
        tagged = self._detect_boxes_tagged(frame_bgr, scale=scale, max_faces=1)
        if not tagged:
            return None, None
        box, src = tagged[0]
        emb = self.embed_box(frame_bgr, box, src=src)
        if emb is None:
            return None, None
        return emb, box

    def _get_embedding_and_box(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        # opencv mode with facenet embeddings: one Haar detection, resnet embedding.
        if self.detector == "opencv" and self._resnet is not None and self._haar is not None:
            return self._embed_opencv(frame_bgr)
        if self._resnet is not None:
            return self._embed_facenet(frame_bgr)
        if self._haar is not None:
            return self._embed_haar(frame_bgr)
        return None, None

    # ── Prototype gallery ─────────────────────────────────────────────────────

    def add_prototype(self, name: str, emb: np.ndarray, *,
                      origin: float = ANCHOR, now: Optional[float] = None) -> str:
        """Fold a (unit-norm) embedding into `name`'s gallery.

        Returns what happened: 'created' | 'merged' | 'inserted' | 'replaced'.

          * close to an existing view (>= tau_dup) → weighted-merge into it, so
            repeated frames of the same pose refine that prototype instead of
            filling the gallery with near-duplicates;
          * genuinely new view with room in its tier  → insert;
          * tier full → replace the most redundant view in THAT tier.

        Tiers never mix: self-learned (ADAPTIVE) prototypes can never evict a
        deliberately enrolled (ANCHOR) one, so a bad auto-capture is always
        recoverable via reset_adaptive() and enrolment data is untouchable.
        """
        now = time.time() if now is None else now
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        P = self._protos.get(name)
        if P is None or len(P) == 0:
            self._protos[name] = emb[None, :].copy()
            self._meta[name]   = np.array([[1.0, origin, now, now]], np.float32)
            return "created"

        sims = P @ emb
        j    = int(np.argmax(sims))
        if float(sims[j]) >= self.tau_dup:          # same view → refine in place
            w = float(self._meta[name][j, 0])
            v = (P[j] * w + emb) / (w + 1.0)
            self._protos[name][j] = v / (np.linalg.norm(v) + 1e-8)
            self._meta[name][j, 0] = min(w + 1.0, W_CAP)
            self._meta[name][j, 3] = now
            return "merged"

        tier_idx = np.where(self._meta[name][:, 1] == origin)[0]
        cap = self.k_anchor if origin == ANCHOR else self.k_adapt
        if len(tier_idx) < cap:                     # new view, room available
            self._protos[name] = np.vstack([P, emb[None, :]]).astype(np.float32)
            self._meta[name]   = np.vstack(
                [self._meta[name], [[1.0, origin, now, now]]]).astype(np.float32)
            return "inserted"

        v = self._most_redundant(name, tier_idx)    # tier full → evict its twin
        self._protos[name][v] = emb
        self._meta[name][v]   = [1.0, origin, now, now]
        return "replaced"

    def _most_redundant(self, name: str, idxs: np.ndarray) -> int:
        """Row index (drawn from `idxs`) that carries the least unique information:
        the weaker half of the CLOSEST pair in that tier.

        Always removing one of the two most similar rows keeps the surviving set's
        minimum pairwise distance non-decreasing — i.e. the gallery keeps spanning
        different poses instead of hoarding copies of the easy frontal view.
        """
        idxs = np.asarray(idxs)
        if len(idxs) == 1:
            return int(idxs[0])
        S = self._protos[name][idxs] @ self._protos[name][idxs].T
        np.fill_diagonal(S, -np.inf)
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        ma, mb = self._meta[name][idxs[a]], self._meta[name][idxs[b]]
        # lower weight loses; tie → the one hit longest ago loses
        return int(idxs[a]) if (ma[0], ma[3]) <= (mb[0], mb[3]) else int(idxs[b])

    def _reduce_tiers(self, name: str) -> None:
        """Evict most-redundant rows until each tier is within its cap."""
        for origin, cap in ((ANCHOR, self.k_anchor), (ADAPTIVE, self.k_adapt)):
            while True:
                idx = np.where(self._meta[name][:, 1] == origin)[0]
                if len(idx) <= cap:
                    break
                drop = self._most_redundant(name, idx)
                self._protos[name] = np.delete(self._protos[name], drop, axis=0)
                self._meta[name]   = np.delete(self._meta[name],   drop, axis=0)

    def _centroid(self, name: str) -> np.ndarray:
        """Weighted mean of a person's prototypes — only for the legacy npz field."""
        P, M = self._protos[name], self._meta[name]
        v = (P * M[:, 0:1]).sum(axis=0)
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)

    def _score(self, name: str, emb: np.ndarray) -> tuple[float, float]:
        """(retain_score, acquire_score) of `emb` against one person's gallery.

        retain  = best-matching view. Good for "is this still the person I hold?".
        acquire = mean of the top TWO views once the gallery is big enough, so one
                  freak prototype cannot by itself admit a stranger. A genuinely new
                  pose lights up only one row and is instead rescued by hysteresis.
        """
        s = np.sort(self._protos[name] @ emb)[::-1]
        smax = float(s[0])
        sacq = float(s[:2].mean()) if len(s) >= 4 else smax
        return smax, sacq

    def _match(self, emb: np.ndarray,
               sticky: Optional[str] = None) -> tuple[Optional[str], float]:
        """Identify one embedding, with hysteresis around the caller's held identity.

        `sticky` is who the caller currently believes this face is. That identity is
        kept on the LOWER retain_threshold and is only displaced by a rival that both
        clears the acquire threshold and beats it by switch_margin — so a brief dip
        (turning the head) no longer flips the label to unknown and back.
        """
        if not self._protos:
            return None, 0.0
        scored = {n: self._score(n, emb) for n in self._protos}
        best   = max(scored, key=lambda n: scored[n][1])
        s_acq_best = scored[best][1]
        second = max((scored[n][1] for n in scored if n != best), default=-1.0)

        if sticky in scored:
            s_ret_sticky = scored[sticky][0]
            if (s_ret_sticky >= self.retain_threshold
                    and s_ret_sticky >= s_acq_best - self.switch_margin):
                return sticky, s_ret_sticky
            if (s_acq_best >= self.threshold
                    and (s_acq_best - s_ret_sticky) >= self.switch_margin):
                return best, s_acq_best
            return None, s_ret_sticky

        ok = (s_acq_best >= self.threshold
              and (len(scored) < 2 or (s_acq_best - second) >= self.id_margin))
        return (best if ok else None), s_acq_best

    def gallery_size(self, name: str) -> tuple[int, int]:
        """(n_anchor, n_adaptive) prototypes stored for `name`."""
        M = self._meta.get(name)
        if M is None or len(M) == 0:
            return (0, 0)
        return (int((M[:, 1] == ANCHOR).sum()), int((M[:, 1] == ADAPTIVE).sum()))

    def reset_adaptive(self, name: Optional[str] = None) -> int:
        """Drop self-learned prototypes (all people, or just `name`). Returns how
        many rows were removed. The undo button for adaptive capture."""
        removed = 0
        for n in ([name] if name else list(self._protos)):
            M = self._meta.get(n)
            if M is None:
                continue
            keep = M[:, 1] != ADAPTIVE
            removed += int((~keep).sum())
            if keep.all():
                continue
            if not keep.any():        # would leave the person with no views at all
                continue
            self._protos[n] = self._protos[n][keep]
            self._meta[n]   = M[keep]
        return removed

    # ── Public API ────────────────────────────────────────────────────────────

    def enroll(self, name: str, frame_bgr: np.ndarray) -> bool:
        """
        Add the face in one BGR frame to `name`'s gallery as an ENROLLED view.

        Repeated calls build up a multi-view profile: frames of the same pose refine
        one prototype, genuinely different poses become new ones. Call across several
        frames (and ideally several head angles) for a robust profile.

        Returns True if a face was found and enrolled, False otherwise.
        """
        emb, _box = self._get_embedding_and_box(frame_bgr)
        if emb is None:
            return False
        self.add_prototype(name, emb, origin=ANCHOR)
        self._counts[name] = self._counts.get(name, 0) + 1
        return True

    def enroll_from_camera(
        self,
        name: str,
        camera_index: int = 0,
        n_captures: int = 15,
        countdown: int = 3,
        poses: Optional[list] = None,
    ) -> bool:
        """
        Guided multi-angle enrollment: opens the webcam and walks the person through
        several head poses, storing each as a DISTINCT view.

        Capturing 30 frontal frames does not make recognition angle-robust — they all
        collapse into one view. Pose coverage is what matters, so this prompts for
        front / left / right / up / down and refuses frames that merely repeat a view
        already stored (otherwise ignoring the prompts silently re-records the front).

        Returns True if at least one frame was successfully enrolled.
        """
        poses = poses or DEFAULT_POSES
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[FaceID] Cannot open camera {camera_index}")
            return False

        per_pose = max(1, math.ceil(n_captures / len(poses)))
        captured = 0
        fps_est  = cap.get(cv2.CAP_PROP_FPS) or 30
        win      = f"Enroll: {name}"

        print(f"[FaceID] Enrolling '{name}' — {len(poses)} poses × {per_pose} frames. "
              "Follow the on-screen prompt.")

        def _hud(frame, line1, line2, col):
            o = frame.copy()
            cv2.putText(o, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.putText(o, line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            na, _ = self.gallery_size(name)
            cv2.putText(o, f"stored {na} distinct view(s)", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(win, o)
            cv2.waitKey(1)

        try:
            for pi, (prompt, _tag) in enumerate(poses):
                # Prompt + countdown for this pose
                t_end = time.time() + max(1.5, countdown)
                while time.time() < t_end:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    left = int(t_end - time.time()) + 1
                    _hud(frame, prompt, f"starting in {left}s …", (255, 255, 0))

                got, attempts = 0, 0
                max_att = per_pose * 8
                while got < per_pose and attempts < max_att:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    attempts += 1

                    emb, _box = self._get_embedding_and_box(frame)
                    if emb is None:
                        _hud(frame, prompt, "no face found — adjust position",
                             (0, 0, 255))
                        continue
                    # After the first pose, insist on a genuinely different view.
                    P = self._protos.get(name)
                    if pi > 0 and P is not None and len(P) and \
                            float((P @ emb).max()) >= self.tau_dup:
                        _hud(frame, prompt,
                             "turn further — I already have this view", (0, 165, 255))
                        continue

                    self.add_prototype(name, emb, origin=ANCHOR)
                    self._counts[name] = self._counts.get(name, 0) + 1
                    got += 1
                    captured += 1
                    _hud(frame, f"{prompt}  [{got}/{per_pose}]",
                         f"pose {pi + 1}/{len(poses)} — hold still", (0, 255, 0))

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if captured > 0:
            na, nd = self.gallery_size(name)
            print(f"[FaceID] '{name}' enrolled: {na} distinct view(s) "
                  f"from {self._counts.get(name, 0)} frames.")
        else:
            print(f"[FaceID] Enrollment failed — no faces captured for '{name}'.")
        return captured > 0

    def identify(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[str], float, Optional[tuple]]:
        """
        Identify the most prominent face in frame_bgr.

        Returns:
            person_id  — matched name, or None if no face / no match above threshold
            similarity — cosine similarity of best match (0–1); 0.0 if no face
            box        — (x1, y1, x2, y2) in pixels, or None if no face found
        """
        emb, box = self._get_embedding_and_box(frame_bgr)
        if emb is None:
            return None, 0.0, box
        pid, sim = self._match(emb)
        return pid, sim, box

    def identify_all(
        self,
        frame_bgr:  np.ndarray,
        max_faces:  int   = 4,
        scale:      float = _DET_SCALE,
        *,
        sticky:     Optional[str] = None,
        with_emb:   bool = False,
    ) -> list[tuple]:
        """
        Identify every face visible in the frame.

        Returns:
            List of (person_id, similarity, box) — one entry per detected face.
            person_id is None for faces below the similarity threshold.
            Faces are ordered largest-first.
            Returns [] when no faces are found.

        Args:
            sticky:   Identity the caller currently holds. Hysteresis is applied to
                      the LARGEST face only, so a second person in frame is never
                      pulled toward the held identity.
            with_emb: Yield (person_id, similarity, box, embedding) 4-tuples instead.
                      The embedding is computed either way — this just stops it being
                      thrown away, so adaptive capture costs no extra compute.
        """
        def _pack(pid, sim, box, emb, is_primary):
            return (pid, sim, box, emb) if with_emb else (pid, sim, box)

        # ── OpenCV mode: one Haar pass locates every face, facenet embeds each ──
        if self.detector == "opencv" and self._resnet is not None and self._haar is not None:
            tagged = self._detect_boxes_tagged(frame_bgr, scale=scale, max_faces=max_faces)
            results = []
            for i, (box, src) in enumerate(tagged):
                emb = self.embed_box(frame_bgr, box, src=src)
                if emb is None:
                    continue
                # tagged is largest-first, so i == 0 is the primary face
                pid, sim = self._match(emb, sticky=sticky if i == 0 else None)
                results.append(_pack(pid, float(sim), box, emb, i == 0))
            return results

        # ── Haar fallback: single face only ──────────────────────────────────
        if self._mtcnn_all is None:
            if self._haar is None:
                return []
            emb, box = self._embed_haar(frame_bgr)
            if box is None:
                return []
            if emb is None:
                return [_pack(None, 0.0, box, None, True)]
            pid, sim = self._match(emb, sticky=sticky)
            return [_pack(pid, float(sim), box, emb, True)]

        # ── FaceNet path ─────────────────────────────────────────────────────
        # Downscale for faster MTCNN (boxes scaled back to original coords)
        if scale != 1.0:
            h0, w0 = frame_bgr.shape[:2]
            small  = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)))
        else:
            small  = frame_bgr

        img_rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        boxes, probs = self._mtcnn_all.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return []

        face_tensors = self._mtcnn_all(img_rgb)   # [N,3,160,160] or [3,160,160] or None
        if face_tensors is None:
            return []
        if face_tensors.dim() == 3:               # single face edge case
            face_tensors = face_tensors.unsqueeze(0)

        with torch.no_grad():
            embs = self._resnet(face_tensors)     # [N, 512]

        results = []
        n = min(len(boxes), embs.shape[0], max_faces)
        # Largest box is the primary face — only it gets the sticky/hysteresis pass.
        areas = [_area(tuple(int(v) for v in boxes[i])) for i in range(n)]
        primary_i = int(np.argmax(areas)) if areas else -1
        for i in range(n):
            prob = float(probs[i]) if probs[i] is not None else 0.0
            if prob < 0.90:                       # skip low-confidence detections
                continue
            # Scale box coordinates back to original frame size
            inv = 1.0 / scale
            box = tuple(int(v * inv) for v in boxes[i])

            emb_np = embs[i].cpu().numpy()
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)

            pid, sim = self._match(emb_np,
                                   sticky=sticky if i == primary_i else None)
            results.append(_pack(pid, float(sim), box, emb_np, i == primary_i))

        return results

    def detect_boxes(
        self, frame_bgr: np.ndarray, max_faces: int = 4, scale: float = 0.5,
    ) -> list[tuple]:
        """Locate faces WITHOUT computing identity embeddings — much cheaper than
        identify_all. Returns [(x1,y1,x2,y2), ...] ordered largest-first.

        Used by the detection worker's "hold" phase: between periodic identity
        checks we still need face boxes (for emotion + display) but skip the
        expensive embedding/matching, which also stops the identity from flickering.
        """
        # opencv locator (Haar) — already box-only.
        if self.detector == "opencv" and self._haar is not None:
            return self._detect_boxes_opencv(frame_bgr, scale=scale, max_faces=max_faces)
        # mtcnn locator — detect boxes only (no forward pass through resnet).
        if self._mtcnn_all is not None:
            if scale != 1.0:
                h0, w0 = frame_bgr.shape[:2]
                small  = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)))
            else:
                small  = frame_bgr
            img_rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            boxes, probs = self._mtcnn_all.detect(img_rgb)
            if boxes is None or len(boxes) == 0:
                return []
            inv, out = 1.0 / scale, []
            for i in range(min(len(boxes), max_faces)):
                if probs[i] is None or probs[i] < 0.90:
                    continue
                out.append(tuple(int(v * inv) for v in boxes[i]))
            return out
        # haar pixel fallback.
        if self._haar is not None:
            return self._detect_boxes_opencv(frame_bgr, scale=scale, max_faces=max_faces)
        return []

    def known_people(self) -> list[str]:
        """One entry per PERSON (not per prototype) — `_next_guest_id` scans this."""
        return sorted(self._protos.keys())

    def forget(self, name: str) -> None:
        self._protos.pop(name, None)
        self._meta.pop(name, None)
        self._counts.pop(name, None)

    def rename(self, old_name: str, new_name: str) -> bool:
        """Move a person's gallery from `old_name` to `new_name` — used when a
        provisionally-enrolled face (e.g. 'guest_3') tells us their real name.

        If `new_name` already exists the two galleries are CONCATENATED, not averaged:
        a guest gallery and an existing one are the same person captured in different
        sessions and poses, and averaging them would destroy exactly the pose diversity
        this gallery exists to accumulate. Tiers are then trimmed back to their caps.
        Returns True if `old_name` existed.
        """
        if old_name == new_name or old_name not in self._protos:
            return False
        P_old = self._protos.pop(old_name)
        M_old = self._meta.pop(old_name)
        n_old = self._counts.pop(old_name, 1)
        if new_name in self._protos:
            self._protos[new_name] = np.vstack(
                [self._protos[new_name], P_old]).astype(np.float32)
            self._meta[new_name] = np.vstack(
                [self._meta[new_name], M_old]).astype(np.float32)
            self._counts[new_name] = self._counts.get(new_name, 1) + n_old
            self._reduce_tiers(new_name)
        else:
            self._protos[new_name] = P_old
            self._meta[new_name]   = M_old
            self._counts[new_name] = n_old
        return True

    # ── Persistence ───────────────────────────────────────────────────────────

    SCHEMA = 2

    def save(self, path: str) -> None:
        """Save the prototype galleries to a .npz file (schema 2).

        Ragged per-person galleries are stored flat — `protos` (M,512) plus
        `proto_owner` indexing into `names` — because a per-person object array
        could not be read back with allow_pickle=False.

        The legacy `embeddings` (P,512) per-person centroid is still written so an
        older checkout, or any external reader, keeps working. Schema-2 readers
        ignore it.
        """
        names = list(self._protos)
        if not names:
            np.savez(path, schema=np.array([self.SCHEMA], np.int32),
                     names=np.array([], dtype="<U1"),
                     counts=np.array([], np.int64),
                     embeddings=np.zeros((0, 512), np.float32),
                     protos=np.zeros((0, 512), np.float32),
                     proto_owner=np.array([], np.int32),
                     proto_meta=np.zeros((0, 4), np.float32))
            print(f"[FaceID] saved 0 people → {os.path.abspath(path)}")
            return
        owner = np.concatenate(
            [np.full(len(self._protos[n]), i, np.int32) for i, n in enumerate(names)])
        np.savez(
            path,
            schema      = np.array([self.SCHEMA], np.int32),
            names       = np.array(names),
            counts      = np.array([self._counts.get(n, 1) for n in names], np.int64),
            embeddings  = np.stack([self._centroid(n) for n in names]),   # legacy
            protos      = np.vstack([self._protos[n] for n in names]).astype(np.float32),
            proto_owner = owner,
            proto_meta  = np.vstack([self._meta[n] for n in names]).astype(np.float32),
        )
        views = sum(len(self._protos[n]) for n in names)
        print(f"[FaceID] saved {len(names)} people / {views} views "
              f"→ {os.path.abspath(path)}")

    def load(self, path: str) -> bool:
        """Load galleries from a .npz file. Reads schema 2, and MIGRATES the old
        single-averaged-embedding format (schema 1) in place. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            data   = np.load(path, allow_pickle=False)
            schema = int(data["schema"][0]) if "schema" in data.files else 1
            names  = data["names"].tolist()
            counts = data["counts"].tolist()
            self._protos, self._meta = {}, {}

            if schema >= 2 and "protos" in data.files:
                owner, protos = data["proto_owner"], data["protos"]
                meta = data["proto_meta"]
                for i, n in enumerate(names):
                    rows = np.where(owner == i)[0]
                    self._protos[n] = protos[rows].astype(np.float32)
                    self._meta[n]   = meta[rows].astype(np.float32)
                views = sum(len(v) for v in self._protos.values())
                print(f"[FaceID] loaded {len(names)} people / {views} views from {path}")
            else:
                now, E = time.time(), data["embeddings"]
                for i, n in enumerate(names):
                    v = E[i].astype(np.float32)
                    v = v / (np.linalg.norm(v) + 1e-8)
                    self._protos[n] = v[None, :]
                    self._meta[n]   = np.array(
                        [[min(float(counts[i]), W_CAP), ANCHOR, now, now]], np.float32)
                print(f"[FaceID] migrated {len(names)} legacy profile(s) to the "
                      f"multi-view format — each has ONE view so far; re-enroll with "
                      f"the guided pose prompts, or let adaptive capture fill in "
                      f"side views as you talk")

            self._counts = {n: int(c) for n, c in zip(names, counts)}
            return True
        except Exception as exc:
            print(f"[FaceID] load failed: {exc}")
            return False
