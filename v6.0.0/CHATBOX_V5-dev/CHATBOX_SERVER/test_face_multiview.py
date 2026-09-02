"""
Multi-view face recognition: gallery, hysteresis, adaptive capture.

Covers the fix for "recognition only works at one angle and flickers":
  * a person is stored as SEVERAL prototype views, matched by best-of, instead of
    one averaged centroid that matched no pose well;
  * the held identity survives a brief off-angle dip (hysteresis + miss-grace);
  * new views are learned automatically, behind a gate that must refuse to learn
    from anyone but the confirmed person.

No camera and no models — galleries are hand-built via FaceIdentifier.__new__.
Run directly (python3 test_face_multiview.py) or under pytest.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.face_webcam.face_id import (          # noqa: E402
    ADAPTIVE, ANCHOR, W_CAP, FaceIdentifier, _dedupe_boxes,
)
from modules.face_webcam.webcam_loop import _DetectionWorker  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _fid(**over) -> FaceIdentifier:
    """A FaceIdentifier with tuning fields set but no models loaded."""
    f = FaceIdentifier.__new__(FaceIdentifier)
    f._protos, f._meta, f._counts = {}, {}, {}
    f.threshold, f.retain_threshold = 0.75, 0.62
    f.switch_margin, f.id_margin = 0.08, 0.05
    f.tau_dup, f.k_anchor, f.k_adapt = 0.92, 12, 6
    for k, v in over.items():
        setattr(f, k, v)
    return f


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _worker(**over) -> _DetectionWorker:
    """A _DetectionWorker with only the fields the gate touches."""
    import threading
    w = _DetectionWorker.__new__(_DetectionWorker)
    w._lock = threading.Lock()
    w._face_id = _fid()
    w._confirmed_pid = "jay"
    w._id_votes, w._id_samples = {"jay": 10}, 10
    w._id_confirm_ratio = 0.6
    w._saw_face_window = True
    w._id_miss_grace, w._miss_streak = 2, 0
    w._adapt_enabled = True
    w._adapt_floor, w._adapt_interval = 0.75, 20.0
    w._adapt_max, w._adapt_count = 20, 0
    w._adapt_min_px, w._adapt_min_sharp = 90, 40.0
    w._adapt_buf, w._adapt_dirty = [], False
    w._adapt_last, w._adapt_events = {}, []
    for k, v in over.items():
        setattr(w, k, v)
    return w


# ── 1. the core regression: several views beat one centroid ──────────────────

def test_multiprototype_beats_centroid():
    f = _fid()
    f.add_prototype("jay", _unit([1, 0, 0]))     # e.g. frontal
    f.add_prototype("jay", _unit([0, 1, 0]))     # e.g. profile
    probe = _unit([0.05, 1, 0])                  # a profile-ish view

    centroid_sim = float(f._centroid("jay") @ probe)
    best_view_sim = float((f._protos["jay"] @ probe).max())

    assert centroid_sim < f.threshold, "averaging should lose this match"
    assert best_view_sim >= f.threshold, "best-of-views should win it"
    assert f._match(probe)[0] == "jay"
    print(f"1. multi-view beats centroid: centroid={centroid_sim:.3f} (fails) "
          f"vs best-view={best_view_sim:.3f} (matches) ✓")


# ── 2. same view refines, new view is added ──────────────────────────────────

def test_novelty_merge_vs_insert():
    f = _fid()
    assert f.add_prototype("jay", _unit([1, 0, 0])) == "created"

    assert f.add_prototype("jay", _unit([1, 0.02, 0])) == "merged"
    assert len(f._protos["jay"]) == 1
    assert f._meta["jay"][0, 0] == 2.0                    # weight grew

    assert f.add_prototype("jay", _unit([0, 1, 0])) == "inserted"
    assert len(f._protos["jay"]) == 2
    print("2. novelty: near-duplicate merges (weight+1), new pose inserts ✓")


def test_merge_weight_is_capped():
    f = _fid()
    f.add_prototype("jay", _unit([1, 0, 0]))
    for _ in range(int(W_CAP) + 20):
        f.add_prototype("jay", _unit([1, 0.01, 0]))
    assert f._meta["jay"][0, 0] == W_CAP
    print(f"3. prototype weight caps at {W_CAP:.0f} (keeps adapting, never freezes) ✓")


# ── 3. eviction keeps the gallery spanning poses ─────────────────────────────

def test_eviction_preserves_spread():
    f = _fid(k_anchor=4)
    for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
        f.add_prototype("jay", _unit(v))

    def worst_pair(name):
        S = f._protos[name] @ f._protos[name].T
        np.fill_diagonal(S, -np.inf)
        return float(S.max())

    before = worst_pair("jay")
    for i in range(20):                       # flood with near-duplicates of one pose
        f.add_prototype("jay", _unit([1, 0.30 + 0.001 * i, 0]))

    assert len(f._protos["jay"]) == 4, "cap must hold"
    assert worst_pair("jay") < 0.95, "gallery collapsed into duplicates"
    print(f"4. eviction preserves spread: {len(f._protos['jay'])} views, "
          f"closest pair {before:.3f} → {worst_pair('jay'):.3f} ✓")


def test_adaptive_never_evicts_anchors():
    f = _fid(k_anchor=2, k_adapt=2)
    for i in range(2):
        f.add_prototype("jay", _unit([1, i * 0.5, 0]), origin=ANCHOR)
    for i in range(6):
        f.add_prototype("jay", _unit([0, 1, i * 0.5]), origin=ADAPTIVE)
    assert f.gallery_size("jay") == (2, 2)
    assert f.reset_adaptive("jay") == 2
    assert f.gallery_size("jay") == (2, 0), "enrolled views must survive the reset"
    print("5. tiers: self-learned views never displace enrolled ones; reset keeps anchors ✓")


# ── 4. hysteresis ────────────────────────────────────────────────────────────

def test_hysteresis_retain_vs_acquire():
    f = _fid()
    f.add_prototype("jay", _unit([1, 0, 0]))
    f.add_prototype("hj", _unit([0, 1, 0]))

    dip = _unit([0.68, 0.73, 0])              # ambiguous: neither clears acquire
    assert f._match(dip)[0] is None, "cold, an ambiguous face should not be claimed"
    assert f._match(dip, sticky="jay")[0] == "jay", "held identity should survive a dip"

    clear_hj = _unit([0.1, 1, 0])
    assert f._match(clear_hj, sticky="jay")[0] == "hj", "a clear rival must take over"
    print("6. hysteresis: dip keeps the held identity, a clear rival still switches ✓")


def test_runner_up_margin_needs_two_people():
    f = _fid()
    f.add_prototype("solo", _unit([1, 0, 0]))
    assert f._match(_unit([1, 0.05, 0]))[0] == "solo"   # margin not applied at n=1
    print("7. runner-up margin only applies once 2+ people are enrolled ✓")


# ── 5. the adaptive-capture gate (anti-poisoning) ────────────────────────────

def _buf(sim, n=10):
    return [(sim, _unit(np.random.rand(8))) for _ in range(n)]


def test_adaptive_gate_accepts_clean_window():
    w = _worker(_adapt_buf=_buf(0.80))
    w._maybe_adapt(now=1000.0)
    assert len(w._adapt_events) == 1
    assert w._adapt_events[0]["pid"] == "jay"
    print("8. adaptive gate: a clean, confirmed, single-face window learns one view ✓")


def test_adaptive_gate_rejects_poison():
    cases = {
        "unconfirmed":     _worker(_confirmed_pid=None, _adapt_buf=_buf(0.80)),
        "two faces":       _worker(_adapt_dirty=True,   _adapt_buf=_buf(0.80)),
        "below floor":     _worker(_adapt_buf=_buf(0.55)),
        "rival in window": _worker(_id_votes={"jay": 6, "hj": 4}, _adapt_buf=_buf(0.80)),
        "weak vote":       _worker(_id_votes={"jay": 3}, _adapt_buf=_buf(0.80)),
        "already covered": _worker(_adapt_buf=_buf(0.99)),
        "empty buffer":    _worker(_adapt_buf=[]),
    }
    for label, w in cases.items():
        w._maybe_adapt(now=1000.0)
        assert not w._adapt_events, f"gate must refuse to learn — {label}"
    print(f"9. adaptive gate refuses all {len(cases)} poisoning cases "
          f"({', '.join(cases)}) ✓")


def test_adaptive_rate_limit_and_cap():
    w = _worker(_adapt_buf=_buf(0.80))
    w._maybe_adapt(now=1000.0)
    w._adapt_buf, w._adapt_dirty = _buf(0.80), False
    w._maybe_adapt(now=1001.0)                       # 1 s later — inside the interval
    assert len(w._adapt_events) == 1, "rate limit should suppress the second"
    w._adapt_buf = _buf(0.80)
    w._maybe_adapt(now=1000.0 + w._adapt_interval + 1)
    assert len(w._adapt_events) == 2, "should learn again after the interval"

    w2 = _worker(_adapt_buf=_buf(0.80), _adapt_count=20)   # session cap reached
    w2._maybe_adapt(now=5000.0)
    assert not w2._adapt_events
    print("10. adaptive capture respects the per-person interval and session cap ✓")


# ── 6. worker vote/grace behaviour ───────────────────────────────────────────

def test_miss_grace_and_rival_drop():
    # face present but unrecognised → hold through one window, drop on the next
    w = _worker(_id_votes={}, _id_samples=0, _adapt_enabled=False)
    w._confirm_identity()
    assert w._confirmed_pid == "jay", "a hard pose should not drop the label at once"
    w._saw_face_window = True
    w._confirm_identity()
    assert w._confirmed_pid is None, "grace must expire"

    # nobody in frame at all → drop immediately
    w = _worker(_id_votes={}, _id_samples=0, _saw_face_window=False,
                _adapt_enabled=False)
    w._confirm_identity()
    assert w._confirmed_pid is None

    # a different person → drop immediately
    w = _worker(_id_votes={"hj": 8}, _id_samples=10, _adapt_enabled=False)
    w._confirm_identity()
    assert w._confirmed_pid != "jay"
    print("11. miss-grace holds a hard pose, drops instantly on an empty frame/rival ✓")


def test_empty_frames_do_not_dilute_the_vote():
    """The flicker bug: frames with NO face used to count in the denominator."""
    w = _worker(_id_votes={"jay": 6}, _id_samples=6, _adapt_enabled=False)
    w._confirm_identity()                     # 6/6 — only face-bearing frames counted
    assert w._confirmed_pid == "jay"
    print("12. only frames containing a face count toward the confirmation ratio ✓")


# ── 7. detection dedupe ──────────────────────────────────────────────────────

def test_dedupe_frontal_beats_profile():
    frontal = (100, 100, 200, 200)
    overlapping_profile = (130, 105, 225, 205)
    kept = _dedupe_boxes([(overlapping_profile, "profile-l"), (frontal, "frontal")])
    assert len(kept) == 1 and kept[0][1] == "frontal"

    nested = (110, 110, 190, 190)
    assert len(_dedupe_boxes([(nested, "profile-l"), (frontal, "frontal")])) == 1

    far = (400, 100, 500, 200)
    assert len(_dedupe_boxes([(frontal, "frontal"), (far, "profile-r")])) == 2
    print("13. dedupe: one box per head (frontal wins), separate faces both kept ✓")


# ── 8. persistence ───────────────────────────────────────────────────────────

def test_load_v1_npz_migrates():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "v1.npz")
    emb = _unit(np.random.rand(512))[None, :]
    np.savez(path, names=np.array(["jay"]), embeddings=emb, counts=np.array([50]))

    f = _fid()
    assert f.load(path)
    assert f._protos["jay"].shape == (1, 512)
    assert f._meta["jay"][0, 0] == 50.0          # weight from the old count
    assert f._meta["jay"][0, 1] == ANCHOR
    assert f._counts["jay"] == 50
    assert f.known_people() == ["jay"]
    print("14. legacy single-embedding faces.npz migrates to one enrolled view ✓")


def test_save_load_roundtrip_v2():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "v2.npz")
    a = _fid()
    for name, k in (("jay", 5), ("hj", 3)):
        for i in range(k):
            a.add_prototype(name, _unit(np.random.rand(512)),
                            origin=ANCHOR if i % 2 else ADAPTIVE)
        a._counts[name] = k * 2
    a.save(path)

    b = _fid()
    assert b.load(path)
    for name in a._protos:
        assert np.allclose(a._protos[name], b._protos[name])
        assert np.allclose(a._meta[name], b._meta[name])
    assert b._counts == a._counts

    raw = np.load(path, allow_pickle=False)      # must never need pickle
    assert raw["embeddings"].shape == (2, 512)   # legacy centroid still written
    print("15. v2 npz roundtrips exactly, loads with allow_pickle=False, "
          "keeps the legacy centroid ✓")


def test_known_people_is_one_entry_per_person():
    """_next_guest_id() scans this to pick a free guest_N — many views, one entry."""
    f = _fid()
    for i in range(8):
        f.add_prototype("jay", _unit(np.random.rand(16)))
    assert f.known_people() == ["jay"]
    print("16. known_people() lists each person once, whatever the gallery size ✓")


if __name__ == "__main__":
    test_multiprototype_beats_centroid()
    test_novelty_merge_vs_insert()
    test_merge_weight_is_capped()
    test_eviction_preserves_spread()
    test_adaptive_never_evicts_anchors()
    test_hysteresis_retain_vs_acquire()
    test_runner_up_margin_needs_two_people()
    test_adaptive_gate_accepts_clean_window()
    test_adaptive_gate_rejects_poison()
    test_adaptive_rate_limit_and_cap()
    test_miss_grace_and_rival_drop()
    test_empty_frames_do_not_dilute_the_vote()
    test_dedupe_frontal_beats_profile()
    test_load_v1_npz_migrates()
    test_save_load_roundtrip_v2()
    test_known_people_is_one_entry_per_person()
    print("\nAll multi-view face recognition tests passed.")
