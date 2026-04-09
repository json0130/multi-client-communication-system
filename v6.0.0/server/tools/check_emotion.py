"""
tools/check_emotion.py
=======================
Checkpoint 5 — verifies the Emotion module (tracker + full module).

Run from project root:
    python3 tools/check_emotion.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — model file present:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 5 — Emotion module
=======================================================
  PASS  EmotionTracker imports
  PASS  EmotionModule imports
  PASS  EmotionResult dataclass works

  -- EmotionTracker unit tests --
  PASS  Returns neutral before enough data
  PASS  Detects dominant emotion after 5 detections
  PASS  changed=True when emotion flips
  PASS  get_distribution() returns percentages summing to 100
  PASS  reset() clears state

  -- EmotionModule lifecycle --
  [EmotionModule] Face cascade loaded.
  [EmotionModule] EfficientNet B0 loaded on cpu.
  PASS  initialize() succeeded
  PASS  is_available() is True
  PASS  get_status() has required keys
  PASS  get_current() returns (str, float)

Result: 12/12 checks passed
Emotion module is ready. Proceed to Step 6 (RAG module).
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — model file NOT found:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  (tracker tests all PASS)
  [EmotionModule] Face cascade loaded.
  [EmotionModule] Model file not found: ./models/efficientnet_...pth
    Set EMOTION_MODEL_PATH in your .env to point to the .pth file.
  FAIL  initialize() succeeded
         → Set EMOTION_MODEL_PATH=./path/to/model.pth in your .env file

Result: 9/12 checks passed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE: Tracker tests (8 checks) pass with or without the model file.
      Only the last 4 checks need the model.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 5 — Emotion module")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Imports ────────────────────────────────────────────
    total += 1
    try:
        from modules.emotion.emotion_tracker import EmotionTracker
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("EmotionTracker imports", ok):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    total += 1
    try:
        from modules.emotion.emotion_module import EmotionModule, EmotionResult
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("EmotionModule imports", ok):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    from modules.emotion.emotion_tracker import EmotionTracker
    from modules.emotion.emotion_module import EmotionModule, EmotionResult

    # ── 2. EmotionResult dataclass ────────────────────────────
    total += 1
    r = EmotionResult("happy", 85.0, True, {"happy": 100.0}, "success")
    if check("EmotionResult dataclass works",
             r.emotion == "happy" and r.confidence == 85.0 and r.changed):
        passed += 1

    # ── 3. EmotionTracker unit tests ──────────────────────────
    print("\n  -- EmotionTracker unit tests --")

    total += 1
    t = EmotionTracker(window_size=5)
    em, cf, changed = t.get_stable()
    if check("Returns neutral before enough data",
             em == "neutral" and not changed):
        passed += 1

    total += 1
    t2 = EmotionTracker(window_size=5)
    for _ in range(5):
        t2.add("happy", 80.0)
    em, cf, changed = t2.get_stable(change_threshold=20.0)
    if check("Detects dominant emotion after 5 detections",
             em == "happy" and cf > 0):
        passed += 1

    total += 1
    t3 = EmotionTracker(window_size=5)
    for _ in range(3):
        t3.add("neutral", 75.0)
    t3.get_stable()                        # lock in neutral
    for _ in range(5):
        t3.add("happy", 90.0)
    em, cf, changed = t3.get_stable(change_threshold=20.0)
    if check("changed=True when emotion flips",
             em == "happy" and changed):
        passed += 1

    total += 1
    t4 = EmotionTracker(window_size=5)
    t4.add("happy", 80.0)
    t4.add("happy", 70.0)
    t4.add("neutral", 60.0)
    dist = t4.get_distribution()
    total_pct = sum(dist.values())
    if check("get_distribution() returns percentages summing to 100",
             abs(total_pct - 100.0) < 0.1):
        passed += 1

    total += 1
    t5 = EmotionTracker(window_size=5)
    for _ in range(5):
        t5.add("happy", 80.0)
    t5.reset()
    if check("reset() clears state",
             t5.stable_emotion == "neutral" and len(t5._emotions) == 0):
        passed += 1

    # ── 4. EmotionModule lifecycle ────────────────────────────
    print("\n  -- EmotionModule lifecycle --")

    total += 1
    module = EmotionModule()
    init_ok = module.initialize()
    if check("initialize() succeeded", init_ok,
             "Set EMOTION_MODEL_PATH=./path/to/model.pth in your .env file"):
        passed += 1

    total += 1
    if check("is_available() is True", module.is_available()):
        passed += 1

    total += 1
    status = module.get_status()
    required = {"module", "available", "model_loaded",
                "face_cascade_loaded", "current_emotion"}
    if check("get_status() has required keys",
             required.issubset(status.keys())):
        passed += 1

    total += 1
    em, cf = module.get_current()
    if check("get_current() returns (str, float)",
             isinstance(em, str) and isinstance(cf, float)):
        passed += 1

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Emotion module is ready. Proceed to Step 6 (RAG module).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()