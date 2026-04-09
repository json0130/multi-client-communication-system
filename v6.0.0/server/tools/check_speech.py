"""
tools/check_speech.py
=====================
Checkpoint 4 — verifies the Speech module (Faster-Whisper STT).

Run from project root:
    python3 tools/check_speech.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — faster-whisper installed, no audio file:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 4 — Speech module
=======================================================
  PASS  SpeechModule imports
  PASS  SpeechResult dataclass works
  PASS  get_status() returns a dict with required keys
  [SpeechModule] Loaded Whisper 'base' on cpu (int8)
  PASS  initialize() succeeded
  PASS  is_available() is True
  PASS  Rejects audio that is too short
  PASS  Rejects invalid WAV bytes

  No test audio file provided — skipping live transcription test.
  To test with real audio run:
      python3 tools/check_speech.py path/to/audio.wav

Result: 7/7 checks passed
Speech module is ready. Proceed to Step 5 (Emotion module).
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — with a real WAV file:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ... (same as above) ...
  [SpeechModule] Transcribed: 'hello how are you' (87.3% conf, lang=en)
  PASS  Live transcription succeeded
  PASS  Transcription text is not empty
  PASS  Confidence is between 0 and 100

Result: 10/10 checks passed
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — faster-whisper NOT installed:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PASS  SpeechModule imports
  PASS  SpeechResult dataclass works
  PASS  get_status() returns a dict with required keys
  [SpeechModule] faster-whisper not installed. Run: pip install faster-whisper
  FAIL  initialize() succeeded
         → pip install faster-whisper

Result: 3/5 checks passed
=======================================================
"""

import sys
import os
import struct
import wave
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def make_silent_wav(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file containing silence — used to test validation."""
    num_frames = int(sample_rate * duration_sec)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)       # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * num_frames)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def run():
    print("=" * 55)
    print("Checkpoint 4 — Speech module")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Import ─────────────────────────────────────────────
    total += 1
    try:
        from modules.speech.speech_module import SpeechModule, SpeechResult
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("SpeechModule imports", ok):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    from modules.speech.speech_module import SpeechModule, SpeechResult

    # ── 2. SpeechResult dataclass ─────────────────────────────
    total += 1
    r = SpeechResult(True, "hello", 90.0, "en", "")
    if check("SpeechResult dataclass works",
             r.success and r.transcription == "hello" and r.confidence == 90.0):
        passed += 1

    # ── 3. Status before init ─────────────────────────────────
    total += 1
    module = SpeechModule()
    status = module.get_status()
    required_keys = {"module", "available", "model_size", "device", "max_audio_sec"}
    if check("get_status() returns a dict with required keys",
             required_keys.issubset(status.keys())):
        passed += 1

    # ── 4. Initialize ─────────────────────────────────────────
    total += 1
    init_ok = module.initialize()
    if check("initialize() succeeded", init_ok,
             "pip install faster-whisper"):
        passed += 1

    total += 1
    if check("is_available() is True", module.is_available()):
        passed += 1

    if not module.is_available():
        print("\n  Skipping validation + transcription tests.")
        _summary(passed, total)
        return

    # ── 5. WAV validation ─────────────────────────────────────
    total += 1
    too_short = make_silent_wav(0.05)   # 50ms — below 100ms minimum
    result = module.transcribe_bytes(too_short)
    if check("Rejects audio that is too short",
             not result.success and "short" in result.error.lower()):
        passed += 1

    total += 1
    bad_bytes = b"this is not a wav file at all"
    result = module.transcribe_bytes(bad_bytes)
    if check("Rejects invalid WAV bytes", not result.success):
        passed += 1

    # ── 6. Live transcription (optional) ─────────────────────
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None

    if not audio_file:
        print()
        print("  No test audio file provided — skipping live transcription test.")
        print("  To test with real audio run:")
        print("      python3 tools/check_speech.py path/to/audio.wav")
    else:
        if not os.path.exists(audio_file):
            print(f"\n  Audio file not found: {audio_file}")
        else:
            print(f"\n  Testing with: {audio_file}")
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()

            total += 1
            result = module.transcribe_bytes(audio_bytes)
            if check("Live transcription succeeded", result.success,
                     result.error):
                passed += 1

            total += 1
            if check("Transcription text is not empty",
                     bool(result.transcription.strip())):
                passed += 1

            total += 1
            if check("Confidence is between 0 and 100",
                     0.0 <= result.confidence <= 100.0):
                passed += 1

            if result.success:
                print(f"  Transcription : '{result.transcription}'")
                print(f"  Confidence    : {result.confidence:.1f}%")
                print(f"  Language      : {result.language}")

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Speech module is ready. Proceed to Step 5 (Emotion module).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()