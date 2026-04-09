"""
tools/check_config.py
=====================
Checkpoint 1 — verifies core/config.py loads correctly.

Run from the project root:
    python -m tools.check_config

What it checks:
  1. Config loads without crashing
  2. All sub-configs exist with correct types
  3. Required env vars are present (SUPABASE_URL, SUPABASE_KEY)
  4. Defaults are populated for optional vars
  5. The frozen dataclass cannot be mutated (safety guarantee)
"""

import sys
import os

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, detail: str = ""):
    status = "  PASS" if condition else "  FAIL"
    print(f"{status}  {label}" + (f"\n         → {detail}" if detail else ""))
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 1 — core/config.py")
    print("=" * 55)

    passed = 0
    total = 0

    # ── 1. Import without crash ───────────────────────────────
    total += 1
    try:
        from core.config import cfg, AppConfig, DatabaseConfig
        ok = True
    except Exception as e:
        ok = False
        detail = str(e)
    if check("Module imports without error", ok, detail if not ok else ""):
        passed += 1

    if not ok:
        print("\nCannot continue — fix import error first.")
        return

    # ── 2. cfg is not None (env vars present) ────────────────
    total += 1
    if check("cfg singleton is not None (env vars set)",
             cfg is not None,
             "Set SUPABASE_URL and SUPABASE_KEY in your .env file"):
        passed += 1
    else:
        print("\nSkipping remaining checks — required env vars missing.")
        _print_summary(passed, total)
        return

    # ── 3. Sub-config types ───────────────────────────────────
    sub_checks = [
        ("cfg.db is DatabaseConfig",          isinstance(cfg.db,     __import__('core.config', fromlist=['DatabaseConfig']).DatabaseConfig)),
        ("cfg.llm has ollama_model",          bool(cfg.llm.ollama_model)),
        ("cfg.speech has whisper_model_size", bool(cfg.speech.whisper_model_size)),
        ("cfg.emotion has model_path",        bool(cfg.emotion.model_path)),
        ("cfg.rag has embed_model",           bool(cfg.rag.embed_model)),
        ("cfg.server has port (int)",         isinstance(cfg.server.port, int)),
    ]
    for label, cond in sub_checks:
        total += 1
        if check(label, cond):
            passed += 1

    # ── 4. Defaults populated ─────────────────────────────────
    total += 1
    if check("Ollama default host is 127.0.0.1",
             cfg.llm.ollama_host == "127.0.0.1"):
        passed += 1

    total += 1
    if check("Server default port is 5000",
             cfg.server.port == 5000):
        passed += 1

    # ── 5. Frozen (immutable) ─────────────────────────────────
    total += 1
    try:
        cfg.server.port = 9999  # type: ignore
        mutation_blocked = False
    except Exception:
        mutation_blocked = True
    if check("Config is immutable (frozen dataclass)", mutation_blocked):
        passed += 1

    # ── Summary ───────────────────────────────────────────────
    _print_summary(passed, total)

    # Print the loaded values for visual inspection
    if cfg:
        print("\nLoaded values:")
        print(f"  DB URL       : {cfg.db.url[:30]}...")
        print(f"  Ollama       : {cfg.llm.ollama_host}:{cfg.llm.ollama_port}  model={cfg.llm.ollama_model}")
        print(f"  Whisper      : {cfg.speech.whisper_model_size} / {cfg.speech.whisper_device}")
        print(f"  Emotion path : {cfg.emotion.model_path}")
        print(f"  RAG index    : {cfg.rag.index_dir}")
        print(f"  OpenAI key   : {'set' if cfg.llm.openai_api_key else 'not set (optional)'}")


def _print_summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Config layer is ready. Proceed to Step 2 (data layer).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()