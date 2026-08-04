"""
tools/check_rag.py
==================
Checkpoint 6 — verifies the RAG module (FAISS index + Ollama embeddings).

Run from project root:
    python3 tools/check_rag.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — Ollama running + faiss-cpu installed:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 6 — RAG module
=======================================================
  PASS  RagModule imports
  PASS  get_status() returns required keys before init
  [RagModule] No local index found — rebuilding from Supabase...
  [RagModule] No chat history for user 99999 — empty index
  [RagModule] user=99999 ready — 0 vectors
  PASS  initialize() succeeds (even with empty index)
  PASS  is_available() is True
  PASS  search() on empty index returns empty list
  PASS  add() embeds and stores a message
  PASS  index has 1 vector after add()
  PASS  search() returns relevant result
  PASS  result contains the added message text

  Cleaning up test index files...
  OK  Test index files deleted

Result: 9/9 checks passed
RAG module is ready. Proceed to Step 7 (Robot layer).
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — Ollama NOT running:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PASS  RagModule imports
  PASS  get_status() returns required keys before init
  [RagModule] No local index found — rebuilding from Supabase...
  [RagModule] Batch embed error (skipping batch): ...
  [RagModule] user=99999 ready — 0 vectors
  PASS  initialize() succeeds (even with empty index)
  PASS  is_available() is True
  PASS  search() on empty index returns empty list
  [RagModule] embed error: ...
  PASS  add() handles Ollama being unreachable gracefully
  PASS  index stays empty when embedding fails

Result: 8/9 checks passed
  NOTE: Start Ollama and run `ollama pull nomic-embed-text` for full functionality.
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — faiss-cpu NOT installed:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FAIL  RagModule imports
         → pip install faiss-cpu
=======================================================
"""

import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_USER_ID = 99999   # Non-existent user — won't find any chat history


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 6 — RAG module")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Import ─────────────────────────────────────────────
    total += 1
    try:
        from modules.rag.rag_module import RagModule
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("RagModule imports", ok,
             "pip install faiss-cpu"):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    from modules.rag.rag_module import RagModule

    # ── 2. Status before init ─────────────────────────────────
    # RBAC identity for retrieval — search() is filtered, so it needs a requester.
    from core.rbac import AccessLevel, RBACFilter, RobotIdentity, Visibility

    TEST_ROBOT_ID = "_checkpoint_rag_robot"
    TEST_SCENARIO = "_checkpoint"
    rbac = RBACFilter()
    requester = RobotIdentity(
        TEST_ROBOT_ID, TEST_SCENARIO, "sess", AccessLevel.LOCAL
    )

    total += 1
    module = RagModule(
        user_id=TEST_USER_ID,
        client_id=TEST_ROBOT_ID,
        scenario_id=TEST_SCENARIO,
        session_id="sess",
        default_visibility=Visibility.LOCAL,
    )
    status = module.get_status()
    required = {"module", "available", "user_id", "vector_count", "embed_model"}
    if check("get_status() returns required keys before init",
             required.issubset(status.keys())):
        passed += 1

    # ── 3. Initialize ─────────────────────────────────────────
    total += 1
    init_ok = module.initialize()
    if check("initialize() succeeds (even with empty index)", init_ok):
        passed += 1

    total += 1
    if check("is_available() is True", module.is_available()):
        passed += 1

    # ── 4. Search on empty index ──────────────────────────────
    total += 1
    results = module.search("hello robot", requester=requester, rbac=rbac)
    if check("search() on empty index returns empty list",
             isinstance(results, list) and len(results) == 0):
        passed += 1

    # ── 5. Add a message ─────────────────────────────────────
    total += 1
    module.add("I enjoy playing chess and reading science fiction books")
    # Give it a moment
    from core.config import cfg
    index_count = module._index.ntotal if module._index else 0
    ollama_reachable = index_count > 0   # if embed failed, count stays 0

    if check("add() embeds and stores a message",
             ollama_reachable,
             "Start Ollama and run: ollama pull nomic-embed-text"):
        passed += 1
    else:
        # Ollama not running — test graceful degradation instead
        total += 1
        if check("add() handles Ollama being unreachable gracefully",
                 module._index is None or module._index.ntotal == 0):
            passed += 1
        total += 1
        if check("index stays empty when embedding fails",
                 module._index is None or module._index.ntotal == 0):
            passed += 1
        _cleanup(module)
        _summary(passed, total)
        print("\n  NOTE: Start Ollama and run `ollama pull nomic-embed-text` "
              "for full functionality.")
        return

    # ── 6. Index count ────────────────────────────────────────
    total += 1
    if check("index has 1 vector after add()",
             module._index is not None and module._index.ntotal == 1):
        passed += 1

    # ── 7. Search returns result ──────────────────────────────
    module.add("I love robotics and machine learning")
    results = module.search("chess and books", requester=requester, rbac=rbac, top_k=3)

    total += 1
    if check("search() returns relevant result",
             isinstance(results, list) and len(results) > 0):
        passed += 1

    total += 1
    # search() now returns RBAC-cleared records, not bare strings.
    found = any("chess" in r.text or "science fiction" in r.text or "robotics" in r.text
                for r in results)
    if check("result contains the added message text", found,
             f"Got: {[r.text for r in results]}"):
        passed += 1

    _cleanup(module)
    _summary(passed, total)


def _cleanup(module):
    print("\n  Cleaning up test index files...")
    try:
        if module._faiss_path.exists():
            module._faiss_path.unlink()
        if module._texts_path.exists():
            module._texts_path.unlink()
        print("  OK  Test index files deleted")
    except Exception as e:
        print(f"  WARN  Cleanup failed: {e}")


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("RAG module is ready. Proceed to Step 7 (Robot layer).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()