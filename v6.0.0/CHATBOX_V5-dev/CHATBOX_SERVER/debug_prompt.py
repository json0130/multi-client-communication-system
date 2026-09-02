"""
Show what feeds the system prompt for one person: the 3 modules' outputs + the
final assembled prompt. Read-only (never writes the graph).

  python3 debug_prompt.py --name jay --msg "do we have anything in common?"
  python3 debug_prompt.py --name jay --culture Korean   # preview a culture tag (in-memory only)

Modules:
  1. KG retrieval  — interests / common ground / notes / culture block (_person_memory)
  2. Embedding RAG — top relevant past turns for --msg (SessionRAG over transcripts)
  3. BN overlay    — rank_suggestions: culture priors → observed clamp → propagation → posteriors
"""

from __future__ import annotations

import argparse

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.topics import (
    person_interests, related_common_ground, person_related_pairs, normalize_label,
)
from modules.graph_relationship.cultures import person_culture, culture_priors
from modules.preference_model import rank_suggestions
from modules.session_store import SessionStore
from modules.face_webcam.webcam_loop import WebcamKGLoop

_BAR = "═" * 78


def _h(title: str) -> None:
    print(f"\n{_BAR}\n {title}\n{_BAR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="jay")
    p.add_argument("--msg", default="do we have anything in common?")
    p.add_argument("--kg", default="kg_state.json")
    p.add_argument("--sessions-db", default="sessions.db")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--robot", default="chatbox")
    p.add_argument("--culture", default=None,
                   help="Preview a culture tag IN MEMORY (not saved) to see the culture/BN path")
    args = p.parse_args()

    store = InMemoryGraphStore()
    store.load(args.kg)
    pid = args.name

    if args.culture:
        from modules.culture_seed import assign_person_culture
        assign_person_culture(store, pid, args.culture)   # in-memory only (never saved)
        print(f"[preview] tagged {pid} → {args.culture} IN MEMORY (not saved)")

    print(f"\nPerson: {pid}   |   user message: {args.msg!r}")

    # ── Module 1: KG retrieval ────────────────────────────────────────────────
    _h("MODULE 1 — KG RETRIEVAL  (structured memory)")
    interests = person_interests(store, pid)
    print("Interests (interest → topics):")
    for interest, topics in interests:
        print(f"   • {interest.label}: {', '.join(t.label for t in topics) or '—'}")
    cg = related_common_ground(store, pid, args.robot)
    print(f"Common ground (direct): {cg['direct'] or '—'}")
    print(f"Common ground (related bridges): "
          f"{[f'{a}~{b}' for a,b in cg['bridges']] or '—'}")
    print(f"Related interest pairs: {[f'{a}~{b}' for a,b in person_related_pairs(store,pid)] or '—'}")
    cid = person_culture(store, pid)
    print(f"Culture tag (belongs_to_culture): {cid or '— (none — untagged)'}")

    # ── Module 2: Embedding RAG ───────────────────────────────────────────────
    _h("MODULE 2 — EMBEDDING RAG  (relevant past turns for the message)")
    hits = []
    try:
        from modules.graph_relationship.embedding import ollama_embed_fn
        from modules.session_rag import SessionRAG
        embed_fn = ollama_embed_fn(model=args.embed_model)
        rag = SessionRAG(SessionStore(args.sessions_db), embed_fn)
        added = rag.reindex()
        print(f"(embeddings: {args.embed_model}; indexed {added} new turn(s))")
        hits = rag.search(args.msg, top_k=5, person_id=pid)
        if hits:
            for h in hits:
                print(f"   [{h.get('score', 0):.3f}] ({h['ts'][:10]}) "
                      f"child: {h.get('child','')!r}")
        else:
            print("   (no relevant embedded turns — nothing retrieved)")
    except Exception as exc:  # noqa: BLE001
        print(f"   RAG unavailable ({exc}) — is Ollama up? Skipping embedding module.")

    # ── Module 3: BN overlay (preference model) ───────────────────────────────
    _h("MODULE 3 — BN OVERLAY  (rank_suggestions: priors → evidence → propagation)")
    if cid:
        print("Culture priors (base rates):")
        for _ck, label, prior in culture_priors(store, cid):
            print(f"   {prior:.2f}  {label}")
    else:
        print("Culture priors: — (person not tagged; overlay uses related-topic only)")
    observed = sorted({normalize_label(t.label)
                       for _i, ts in interests for t in ts})
    print(f"Observed (clamped to 0.90): {observed or '—'}")
    ranked = rank_suggestions(store, pid, k=6, floor=0.35)
    print("Ranked suggestions (UNOBSERVED, posterior desc):")
    if ranked:
        for nid, post in ranked:
            n = store.get_node(nid)
            slug = normalize_label(n.label) if n else nid
            base = dict((normalize_label(l), pr) for _c, l, pr in culture_priors(store, cid)).get(slug) if cid else None
            tag = f"  (prior {base:.2f} → posterior {post:.3f})" if base is not None else f"  (posterior {post:.3f})"
            print(f"   {post:.3f}  {n.label if n else nid}{tag}")
    else:
        print("   (nothing ≥ floor 0.35)")

    # ── Final assembled system prompt ─────────────────────────────────────────
    _h("FINAL SYSTEM PROMPT  (exactly what the LLM receives)")
    loop = WebcamKGLoop.__new__(WebcamKGLoop)
    loop.store = store
    loop.robot_id = args.robot
    loop._robot_display = {"chatbox": "ChatBox", "ellebot": "ElleBot"}.get(args.robot, args.robot)
    print(loop._build_system_prompt(pid, rag_hits=hits))
    print(_BAR)


if __name__ == "__main__":
    main()
