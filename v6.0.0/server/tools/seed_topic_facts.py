"""
tools/seed_topic_facts.py
=========================
A DRAFT of what each robot may say about its own topics.

    python3 tools/seed_topic_facts.py --dry-run    # print, write nothing
    python3 tools/seed_topic_facts.py              # insert
    python3 tools/seed_topic_facts.py --report     # what is still missing

EVERY ROW HERE IS UNVERIFIED AND MOST ARE DELIBERATELY VAGUE.
The only defensible source in the repository is robots.robot_role — three
sentences per robot — plus the topic vocabulary itself. Nothing states a
model, a dataset, a number or a paper, so nothing here does either. That is
the point: a live run had the LLM fill exactly this gap with "Extended
Kalman Filters" and "BM25", neither of which appears anywhere in this
system. A vague true row beats a specific invented one, and every row is
marked UNVERIFIED in the prompt until a researcher confirms it.

The `limitation` rows are the ones that make a demo honest, and they are the
ones only a researcher can write. They are seeded as explicit placeholders so
the gap is visible in tools/check_facts.py rather than silent.

WHAT TO DO WITH THIS
Correct it. Replace the vague rows with what is actually true, add the model
and dataset rows that only you know, delete anything wrong, and mark what
survives as verified. Until then the robots will describe their approach in
general terms and decline to quote specifics, which is the intended
behaviour for unreviewed content.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROLE = "drafted from robots.robot_role"
RESEARCHER = "supplied by the researcher — not independently checkable"

# (topic_id, robot_id | None, kind, fact)
# Claims are traceable to robot_role or to the topic label. Where robot_role
# says nothing, the row says nothing specific either.
DRAFT = [
    # ══ Silbot: human-aware navigation ═══════════════════════════════════════
    # Supplied by the researcher. Not independently checkable from the repo,
    # so unverified until confirmed.
    ("topic:mapping-and-localisation", "silbot_01", "method",
     "Visual SLAM / RGB-D SLAM: builds a map of the space and tracks the "
     "robot's own position within it.", False, RESEARCHER),
    ("topic:mapping-and-localisation", "silbot_01", "hardware",
     "An RGB-D camera, with wheel odometry and IMU used where available.",
     False, RESEARCHER),
    ("topic:mapping-and-localisation", "silbot_01", "limitation",
     "Degrades in poor lighting, against reflective surfaces, when people "
     "occlude the view, and in spaces with few distinctive visual features.",
     False, RESEARCHER),
    ("topic:social-robot-navigation", "silbot_01", "method",
     "Plans paths through crowded spaces that respect personal space and "
     "social norms, rather than treating people as ordinary obstacles.",
     False, ROLE),
    ("topic:social-robot-navigation", "silbot_01", "limitation",
     "Assumes a reasonably observable environment. Sudden pedestrian "
     "movement, dense crowds, occlusion, narrow spaces and ambiguous human "
     "intent are all hard cases.", False, RESEARCHER),
    ("topic:social-robot-navigation", None, "limitation",
     "A collision-free path is not automatically a socially comfortable one "
     "— that gap is the open research problem, not a bug in the "
     "implementation.", False, RESEARCHER),
    ("topic:multi-robot-coordination", "silbot_01", "method",
     "Coordination between robots sharing the same space. Part of Silbot's "
     "declared area.", False, ROLE),

    # ══ ChatBox: conversational AI ═══════════════════════════════════════════
    # These four ARE checkable in this repository, so they ship verified.
    ("topic:large-language-models", "chatbox_01", "model",
     "Qwen2.5 at 7B parameters, served locally through Ollama.", True,
     "verified: OLLAMA_MODEL in .env"),
    ("topic:retrieval-augmented-generation", "chatbox_01", "method",
     "Dense vector retrieval over a FAISS IndexFlatL2 index — an exact "
     "search, not an approximate one — with passages embedded locally by "
     "nomic-embed-text.", True, "verified: modules/rag/rag_module.py"),
    ("topic:retrieval-augmented-generation", "chatbox_01", "dataset",
     "The index is built per user from that user's own stored interaction "
     "history, rebuilt from Supabase when no local index exists.", True,
     "verified: modules/rag/rag_module.py"),
    ("topic:retrieval-augmented-generation", "chatbox_01", "limitation",
     "When retrieval finds nothing relevant the system must say the "
     "information is unavailable rather than generate a specific answer.",
     False, RESEARCHER),
    ("topic:long-term-interaction", "chatbox_01", "method",
     "Long-context memory, so a robot can hold a coherent conversation "
     "across many turns and across separate visits.", False, ROLE),
    ("topic:conversational-memory", "chatbox_01", "method",
     "Retains what a visitor said earlier and uses it in later answers.",
     False, ROLE),
    ("topic:knowledge-graphs", "chatbox_01", "method",
     "Structured knowledge the conversational system can retrieve from, "
     "alongside the vector index.", False, ROLE),

    # ══ Navel: emotion-aware interaction ═════════════════════════════════════
    # The taxonomy and the model ARE in this repository — see the source
    # notes. They differ from the generic "RetinaFace or MTCNN" answer, so
    # the repository is used and the discrepancy is flagged to the researcher.
    ("topic:facial-expression-analysis", "navel_01", "model",
     "EfficientNet-B0 served as ONNX (the hsemotion backend), with a larger "
     "B2 variant available; roughly 5 ms per frame on CPU.", True,
     "verified: modules/face_webcam/emotion_detector.py"),
    ("topic:facial-expression-analysis", "navel_01", "dataset",
     "Trained on AffectNet and VGAF. An older in-house EfficientNet-B0 "
     "trained on HQRAF is kept as a fallback backend.", True,
     "verified: modules/face_webcam/emotion_detector.py"),
    ("topic:facial-expression-analysis", "navel_01", "method",
     "Seven expression classes — angry, disgust, fear, happy, neutral, sad, "
     "surprise. The 8-class variant maps contempt onto disgust. Face "
     "detection for the PyTorch backend is a Haar cascade.", True,
     "verified: modules/face_webcam/emotion_detector.py"),
    ("topic:facial-expression-analysis", "navel_01", "limitation",
     "Affected by lighting, camera angle, occlusion, individual differences "
     "and genuinely ambiguous expressions.", False, RESEARCHER),
    ("topic:emotion-recognition", "navel_01", "method",
     "Combines facial cues with conversational context to estimate how a "
     "visitor is responding, and adapts communication style to match. "
     "Scores are placed on Russell's (1980) valence-arousal circumplex.",
     True, "verified: emotion_detector.py _VA_TABLE"),
    ("topic:emotion-recognition", "navel_01", "limitation",
     "Emotion recognition is inherently uncertain, and accuracy on real "
     "visitors differs substantially from controlled datasets because of "
     "lighting, pose, cultural difference and natural variation in how "
     "people express themselves.", False, RESEARCHER),
    ("topic:non-verbal-interaction", "navel_01", "method",
     "Communication carried by something other than the words themselves.",
     False, ROLE),
    ("topic:social-signals", "navel_01", "method",
     "Cues a person gives off during interaction, used to adapt the robot's "
     "behaviour.", False, ROLE),
    ("topic:human-robot-trust", "navel_01", "method",
     "How a visitor's confidence in the robot develops over an interaction.",
     False, ROLE),

    # ══ Unowned topics: background, explicitly nobody's result ═══════════════
    ("topic:speech-recognition", None, "limitation",
     "No project here claims speech recognition as its research area. It is "
     "infrastructure the demo runs on, not a result being presented.", True,
     "verified: no robot declares this topic"),
    ("topic:text-to-speech", None, "limitation",
     "No project claims text-to-speech as its research area. It is how the "
     "robots speak, not what they study.", True,
     "verified: no robot declares this topic"),
    ("topic:robot-hardware", None, "limitation",
     "The platforms are off-the-shelf. The contribution is in the software "
     "each robot runs, not the hardware.", True,
     "verified: no robot declares this topic"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print every unverified fact with its id, for review")
    ap.add_argument("--verify", nargs="*", metavar="ID",
                    help="mark fact ids as researcher-confirmed; "
                         "--verify all marks every unverified row")
    args = ap.parse_args()

    from data import demo_facts_repo as repo

    if args.list or args.verify is not None:
        from data.connection import get_client
        rows = (get_client().table(repo.TABLE).select("*")
                .eq("verified", False).order("topic_id").execute().data or [])
        if args.verify is None:
            print(f"  {len(rows)} unverified fact(s). Confirm with "
                  f"--verify <id> [<id> ...], or --verify all\n")
            for r in rows:
                print(f"  [{r['id']:>4}] {r['kind']:<11} {r['topic_id'][6:]:<30} "
                      f"{r.get('robot_id') or '(general)'}")
                print(f"         {r['fact']}")
            return 0
        targets = ([r["id"] for r in rows] if args.verify == ["all"]
                   else [int(x) for x in args.verify])
        done = sum(1 for t in targets if repo.set_verified(t, True))
        print(f"  marked {done}/{len(targets)} fact(s) verified.")
        return 0

    if args.report:
        rows = repo.coverage()
        print(f"  {'topic':<34} {'facts':>6} {'verified':>9}  gap")
        print("  " + "-" * 68)
        for r in rows:
            gap = "" if r["facts"] else "NO GROUNDING — answers will be general only"
            if r["facts"] and not r["verified_facts"]:
                gap = "nothing reviewed yet"
            print(f"  {r['label'][:33]:<34} {r['facts']:>6} "
                  f"{r['verified_facts']:>9}  {gap}")
        print(f"\n  {repo.unverified_count()} unverified fact(s) in total.")
        return 0

    nv = sum(1 for r in DRAFT if r[4])
    print(f"  {len(DRAFT)} rows — {nv} verified against this repository, "
          f"{len(DRAFT)-nv} supplied by the researcher and unconfirmed.")
    if args.dry_run:
        for t, r, k, f, v, _s in DRAFT:
            flag = "OK " if v else "   "
            print(f"  {flag}[{k:<10}] {t[6:]:<30} {r or '(general)':<11} {f[:52]}")
        print("\n  --dry-run: nothing written.")
        return 0

    written = 0
    for topic_id, robot_id, kind, fact, verified, source in DRAFT:
        if repo.add(topic_id, kind, fact, robot_id=robot_id,
                    verified=verified, source=source) is not None:
            written += 1
    nv = sum(1 for r in DRAFT if r[4])
    print(f"  wrote {written}/{len(DRAFT)} rows — {nv} verified against this "
          f"repository, {len(DRAFT)-nv} awaiting researcher confirmation.")
    print("  Review them with:  python3 tools/seed_topic_facts.py --report")
    return 0 if written == len(DRAFT) else 1


if __name__ == "__main__":
    sys.exit(main())
