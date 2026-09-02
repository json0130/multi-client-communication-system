"""
tools/fixtures/questions.py
===========================
The visitor question corpus. VERSIONED AND FIXED.

Runs are only comparable if the question set does not move. Changing a question
changes what the correction-rate curve is measuring, so treat edits the way you
would treat editing a benchmark: bump CORPUS_VERSION and do not mix runs across
versions.

Each entry carries the topic it is INTENDED to resolve to, which is what lets
resolution accuracy be measured separately from routing quality. The two have
different fixes — a resolution failure means the matcher or the vocabulary is
wrong, a routing failure means the graph is wrong — and a single "the system got
it wrong" number cannot tell you which.

Three deliberate categories:

  intended = "<topic id>"   should resolve there
  intended = None           should resolve to NOTHING; exercises the fail-open
                            path, where routing falls back to whoever was asked
  ambiguous = True          plausibly spans two topics. Word overlap ties and
                            resolution returns None by design, because a
                            question that names two subjects has named neither.
                            These are also where routing is genuinely uncertain
                            and where exploration should fire once resolution
                            improves enough to reach the graph.
"""

from __future__ import annotations

CORPUS_VERSION = "1.0"


def q(text, intended=None, ambiguous=False):
    return {"text": text, "intended": intended, "ambiguous": ambiguous}


# KNOWN RESOLVER FAILURES, deliberately left in.
# Seven of these resolve to None under the current word-overlap matcher even
# though a person would place them immediately — "how do you remember a
# conversation" shares no word with "conversational memory" (conversation vs
# conversational), and "how do you navigate around people" shares none with
# "social robot navigation". They are kept because they are things visitors
# actually say, and because resolution accuracy is a metric the harness reports
# rather than a number to be tuned away. Editing them to match the matcher would
# be tuning the benchmark to the system.
#
# Baseline at corpus v1.0:
#   word overlap only      36/43 (84%)  — 7 unresolved, 0 mis-resolved
#   with stemming          39/43 (91%)  — 4 unresolved, 0 mis-resolved
#
# All failures in BOTH configurations are unresolved rather than mis-resolved,
# which is the failure class that matters: unresolved loses data, mis-resolved
# writes an observation against the wrong topic that is indistinguishable from a
# real one afterwards. Report the two separately, always.
#
# The four that survive stemming are genuine ambiguities, not morphology:
# 'emotion model' spans two topics, 'cues' is a synonym for 'signals' rather
# than an inflection, 'robot' appears in three topic labels and 'speech' in two.
# Those need either a richer vocabulary or a semantic matcher, and neither is
# obviously worth it while the failure stays safe.

QUESTIONS = [
    # ── retrieval augmented generation ───────────────────────────────────────
    q("how does retrieval augmented generation work", "topic:retrieval-augmented-generation"),
    q("what is retrieval augmented generation", "topic:retrieval-augmented-generation"),
    q("can you explain augmented generation to me", "topic:retrieval-augmented-generation"),
    q("why is retrieval better than just generation", "topic:retrieval-augmented-generation"),

    # ── large language models ────────────────────────────────────────────────
    q("which large language models do you run", "topic:large-language-models"),
    q("how big are the language models you use", "topic:large-language-models"),
    q("do the language models run locally", "topic:large-language-models"),

    # ── conversational memory ────────────────────────────────────────────────
    q("how do you remember a conversation", "topic:conversational-memory"),
    q("what happens to conversational memory between visits", "topic:conversational-memory"),
    q("does your memory persist after I leave", "topic:conversational-memory"),

    # ── knowledge graphs ─────────────────────────────────────────────────────
    q("what are the knowledge graphs used for", "topic:knowledge-graphs"),
    q("how do knowledge graphs help the robots", "topic:knowledge-graphs"),

    # ── emotion recognition ──────────────────────────────────────────────────
    q("how does emotion recognition work", "topic:emotion-recognition"),
    q("can you tell what emotion I am feeling", "topic:emotion-recognition"),
    q("is the emotion model accurate", "topic:emotion-recognition"),

    # ── facial expression analysis ───────────────────────────────────────────
    q("what facial expression are you seeing", "topic:facial-expression-analysis"),
    q("how do you analyse a facial expression", "topic:facial-expression-analysis"),

    # ── social signals ───────────────────────────────────────────────────────
    q("which social signals do you pick up on", "topic:social-signals"),
    q("do you read social cues from posture", "topic:social-signals"),

    # ── human robot trust ────────────────────────────────────────────────────
    q("how do you build trust with a person", "topic:human-robot-trust"),
    q("what makes someone trust a robot", "topic:human-robot-trust"),

    # ── social robot navigation ──────────────────────────────────────────────
    q("how do you navigate around people", "topic:social-robot-navigation"),
    q("what is social navigation", "topic:social-robot-navigation"),

    # ── mapping and localisation ─────────────────────────────────────────────
    q("how do you build a map of the lab", "topic:mapping-and-localisation"),
    q("what does localisation mean here", "topic:mapping-and-localisation"),

    # ── multi robot coordination ─────────────────────────────────────────────
    q("how do the robots coordinate with each other", "topic:multi-robot-coordination"),
    q("who decides which robot answers", "topic:multi-robot-coordination"),

    # ── robot hardware ───────────────────────────────────────────────────────
    q("what hardware is inside you", "topic:robot-hardware"),
    q("which robot hardware was hardest to build", "topic:robot-hardware"),

    # ── speech recognition ───────────────────────────────────────────────────
    q("how good is your speech recognition", "topic:speech-recognition"),
    q("does speech recognition work in a noisy room", "topic:speech-recognition"),

    # ── text to speech ───────────────────────────────────────────────────────
    q("how is your text to speech generated", "topic:text-to-speech"),
    q("why does the speech sound synthetic", "topic:text-to-speech"),

    # ── Should NOT resolve ───────────────────────────────────────────────────
    # Real things visitors say. Routing must fall back to whoever was asked, and
    # no observation may be written — there is no edge these belong to.
    q("what is the weather like today", None),
    q("how long have you been at this university", None),
    q("are you going to take my job", None),
    q("can I have a photo with you", None),
    q("who funds all of this", None),
    q("hello", None),

    # ── Ambiguous: plausibly two topics ──────────────────────────────────────
    # 'recognition' alone spans emotion and speech; 'social' spans navigation
    # and signals. Resolution ties and returns None, which is correct — the
    # question named two subjects, so it named neither.
    q("how does recognition work on the robot", None, ambiguous=True),
    q("tell me about the social side of the research", None, ambiguous=True),
    # Reads ambiguous, but this vocabulary contains exactly one topic with
    # 'models' in it, so resolution is unambiguous and correct here.
    q("what models do you use", "topic:large-language-models"),
    q("how does the robot understand speech and emotion", None, ambiguous=True),
]


def by_category() -> dict:
    return {
        "resolvable": [x for x in QUESTIONS if x["intended"]],
        "unresolvable": [x for x in QUESTIONS if not x["intended"] and not x["ambiguous"]],
        "ambiguous": [x for x in QUESTIONS if x["ambiguous"]],
    }
