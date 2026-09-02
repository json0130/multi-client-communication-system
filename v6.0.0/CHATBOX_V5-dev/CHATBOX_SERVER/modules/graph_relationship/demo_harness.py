"""
KG × PAD integration harness — proves per-person KG adaptation.

Wires the real end-to-end loop with NO hardware and NO webcam:

    KGBridge.pre_turn  →  PADPipelineAdapter.process_turn  →  KGBridge.post_turn

A shared InMemoryGraphStore persists across the whole run.

Faked inputs (the only stubs):
  person_id      — typed by the user or scripted
  camera_emotion — typed by the user or scripted (happy/sad/neutral/angry/…)
  robot_id       — flag --robot chatbox|ellebot  (default: chatbox)

Usage:
    python3 -m modules.graph_relationship.demo_harness              # interactive
    python3 -m modules.graph_relationship.demo_harness --scripted   # automated proof
    python3 -m modules.graph_relationship.demo_harness --scripted --llm
    python3 -m modules.graph_relationship.demo_harness --llm --model qwen2.5:7b
    python3 -m modules.graph_relationship.demo_harness --robot ellebot --obsidian
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# ── Module imports ────────────────────────────────────────────────────────────
from .schema import (
    Embodiment,
    PersonNode,
    RobotNode,
)
from .store import InMemoryGraphStore
from .kg_bridge import KGBridge, derive_tier, _tier_from_edges
from .interactions import count_person_turns, get_interaction, set_closeness

try:
    from ..pad_persona.pipeline_adapter import PADPipelineAdapter
except ImportError:
    _here = os.path.dirname(os.path.dirname(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from pad_persona.pipeline_adapter import PADPipelineAdapter  # type: ignore

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_EMOTIONS = frozenset({
    "happy", "sad", "neutral", "angry", "calm",
    "fear", "disgust", "surprise",
})
VALID_ROBOTS = frozenset({"chatbox", "ellebot"})

_ROBOT_DISPLAY = {"chatbox": "ChatBox", "ellebot": "ElleBot"}

_DEFAULT_SCENARIO = "Hi! Can we be friends today?"

_TIER_COLOUR = {
    "unknown": "\033[90m",
    "visitor": "\033[33m",
    "known":   "\033[36m",
    "close":   "\033[32m",
}
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RST   = "\033[0m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Stateless wrapper around Ollama's OpenAI-compatible API.

    Each call is independent (no history buffer) — intentional for the harness
    so that back-to-back person comparisons reflect ONLY the PAD system_prompt
    differences, not accumulated conversation context.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "127.0.0.1",
        port: int = 11434,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package required — pip install openai")
        self.model     = model
        self.available = False
        self._client   = _OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="ollama",
        )

    def connect(self) -> bool:
        try:
            self._client.models.list()
            self.available = True
            print(f"  [LLM] Connected — model: {self.model}")
            return True
        except Exception as exc:
            print(f"  [LLM] Cannot reach Ollama at {self._client.base_url}: {exc}")
            print("        Start Ollama with:  ollama serve")
            return False

    def respond(self, system_prompt: str, user_message: str) -> str:
        """Send one stateless turn and return the raw model string."""
        if not self.available:
            return "(LLM unavailable)"
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_message},
                ],
                temperature=0.7,
                stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"(LLM error: {exc})"


# ── Store helpers ─────────────────────────────────────────────────────────────

def _relationship_snapshot(
    store: InMemoryGraphStore,
    person_id: str,
    robot_id: str,
) -> dict:
    # rapport / trust / interaction_count all live on the InteractionNode now.
    interaction = get_interaction(store, person_id, robot_id)
    return {
        "rapport":           interaction.rapport if interaction else 0.0,
        "trust":             interaction.trust if interaction else 0.0,
        "interaction_count": count_person_turns(store, person_id),
    }


def _ensure_person(store: InMemoryGraphStore, person_id: str) -> None:
    if store.get_node(person_id) is None:
        store.upsert_node(PersonNode(id=person_id, display_name=person_id))


def _ensure_robot(store: InMemoryGraphStore, robot_id: str) -> None:
    if store.get_node(robot_id) is None:
        emb = Embodiment.CAT if robot_id.lower() == "chatbox" else Embodiment.ELEPHANT
        store.upsert_node(RobotNode(id=robot_id, name=robot_id, embodiment=emb))


# ── TEST AID ──────────────────────────────────────────────────────────────────

def seed_relationship(
    store: InMemoryGraphStore,
    person_id: str,
    robot_id: str,
    rapport: float = 0.5,
    trust: float = 0.5,
) -> None:
    """
    TEST AID — NOT production logic.

    Sets rapport and trust on the pair's InteractionNode so the harness can push
    a person into 'known' / 'close' tier without running enough interaction turns.

    Tier thresholds: score = (rapport + trust) / 2
      score > 0.70 → "close"  |  score > 0.45 → "known"
    """
    rapport = max(0.0, min(1.0, rapport))
    trust   = max(0.0, min(1.0, trust))
    _ensure_person(store, person_id)
    _ensure_robot(store, robot_id)
    set_closeness(store, person_id, robot_id, rapport=rapport, trust=trust,
                  source="harness:seed")
    score      = (rapport + trust) / 2.0
    tier_after = "close" if score > 0.70 else ("known" if score > 0.45 else "visitor")
    tc         = _TIER_COLOUR.get(tier_after, "")
    print(f"  [TEST AID] seeded {person_id}: rapport={rapport:.2f}  trust={trust:.2f}"
          f"  score={score:.2f}  → tier now {tc}{tier_after}{_RST}")


# ── End-of-session knowledge extraction ────────────────────────────────────────

def run_session_extraction(h, llm, matcher=None) -> None:
    """Distill each of this meetup's session transcripts into graph updates —
    closeness (rapport/trust) on the InteractionNode and new human interests →
    topics — via the LLM with deterministic guards (graph_relationship.extraction).

    `matcher` selects which robot capability covers a topic (keyword by default,
    or the embedding matcher when --embed is on) so shared topics link up.
    """
    from .extraction import extract_and_apply
    from .interactions import unextracted_turns, mark_session_extracted
    # Respect any external edits (viz deletions) before extracting.
    if h.kg_path:
        h.store.reload(h.kg_path)
    sessions = h.bridge.current_sessions()
    if not sessions:
        print("  (no session this run — nothing to extract)")
        return
    print("Extracting knowledge from this session …")
    for pid, sid in sessions.items():
        sess = h.store.get_node(sid)
        if sess is None or sess.node_type != "session":
            continue
        # Only the turns of THIS session not yet extracted — not the whole history.
        turns = unextracted_turns(sess)
        if not turns:
            print(f"  {_GREEN}{pid}{_RST}: no new turns since last extract")
            continue
        _update, s = extract_and_apply(
            h.store, pid, h.robot_id, turns, llm.respond,
            matcher=matcher, source="extraction")
        mark_session_extracted(h.store, sid)
        ints = s["interests_added"]
        int_str = ("  interests: " + ", ".join(
            f"{lab}→{'/'.join(ts)}" if ts else lab for lab, ts, _sm in ints)
        ) if ints else "  (no new interests)"
        print(f"  {_GREEN}{pid}{_RST}: Δrapport {s['rapport_delta']:+.2f}"
              f"  Δtrust {s['trust_delta']:+.2f}{int_str}")
        for item, tl in s.get("capability_links", []):
            print(f"      {_DIM}↳ shared topic '{tl}' — chatbox [{item}]{_RST}")
    if h.kg_path:
        h.store.save(h.kg_path)


# ── Graph export ──────────────────────────────────────────────────────────────

def _edge_label(edge) -> str:
    et = edge.edge_type
    if hasattr(edge, "weight"):
        return f"{et}={edge.weight:.2f}"
    if hasattr(edge, "count"):
        return f"{et}={edge.count}"
    if hasattr(edge, "value"):
        return f"{et}={edge.value:.2f}"
    return et


def export_graph_json(store: InMemoryGraphStore, out_path: str) -> None:
    data = {
        "nodes": [n.model_dump(mode="json") for n in store._nodes.values()],
        "edges": [e.model_dump(mode="json") for e in store._edges.values()],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"  → {os.path.abspath(out_path)}")


def export_graph_html(store: InMemoryGraphStore, out_path: str) -> None:
    try:
        from pyvis.network import Network  # type: ignore
    except ImportError:
        print("  pyvis not installed — run:  pip install pyvis")
        json_fallback = out_path.replace(".html", ".json")
        print("  Falling back to JSON export:")
        export_graph_json(store, json_fallback)
        return

    net = Network(
        height="780px", width="100%", directed=True,
        notebook=False, bgcolor="#1a1a2e", font_color="#e0e0e0",
    )
    net.set_options("""{
        "nodes": {"font": {"size": 14, "face": "monospace"}, "borderWidth": 2},
        "edges": {
            "font": {"size": 10, "color": "#bbbbbb", "align": "middle"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "physics": {
            "enabled": true, "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 120}
        }
    }""")

    _NODE_COLOUR = {"robot": "#4A90D9", "person": "#5CB85C",
                    "topic": "#F0AD4E", "event": "#D9534F"}
    _NODE_SHAPE  = {"robot": "star", "person": "ellipse",
                    "topic": "box",  "event":  "diamond"}

    for node in store._nodes.values():
        label = getattr(node, "display_name", None) or node.id
        tip   = [f"type: {node.node_type}", f"id: {node.id}"]
        if hasattr(node, "embodiment"):
            tip.append(f"embodiment: {node.embodiment}")
        net.add_node(
            node.id, label=label,
            color=_NODE_COLOUR.get(node.node_type, "#AAAAAA"),
            shape=_NODE_SHAPE.get(node.node_type, "ellipse"),
            title="<br>".join(tip),
            size=30 if node.node_type == "robot" else 20,
        )

    for edge in store._edges.values():
        tip = (f"type: {edge.edge_type}<br>"
               f"source: {edge.source_id}<br>"
               f"target: {edge.target_id}")
        net.add_edge(edge.source_id, edge.target_id,
                     label=_edge_label(edge), title=tip)

    net.write_html(out_path)
    print(f"  → {os.path.abspath(out_path)}")


def export_obsidian_vault(store: InMemoryGraphStore, vault_dir: str) -> None:
    os.makedirs(vault_dir, exist_ok=True)

    def _slug(node_id: str, node_type: str) -> str:
        return f"{node_type}_{node_id.replace('-', '_')}"

    for node in store._nodes.values():
        ns       = _slug(node.id, node.node_type)
        outgoing = [e for e in store._edges.values() if e.source_id == node.id]
        incoming = [e for e in store._edges.values()
                    if e.target_id == node.id and e.source_id != node.id]

        lines: list[str] = [f"# {node.node_type}: {node.id}", ""]
        lines.append(f"**type**: `{node.node_type}`")
        if hasattr(node, "embodiment"):
            lines.append(f"**embodiment**: `{node.embodiment}`")
        lines.append("")
        if outgoing:
            lines.append("## Outgoing edges")
            for e in outgoing:
                tgt = store._nodes.get(e.target_id)
                if tgt:
                    lines.append(f"- `{_edge_label(e)}` → [[{_slug(tgt.id, tgt.node_type)}]]")
            lines.append("")
        if incoming:
            lines.append("## Incoming edges")
            for e in incoming:
                src = store._nodes.get(e.source_id)
                if src:
                    lines.append(f"- [[{_slug(src.id, src.node_type)}]] → `{_edge_label(e)}`")
            lines.append("")

        with open(os.path.join(vault_dir, f"{ns}.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    print(f"  → {os.path.abspath(vault_dir)}  ({len(store._nodes)} markdown files)")


# ── Harness class ─────────────────────────────────────────────────────────────

class Harness:
    """
    Owns the shared InMemoryGraphStore and runs one turn at a time.

    The store is created once at construction and lives for the whole run —
    per-person state MUST persist across turns; that is the entire point.
    """

    def __init__(
        self,
        robot_id: str = "chatbox",
        obsidian: bool = False,
        llm: Optional[LLMClient] = None,
        kg_path: Optional[str] = None,
    ) -> None:
        self.store    = InMemoryGraphStore()
        self.bridge   = KGBridge(self.store)
        self.robot_id = robot_id
        self.obsidian = obsidian
        self.llm      = llm
        # When set, the graph is persisted to this file after every turn so the
        # standalone viz server (graph_relationship.viz) can poll it live.
        self.kg_path  = kg_path
        # Load an existing graph (e.g. seeded subgraphs) so per-turn saves append
        # to it instead of overwriting it with a fresh, empty store.
        if kg_path and os.path.exists(kg_path):
            self.store.load(kg_path)
        self.turn_n   = 0
        self._pad_adapters: dict[str, PADPipelineAdapter] = {}

    def _adapter(self, robot_id: str) -> PADPipelineAdapter:
        if robot_id not in self._pad_adapters:
            self._pad_adapters[robot_id] = PADPipelineAdapter(robot_id)
        return self._pad_adapters[robot_id]

    # ── Core turn ─────────────────────────────────────────────────────────────

    def run_turn(
        self,
        person_id: str,
        robot_id: str,
        emotion: str,
        user_message: Optional[str] = None,
    ) -> dict:
        """
        Run one full KG → PAD → KG loop and print a readable trace.

        user_message: if provided AND self.llm is connected, calls the LLM with
        the PAD-generated system_prompt and prints the verbal response.  This
        makes the per-person PAD differences audible, not just numeric.
        """
        self.turn_n += 1

        # 0. Re-sync from disk so external edits (e.g. deletions made in the viz)
        #    are respected instead of being overwritten by our in-memory graph.
        if self.kg_path:
            self.store.reload(self.kg_path)

        # 1. KG → bridge input (tier, blended v/a, slow-edge memory)
        bi = self.bridge.pre_turn(person_id, robot_id, emotion)

        # 2. PAD update — tier drives D-axis; V/A from camera_emotion via bridge
        pad_result = self._adapter(robot_id).process_turn(
            valence=bi.valence,
            arousal=bi.arousal,
            relationship_tier=bi.tier,
            memory_context=bi.structured_memory,
        )

        # 3. Verbal response FIRST (needs pad_result) so it can be stored on the
        #    session Event node by post_turn. Printed below in trace order.
        verbal: Optional[str] = None
        if user_message is not None and self.llm is not None and self.llm.available:
            verbal = self.llm.respond(pad_result["system_prompt"], user_message)

        # 4. Write back mood + attention, and append this turn to the current
        #    session Event node (person↔robot connect through the event).
        self.bridge.post_turn(person_id, robot_id, pad_result,
                              emotion=emotion, child_message=user_message,
                              reply=verbal)

        # 5. Read relationship snapshot AFTER post_turn
        rel = _relationship_snapshot(self.store, person_id, robot_id)

        # 6. Print PAD trace
        p, a, d  = pad_result["pad_state"]
        desc     = pad_result["descriptors"]
        mem_str  = bi.structured_memory or ""
        tc       = _TIER_COLOUR.get(bi.tier, "")

        print(
            f"turn {self.turn_n:3d} | "
            f"person={person_id:<10s} robot={robot_id:<8s} emotion={emotion}"
        )
        print(
            f"         → tier={tc}{bi.tier:<8s}{_RST}"
            f"  v={bi.valence:+.2f}  a={bi.arousal:+.2f}"
        )
        print(
            f"         → PAD  P={p:+.3f}  A={a:+.3f}  D={d:+.3f}"
            f"   descriptors={desc['pleasure']}/{desc['arousal']}/{desc['dominance']}"
        )
        print(f"         → mem={mem_str!r}")
        print(
            f"         [graph]  rapport={rel['rapport']:.2f}"
            f"  trust={rel['trust']:.2f}"
            f"  interaction_count={rel['interaction_count']}"
        )

        if verbal is not None:
            display = _ROBOT_DISPLAY.get(robot_id, robot_id.title())
            print(f"         {_DIM}[child]   {user_message!r}{_RST}")
            print(f"         {_BOLD}[{display}]{_RST}  {_GREEN}{verbal}{_RST}")

        # 7. Persist the graph so the live viz server can pick it up within ~1s.
        if self.kg_path:
            self.store.save(self.kg_path)

        return pad_result

    # ── Summary view ──────────────────────────────────────────────────────────

    def show_people(self) -> None:
        persons = [n for n in self.store._nodes.values() if n.node_type == "person"]
        if not persons:
            print("  (no people in store yet)")
            return
        print(f"  {'Person':<12s}  {'Tier':<10s}  {'Count':>5s}  {'Rapport':>7s}  {'Trust':>5s}")
        print("  " + "─" * 48)
        for node in sorted(persons, key=lambda n: n.id):
            tier = derive_tier(node.id, self.robot_id, self.store)
            rel  = _relationship_snapshot(self.store, node.id, self.robot_id)
            tc   = _TIER_COLOUR.get(tier, "")
            print(
                f"  {node.id:<12s}  {tc}{tier:<10s}{_RST}"
                f"  {rel['interaction_count']:>5d}"
                f"  {rel['rapport']:>7.2f}"
                f"  {rel['trust']:>5.2f}"
            )

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, out_dir: str = ".") -> None:
        print("Exporting graph …")
        export_graph_html(self.store, os.path.join(out_dir, "graph_snapshot.html"))
        if self.obsidian:
            export_obsidian_vault(self.store, os.path.join(out_dir, "vault"))


# ── System-prompt excerpt helper ──────────────────────────────────────────────

def _sysprompt_key_lines(system_prompt: str) -> tuple[str, str]:
    """Extract the mood line and relationship-note line from a PAD system prompt."""
    mood_line = ""
    rel_line  = ""
    for line in system_prompt.splitlines():
        stripped = line.strip()
        if "Right now respond in a" in stripped:
            mood_line = stripped
        elif any(k in stripped for k in (
            "You know this person", "This person is a new face",
            "You are meeting this person", "like family",
        )):
            rel_line = stripped
    return mood_line, rel_line


# ── Scripted proof sequence ───────────────────────────────────────────────────

_DIVIDER = "─" * 72


def run_scripted(
    robot_id: str = "chatbox",
    obsidian: bool = False,
    llm: Optional[LLMClient] = None,
    scenario: str = _DEFAULT_SCENARIO,
    kg_path: Optional[str] = None,
) -> None:
    """
    Fixed proof sequence — no typing required.

    With --llm, also sends the same `scenario` sentence to the LLM for
    alice (close), bob (visitor), and casey (unknown) back-to-back so you
    can SEE the verbal divergence, not just the PAD numbers.
    """
    h = Harness(robot_id=robot_id, obsidian=obsidian, llm=llm, kg_path=kg_path)

    llm_on = llm is not None and llm.available
    display = _ROBOT_DISPLAY.get(robot_id, robot_id.title())

    print("=" * 72)
    print("  SCRIPTED DEMO — KG × PAD per-person adaptation")
    print(f"  robot: {robot_id}" + (f"   |   LLM: {llm.model}" if llm_on else "   |   LLM: off"))
    if llm_on:
        print(f"  scenario: {scenario!r}")
    print("=" * 72)

    # ── PROOF 1: same person across many turns ────────────────────────────────
    print()
    print(_DIVIDER)
    print("PROOF 1 — alice × 6 turns (happy): tier escalation, D-axis shift")
    print("  Warm-up turns — LLM skipped to keep output concise.")
    print(_DIVIDER)
    print()
    for _ in range(6):
        h.run_turn("alice", robot_id, "happy")   # no LLM on warm-up turns
        print()

    # ── bob: 2 turns ─────────────────────────────────────────────────────────
    print(_DIVIDER)
    print("PROOF 2 setup — bob × 2 turns (neutral): builds visitor tier")
    print(_DIVIDER)
    print()
    for _ in range(2):
        h.run_turn("bob", robot_id, "neutral")
        print()

    # ── Boost alice to 'close' ────────────────────────────────────────────────
    print(_DIVIDER)
    print("BOOST alice — rapport=0.75  trust=0.75  (score=0.75 > 0.70 → 'close')")
    print(_DIVIDER)
    print()
    seed_relationship(h.store, "alice", robot_id, rapport=0.75, trust=0.75)
    print()

    # ── PROOF 2: PAD divergence ───────────────────────────────────────────────
    print(_DIVIDER)
    print("PROOF 2 — alice (close) vs bob (visitor) vs casey (unknown)")
    print("  Same emotion: happy" + (f"   |   Same scenario: {scenario!r}" if llm_on else ""))
    print("  Watch D-axis diverge. With --llm: watch verbal responses diverge.")
    print(_DIVIDER)
    print()

    print("[alice — close tier]")
    alice_result = h.run_turn("alice", robot_id, "happy",
                               user_message=scenario if llm_on else None)
    print()

    print("[bob — visitor tier]")
    bob_result   = h.run_turn("bob",   robot_id, "happy",
                               user_message=scenario if llm_on else None)
    print()

    # ── PROOF 3: cold-start ───────────────────────────────────────────────────
    print(_DIVIDER)
    print("PROOF 3 — casey cold-start (never seen before)")
    print(_DIVIDER)
    print()
    casey_result = h.run_turn("casey", robot_id, "happy",
                               user_message=scenario if llm_on else None)
    print()

    # ── Verbal diff summary (only when LLM ran) ───────────────────────────────
    if llm_on:
        print(_DIVIDER)
        print(f"VERBAL DIFF SUMMARY — same scenario: {scenario!r}")
        print(f"Shows HOW the PAD system_prompt encodes tier → different {display} voice")
        print(_DIVIDER)
        for label, result in [
            ("alice (close)",   alice_result),
            ("bob   (visitor)", bob_result),
            ("casey (unknown)", casey_result),
        ]:
            mood_line, rel_line = _sysprompt_key_lines(result["system_prompt"])
            desc = result["descriptors"]
            print(f"\n  {_BOLD}{label}{_RST}")
            print(f"    PAD mood  : {desc['pleasure']}/{desc['arousal']}/{desc['dominance']}")
            print(f"    sys mood  : {_DIM}{mood_line}{_RST}")
            print(f"    sys tier  : {_DIM}{rel_line}{_RST}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(_DIVIDER)
    print("Final people state")
    print(_DIVIDER)
    h.show_people()
    print()

    print(_DIVIDER)
    h.export(".")
    print()

    print("=" * 72)
    print("  SCRIPTED DEMO COMPLETE")
    print("=" * 72)


# ── Interactive loop ──────────────────────────────────────────────────────────

_HELP = """\
Commands
────────
  <person> <emotion>              run one turn (PAD only)       e.g.  alice happy
  <person> <emotion> <message…>   run + LLM verbal response     e.g.  alice happy Hi can we play?
  boost <person> [r] [t]          TEST AID: seed rapport/trust  e.g.  boost alice 0.75 0.75
  robot <chatbox|ellebot>         switch active robot
  who                             list all known people and their tiers
  graph                           export graph snapshot right now
  extract                         distill this session → interests + closeness (needs --llm)
  relink                          re-match capabilities → topics (embedding if --embed)
  help                            show this message
  q / quit                        exit (extracts this session, then exports graph)

Emotions:  happy  sad  neutral  angry  calm  fear  disgust  surprise

LLM note:
  Pass --llm when launching to enable verbal responses.
  Include a message after the emotion to trigger the LLM for that turn,
  e.g.   alice happy What should we do today?
  Or use --scenario TEXT to set a default message used for every turn.
"""


def run_interactive(
    robot_id: str = "chatbox",
    obsidian: bool = False,
    llm: Optional[LLMClient] = None,
    scenario: Optional[str] = None,
    kg_path: Optional[str] = None,
    matcher=None,
) -> None:
    h       = Harness(robot_id=robot_id, obsidian=obsidian, llm=llm, kg_path=kg_path)
    llm_on  = llm is not None and llm.available

    print("=" * 72)
    print("  KG × PAD Integration Harness  —  interactive mode")
    print(f"  robot: {robot_id}"
          + (f"   |   LLM: {llm.model}" if llm_on else "   |   LLM: off (use --llm)"))
    if llm_on and scenario:
        print(f"  default scenario: {scenario!r}  (overridable per turn)")
    print("  type 'help' for commands")
    print("=" * 72)
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            break

        if cmd == "help":
            print(_HELP)
            continue

        if cmd == "who":
            h.show_people()
            print()
            continue

        if cmd == "graph":
            h.export(".")
            print()
            continue

        if cmd == "robot":
            if len(parts) < 2 or parts[1].lower() not in VALID_ROBOTS:
                print(f"  Usage: robot <{'|'.join(sorted(VALID_ROBOTS))}>")
            else:
                h.robot_id = parts[1].lower()
                print(f"  Robot switched to: {h.robot_id}")
            print()
            continue

        if cmd == "boost":
            if len(parts) < 2:
                print("  Usage: boost <person> [rapport=0.5] [trust=0.5]")
                print()
                continue
            person_id = parts[1]
            try:
                rapport = float(parts[2]) if len(parts) > 2 else 0.5
                trust   = float(parts[3]) if len(parts) > 3 else 0.5
            except ValueError:
                print("  rapport and trust must be floats in [0, 1]")
                print()
                continue
            seed_relationship(h.store, person_id, h.robot_id, rapport, trust)
            if h.kg_path:
                h.store.save(h.kg_path)
            print()
            continue

        if cmd == "extract":
            if not llm_on:
                print("  extraction needs the LLM — run with --llm")
            else:
                run_session_extraction(h, llm, matcher=matcher)
            print()
            continue

        if cmd == "relink":
            from .topics import relink_capability_topics
            linked = relink_capability_topics(h.store, h.robot_id, matcher=matcher)
            how = "embedding" if matcher is not None else "keyword"
            if linked:
                print(f"  re-linked ({how}) capability → topic:")
                for item, tl in linked:
                    print(f"      chatbox [{item}] → {tl}")
            else:
                print(f"  no new capability→topic links ({how} matcher)")
            if h.kg_path:
                h.store.save(h.kg_path)
            print()
            continue

        # ── <person> <emotion> [message…] ─────────────────────────────────────
        if len(parts) >= 2:
            person_id = parts[0]
            emotion   = parts[1].lower()

            if emotion not in VALID_EMOTIONS:
                print(f"  Unknown emotion '{emotion}'.")
                print(f"  Valid: {', '.join(sorted(VALID_EMOTIONS))}")
                print()
                continue

            # Message: inline tokens ≥3, else --scenario default, else None
            if len(parts) >= 3:
                user_message = " ".join(parts[2:])
            elif scenario:
                user_message = scenario
            else:
                user_message = None

            h.run_turn(person_id, h.robot_id, emotion,
                       user_message=user_message if llm_on else None)
            print()
            continue

        # ── single token: just a person name → prompt for rest ────────────────
        if len(parts) == 1 and cmd not in VALID_ROBOTS:
            person_id = parts[0]
            try:
                emotion = input(f"  emotion for {person_id}? ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if emotion not in VALID_EMOTIONS:
                print(f"  Unknown emotion '{emotion}'.")
                print(f"  Valid: {', '.join(sorted(VALID_EMOTIONS))}")
                print()
                continue

            user_message = None
            if llm_on:
                if scenario:
                    user_message = scenario
                else:
                    try:
                        typed = input("  child says: ").strip()
                        user_message = typed if typed else None
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break

            h.run_turn(person_id, h.robot_id, emotion, user_message=user_message)
            print()
            continue

        print(f"  Unrecognised input: {raw!r}  —  type 'help' for commands")
        print()

    # End-of-session trigger: distill the meetup's transcript into graph updates.
    if llm_on:
        print()
        run_session_extraction(h, llm, matcher=matcher)

    print()
    print("Exiting — writing final graph …")
    h.export(".")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="KG × PAD integration harness — proves per-person KG adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scripted", action="store_true",
                    help="Run the fixed proof sequence then exit")
    ap.add_argument("--robot", choices=sorted(VALID_ROBOTS), default="chatbox",
                    metavar="ROBOT",
                    help="Robot persona: chatbox (default) or ellebot")
    ap.add_argument("--obsidian", action="store_true",
                    help="Also write an Obsidian markdown vault to ./vault/")
    ap.add_argument("--llm", action="store_true",
                    help="Enable verbal LLM responses via Ollama")
    ap.add_argument("--model", default="qwen2.5:7b", metavar="MODEL",
                    help="Ollama model name (default: qwen2.5:7b)")
    ap.add_argument("--scenario", default=None, metavar="TEXT",
                    help=(f"Fixed child message for every LLM turn "
                          f"(default: {_DEFAULT_SCENARIO!r})"))
    ap.add_argument("--kg-path", default=None, metavar="FILE",
                    help=("Persist the graph to FILE after every turn so the "
                          "live viz server (python3 -m modules.graph_relationship."
                          "viz.server) can poll it. e.g. --kg-path kg_state.json"))
    ap.add_argument("--embed", action="store_true",
                    help="Match extracted topics to robot capabilities by embedding "
                         "similarity (Ollama) instead of keywords, so near-matches "
                         "(addition→math, planets→space) connect")
    ap.add_argument("--embed-model", default="nomic-embed-text", metavar="MODEL",
                    help="Ollama embedding model (default: nomic-embed-text)")
    ap.add_argument("--embed-floor", type=float, default=0.5, metavar="F",
                    help="Min cosine similarity to link a topic to a capability "
                         "(default: 0.5)")
    args = ap.parse_args(argv)

    # ── Topic matcher (keyword default; embedding when --embed) ────────────────
    matcher = None
    if args.embed:
        from .embedding import make_embedding_matcher, ollama_embed_fn
        matcher = make_embedding_matcher(
            ollama_embed_fn(model=args.embed_model), floor=args.embed_floor)
        print(f"  [embed] topic matching via {args.embed_model} "
              f"(floor {args.embed_floor})")

    # ── LLM setup ─────────────────────────────────────────────────────────────
    llm: Optional[LLMClient] = None
    if args.llm:
        if not _OPENAI_AVAILABLE:
            print("ERROR: openai package not installed — pip install openai")
            sys.exit(1)
        llm = LLMClient(model=args.model)
        if not llm.connect():
            print("  Continuing without LLM (pass --no-llm to suppress this).")
            llm = None   # graceful degradation

    scenario = args.scenario or (_DEFAULT_SCENARIO if args.llm else None)

    if args.scripted:
        run_scripted(robot_id=args.robot, obsidian=args.obsidian,
                     llm=llm, scenario=scenario or _DEFAULT_SCENARIO,
                     kg_path=args.kg_path)
    else:
        run_interactive(robot_id=args.robot, obsidian=args.obsidian,
                        llm=llm, scenario=scenario, kg_path=args.kg_path,
                        matcher=matcher)


if __name__ == "__main__":
    main()
