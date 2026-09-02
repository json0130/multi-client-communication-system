"""
Live webcam → FaceIdentifier + EmotionProcessor → KGBridge → PAD → LLM loop.

All interaction happens inside the OpenCV window — no terminal typing needed.

Run from CHATBOX_SERVER/:
    python3 -m modules.face_webcam.webcam_loop --mode enroll --name jay
    python3 -m modules.face_webcam.webcam_loop --mode run --llm

In-window keyboard controls:
    T       — open chat input box (type message, Enter=send, Esc=cancel)
    E       — open enroll box (type name, Enter=capture 12 frames, Esc=cancel)
    B       — boost current person's rapport+trust (+0.15 each)
    S       — save faces.npz
    Q / Esc — quit and auto-save faces

Tier progression (at 1 tick/sec, happy emotion):
    unknown → visitor : tick 1   (first interaction)
    visitor → known   : tick 6   (count > 5)
    known   → close   : ~34 s    (rapport+trust average > 0.70 via auto-increment)
    OR press B 5 ×               (instant +0.15 each press)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import socket
import sys
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


@contextlib.contextmanager
def _mute_stderr():
    """
    Redirect file-descriptor 2 to /dev/null for the duration of the block.

    Qt's QFontDatabase writes 'Cannot find font directory' via qWarning() which
    bypasses Python's sys.stderr and QT_LOGGING_RULES — only an fd-level
    redirect suppresses it.  Used only around cv2.namedWindow / cv2.imshow
    so real errors are never hidden.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved   = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# ── Import path resolution ────────────────────────────────────────────────────

def _add_server_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

_SERVER_ROOT = _add_server_root()

# ── KG + PAD ──────────────────────────────────────────────────────────────────

from modules.graph_relationship.kg_bridge import KGBridge
from modules.graph_relationship.store import InMemoryGraphStore
from modules.pad_persona.pipeline_adapter import PADPipelineAdapter
from modules.face_webcam.face_id import FaceIdentifier
from modules.face_webcam.emotion_detector import EmotionDetector
from modules.face_webcam.demographics import DemographicsDetector
from modules.session_store import SessionStore, DEFAULT_DB as _DEFAULT_SESSIONS_DB

# ── LLM ───────────────────────────────────────────────────────────────────────

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FACES   = "faces.npz"
# Isolated KG file for the culture branch — kept separate from the shared
# kg_state.json so migration/consolidation tools on OTHER branches can't clobber
# the culture graph. Override with --kg.
_DEFAULT_KG      = "kg_culture.json"
_DEFAULT_ROBOT   = "chatbox"
_DEFAULT_TICK    = 1.0
_DEFAULT_THRESH  = 0.75
_DEFAULT_CAMERA  = 0
_DEFAULT_MODEL   = "qwen2.5:7b"

# Input mode states
_MODE_IDLE   = 0
_MODE_CHAT   = 1
_MODE_ENROLL = 2

# BGR colours
_C_CLOSE   = (80,  220, 60)
_C_KNOWN   = (60,  200, 200)
_C_VISITOR = (40,  170, 255)
_C_UNKNOWN = (60,  60,  230)
_C_WHITE   = (255, 255, 255)
_C_BLACK   = (0,   0,   0)
_C_YELLOW  = (0,   215, 215)
_C_GRAY    = (140, 140, 140)
_C_GREEN   = (70,  220, 90)
_C_PINK    = (200, 130, 255)
_C_CYAN    = (210, 230, 20)
_C_DARK    = (20,  20,  20)

_TIER_COL = {
    "close":   _C_CLOSE,
    "known":   _C_KNOWN,
    "visitor": _C_VISITOR,
    "unknown": _C_UNKNOWN,
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM action-tag parsing + ESP32 dispatch
# ─────────────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r'^\[([A-Z_]+)\]', re.ASCII)

# Map LLM action tags → ESP32 validExpressions[]
_TAG_TO_ESP32: dict[str, str] = {
    "GREETING": "greeting",
    "WAVE":     "wave",
    "NOD":      "head_nod",
    "CONFUSED": "confused",
    "SAD":      "sad",
    "ANGRY":    "angry",
    "SHRUG":    "shrug",
    "POINT":    "point",
    "DANCE":    "seq_dance",
    "SLEEP":    "sleep",
    "IDLE":     "idle",
    "HAPPY":    "ears_wiggle",
    "SURPRISE": "ears_perk",
    "EARS":     "ears_perk",
}


_LEAK_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                 "\nuser", "\nUser", "\nassistant", "\nAssistant")


def _clean_reply(text: Optional[str]) -> str:
    """Trim a model reply at any leaked ChatML / next-turn marker (qwen sometimes
    keeps generating past its turn — Chinese text + a fake 'user:' turn)."""
    s = (text or "").strip()
    cut = len(s)
    for mark in _LEAK_MARKERS:
        i = s.find(mark)
        if i != -1:
            cut = min(cut, i)
    return s[:cut].strip()


def _parse_llm_response(text: str) -> tuple[str, str]:
    """Split '[TAG] body text' into ('TAG', 'body text'). Returns ('', text) if no tag."""
    m = _TAG_RE.match(text.strip())
    if m:
        return m.group(1), text[m.end():].strip()
    return "", text.strip()


def _send_esp32(expression: str, host: str, port: int = 8888,
                timeout: float = 0.5) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall((expression + "\n").encode())
        print(f"[ESP32] → {expression!r}")
    except OSError as exc:
        print(f"[ESP32] send failed ({host}:{port}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# First-impression: name learning (auto-enrol unknown faces → learn their name)
# ─────────────────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(
    r"\b(?:my name is|my name's|i am|i'm|im|call me|this is|name is|name's|names)"
    r"\s+([A-Za-z][A-Za-z'\-]{1,19})",
    re.IGNORECASE,
)
# Words that follow the trigger phrases but are NOT names — avoids "I'm fine" /
# "I am not sure" being read as the name "fine" / "not".
_NAME_STOPWORDS = frozenset({
    "not", "sorry", "fine", "good", "great", "okay", "ok", "here", "just",
    "really", "so", "very", "doing", "the", "a", "an", "sure", "happy", "sad",
    "tired", "hungry", "back", "done", "trying", "going", "feeling", "glad",
    "curious", "bored", "excited", "confused", "afraid", "scared", "korean", "maori",
})


def _extract_name(text: Optional[str]) -> Optional[str]:
    """Pull a first name out of a self-introduction, or None. Rejects obvious
    non-names (feelings/filler) via a small stoplist."""
    m = _NAME_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip("-'")
    if len(raw) < 2 or raw.lower() in _NAME_STOPWORDS:
        return None
    return raw


def _slug_name(name: str) -> str:
    """Lowercase single-token id from a captured name (faces keyed 'jay', 'alice')."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Ollama wrapper with optional rolling chat history."""

    def __init__(self, model: str = _DEFAULT_MODEL,
                 host: str = "127.0.0.1", port: int = 11434):
        self.model     = model
        self.available = False
        self._base_url = f"http://{host}:{port}/v1"
        self._client   = None

    def connect(self) -> bool:
        if not _OPENAI_AVAILABLE:
            print("[LLM] openai package not installed — run: pip install openai")
            return False
        try:
            self._client = _OpenAI(base_url=self._base_url, api_key="ollama")
            self._client.models.list()
            self.available = True
            print(f"[LLM] connected — model: {self.model}")
        except Exception as exc:
            print(f"[LLM] Ollama unavailable: {exc}")
        return self.available

    def respond(self, system_prompt: str, user_msg: str,
                history: list[tuple[str, str]] | None = None,
                max_tokens: int = 140, json_mode: bool = False) -> str:
        """
        Args:
            history: list of (user_text, assistant_text) pairs from previous turns.
                     Injected as alternating user/assistant messages before user_msg.
            max_tokens: reply budget. 140 suits a short spoken reply; JSON extraction
                     (topics/closeness) needs far more or the JSON truncates mid-object.
            json_mode: for extraction calls — request a strict JSON object
                     (response_format), temperature 0, and DROP the spoken-reply stop
                     strings. This near-eliminates the prose-wrapper / truncation
                     misfires that otherwise silently drop a whole turn's topics, and
                     makes extraction deterministic. Not used for spoken replies.
        """
        if not self.available or self._client is None:
            return "[LLM not connected — run with --llm]"
        try:
            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            for u, b in (history or []):
                messages.append({"role": "user",      "content": u})
                messages.append({"role": "assistant", "content": b})
            messages.append({"role": "user", "content": user_msg})
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["temperature"] = 0.0
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["temperature"] = 0.7
                # Stop if the model tries to continue past its turn / open a new one
                # (qwen sometimes leaks ChatML tokens or a fake "user:" turn). NOT
                # applied in json_mode — a stray "\nassistant" inside JSON would
                # truncate a valid object.
                kwargs["stop"] = ["<|im_start|>", "<|im_end|>",
                                  "\nuser", "\nUser", "\nassistant"]
            resp = self._client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            # In json_mode the content is a clean JSON object — skip _clean_reply so
            # it can never trim JSON that happens to contain a leak-marker substring.
            return (content or "").strip() if json_mode else _clean_reply(content)
        except Exception as exc:
            return f"[LLM error: {exc}]"


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def _text(frame, msg: str, pos: tuple,
          scale: float = 0.62, col=_C_WHITE, thickness: int = 2) -> None:
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, col, thickness, cv2.LINE_AA)


def _panel(frame, x1: int, y1: int, x2: int, y2: int,
           col=_C_DARK, alpha: float = 0.72) -> None:
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def _bar(frame, x: int, y: int, w: int, h: int,
         value: float, col, bg_col=_C_GRAY) -> None:
    """Draw a horizontal progress bar for a 0–1 float."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_col, 1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), col, -1)


def draw_overlay(
    frame: np.ndarray,
    *,
    # primary person (drives bottom panel)
    person_id:   Optional[str],
    sim:         float,
    box:         Optional[tuple],
    emotion:     str,
    e_conf:      float,
    tier:        str,
    pad_state:   Optional[tuple],
    descriptors: Optional[dict],
    rapport:     float,
    trust:       float,
    # emotion V/A values (weighted-blend from softmax)
    va:          tuple = (0.0, 0.0),
    # chat display
    last_user_msg: Optional[str],
    last_verbal:   Optional[str],
    # header
    robot_name: str,
    tick:       int,
    fps:        float,
    # input box
    input_mode:    int   = _MODE_IDLE,
    input_text:    str   = "",
    input_error:   str   = "",
    cursor_on:     bool  = True,
    # misc
    enroll_capturing: bool = False,
    enroll_progress:  int  = 0,
    enroll_total:     int  = 12,
    enroll_prompt:    str  = "",
    llm_on:           bool = False,
    # ALL detected faces this tick  ← new
    all_detections: list  = [],
) -> np.ndarray:
    frame = frame.copy()
    h, w  = frame.shape[:2]
    tc    = _TIER_COL.get(tier, _C_UNKNOWN)

    # ── Face bounding boxes — one per detected face ───────────────────────────
    draw_list = all_detections if all_detections else (
        [{"person_id": person_id, "sim": sim, "box": box,
          "emotion": emotion, "e_conf": e_conf, "tier": tier}]
        if box is not None else []
    )
    for det in draw_list:
        dbox = det.get("box")
        if dbox is None:
            continue
        x1, y1, x2, y2 = int(dbox[0]), int(dbox[1]), int(dbox[2]), int(dbox[3])
        dtier = det.get("tier", "unknown")
        dpid  = det.get("person_id")
        dcol  = _TIER_COL.get(dtier, _C_UNKNOWN) if dpid else _C_UNKNOWN
        cv2.rectangle(frame, (x1, y1), (x2, y2), dcol, 2)
        # Corner accents
        clen = 18
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*clen, cy), dcol, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy*clen), dcol, 3)
        # Name + similarity above box. A negative sim means "identity held from the
        # last check, not measured this frame" — show 'held' rather than a number.
        dsim = det.get("sim", 0.0)
        sim_str = "held" if dsim < 0 else f"{dsim:.2f}"
        name_label = f"{dpid}  {sim_str}" if dpid else f"?  {sim_str}"
        _text(frame, name_label, (x1 + 4, max(y1 - 20, 18)), 0.58, dcol)
        # Emotion below name (inside top of box area)
        dem, dec = det.get("emotion", ""), det.get("e_conf", 0.0)
        if dem:
            emo_str = f"{dem} {dec:.0f}%" if dec > 1 else dem
            _text(frame, emo_str, (x1 + 4, max(y1 - 5, 34)), 0.46, dcol, 1)
        # Culture: region/heritage + age band, drawn under the box (display only)
        region = det.get("region", "")
        if region:
            lock = " *" if det.get("demo_locked") else ""
            rconf = det.get("region_conf", 0.0)
            _text(frame, f"{region} {rconf:.0%}{lock}",
                  (x1 + 4, y2 + 18), 0.50, _C_CYAN, 1)
            age, stage = det.get("age", ""), det.get("age_stage", "")
            if age:
                age_str = f"{age} ({stage})" if stage else age
                _text(frame, age_str, (x1 + 4, y2 + 38), 0.46, _C_CYAN, 1)

    # ── Top bar ───────────────────────────────────────────────────────────────
    _panel(frame, 0, 0, w, 38)
    _text(frame, f"{robot_name}  |  tick {tick}  |  {fps:.1f} fps",
          (8, 26), 0.60, _C_YELLOW)

    # LLM status badge (right side of top bar)
    if llm_on:
        llm_label, llm_col = "LLM ON", _C_GREEN
    else:
        llm_label, llm_col = "no LLM", _C_UNKNOWN
    _text(frame, llm_label, (w - 100, 26), 0.50, llm_col, 1)

    # ── Bottom info panel ─────────────────────────────────────────────────────
    # Decide height based on how much chat text we have
    chat_extra = 0
    if last_user_msg:
        chat_extra += 22
    if last_verbal:
        chat_extra += 22 * max(1, (len(last_verbal) + 54) // 55)

    panel_top = h - (130 + chat_extra)
    _panel(frame, 0, panel_top, w, h)
    cv2.line(frame, (0, panel_top), (w, panel_top), tc, 1)

    y = panel_top + 22

    # Person + emotion row
    pname = person_id or "unknown"
    _text(frame, f"Person: {pname}", (8, y), 0.60, _C_WHITE)
    ec_str = f"Emotion: {emotion}  {e_conf:.0f}%" if e_conf > 1 else f"Emotion: {emotion}"
    _text(frame, ec_str, (w // 2, y), 0.60, _C_WHITE)
    y += 26

    # Tier label + rapport/trust bars
    _text(frame, f"Tier:  {tier.upper()}", (8, y), 0.68, tc, 2)
    bx, bw = w // 2, 90
    _text(frame, f"R {rapport:.2f}", (bx - 6, y), 0.50, _C_GRAY, 1)
    _bar(frame, bx + 40, y - 13, bw, 11, rapport, tc)
    _text(frame, f"T {trust:.2f}",  (bx - 6, y + 16), 0.50, _C_GRAY, 1)
    _bar(frame, bx + 40, y + 3,  bw, 11, trust,   tc)
    # Threshold marker at 0.70
    mx = bx + 40 + int(bw * 0.70)
    cv2.line(frame, (mx, y - 15), (mx, y + 16), _C_YELLOW, 1)
    y += 32

    # PAD values
    if pad_state is not None:
        p, a, d = pad_state
        _text(frame, f"PAD   P={p:+.2f}   A={a:+.2f}   D={d:+.2f}", (8, y), 0.60, _C_WHITE)
    y += 22

    # Emotion V/A (weighted softmax blend from camera)
    ev, ea = va
    _text(frame, f"V/A   V={ev:+.2f}   A={ea:+.2f}", (8, y), 0.56, _C_CYAN)
    y += 22

    # Mood descriptors
    if descriptors:
        mood = (f"Mood:  {descriptors.get('pleasure','?')} / "
                f"{descriptors.get('arousal','?')} / "
                f"{descriptors.get('dominance','?')}")
        _text(frame, mood, (8, y), 0.54, _C_GRAY)
    y += 22

    # Chat exchange
    if last_user_msg:
        short = last_user_msg[:52] + ("…" if len(last_user_msg) > 52 else "")
        _text(frame, f"[you]   \"{short}\"", (8, y), 0.52, _C_PINK, 1)
        y += 22
    if last_verbal:
        words, buf, lines = last_verbal.split(), "", []
        for ww in words:
            if len(buf) + len(ww) + 1 > 55:
                lines.append(buf); buf = ww
            else:
                buf = (buf + " " + ww).strip()
        if buf:
            lines.append(buf)
        for ln in lines[:3]:
            _text(frame, f"[{robot_name}]  \"{ln}\"", (8, y), 0.52, _C_GREEN, 1)
            y += 20

    # ── Input box (replaces controls hint when active) ────────────────────────
    BOX_H = 46
    box_y = h - BOX_H
    _panel(frame, 0, box_y, w, h, col=_C_DARK, alpha=0.88)
    cv2.line(frame, (0, box_y), (w, box_y),
             _C_CYAN if input_mode != _MODE_IDLE else _C_GRAY, 2)

    if input_mode == _MODE_IDLE:
        if enroll_capturing:
            pct = int(enroll_progress / enroll_total * 100)
            bar_w = int((w - 200) * enroll_progress / enroll_total)
            cv2.rectangle(frame, (8, box_y + 14), (8 + bar_w, box_y + 30), _C_GREEN, -1)
            _text(frame, f"Capturing …  {enroll_progress}/{enroll_total}  ({pct}%)",
                  (8, box_y + 34), 0.52, _C_GREEN, 1)
        elif llm_on:
            _text(frame, "T=chat   E=enroll   B=boost   S=save   Q=quit",
                  (8, box_y + 30), 0.50, _C_GRAY, 1)
        else:
            _text(frame, "T=chat   E=enroll   B=boost   S=save   Q=quit",
                  (8, box_y + 18), 0.50, _C_GRAY, 1)
            _text(frame, "restart with --llm flag to enable AI chat",
                  (8, box_y + 36), 0.45, _C_UNKNOWN, 1)

    else:
        label = "Chat:" if input_mode == _MODE_CHAT else "Enroll name:"
        cursor = "|" if cursor_on else " "
        display_text = input_text + cursor

        # Prompt label
        _text(frame, label, (8, box_y + 30), 0.56, _C_CYAN, 1)

        # Input text area with border
        tx = 130
        cv2.rectangle(frame, (tx - 4, box_y + 10), (w - 130, box_y + 40),
                      _C_GRAY, 1)
        # Clip displayed text to fit the box (show last N chars)
        max_chars = 52
        disp = display_text[-max_chars:] if len(display_text) > max_chars else display_text
        _text(frame, disp, (tx, box_y + 30), 0.56, _C_WHITE, 1)

        # Hints on the right
        hints = "Enter=send  Esc=cancel" if input_mode == _MODE_CHAT else "Enter=confirm  Esc=cancel"
        _text(frame, hints, (w - 260, box_y + 30), 0.44, _C_GRAY, 1)

        # Error message
        if input_error:
            _text(frame, input_error, (tx, box_y + 10), 0.45, _C_UNKNOWN, 1)

    # Enrollment progress overlay (centre)
    if enroll_capturing:
        pct = int(enroll_progress / enroll_total * 100)
        banner_w, banner_h = 340, 60
        bx0 = (w - banner_w) // 2
        by0 = h // 2 - banner_h // 2
        # Pose instruction sits ABOVE the progress banner — it is the thing the
        # person has to act on, so it gets the prominent slot.
        if enroll_prompt:
            pw = 560
            px0 = max(4, (w - pw) // 2)
            _panel(frame, px0, by0 - 54, min(w - 4, px0 + pw), by0 - 10,
                   col=_C_DARK, alpha=0.9)
            _text(frame, enroll_prompt, (px0 + 14, by0 - 24), 0.62, _C_CYAN, 2)
        _panel(frame, bx0, by0, bx0 + banner_w, by0 + banner_h,
               col=(0, 80, 0), alpha=0.85)
        cv2.rectangle(frame, (bx0, by0), (bx0 + banner_w, by0 + banner_h),
                      _C_GREEN, 2)
        _text(frame, f"Capturing …  {enroll_progress}/{enroll_total}",
              (bx0 + 20, by0 + 22), 0.7, _C_GREEN, 2)
        bar_inner = int((banner_w - 40) * enroll_progress / enroll_total)
        cv2.rectangle(frame, (bx0 + 20, by0 + 34), (bx0 + banner_w - 20, by0 + 50),
                      _C_GRAY, 1)
        if bar_inner > 0:
            cv2.rectangle(frame, (bx0 + 20, by0 + 34),
                          (bx0 + 20 + bar_inner, by0 + 50), _C_GREEN, -1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# KG rapport/trust helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_rapport_trust(
    store: InMemoryGraphStore, person_id: str, robot_id: str
) -> tuple[float, float]:
    from modules.graph_relationship.interactions import get_interaction
    interaction = get_interaction(store, person_id, robot_id)
    if interaction is None:
        return 0.0, 0.0
    return interaction.rapport, interaction.trust


def _dump_kg(store: InMemoryGraphStore, robot_id: str) -> None:
    """Print all KG nodes and edges to stdout for debugging."""
    nodes = list(store._nodes.values())
    edges = list(store._edges.values())
    print("\n" + "=" * 60)
    print(f"  KG DUMP  ({len(nodes)} nodes, {len(edges)} edges)")
    print("=" * 60)
    for node in nodes:
        print(f"  [NODE] {node.node_type:8s}  id={node.id!r}")
    print()
    for edge in edges:
        val = getattr(edge, "value",  None)
        wt  = getattr(edge, "weight", None)
        cnt = getattr(edge, "count",  None)
        extra = ""
        if val  is not None: extra = f"  value={val:+.3f}"
        if wt   is not None: extra = f"  weight={wt:.3f}"
        if cnt  is not None: extra = f"  count={cnt}"
        print(f"  [EDGE] {edge.edge_type:18s}  {edge.source_id!r:12s} → {edge.target_id!r:12s}{extra}")
    print("=" * 60 + "\n")


def _update_rapport_trust(
    store: InMemoryGraphStore,
    person_id: str,
    robot_id:  str,
    delta: float,
    verbose: bool = False,
) -> None:
    from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment
    from modules.graph_relationship.interactions import set_closeness

    if store.get_node(person_id) is None:
        store.upsert_node(PersonNode(id=person_id))
    if store.get_node(robot_id) is None:
        store.upsert_node(RobotNode(id=robot_id, name=robot_id,
                                    embodiment=Embodiment.CAT))
    r_cur, t_cur = _read_rapport_trust(store, person_id, robot_id)
    r_new = min(1.0, r_cur + delta)
    t_new = min(1.0, t_cur + delta)
    # Closeness lives on the pair's InteractionNode.
    set_closeness(store, person_id, robot_id, rapport=r_new, trust=t_new,
                  source="webcam_loop")
    if verbose:
        print(f"[Boost] {person_id} → rapport={r_new:.2f}  trust={t_new:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Background detection worker
# ─────────────────────────────────────────────────────────────────────────────

class _DetectionWorker(threading.Thread):
    """
    Runs MTCNN + ResNet + emotion in a daemon thread.

    The main display loop submits a frame via submit() and reads
    get_results() without ever blocking — detection latency is completely
    hidden from the display frame-rate.

    Design:
    - Only the latest submitted frame is processed; stale frames are dropped.
    - Per-person emotion smoothers live here so the smoothing window is never
      shared between different identities.
    - Capped at max_faces (default 4) and downscaled by det_scale (default 0.5)
      for fast MTCNN inference.
    """

    def __init__(
        self,
        face_id,
        emotion_backend: str   = 'hsemotion',
        max_faces:       int   = 4,
        det_scale:       float = 0.5,
        detect_emotion:  bool  = True,
        detect_demographics: bool = False,
        demographics_backend: str = 'fairface',
        id_check_interval: float = 3.0,
        id_sample_window:  float = 1.0,
        id_confirm_ratio:  float = 0.6,
        id_miss_grace:     int   = 2,
        id_debug:          bool  = False,
        adapt_enabled:     bool  = True,
        adapt_floor:       Optional[float] = None,
        adapt_interval:    float = 20.0,
        adapt_max:         int   = 20,
        adapt_min_px:      int   = 90,
        adapt_min_sharp:   float = 40.0,
    ):
        super().__init__(daemon=True, name="detection-worker")
        self._face_id         = face_id
        self._emotion_backend = emotion_backend
        self._max_faces       = max_faces
        self._det_scale       = det_scale
        self._detect_emotion  = detect_emotion

        # ── Sampled identity (sample-and-hold) ───────────────────────────────
        # Face recognition flickers if we re-identify every frame. Instead we run
        # a short SAMPLE window (id_sample_window s) where we identify each frame
        # and vote, confirm the identity if it was recognised in >= id_confirm_ratio
        # of those frames, then HOLD that identity for id_check_interval s — during
        # which we only detect boxes (for emotion/display), never re-identify. This
        # stabilises the label and skips the expensive embedding most of the time.
        self._id_check_interval = id_check_interval
        self._id_sample_window  = id_sample_window
        self._id_confirm_ratio  = id_confirm_ratio
        self._id_phase   = 'sample'   # 'sample' | 'hold'
        self._id_phase_t = 0.0        # when the current phase started (0 = uninit)
        self._id_votes: dict[str, int] = {}   # pid -> frames recognised this window
        self._id_samples = 0          # frames WITH A FACE processed this sample window
        self._confirmed_pid: Optional[str] = None   # currently held primary identity
        # A face was seen at all this window (even an unrecognised one). Distinguishes
        # "hard pose, couldn't name them" from "nobody there" — the first deserves a
        # grace window, the second must drop the label immediately.
        self._saw_face_window = False
        self._id_miss_grace = id_miss_grace
        self._miss_streak   = 0
        self._id_debug      = id_debug

        # ── Adaptive capture: learn a new VIEW of a known person automatically ──
        # Decided once per sample window (never per frame) because the corroboration
        # signal — the vote ratio — is a window property. The worker only EMITS
        # events; the main thread applies them, so the gallery keeps a single writer.
        self._adapt_enabled  = adapt_enabled
        self._adapt_floor    = (adapt_floor if adapt_floor is not None
                                else getattr(face_id, "threshold", 0.75))
        self._adapt_interval = adapt_interval
        self._adapt_max      = adapt_max
        self._adapt_min_px   = adapt_min_px
        self._adapt_min_sharp = adapt_min_sharp
        self._adapt_buf: list = []      # [(sim, emb)] candidates this window
        self._adapt_dirty     = False   # window saw 0 or >1 faces → not usable
        self._adapt_last: dict = {}     # pid -> last adoption timestamp
        self._adapt_count     = 0       # adoptions this session
        self._adapt_events: list = []   # queued for the main thread

        # Per-person smoothers (only ever touched from this thread — no lock needed).
        # Skipped entirely when emotion detection is disabled (no models loaded).
        self._per_emotion: dict[str, EmotionDetector] = {}
        self._unknown_emotion = (
            EmotionDetector.create(emotion_backend) if detect_emotion else None
        )

        # Demographics (region/heritage + age) — display-only ("--culture"). Runs
        # the model per face and votes to a stable estimate; no KG/prompt use.
        self._detect_demographics = detect_demographics
        self._demographics_backend = demographics_backend
        self._per_demo: dict[str, DemographicsDetector] = {}
        self._last_demo: dict = {}   # key -> last DemographicsResult (throttle cache)
        self._unknown_demo = (
            DemographicsDetector.create(demographics_backend)
            if detect_demographics else None
        )
        # FairFace is ~15 ms/face; run every Nth worker cycle to keep it light —
        # the vote window keeps the last estimate visible between inferences.
        self._demo_every = 3
        self._demo_cycle = 0

        self._lock    = threading.Lock()
        self._frame   = None
        self._results: list = []
        self._event   = threading.Event()
        # NOTE: must NOT be named `_stop` — that shadows threading.Thread._stop,
        # which Thread.join() calls internally (→ 'Event' object is not callable).
        self._stop_evt = threading.Event()

    # ── Public API (called from main thread) ──────────────────────────────────

    def submit(self, frame: np.ndarray) -> None:
        """Drop latest frame in; any unprocessed previous frame is discarded."""
        with self._lock:
            self._frame = frame
        self._event.set()

    def get_results(self) -> list:
        """Non-blocking — returns last completed detection list."""
        with self._lock:
            return list(self._results)

    def stop(self) -> None:
        self._stop_evt.set()
        self._event.set()  # unblock the wait

    # ── Worker loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop_evt.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()
            if self._stop_evt.is_set():
                break

            with self._lock:
                frame = self._frame
            if frame is None:
                continue

            now = time.time()
            if self._id_phase_t == 0.0:
                self._id_phase_t = now   # first frame → start the sample window

            phase_elapsed = now - self._id_phase_t

            if self._id_phase == 'sample':
                # Identify every frame this window and vote on the primary face.
                # `sticky` gives face_id the identity we currently hold so it can
                # apply hysteresis (keep it through a brief off-angle dip).
                # `with_emb` keeps the embedding that was computed anyway, so
                # adaptive capture costs no extra inference.
                full = self._face_id.identify_all(
                    frame, max_faces=self._max_faces, scale=self._det_scale,
                    sticky=self._confirmed_pid, with_emb=self._adapt_enabled,
                )
                if self._adapt_enabled:
                    self._collect_adapt(frame, full)
                    raw = [(p, s, b) for (p, s, b, _e) in full]
                else:
                    raw = full
                primary = self._primary_pid(raw)
                # Only frames that actually CONTAIN a face count toward the vote.
                # Counting empty frames inflates the denominator, so looking away
                # (no box at all) used to push the ratio under id_confirm_ratio and
                # drop a perfectly good identity — a direct cause of label flicker.
                if raw:
                    self._id_samples += 1
                    self._saw_face_window = True
                    if primary is not None:
                        self._id_votes[primary] = self._id_votes.get(primary, 0) + 1

                # Show the already-held identity (no flicker); before the first
                # confirmation, show the raw guesses so something appears.
                raw_for_results = (
                    self._relabel_primary(raw, self._confirmed_pid)
                    if self._confirmed_pid is not None else raw
                )

                if self._id_debug:
                    self._debug_frame(frame, raw, 'sample')

                if phase_elapsed >= self._id_sample_window:
                    self._confirm_identity()      # decide held identity from votes
                    self._id_phase   = 'hold'
                    self._id_phase_t = now
            else:  # hold — detect boxes only, reuse the confirmed identity
                boxes = self._face_id.detect_boxes(
                    frame, max_faces=self._max_faces, scale=self._det_scale,
                )
                raw_for_results = self._boxes_to_raw(boxes, self._confirmed_pid)
                if phase_elapsed >= self._id_check_interval:
                    self._id_phase   = 'sample'   # time to re-check identity
                    self._id_phase_t = now
                    self._start_sample_window()

            # Throttle the (heavier) demographics model — infer every Nth cycle,
            # reuse the last voted estimate on the cycles we skip.
            run_demo = False
            if self._detect_demographics:
                self._demo_cycle += 1
                run_demo = (self._demo_cycle % self._demo_every) == 0

            results = self._build_results(frame, raw_for_results, run_demo)

            with self._lock:
                self._results = results

    # ── Sampled-identity helpers ──────────────────────────────────────────────

    @staticmethod
    def _box_area(box) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _primary_idx(cls, raw: list) -> int:
        """Index of the largest-box entry in [(pid, sim, box), ...], or -1."""
        best_i, best_a = -1, -1
        for i, (_, _, box) in enumerate(raw):
            a = cls._box_area(box)
            if a > best_a:
                best_a, best_i = a, i
        return best_i

    @classmethod
    def _primary_pid(cls, raw: list):
        i = cls._primary_idx(raw)
        return raw[i][0] if i >= 0 else None

    @classmethod
    def _relabel_primary(cls, raw: list, pid) -> list:
        """Force the largest-box face's person_id to `pid` (stable display label)."""
        i = cls._primary_idx(raw)
        if i < 0:
            return raw
        out = list(raw)
        _, sim, box = out[i]
        out[i] = (pid, sim, box)
        return out

    # sim sentinel for a HOLD-phase box: identity is REUSED, not measured this frame.
    # (Reporting 1.0 here would be a fabricated perfect match and would mislead any
    # threshold calibration done from the overlay/debug trace.)
    HELD_SIM = -1.0

    @classmethod
    def _boxes_to_raw(cls, boxes: list, pid) -> list:
        """Bare boxes → (pid, sim, box): the largest gets the held identity, the
        rest are unknown. `boxes` arrives largest-first from detect_boxes."""
        if not boxes:
            return []
        return ([(pid, cls.HELD_SIM, boxes[0])]
                + [(None, 0.0, b) for b in boxes[1:]])

    def _start_sample_window(self) -> None:
        """Reset per-window accumulators at the start of a SAMPLE phase."""
        self._id_votes        = {}
        self._id_samples      = 0
        self._saw_face_window = False
        self._adapt_buf       = []
        self._adapt_dirty     = False

    # ── Adaptive capture ──────────────────────────────────────────────────────

    def _quality_ok(self, frame, box) -> bool:
        """Reject faces too small, clipped by the frame edge, or motion-blurred —
        a bad crop would be memorised as a 'new view' and pollute the gallery."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        if x1 < 4 or y1 < 4 or x2 > w - 4 or y2 > h - 4:
            return False
        if min(x2 - x1, y2 - y1) < self._adapt_min_px:
            return False
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= self._adapt_min_sharp

    def _collect_adapt(self, frame, full: list) -> None:
        """Buffer this frame's embedding as an adoption candidate (or poison the
        window). Called per sampled frame; `full` holds 4-tuples with embeddings."""
        if len(full) != 1:
            # 0 faces = nothing to learn; >1 = we cannot be sure which is whom.
            self._adapt_dirty = True
            return
        pid, sim, box, emb = full[0]
        if emb is None:
            return
        # Never learn from a DIFFERENT enrolled person (that is the poisoning case).
        # Unknown is allowed: an unrecognised pose of the held person is the target.
        if pid is not None and pid != self._confirmed_pid:
            self._adapt_dirty = True
            return
        if self._quality_ok(frame, box):
            self._adapt_buf.append((float(sim), emb))

    def _maybe_adapt(self, now: float) -> None:
        """Decide, at end of window, whether to learn a new view. Every condition
        must hold — this is the anti-poisoning gate."""
        if self._adapt_dirty or not self._adapt_buf:
            return
        pid = self._confirmed_pid
        if pid is None:                                     # (1) nobody confirmed
            return
        if len(self._id_votes) > 1:                         # (2) ambiguous window
            return
        if self._id_samples <= 0:
            return
        if (self._id_votes.get(pid, 0) / self._id_samples
                < self._id_confirm_ratio):                  # (3) weak confirmation
            return
        if now - self._adapt_last.get(pid, 0.0) < self._adapt_interval:
            return                                          # (4) rate limit
        if self._adapt_count >= self._adapt_max:            # (5) session cap
            return

        sims = [s for s, _ in self._adapt_buf]
        med  = float(np.median(sims))
        tau  = getattr(self._face_id, "tau_dup", 0.92)
        if med >= tau:                                      # (6) view already covered
            return
        if med < self._adapt_floor:                         # (7) impostor territory
            return
        if (med < getattr(self._face_id, "threshold", 0.75)
                and len(self._adapt_buf) < self._id_samples * 0.8):
            return                                          # (8) marginal → unanimity

        # Take the frame nearest the band centre: most typical of the new pose,
        # rather than the most extreme (and least trustworthy) outlier.
        target = min(med + 0.03, tau - 0.01)
        _s, emb = min(self._adapt_buf, key=lambda t: abs(t[0] - target))
        with self._lock:
            self._adapt_events.append({"pid": pid, "emb": emb, "sim": med, "t": now})
        self._adapt_last[pid] = now
        self._adapt_count += 1

    def pop_adapt_events(self) -> list:
        """Main thread drains queued adoptions (worker never mutates the gallery)."""
        with self._lock:
            ev, self._adapt_events = self._adapt_events, []
            return ev

    def _debug_frame(self, frame, raw: list, phase: str) -> None:
        """--id-debug: one line per sampled frame, for threshold calibration."""
        if not raw:
            print(f"[id] {phase}: no face   held={self._confirmed_pid}")
            return
        i = self._primary_idx(raw)
        pid, sim, box = raw[i][0], raw[i][1], raw[i][2]
        area = self._box_area(box)
        print(f"[id] {phase}: box={box} area={area} "
              f"best={pid or '?'}:{sim:.3f} faces={len(raw)} "
              f"held={self._confirmed_pid} votes={self._id_votes}/{self._id_samples}")

    def _confirm_identity(self) -> None:
        """Decide the held identity from this sample window's votes: the most-voted
        person, but only if recognised in >= id_confirm_ratio of the frames that
        actually contained a face.

        Miss-grace: if this window failed to confirm anyone but a face WAS present and
        no rival was seen, keep the previous identity for up to id_miss_grace windows —
        that pattern means "hard pose", not "different person". A rival vote or an
        empty window (nobody there) drops the identity immediately.
        """
        cand = None
        if self._id_samples > 0 and self._id_votes:
            best_pid, best_cnt = max(self._id_votes.items(), key=lambda kv: kv[1])
            ratio = best_cnt / self._id_samples
            if ratio >= self._id_confirm_ratio:
                cand = best_pid

        if cand is not None:
            self._miss_streak = 0
        elif self._confirmed_pid is not None:
            rival = any(p != self._confirmed_pid for p in self._id_votes)
            if rival or not self._saw_face_window:
                self._miss_streak = 0          # someone else, or truly gone → drop now
            else:
                self._miss_streak += 1
                if self._miss_streak < self._id_miss_grace:
                    cand = self._confirmed_pid  # hard pose → keep holding
                else:
                    self._miss_streak = 0

        self._confirmed_pid = cand
        # Adoption is decided BEFORE the window state is cleared — it needs the votes.
        if self._adapt_enabled:
            self._maybe_adapt(time.time())
        self._start_sample_window()

    def _build_results(self, frame, raw: list, run_demo: bool) -> list:
        """Turn a [(person_id, sim, box), ...] list into the per-face result dicts
        (emotion + demographics), shared by both the sample and hold phases."""
        results = []
        for person_id, sim, box in raw:
            if not self._detect_emotion:
                # Emotion disabled for this pass — face identification only.
                emo, e_conf, ev, ea = "neutral", 0.0, 0.0, 0.0
            elif person_id is not None:
                if person_id not in self._per_emotion:
                    self._per_emotion[person_id] = EmotionDetector.create(
                        self._emotion_backend
                    )
                emo, e_conf, ev, ea = self._per_emotion[person_id].detect(
                    frame, box=box, smooth=True
                )
            else:
                emo, e_conf, ev, ea = self._unknown_emotion.detect(
                    frame, box=box, smooth=False
                )

            # ── Demographics (region/heritage + age) — display only ────────
            demo = None
            if self._detect_demographics:
                key = person_id or DemographicsDetector.__name__  # shared unknown
                if person_id is not None and key not in self._per_demo:
                    self._per_demo[key] = DemographicsDetector.create(
                        self._demographics_backend)
                det = self._per_demo.get(key, self._unknown_demo)
                if run_demo:
                    demo = det.detect(frame, box=box, smooth=True)
                    self._last_demo[key] = demo
                else:
                    demo = self._last_demo.get(key)

            results.append({
                "person_id": person_id,
                "sim":       sim,
                "box":       box,
                "emotion":   emo,
                "e_conf":    e_conf,
                "va":        (ev, ea),
                "region":      demo.region      if demo else "",
                "region_conf": demo.region_conf if demo else 0.0,
                "age":         demo.age         if demo else "",
                "age_stage":   demo.age_stage   if demo else "",
                "demo_locked": demo.locked      if demo else False,
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

class WebcamKGLoop:
    """
    Webcam → face_id + emotion → KGBridge → PAD → LLM, with in-window chat UI.

    Press T in the OpenCV window to open the chat box.
    Press E to enroll a new face by typing a name in the window.
    """

    def __init__(
        self,
        robot_id:        str   = _DEFAULT_ROBOT,
        faces_path:      str   = _DEFAULT_FACES,
        kg_path:         str   = _DEFAULT_KG,
        threshold:       float = _DEFAULT_THRESH,
        tick_interval:   float = _DEFAULT_TICK,
        llm_client:      Optional[LLMClient] = None,
        show_window:     bool  = True,
        emotion_backend: str   = 'hsemotion',
        esp32_host:      str   = "",
        esp32_port:      int   = 8888,
        spec_dir:        Optional[str] = None,
        seed:            bool  = True,
        matcher                = None,
        embed_fn               = None,
        sessions_db:     str   = _DEFAULT_SESSIONS_DB,
        pad_enabled:     bool  = False,
        emotion_enabled: bool  = False,
        demographics_enabled: bool = False,
        demographics_backend: str  = 'fairface',
        detector:        str   = 'opencv',
        id_check_interval: float = 3.0,
        id_sample_window:  float = 1.0,
        id_confirm_ratio:  float = 0.6,
        id_miss_grace:     int   = 2,
        id_debug:          bool  = False,
        multi_pose:        bool  = True,
        retain_threshold:  float = 0.62,
        switch_margin:     float = 0.08,
        id_margin:         float = 0.05,
        proto_dup:         float = 0.92,
        proto_max:         int   = 12,
        proto_adapt_max:   int   = 6,
        adapt_enabled:     bool  = True,
        adapt_floor:       Optional[float] = None,
        adapt_interval:    float = 20.0,
        adapt_max:         int   = 20,
        adapt_min_px:      int   = 90,
        debug_prompt:    bool  = False,
    ):
        self.robot_id      = robot_id
        self.faces_path    = faces_path
        self.kg_path       = kg_path
        self.tick_interval = tick_interval
        self.llm           = llm_client
        self.show_window   = show_window

        self.face_id = FaceIdentifier(
            threshold=threshold, detector=detector,
            multi_pose=multi_pose, retain_threshold=retain_threshold,
            switch_margin=switch_margin, id_margin=id_margin,
            tau_dup=proto_dup, k_anchor=proto_max, k_adapt=proto_adapt_max,
        )
        if os.path.exists(faces_path):
            self.face_id.load(faces_path)
        else:
            print(f"[WebcamLoop] No face DB at '{faces_path}' — starting empty")
        # Debounced persistence for views learned at runtime (see _drain_adapt_events).
        self._faces_dirty     = False
        self._last_faces_save = time.time()

        self.store   = InMemoryGraphStore()
        if os.path.exists(kg_path):
            self.store.load(kg_path)
        else:
            print(f"[WebcamLoop] No KG at '{kg_path}' — starting fresh")

        # Seed authored robot/human subgraphs from spec files so a recognized
        # person has KG info to retrieve immediately (idempotent — deterministic ids).
        if seed:
            spec_dir = spec_dir or os.path.join(
                _SERVER_ROOT, "modules", "graph_relationship", "specs"
            )
            try:
                from modules.graph_relationship.seed import seed_all
                seed_all(self.store, spec_dir)
                if self.kg_path:
                    self.store.save(self.kg_path)
            except Exception as exc:  # noqa: BLE001 — seeding is best-effort
                print(f"[WebcamLoop] seed skipped ({spec_dir}): {exc}")

        self.bridge  = KGBridge(self.store)
        self._adapters: dict[str, PADPipelineAdapter] = {}

        self._emotion_backend  = emotion_backend
        self._esp32_host       = esp32_host
        self._esp32_port       = esp32_port

        # PAD persona engine + emotion detection are disabled for this pass
        # (face-reco → KG-through-conversation only). Re-enable via CLI later.
        self._pad_enabled      = pad_enabled
        self._emotion_enabled  = emotion_enabled
        # Demographics (region/heritage + age) — display only, no KG/prompt use.
        self._demographics_enabled = demographics_enabled
        self._demographics_backend = demographics_backend
        # Sampled identity (sample-and-hold) — check who it is every id_check_interval
        # seconds via a id_sample_window-second vote instead of re-identifying 24/7.
        self._id_check_interval = id_check_interval
        self._id_sample_window  = id_sample_window
        self._id_confirm_ratio  = id_confirm_ratio
        self._id_miss_grace     = id_miss_grace
        self._id_debug          = id_debug
        # Adaptive capture (learn new views of a known person during conversation).
        self._adapt_enabled  = adapt_enabled
        self._adapt_floor    = adapt_floor
        self._adapt_interval = adapt_interval
        self._adapt_max      = adapt_max
        self._adapt_min_px   = adapt_min_px
        # Print the 3 prompt-feeding modules (KG / RAG / BN) at each chat turn.
        self._debug_prompt = debug_prompt
        self._matcher          = matcher
        self._embed_fn         = embed_fn   # for on-demand topic consolidation (Feature 2)
        # Conversation transcripts live in SQLite (not the graph). The graph keeps
        # only Interaction (rapport/trust/count) + topics/interests.
        self._session_store    = SessionStore(sessions_db)
        # RAG over the transcript store (needs embeddings); None when --no-embed.
        self._session_rag = None
        if embed_fn is not None:
            from modules.session_rag import SessionRAG
            self._session_rag = SessionRAG(self._session_store, embed_fn)
        # person_id -> this run's session id (uuid). Populated on the first turn.
        self._run_sessions: dict[str, str] = {}
        # person_id -> (valence, emotion_label) last persisted — dirty-check so the
        # per-tick mood write only hits disk when it actually changes.
        self._last_mood: dict[str, tuple] = {}

        # Per-person rolling chat history (last 5 turns)
        self._chat_history: dict[str, deque] = {}

        self._robot_display = {"chatbox": "ChatBox", "ellebot": "ElleBot"}.get(
            robot_id.lower(), robot_id
        )
        self._last_pad_result: Optional[dict] = None

        # Option B: the live topic label (a 2nd, non-reply LLM call) is computed OFF
        # the main/display thread. Worker threads compute it and hand the result back
        # here; the main loop drains this and applies the store write itself (the
        # InMemoryGraphStore is single-threaded — only the main loop mutates it).
        # Items: (person_id, robot_id, topic_or_None).
        self._pending_topics: "queue.Queue" = queue.Queue()
        # Mid-session culture auto-attach: async self-declaration detections hand
        # (person_id, culture_label, session_id) back to the main thread to tag.
        self._pending_culture: "queue.Queue" = queue.Queue()
        # Last (person, culture) written to active_state.json for the viz highlight,
        # plus a STICKY current person so brief face-reco dropouts don't make the
        # highlight (and the dimmed culture nodes) flicker on/off.
        self._last_active_written = None
        self._last_active_write_t = 0.0
        self._active_person = None
        self._active_person_t = 0.0
        # First-impression: sequence for provisional 'guest_N' ids (unknown faces).
        self._guest_seq = 0

        # #2: the whole chat pipeline (RAG + prompt build + LLM reply) runs on a
        # dedicated worker thread so the display never freezes while the robot
        # "thinks". The InMemoryGraphStore is NOT thread-safe, so a single RLock
        # serialises every store access that can overlap the worker; the worker holds
        # it ONLY for the fast prompt-build read, never for the slow LLM/RAG calls.
        # Requests/results cross threads via queues; all store WRITES stay on the
        # main thread (_apply_chat_result).
        self._store_lock = threading.RLock()
        self._chat_requests: "queue.Queue" = queue.Queue()
        self._chat_results:  "queue.Queue" = queue.Queue()

        # #3: debounced graph persistence. The frequent per-tick/chat mutations mark
        # the graph dirty; the loop flushes to kg_state.json at most once per
        # _save_min_interval, instead of rewriting the whole file every tick.
        self._kg_dirty = False
        self._last_save_t = 0.0
        self._save_min_interval = 1.0   # seconds between debounced saves
        self._last_save_mtime = 0.0     # mtime of our own last write (external-edit detect)

    def _spawn_topic_detect(self, msg: str, verbal: str, person_id: str,
                            robot_id: str, turn_row_id: Optional[int]) -> None:
        """Run the live-topic LLM label OFF the critical path. Records the topic on
        the (thread-safe) session store, then hands it to the main loop to update the
        conversation node. Never touches self.store from this thread."""
        def _work():
            try:
                topic = self._detect_topic(msg, verbal)
            except Exception:  # noqa: BLE001 — a label failure must never crash chat
                topic = None
            if topic and turn_row_id is not None:
                try:
                    self._session_store.set_turn_topics(turn_row_id, [topic])
                except Exception:  # noqa: BLE001
                    pass
            self._pending_topics.put((person_id, robot_id, topic))
        threading.Thread(target=_work, name="topic-detect", daemon=True).start()

    def _drain_pending_topics(self) -> None:
        """Apply any completed async topic labels to the live conversation node.
        Runs on the MAIN thread each loop iteration; the store write is guarded by
        the store lock so it can't corrupt a concurrent chat-worker prompt read."""
        from modules.graph_relationship.topics import update_conversation
        drained = False
        while True:
            try:
                pid, robot_id, topic = self._pending_topics.get_nowait()
            except queue.Empty:
                break
            if not topic:
                continue
            with self._store_lock:
                conv = update_conversation(self.store, pid, robot_id, topic=topic,
                                           create=False, source="live-topic")
            if conv is not None:
                print(f"  [topic]  recent → {', '.join(conv.topics)}")
                drained = True
        if drained:
            self._mark_kg_dirty()

    # ── mid-session culture auto-attach (self-declaration only) ────────────────

    # Light pre-filter: only bother the LLM when the message plausibly states an
    # origin/background — keeps this to (rare) declaring turns, not every turn.
    _CULTURE_CUES = (
        "i am ", "i'm ", "im ", "from ", "my parents", "my family", "my culture",
        "my background", "my heritage", "grew up", "born in", "descent", "korean",
        "maori", "māori",
    )

    def _spawn_culture_detect(self, msg: str, pid: str, sid: str) -> None:
        """If `pid` is not yet culture-tagged and the message looks like it may state
        an origin, run self-declaration detection OFF the main thread and hand any
        result back to the main loop to tag. Self-declaration ONLY — never inferred
        from face/name/appearance. Once tagged, we stop checking (it persists)."""
        if not (self.llm and self.llm.available):
            return
        from modules.graph_relationship.cultures import person_culture
        if person_culture(self.store, pid) is not None:
            return                                   # already attached — stop checking
        low = (msg or "").lower()
        if not any(cue in low for cue in self._CULTURE_CUES):
            return                                   # no origin cue → skip the LLM call

        def _work():
            try:
                from modules.culture_extraction import detect_self_declared_culture
                label = detect_self_declared_culture([{"child": msg}], self.llm.respond)
            except Exception:  # noqa: BLE001
                label = None
            if label:
                self._pending_culture.put((pid, label, sid))
        threading.Thread(target=_work, name="culture-detect", daemon=True).start()

    def _drain_pending_culture(self) -> None:
        """Apply completed self-declaration detections on the MAIN thread: tag the
        person (belongs_to_culture) so the culture attaches for the REST of the
        session. Idempotent — skips anyone already tagged (first attachment wins)."""
        from modules.graph_relationship.cultures import person_culture
        from modules.culture_seed import assign_person_culture
        changed = False
        while True:
            try:
                pid, label, sid = self._pending_culture.get_nowait()
            except queue.Empty:
                break
            with self._store_lock:
                if person_culture(self.store, pid) is not None:
                    continue                          # already tagged — first wins
                assign_person_culture(self.store, pid, label,
                                      source=f"self-declared:{sid}")
            print(f"      culture: {pid} self-identified as {label} → attached (this session)")
            changed = True
        if changed:
            self._mark_kg_dirty()

    # ── #3: debounced graph persistence ───────────────────────────────────────

    def _mark_kg_dirty(self) -> None:
        """Flag that the graph changed; the loop's _flush_kg debounces the write."""
        self._kg_dirty = True

    def _reload_if_externally_changed(self) -> None:
        """Adopt edits made to the KG file by ANOTHER writer (the viz — e.g. a node
        deletion) so they aren't clobbered by our next save and persist to next run.
        Only reloads when the on-disk mtime is newer than our own last write."""
        kg = getattr(self, "kg_path", None)
        if not kg or self._kg_dirty:        # don't drop our own unsaved changes
            return
        try:
            mt = os.path.getmtime(kg)
        except OSError:
            return
        if self._last_save_mtime and mt > self._last_save_mtime + 0.5:
            with self._store_lock:
                if self.store.reload(kg):
                    self._last_save_mtime = os.path.getmtime(kg)
                    self._last_active_written = None   # force a viz-state refresh
                    print("[WebcamLoop] adopted an external KG edit (viz)")

    def _flush_kg(self, *, force: bool = False) -> None:
        """Persist kg_state.json at most once per _save_min_interval, and only when
        something changed (or `force`). Called every loop iteration on the MAIN
        thread; the actual write holds the store lock so it can't race the chat
        worker. Replaces the old per-tick/per-turn full-graph rewrites."""
        if not self.kg_path or not (self._kg_dirty or force):
            return
        now = time.time()
        if not force and (now - self._last_save_t) < self._save_min_interval:
            return
        with self._store_lock:
            self.store.save(self.kg_path)
        self._kg_dirty = False
        self._last_save_t = now
        try:
            self._last_save_mtime = os.path.getmtime(self.kg_path)   # ignore our own write
        except OSError:
            pass

    # ── #2: threaded chat pipeline ────────────────────────────────────────────

    def _chat_worker_loop(self) -> None:
        """Persistent chat worker: RAG search + prompt build + LLM reply, OFF the
        display thread. Store WRITES are NOT done here — the result is handed back to
        the main thread (_apply_chat_result), which owns the store. The store lock is
        held only for the fast prompt-build read, never across the slow LLM call."""
        while True:
            req = self._chat_requests.get()
            if req is None:                       # shutdown sentinel
                break
            try:
                msg, pid, history = req["msg"], req["pid"], req["history"]
                rag_hits = []
                if self._session_rag and pid:
                    try:
                        rag_hits = self._session_rag.search(
                            msg, top_k=5, person_id=pid)
                    except Exception:  # noqa: BLE001
                        rag_hits = []
                with self._store_lock:
                    if self._pad_enabled and self._last_pad_result:
                        sys_prompt = self._last_pad_result["system_prompt"]
                    else:
                        sys_prompt = self._build_system_prompt(pid, rag_hits=rag_hits)
                    if self._debug_prompt and pid:
                        self._debug_prompt_dump(pid, msg, rag_hits)
                raw_reply = self.llm.respond(sys_prompt, msg, history=history)
                tag, verbal = _parse_llm_response(raw_reply)
                self._chat_results.put({**req, "verbal": verbal, "tag": tag})
            except Exception as exc:  # noqa: BLE001 — never let the worker die
                self._chat_results.put({**req, "verbal": None, "tag": "",
                                        "error": str(exc)})

    def _apply_chat_result(self, res: dict) -> Optional[str]:
        """MAIN thread: print the reply, update history + KG (under the store lock),
        and kick off async topic detection. Returns the spoken reply text (or None)."""
        if res.get("error"):
            print(f"  [chat] LLM error: {res['error']}\n")
            return None
        msg, pid = res["msg"], res["pid"]
        verbal, tag = res["verbal"], res["tag"]
        emotion = res.get("emotion")
        if tag:
            print(f"  [tag]   [{tag}]")
            if self._esp32_host:
                expr = _TAG_TO_ESP32.get(tag)
                if expr:
                    _send_esp32(expr, self._esp32_host, self._esp32_port)
                else:
                    print(f"  [ESP32] no mapping for [{tag}]")
        print(f"  [{self._robot_display}]  \"{verbal}\"\n")
        pid_key = pid or "__unknown__"
        if pid_key not in self._chat_history:
            self._chat_history[pid_key] = deque(maxlen=5)
        self._chat_history[pid_key].append((msg, verbal))
        if pid:
            from modules.graph_relationship.interactions import set_interaction_count
            from modules.graph_relationship.topics import update_conversation
            with self._store_lock:
                sid = self._run_session_id(pid)
                row_id = self._session_store.append_turn(
                    session_id=sid, person_id=pid, robot_id=self.robot_id,
                    emotion=(emotion if self._emotion_enabled else None),
                    child=msg, reply=verbal, topics=None)
                set_interaction_count(
                    self.store, pid, self.robot_id,
                    self._session_store.person_turn_count(pid, self.robot_id),
                    source="session-store")
                update_conversation(
                    self.store, pid, self.robot_id,
                    mood=self._last_mood.get(pid, (None,))[0],
                    emotion=(emotion if self._emotion_enabled else None),
                    create=True, source="live-topic")
                self._mark_kg_dirty()
            # topic label is a further async step (Option B) — off the main thread.
            self._spawn_topic_detect(msg, verbal, pid, self.robot_id, row_id)
            # mid-session culture auto-attach (only while untagged; self-declaration).
            self._spawn_culture_detect(msg, pid, sid)
        return verbal

    def _adapter(self) -> PADPipelineAdapter:
        if self.robot_id not in self._adapters:
            self._adapters[self.robot_id] = PADPipelineAdapter(self.robot_id)
        return self._adapters[self.robot_id]

    def _pipeline_tick(self, person_id: str, emotion: str,
                       va: Optional[tuple] = None):
        bi = self.bridge.pre_turn(person_id, self.robot_id, emotion, camera_va=va)
        pad = self._adapter().process_turn(
            valence=bi.valence, arousal=bi.arousal,
            relationship_tier=bi.tier, memory_context=bi.structured_memory,
            rapport=bi.rapport, trust=bi.trust,
            interaction_count=bi.interaction_count,
        )
        self.bridge.post_turn(person_id, self.robot_id, pad, emotion=emotion)
        p = pad["pad_state"][0]
        if p > 0.05:
            _update_rapport_trust(self.store, person_id, self.robot_id,
                                  delta=0.025 * p)
        self._last_pad_result = pad
        return bi, pad

    # ── KG-only path (PAD/emotion disabled) ────────────────────────────────────

    def _ensure_interaction(self, pid: str) -> None:
        """Ensure the person, robot and their InteractionNode exist in the graph.
        No SessionNode — transcripts live in SQLite now."""
        from modules.graph_relationship.schema import (
            PersonNode, RobotNode, Embodiment,
        )
        from modules.graph_relationship.interactions import get_or_create_interaction
        if self.store.get_node(pid) is None:
            self.store.upsert_node(PersonNode(id=pid, display_name=pid))
        if self.store.get_node(self.robot_id) is None:
            emb = (Embodiment.CAT if self.robot_id.lower() == "chatbox"
                   else Embodiment.ELEPHANT)
            self.store.upsert_node(
                RobotNode(id=self.robot_id, name=self.robot_id, embodiment=emb))
        get_or_create_interaction(self.store, pid, self.robot_id, source=self.robot_id)

    def _run_session_id(self, pid: str) -> str:
        """This run's session id for `pid` (uuid), created on the first turn.
        Ensures the interaction exists. Groups the run's turns in the SQLite store."""
        import uuid
        self._ensure_interaction(pid)
        sid = self._run_sessions.get(pid)
        if sid is None:
            sid = str(uuid.uuid4())
            self._run_sessions[pid] = sid
            self._mark_kg_dirty()   # persist the new interaction (debounced)
        return sid

    def _kg_tick(self, pid: str) -> tuple[str, float, float]:
        """Ensure the person's interaction exists and return (tier, rapport, trust)
        for the overlay. No PAD, no per-tick transcript writes, no SessionNode."""
        from modules.graph_relationship.kg_bridge import derive_tier
        self._ensure_interaction(pid)
        tier = derive_tier(pid, self.robot_id, self.store)
        r, t = _read_rapport_trust(self.store, pid, self.robot_id)
        return tier, r, t

    def _mood_tick(self, pid: str, emotion: Optional[str],
                   va: Optional[tuple]) -> bool:
        """Write the person's FAST MoodEdge from the current camera emotion
        (emotion → valence). No PAD. Returns True if it changed enough to save."""
        from datetime import datetime, timezone
        from modules.graph_relationship.schema import MoodEdge, Provenance
        if va is not None and va[0] is not None:
            valence = float(va[0])
        else:
            from modules.graph_relationship.kg_bridge import emotion_label_to_va
            valence, _a = emotion_label_to_va(emotion)
        valence = max(-1.0, min(1.0, valence))
        self.store.upsert_edge(MoodEdge(
            source_id=pid, target_id=pid,
            provenance=Provenance(source=self.robot_id, confidence=1.0,
                                  timestamp=datetime.now(timezone.utc)),
            value=valence, label=emotion,
        ))
        # Only treat as "changed" (→ a disk save) on an emotion-label change or a
        # real valence shift; the raw detector valence jitters frame-to-frame, so
        # a tight threshold here would save on almost every tick (log spam).
        prev = self._last_mood.get(pid)
        changed = (prev is None or prev[1] != emotion
                   or abs(prev[0] - valence) >= 0.15)
        if changed:
            self._last_mood[pid] = (valence, emotion)
            # Mirror onto the live conversation node if one exists (don't create
            # one just from being seen — only once a conversation has started).
            from modules.graph_relationship.topics import update_conversation
            update_conversation(self.store, pid, self.robot_id,
                                mood=valence, emotion=emotion, create=False,
                                source="live-mood")
        return changed

    # ── Adaptive capture: apply the worker's learned views (main thread only) ──

    def _drain_adapt_events(self, worker) -> None:
        """Fold any views the worker learned into the gallery and persist them.

        Runs on the main thread so `face_id` keeps exactly one writer (the worker
        only queues events). Saves are debounced — enrolment already saves eagerly.
        """
        from modules.face_webcam.face_id import ADAPTIVE
        for ev in worker.pop_adapt_events():
            what = self.face_id.add_prototype(
                ev["pid"], ev["emb"], origin=ADAPTIVE, now=ev["t"])
            if what in ("inserted", "replaced"):
                na, nd = self.face_id.gallery_size(ev["pid"])
                print(f"[FaceID] learned a new view of '{ev['pid']}' "
                      f"(sim {ev['sim']:.2f}) — {na} enrolled + {nd} learned views")
                self._faces_dirty = True
        if self._faces_dirty and time.time() - self._last_faces_save > 30.0:
            self.face_id.save(self.faces_path)
            self._faces_dirty    = False
            self._last_faces_save = time.time()

    # ── First impression: auto-enrol an unknown face + learn their name ────────

    def _next_guest_id(self) -> str:
        """Next free provisional id ('guest_1', 'guest_2', …) not already in the face
        DB — so re-runs don't collide with earlier guests."""
        known = set(self.face_id.known_people())
        self._guest_seq += 1
        while f"guest_{self._guest_seq}" in known:
            self._guest_seq += 1
        return f"guest_{self._guest_seq}"

    def _auto_enroll(self, frame: np.ndarray) -> Optional[str]:
        """Enrol the face in `frame` under a fresh provisional id and persist the face
        DB. Returns the new id, or None if no face was captured. A guest_N is a normal
        PersonNode, so the full culture/interest pipeline runs on them from here."""
        if frame is None:
            print("[FirstImpression] auto-enrol skipped — no face seen recently")
            return None
        gid = self._next_guest_id()
        if not self.face_id.enroll(gid, frame):
            self._guest_seq -= 1          # release the id we reserved
            print("[FirstImpression] auto-enrol: couldn't capture a clean face — "
                  "try again facing the camera")
            return None
        self.face_id.save(self.faces_path)
        print(f"[FirstImpression] new face → enrolled as '{gid}' (guest); "
              "profile + fast link to ChatBox will build as you talk")
        return gid

    def _learn_name(self, old_id: str, new_id: str, display: str) -> bool:
        """Re-key a provisional guest to their real name across the face DB, the graph,
        the transcript store and this run's in-memory maps. Graph writes are held under
        the store lock (the chat worker may be reading concurrently)."""
        if not self.face_id.rename(old_id, new_id):
            return False
        self.face_id.save(self.faces_path)
        try:
            from modules.graph_relationship.rename import rename_person
            with self._store_lock:
                rename_person(self.store, old_id, new_id, self.robot_id,
                              display_name=display)
        except Exception as exc:  # noqa: BLE001 — never crash the loop on rename
            print(f"[FirstImpression] graph rename failed: {exc}")
        self._session_store.rename_person(old_id, new_id)
        # Re-key this run's in-memory per-person maps.
        for d in (self._run_sessions, self._last_mood, self._chat_history):
            if old_id in d:
                d[new_id] = d.pop(old_id)
        self._mark_kg_dirty()
        print(f"[FirstImpression] learned name: '{old_id}' → '{new_id}'")
        return True

    def _detect_topic(self, user_msg: str, reply: str = "") -> Optional[str]:
        """Best-effort 1–3 word label of what's being discussed, via the LLM.
        Used to update the live conversation node. Returns None on any failure."""
        if not (self.llm and self.llm.available):
            return None
        sys = ("You label the topic of a short conversation snippet. Reply with "
               "ONLY the topic as 1-3 lowercase words (a noun phrase) — no "
               "punctuation, no sentence. Examples: 'jazz music', 'the stock "
               "market', 'space travel'.")
        try:
            raw = self.llm.respond(sys, f"Person said: {user_msg}\nRobot said: {reply}")
        except Exception:  # noqa: BLE001
            return None
        topic = (raw or "").strip().strip('".\'').lower()
        # Reject sentences / error strings — keep only short noun phrases.
        if not topic or topic.startswith("[") or len(topic.split()) > 4 or len(topic) > 40:
            return None
        return topic

    # Memory caps — keep the prompt bounded as the graph grows.
    _MAX_INTERESTS = 4
    _MAX_TOPICS_PER_INTEREST = 3
    _MAX_NOTES = 8
    _MAX_NOTES_PER_TOPIC = 2

    def _person_memory(self, pid: Optional[str]) -> str:
        """The 'WHO YOU'RE TALKING TO' body: key interests (capped), common
        ground, and the most recent notes (deduped). Returns "" if nothing known."""
        if not pid:
            return ""
        from modules.graph_relationship.topics import (
            person_interests, person_topic_affinity, related_common_ground,
            person_related_pairs, topic_related,
        )
        from modules.affinity_phrasing import topic_memory_line
        interests = person_interests(self.store, pid)
        lines: list[str] = []

        # Self-declared facts FIRST, as plain recallable facts (not a tentative
        # culture hint). The origin/background is the one people ask "do you
        # remember where I'm from?" about — keep it at the top so the model treats
        # it as known memory, not something to hedge on. Only when self-declared.
        from modules.graph_relationship.cultures import (
            person_culture, person_culture_self_declared,
        )
        if person_culture_self_declared(self.store, pid):
            cnode = self.store.get_node(person_culture(self.store, pid))
            if cnode is not None:
                lines.append(
                    "Facts they told you about themselves (you KNOW these — state "
                    f"them plainly if asked): they're {cnode.label} / that's where "
                    "they're from.")

        if interests:
            # Render each observed topic as a SIGNED, HEDGED line: affinity picks the
            # verb (like / dislike / neutral) and confidence the hedge (clearly /
            # probably / possibly). Dislikes carry an explicit "avoid raising it" so
            # the robot steers away. Richer interests (more topics) still lead; cap
            # the number of interests and topics-per-interest as before.
            aff_by_topic = {t.id: (a, c) for t, a, c in person_topic_affinity(self.store, pid)}
            ranked = sorted(interests, key=lambda it: len(it[1]), reverse=True)
            feeling_lines: list[str] = []
            for _interest, topics in ranked[:self._MAX_INTERESTS]:
                for t in topics[:self._MAX_TOPICS_PER_INTEREST]:
                    aff, conf = aff_by_topic.get(t.id, (0.5, 1.0))
                    feeling_lines.append("  – " + topic_memory_line(t.label, aff, conf))
            if feeling_lines:
                lines.append("How they feel about topics:\n" + "\n".join(feeling_lines))

        # Common ground — direct + RELATED bridges (Feature-2c, point 2): a topic
        # they like that relates to something the robot knows counts as connection.
        cg = related_common_ground(self.store, pid, self.robot_id)
        if cg["direct"]:
            lines.append("Common ground: " + ", ".join(cg["direct"]))
        if cg["bridges"]:
            bl = ", ".join(f"their {p} ~ your {r}" for p, r in cg["bridges"])
            lines.append("You can also connect via related topics: " + bl)
        # Their own related topics, so the robot can bridge/recall across them.
        rp = person_related_pairs(self.store, pid)
        if rp:
            lines.append("Related interests: "
                         + ", ".join(f"{a} ~ {b}" for a, b in rp))

        # Collect notes, then surface the ones with SPECIFIC facts first (proper
        # nouns / quoted titles like "Rafael Nadal", "SZA 'Open Arms'"), then by
        # recency — so concrete memories aren't crowded out by generic ones.
        def _specificity(text: str) -> int:
            score = 2 if ("'" in text or '"' in text) else 0
            words = str(text).split()
            score += sum(1 for w in words[1:] if w[:1].isupper())  # mid-sentence caps
            return score
        # Point 1: gather notes from the person's topics AND one hop across
        # related_topic edges, so related memories surface (e.g. a note on 'hiphop'
        # when they mention 'rap'). Deduped by topic id.
        note_topics: dict = {}
        for _interest, topics in interests:
            for t in topics:
                note_topics[t.id] = t
        for tid in list(note_topics):
            for r in topic_related(self.store, tid):
                note_topics.setdefault(r.id, r)
        collected: list = []
        for t in note_topics.values():
            for n in getattr(t, "notes", []) or []:
                if n.get("person") == pid and n.get("text"):
                    collected.append((_specificity(n["text"]), n.get("ts", ""),
                                      t.label, n["text"]))
        collected.sort(key=lambda x: (x[0], x[1]), reverse=True)  # specific + recent first
        collected = [(ts, label, text) for _s, ts, label, text in collected]
        per_topic: dict = {}
        note_lines: list[str] = []
        for _ts, label, text in collected:
            # Up to _MAX_NOTES_PER_TOPIC per topic so specific facts (e.g. a
            # favourite player/song) aren't hidden behind a generic note.
            if per_topic.get(label, 0) >= self._MAX_NOTES_PER_TOPIC:
                continue
            per_topic[label] = per_topic.get(label, 0) + 1
            note_lines.append(f"  – {label}: {text}")
            if len(note_lines) >= self._MAX_NOTES:
                break
        if note_lines:
            lines.append("What you remember about them:\n" + "\n".join(note_lines))

        # Cultural-background HINT (manual, opt-in). Appended LAST so specific
        # memories still lead — this is a weak background hint, not a fact.
        cblock = self._culture_block(pid)
        if cblock:
            lines.append(cblock)

        return "\n".join(lines)

    def _culture_override_path(self) -> Optional[str]:
        kg = getattr(self, "kg_path", None)
        if not kg:
            return None
        return os.path.join(os.path.dirname(kg), "culture_override.json")

    def _read_culture_override(self) -> Optional[str]:
        """Testing knob (set by the viz culture button): force the robot's ACTIVE
        culture to a culture name (e.g. 'korean'/'maori') or 'generic' (culture off),
        or None/'auto' to follow the person. Read from culture_override.json (next to
        kg_state.json) each turn so it can be flipped live; no kg_path or
        missing/invalid file → None (auto). Never raises."""
        path = self._culture_override_path()
        if not path:
            return None
        try:
            with open(path) as fh:
                v = json.load(fh).get("active_culture")
        except Exception:  # noqa: BLE001 — missing/invalid → auto
            return None
        if not v:
            return None
        v = str(v).strip().lower()
        return None if v in ("", "auto") else v

    def _active_culture_id(self, pid: Optional[str]) -> Optional[str]:
        """Resolve the ACTIVE culture id for this turn (the culture that shapes the
        prompt): override → the current person's belongs_to_culture → generic (None).
        The override 'generic' forces culture OFF (A/B control); a culture name forces
        that culture; otherwise the person's own tag drives it (auto-attach / detach as
        the recognised person changes)."""
        from modules.graph_relationship.cultures import culture_id, person_culture
        ov = self._read_culture_override()
        if ov in ("generic", "none"):
            return None
        if ov:
            return culture_id(ov)          # forced culture (testing override)
        return person_culture(self.store, pid) if pid else None

    _ACTIVE_GRACE = 5.0   # seconds to hold the last person through a face-reco dropout

    def _write_active_state(self, pid: Optional[str], present: bool = False) -> None:
        """Write the CURRENT person + resolved active culture to a sidecar so the viz
        can HIGHLIGHT the active culture/person and dim the rest (display only — no
        graph mutation).

        Sticky, but only through a TRUE dropout: if a recognised person is seen we
        latch them; if a DIFFERENT/unknown face is on camera (`present` but pid=None)
        we drop the latched person IMMEDIATELY (so the viz shows ChatBox only, not the
        previous person); if NO face is present we keep the last person for
        _ACTIVE_GRACE seconds so a missed frame doesn't flicker. Debounced + atomic."""
        kg = getattr(self, "kg_path", None)
        if not kg:
            return
        now = time.time()
        if pid is not None:
            self._active_person = pid
            self._active_person_t = now
        elif present:
            self._active_person = None          # someone else is here → drop at once
        elif (self._active_person is not None
              and now - self._active_person_t > self._ACTIVE_GRACE):
            self._active_person = None          # no face for a while → clear
        eff = self._active_person
        with self._store_lock:
            cid = self._active_culture_id(eff)
        state = (eff, cid, bool(present))
        # Write on change, OR as a ~2s heartbeat so the viz can tell the loop is LIVE
        # (a fresh `ts`); when the loop isn't running the file goes stale and the viz
        # stops dimming. `present` distinguishes "no face" from "unknown face".
        if state == self._last_active_written and (now - self._last_active_write_t) < 2.0:
            return
        self._last_active_written = state
        self._last_active_write_t = now
        try:
            path = os.path.join(os.path.dirname(kg), "active_state.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"person": eff, "culture": cid,
                           "present": bool(present), "ts": now}, fh)
            os.replace(tmp, path)               # atomic swap — no partial reads
        except Exception:  # noqa: BLE001 — display-only, never fail the loop
            pass

    def _culture_block(self, pid: str) -> str:
        """The CULTURAL BACKGROUND content block (topic offers) for the ACTIVE culture,
        or "" when generic. The active culture = override → person's tag → generic
        (see _active_culture_id). Content-first; never asserts what they like.

        Framing: recall-as-fact ONLY when the active culture matches what the person
        actually SELF-DECLARED; otherwise it's the robot's knowledge lens (a manual tag
        or the testing override) — a starting point, never a fact about them."""
        from modules.graph_relationship.cultures import (
            person_culture, person_culture_self_declared,
        )
        from modules.preference_model import rank_suggestions
        cid = self._active_culture_id(pid)
        if not cid:
            return ""
        cnode = self.store.get_node(cid)
        if cnode is None:
            return ""
        # recall-as-fact only if the ACTIVE culture is the one THEY told us about.
        self_declared = (cid == person_culture(self.store, pid)
                         and person_culture_self_declared(self.store, pid))

        # Bayesian preference overlay (Command B): rank what to bring up by
        # posterior (ACTIVE culture priors + propagation from what they already like
        # over related_topic links). rank_suggestions returns UNOBSERVED topics only.
        # Each offer carries ONE cultural fact so the robot has something real to say.
        offers: list[tuple[str, str]] = []   # (label, one_fact_or_"")
        for node_id, _post in rank_suggestions(self.store, pid, k=4, culture_id=cid):
            node = self.store.get_node(node_id)
            if node is None:
                continue
            fact = (getattr(node, "facts", None) or [""])[0]
            offers.append((node.label, fact))

        if self_declared:
            header = (
                f"Background (they told you themselves): {cnode.label}. They stated "
                f"this in an earlier conversation, so you CAN recall it as a fact if "
                f"they ask — e.g. \"you mentioned you're {cnode.label}\". Don't assume "
                "what they like from it, though — ask.")
        else:
            header = (
                f"Cultural knowledge lens: {cnode.label} — background knowledge you can "
                "draw on for this conversation. A starting point, not a fact about them "
                "as a person; ask, don't assume.")
        lines = ["━━━ CULTURAL BACKGROUND ━━━", header]
        if offers:
            offer_lines = "\n".join(
                (f"  – {label}: {fact}" if fact else f"  – {label}")
                for label, fact in offers)
            lines.append(
                f"You know a little about {cnode.label} culture — things you could "
                f"bring up (with a fact to share):\n{offer_lines}\n"
                "If the conversation lulls, you may politely offer ONE of these and "
                "share its fact; drop it immediately if they show no interest. Never "
                "assert what they like — ask. Keep a polite, warm, respectful tone.")
        return "\n".join(lines)

    def _debug_prompt_dump(self, pid: str, msg: str, rag_hits: list) -> None:
        """Print ONLY the 3 modules that feed the prompt (KG retrieval / embedding
        RAG / BN overlay) — not the full prompt. Gated by --debug-prompt."""
        from modules.graph_relationship.topics import (
            person_interests, related_common_ground, normalize_label,
        )
        from modules.graph_relationship.cultures import person_culture, culture_priors
        from modules.preference_model import rank_suggestions
        bar = "═" * 74
        print(f"\n{bar}\n PROMPT DEBUG — {pid}   msg={msg!r}\n{bar}")

        # 1 — KG retrieval
        print("[1] KG RETRIEVAL")
        for interest, topics in person_interests(self.store, pid):
            print(f"     {interest.label}: {', '.join(t.label for t in topics) or '—'}")
        cg = related_common_ground(self.store, pid, self.robot_id)
        print(f"     common ground: {cg['direct'] or '—'}  "
              f"bridges: {[f'{a}~{b}' for a,b in cg['bridges']] or '—'}")
        cid = person_culture(self.store, pid)
        print(f"     culture tag: {cid or '— (untagged)'}")

        # 2 — embedding RAG
        print("[2] EMBEDDING RAG (top relevant past turns)")
        if rag_hits:
            for h in rag_hits:
                print(f"     [{h.get('score',0):.3f}] ({h.get('ts','')[:10]}) "
                      f"{h.get('child','')!r}")
        else:
            print("     (none retrieved / embeddings off)")

        # 3 — BN overlay
        print("[3] BN OVERLAY (rank_suggestions)")
        if cid:
            pri = ", ".join(f"{l}={p:.2f}" for _c, l, p in culture_priors(self.store, cid)[:6])
            print(f"     culture priors: {pri} …")
        obs = sorted({normalize_label(t.label)
                      for _i, ts in person_interests(self.store, pid) for t in ts})
        print(f"     observed(→0.90): {obs or '—'}")
        ranked = rank_suggestions(self.store, pid, k=6)
        print("     suggestions: " + (", ".join(
            f"{(self.store.get_node(i).label if self.store.get_node(i) else i)}={p:.3f}"
            for i, p in ranked) or "—"))
        print(bar + "\n")

    def _build_system_prompt(self, pid: Optional[str], *,
                             rag_hits: Optional[list] = None) -> str:
        """Assemble the system prompt from the seeded RobotNode + retrieved memory,
        in three labelled blocks. Used when PAD is disabled (no PAD system_prompt).
        Mood/emotion is deliberately not injected (kept for the graph/viz only)."""
        from modules.graph_relationship.topics import robot_capability
        personas = [n.descriptor for _e, n in
                    self.store.query_neighbors(self.robot_id, "has_persona")
                    if n.node_type == "persona"]
        roles = [n.descriptor for _e, n in
                 self.store.query_neighbors(self.robot_id, "has_role")
                 if n.node_type == "role"]
        cap = robot_capability(self.store, self.robot_id)
        caps = cap.items if cap else []

        role_word = roles[0] if roles else "friendly companion"

        blocks: list[str] = []
        # ── IDENTITY ──
        ident = ["━━━ IDENTITY ━━━",
                 f"You are {self._robot_display}, a {role_word} robot chatting "
                 f"with someone through a webcam."]
        if personas:
            ident.append(f"Personality: {', '.join(personas)}.")
        if caps:
            ident.append(f"You can: {', '.join(caps)}.")
        blocks.append("\n".join(ident))

        # ── HOW TO REPLY ──
        how_to_reply = (
            "━━━ HOW TO REPLY ━━━\n"
            "• Reply in ENGLISH, in one or two short, warm, spoken sentences. Output "
            "ONLY your single reply — never write the user's next turn.\n"
            "• Begin every reply with an emotion tag in square brackets, e.g. "
            "[HAPPY], [CURIOUS].\n"
            "• ANSWER what they actually ask. Anything in the memory below — a "
            "favourite player/song, or a fact they told you about themselves (where "
            "they're from / their background) — is something you KNOW: answer "
            "DIRECTLY and state it, never say you forgot. Only say you don't remember "
            "when it is genuinely NOT in the memory below — and even then, NEVER "
            "invent or guess a name.\n"
            "• Do NOT state a specific fact — a name, number, title, place, or a 'did "
            "you know…' claim — unless it appears in the memory below (including the "
            "cultural facts). You may chat about a topic in general terms, but never "
            "make up specifics to sound knowledgeable; if you don't have a concrete "
            "fact, say so plainly or ask.\n"
            "• Stay on the topic they actually referenced. If they say 'tell me more "
            "about it', continue the SAME thing from the previous turn — do not switch "
            "to an unrelated topic, and do not force a connection between unrelated "
            "topics.\n"
            "• Weave memories in naturally — don't list them back.\n"
            "• Reply to what they actually said or asked. Do not comment on how they "
            "seem to feel or offer emotional support unless they bring up their "
            "feelings themselves.")
        # Step 4: static per-culture MANNER hint (how to talk), appended to HOW TO
        # REPLY as soft, secondary guidance — separate from the content topic-offer
        # block. Only when the person is tagged to a culture with a non-empty hint;
        # identical text regardless of tier/affect/situation (no dynamics — Approach 2).
        if pid:
            _cid = self._active_culture_id(pid)   # override → person's tag → generic
            _node = self.store.get_node(_cid) if _cid else None
            _hint = getattr(_node, "style_hint", "") if _node else ""
            if _hint:
                how_to_reply += (
                    "\n• Cultural manner (soft guidance, secondary to answering their "
                    f"actual question): {_hint}")
        blocks.append(how_to_reply)

        # ── WHO YOU'RE TALKING TO ──
        if pid:
            who = [f"━━━ WHO YOU'RE TALKING TO: {pid} ━━━"]
            mem = self._person_memory(pid)
            who.append(mem if mem else "You don't remember much about them yet.")
            # NOTE: the detected mood/emotion is intentionally NOT injected into the
            # prompt for now — it pulled replies into unsolicited emotional support.
            # It's still tracked on the graph/conversation node for the viz. Revisit
            # once the emotion model / weighting is improved.
            # RAG: what THEY said before that's relevant now (their own words only —
            # we don't feed the robot's past replies back, to avoid reinforcing any
            # earlier "I forgot" deflections).
            hit_lines = [f'  – ({h["ts"][:10]}) "{h["child"]}"'
                         for h in (rag_hits or []) if h.get("child")]
            if hit_lines:
                who.append("Relevant things they've told you before:\n"
                           + "\n".join(hit_lines))
            # Recent conversation flow: the last few exchanges (persisted, so flow
            # carries across sessions) — oldest → newest, leading up to now.
            ss = getattr(self, "_session_store", None)
            if ss is not None:
                convo = []
                for t in ss.recent_turns(pid, 5):
                    if t.get("child"):
                        convo.append(f"  them: {t['child']}")
                    if t.get("reply"):
                        convo.append(f"  you:  {t['reply']}")
                if convo:
                    who.append("Recent conversation so far (oldest → newest):\n"
                               + "\n".join(convo))
            blocks.append("\n".join(who))
        else:
            blocks.append("━━━ WHO YOU'RE TALKING TO ━━━\n"
                          "You don't recognise this person yet.")

        return "\n\n".join(blocks)

    def _extract_session(self) -> None:
        """End-of-session knowledge extraction: distill each session's transcript
        into interests/topics + rapport/trust deltas, then persist. Needs the LLM."""
        if not (self.llm and self.llm.available):
            print("[WebcamLoop] extraction skipped — LLM not connected")
            return
        # Graph-aware typed TOPIC extraction lives in the app layer (kg_extraction);
        # closeness (rapport/trust) reuses the existing pure extractor for deltas
        # ONLY and the untouched adjust_closeness — its interest logic is not used.
        from modules.kg_extraction import extract_and_apply_topics
        from modules.graph_relationship.extraction import extract as _extract_closeness
        from modules.graph_relationship.interactions import adjust_closeness
        # Respect external edits (e.g. viz deletions) before extracting.
        if self.kg_path and os.path.exists(self.kg_path):
            self.store.reload(self.kg_path)
        people = list(self._run_sessions.keys())   # people talked to this run
        if not people:
            print("[WebcamLoop] no conversation this run — nothing to extract")
            return
        print("[WebcamLoop] extracting knowledge from this session …")
        for pid in people:
            sid = self._run_sessions.get(pid)
            # Un-extracted transcript turns come from the SQLite store now.
            turns = [t for t in self._session_store.unextracted_turns(pid)
                     if t.get("child") or t.get("reply")]
            if not turns:
                print(f"  {pid}: no conversation turns to extract")
                continue

            # JSON extraction needs a big token budget or the response truncates
            # mid-object and every topic in it is silently lost (chat default is 140).
            # json_mode → strict JSON object, temp 0, no spoken-reply stop strings.
            def _json_llm(sysp, usr):
                return self.llm.respond(sysp, usr, max_tokens=900, json_mode=True)

            # (a) Graph-aware typed topics (reuse existing / add genuinely new).
            ts = extract_and_apply_topics(
                self.store, pid, self.robot_id, turns, _json_llm, session_id=sid)

            # (b) Closeness deltas — existing logic, untouched (deltas applied only).
            cu = _extract_closeness(turns, _json_llm)
            if cu.rapport_delta or cu.trust_delta:
                adjust_closeness(self.store, pid, self.robot_id,
                                 d_rapport=cu.rapport_delta, d_trust=cu.trust_delta,
                                 source=f"extraction:{sid}")

            # (c) Culture self-identification — tag belongs_to_culture ONLY when the
            #     person EXPLICITLY states their own background (never from a name,
            #     face, language, or liking a cuisine). Liking kimchi ≠ being Korean.
            from modules.culture_extraction import detect_self_declared_culture
            from modules.culture_seed import assign_person_culture
            declared = detect_self_declared_culture(turns, self.llm.respond)
            if declared:
                from modules.graph_relationship.cultures import person_culture
                if person_culture(self.store, pid) is None:
                    assign_person_culture(self.store, pid, declared,
                                          source=f"self-declared:{sid}")
                    print(f"      culture: {pid} self-identified as {declared} → tagged")

            self._session_store.mark_extracted(pid)

            reinf = ", ".join(lab for lab, _c in ts.get("reinforced", []))
            newt = ", ".join(f"{lab}[{cat}]" for lab, cat, _c in ts.get("added", []))
            rel = ", ".join(f"{a}~{b}" for a, b in ts.get("related", []))
            print(f"  {pid}: Δrapport {cu.rapport_delta:+.2f}  Δtrust {cu.trust_delta:+.2f}")
            print(f"      reused: {reinf or '—'}   new: {newt or '—'}"
                  + (f"   related: {rel}" if rel else "")
                  + (f"   dropped: {len(ts.get('dropped', []))}" if ts.get('dropped') else ""))
            if not ts.get("applied"):
                print("      (topic extraction skipped — LLM JSON parse failed)")
        # Consolidate near-duplicate topics + interests after EVERY extraction.
        self._auto_consolidate()
        self._mark_kg_dirty()

    def _auto_consolidate(self) -> None:
        """Merge near-duplicate topics + interests after every extraction session.
        Applies merges (not a dry-run). No-op without embeddings."""
        if self._embed_fn is None:
            return
        from modules.kg_extraction import (
            consolidate_topics, consolidate_interests, link_related_topics,
            link_cross_namespace_bridges,
        )
        merges = (consolidate_topics(self.store, self._embed_fn, source="auto-consolidate")["merges"]
                  + consolidate_interests(self.store, self._embed_fn, source="auto-consolidate")["merges"])
        if merges:
            print(f"[WebcamLoop] auto-consolidate — merged {len(merges)}:")
            for canon, dup in merges:
                print(f"    '{dup}'  →  '{canon}'")
        else:
            print("[WebcamLoop] auto-consolidate: no near-duplicate topics/interests")
        # Link related-but-distinct topics (rap ~ hiphop) rather than merging them.
        links = link_related_topics(self.store, self._embed_fn, source="auto-related")["links"]
        if links:
            print(f"[WebcamLoop] related-topic links (+{len(links)}):")
            for a, b, sim in links:
                print(f"    '{a}' ~ '{b}'  ({sim})")
        # Step 2: bridge person topic: ↔ culture ck: nodes so evidence can propagate.
        br = link_cross_namespace_bridges(self.store, self._embed_fn, source="auto-bridge")
        bridges = br["exact"] + br["links"]
        if bridges:
            print(f"[WebcamLoop] culture bridges (+{len(bridges)}):")
            for a, b, w in bridges:
                print(f"    '{a}' ~ '{b}'  ({w})")

    def _consolidate_preview(self) -> None:
        """Dry-run: print near-duplicate topics that WOULD merge (non-destructive).
        Applying is a separate, reviewable step: --mode consolidate."""
        if self._embed_fn is None:
            print("[WebcamLoop] consolidation needs embeddings (run without --no-embed)")
            return
        from modules.kg_extraction import consolidate_topics, consolidate_interests
        merges = (consolidate_topics(self.store, self._embed_fn, dry_run=True)["merges"]
                  + consolidate_interests(self.store, self._embed_fn, dry_run=True)["merges"])
        if not merges:
            print("[WebcamLoop] consolidation preview: no near-duplicate topics/interests")
            return
        print(f"[WebcamLoop] consolidation preview — {len(merges)} merge(s), DRY RUN:")
        for canon, dup in merges:
            print(f"    '{dup}'  →  '{canon}'")
        print("    apply with:  python3 -m modules.face_webcam.webcam_loop --mode consolidate")

    def run(self, camera_index: int = _DEFAULT_CAMERA) -> None:  # noqa: C901
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[WebcamLoop] Cannot open camera {camera_index}")
            return

        WIN = "ChatBox KG Loop"
        if self.show_window:
            with _mute_stderr():   # Qt prints QFontDatabase warnings here
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        llm_status = "LLM ready" if (self.llm and self.llm.available) else "no LLM (restart with --llm)"
        mode = ("PAD+emotion" if self._pad_enabled else "KG-only") + \
               (" (emotion on)" if self._emotion_enabled else " (no emotion)")
        print(f"\n[WebcamLoop] robot={self._robot_display}  tick={self.tick_interval}s  "
              f"cam={camera_index}  {llm_status}  mode={mode}")
        print("  T=chat  E=enroll  B=boost  K=dump KG  X=extract  C=consolidate?  "
              "S=save  Q=quit  (all in the OpenCV window)\n")

        # Embed any un-embedded transcript turns once up front so the first chat
        # doesn't stall building the RAG index.
        if self._session_rag is not None:
            added = self._session_rag.reindex()
            if added:
                print(f"[WebcamLoop] RAG: embedded {added} past turn(s)")

        # ── Background detection worker ───────────────────────────────────────
        worker = _DetectionWorker(
            self.face_id,
            emotion_backend=self._emotion_backend,
            max_faces=4,
            det_scale=0.5,
            detect_emotion=self._emotion_enabled,
            detect_demographics=self._demographics_enabled,
            demographics_backend=self._demographics_backend,
            id_check_interval=self._id_check_interval,
            id_sample_window=self._id_sample_window,
            id_confirm_ratio=self._id_confirm_ratio,
            id_miss_grace=self._id_miss_grace,
            id_debug=self._id_debug,
            adapt_enabled=self._adapt_enabled,
            adapt_floor=self._adapt_floor,
            adapt_interval=self._adapt_interval,
            adapt_max=self._adapt_max,
            adapt_min_px=self._adapt_min_px,
        )
        worker.start()

        # ── Chat worker (RAG + prompt + LLM reply) — keeps the display live ───
        chat_thread = threading.Thread(
            target=self._chat_worker_loop, name="chat-worker", daemon=True)
        chat_thread.start()

        # Start DEACTIVATED: no active person/culture until a face is recognised, so
        # the viz doesn't show a stale/phantom culture as active at startup.
        self._write_active_state(None)

        # ── KG state — updated every tick, survives between ticks ─────────────
        # person_id -> {tier, pad_state, descriptors, rapport, trust}
        _kg_state: dict[str, dict] = {}

        # ── Persistent display state ──────────────────────────────────────────
        tick_n          = 0
        last_tick_t     = 0.0
        last_person_id  : Optional[str]   = None
        last_sim        : float           = 0.0
        last_box        : Optional[tuple] = None
        last_emotion    : str             = "neutral"
        last_e_conf     : float           = 0.0
        last_tier       : str             = "unknown"
        last_pad_state  : Optional[tuple] = None
        last_descriptors: Optional[dict]  = None
        last_rapport    : float           = 0.0
        last_trust      : float           = 0.0
        last_user_msg   : Optional[str]   = None
        last_verbal     : Optional[str]   = None
        chat_expire_t   : float           = 0.0
        last_all_detections: list         = []
        last_va         : tuple           = (0.0, 0.0)
        # Most recent frame that actually contained a face — used for auto-enrol so
        # typing while glancing at the keyboard doesn't miss the capture.
        last_face_frame : Optional[np.ndarray] = None

        # ── In-window input state ─────────────────────────────────────────────
        input_mode      : int  = _MODE_IDLE
        input_text      : str  = ""
        input_error     : str  = ""

        # ── Enroll capture state (guided, one pose at a time) ─────────────────
        from modules.face_webcam.face_id import DEFAULT_POSES as _POSES
        enroll_capturing : bool = False
        enroll_name      : str  = ""
        enroll_progress  : int  = 0
        enroll_per_pose  : int  = 3
        enroll_total     : int  = enroll_per_pose * len(_POSES)
        enroll_attempts  : int  = 0
        enroll_max_att   : int  = enroll_total * 8
        enroll_pose_i    : int  = 0     # which pose we are collecting
        enroll_pose_done : int  = 0     # frames accepted for THIS pose
        enroll_prompt    : str  = ""    # on-screen instruction

        fps_t0, fps_frames, fps_display = time.time(), 0, 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WebcamLoop] Frame read failed — camera disconnected?")
                    break

                # Apply any finished chat replies + async topic labels on the main
                # thread (store writes stay single-threaded). Both are cheap and
                # non-blocking when their queues are empty.
                while True:
                    try:
                        _res = self._chat_results.get_nowait()
                    except queue.Empty:
                        break
                    _verbal = self._apply_chat_result(_res)
                    if _verbal is not None:
                        last_verbal   = _verbal
                        chat_expire_t = time.time() + 45.0
                self._reload_if_externally_changed()   # adopt viz deletions (persist)
                self._drain_pending_topics()
                self._drain_pending_culture()   # mid-session culture auto-attach
                self._drain_adapt_events(worker)  # learn new face views (main thread)
                # viz highlight (debounced). `present` = a face (even unknown) is on
                # camera, so an unrecognised person drops the highlight to ChatBox only.
                self._write_active_state(last_person_id, present=bool(last_all_detections))
                # #3: debounced persistence — writes at most once/second when dirty.
                self._flush_kg()

                # FPS tracking
                fps_frames += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 2.0:
                    fps_display = fps_frames / elapsed
                    fps_frames  = 0
                    fps_t0      = time.time()

                # ── Enrollment capture (guided: one head pose at a time) ──────
                if enroll_capturing:
                    if enroll_progress < enroll_total and enroll_attempts < enroll_max_att:
                        enroll_attempts += 1
                        prompt, _tag = _POSES[enroll_pose_i]
                        enroll_prompt = f"{prompt}   [{enroll_pose_done}/{enroll_per_pose}]"

                        emb, _bx = self.face_id._get_embedding_and_box(frame)
                        if emb is None:
                            enroll_prompt = f"{prompt}   (no face — adjust position)"
                        else:
                            # After the first pose insist on a genuinely NEW view,
                            # otherwise ignoring the prompt just re-records the front.
                            P = self.face_id._protos.get(enroll_name)
                            dup = (enroll_pose_i > 0 and P is not None and len(P)
                                   and float((P @ emb).max()) >= self.face_id.tau_dup)
                            if dup:
                                enroll_prompt = f"{prompt}   (turn further — already have this view)"
                            else:
                                from modules.face_webcam.face_id import ANCHOR
                                self.face_id.add_prototype(enroll_name, emb, origin=ANCHOR)
                                self.face_id._counts[enroll_name] = \
                                    self.face_id._counts.get(enroll_name, 0) + 1
                                enroll_progress  += 1
                                enroll_pose_done += 1
                                if enroll_pose_done >= enroll_per_pose:
                                    enroll_pose_done = 0
                                    enroll_pose_i    = min(enroll_pose_i + 1,
                                                           len(_POSES) - 1)
                    else:
                        # Finished
                        if enroll_progress > 0:
                            na, _nd = self.face_id.gallery_size(enroll_name)
                            print(f"[Enroll] '{enroll_name}': {na} distinct view(s) "
                                  f"from {enroll_progress} frames. Saving …")
                            self.face_id.save(self.faces_path)
                            input_error = f"'{enroll_name}' enrolled ({na} views)!"
                        else:
                            print(f"[Enroll] No face found for '{enroll_name}'.")
                            input_error = "No face detected — try again"
                        enroll_capturing = False
                        enroll_name      = ""
                        enroll_progress  = 0
                        enroll_attempts  = 0
                        enroll_pose_i    = 0
                        enroll_pose_done = 0
                        enroll_prompt    = ""

                # ── Submit frame to background worker ─────────────────────────
                if not enroll_capturing:
                    worker.submit(frame)

                # ── Read latest worker results (non-blocking, every frame) ─────
                raw_dets = worker.get_results()

                # Merge worker detections with last known KG state for the overlay
                last_all_detections = []
                for d in raw_dets:
                    pid = d["person_id"]
                    kg  = _kg_state.get(pid, {}) if pid else {}
                    last_all_detections.append({
                        **d,
                        "tier":        kg.get("tier",        "unknown"),
                        "pad_state":   kg.get("pad_state",   None),
                        "descriptors": kg.get("descriptors", None),
                        "rapport":     kg.get("rapport",     0.0),
                        "trust":       kg.get("trust",       0.0),
                    })

                # Update primary-person visual refs every frame
                # (boxes/names track live; tier/PAD come from last tick below)
                primary_det = None
                for d in last_all_detections:
                    if d["person_id"] is not None:
                        primary_det = d
                        break
                if primary_det:
                    last_person_id = primary_det["person_id"]
                    last_sim       = primary_det["sim"]
                    last_box       = primary_det["box"]
                    last_emotion   = primary_det["emotion"]
                    last_e_conf    = primary_det["e_conf"]
                    last_tier      = primary_det["tier"]
                    last_pad_state = primary_det["pad_state"]
                    last_descriptors = primary_det["descriptors"]
                    last_rapport   = primary_det["rapport"]
                    last_trust     = primary_det["trust"]
                    last_va        = primary_det.get("va", (0.0, 0.0))
                elif last_all_detections:
                    first = last_all_detections[0]
                    last_person_id = None
                    last_sim       = first["sim"]
                    last_box       = first["box"]
                    last_emotion   = first["emotion"]
                    last_e_conf    = first["e_conf"]
                else:
                    last_person_id = None
                    last_sim       = 0.0
                    last_box       = None
                # Remember the latest frame that had a face, for a robust auto-enrol.
                if last_box is not None:
                    last_face_frame = frame

                # ── KG / PAD tick (every tick_interval) ───────────────────────
                now = time.time()
                if not enroll_capturing and now - last_tick_t >= self.tick_interval:
                    last_tick_t = now
                    tick_n += 1
                    mood_dirty = False

                    # The per-tick KG reads/writes share the store with the chat
                    # worker, so hold the store lock for the whole tick (fast: µs–ms).
                    with self._store_lock:
                        for d in raw_dets:
                            pid = d["person_id"]
                            if pid is None:
                                continue
                            if self._pad_enabled:
                                bi, pad = self._pipeline_tick(pid, d["emotion"], va=d.get("va"))
                                r, t    = _read_rapport_trust(self.store, pid, self.robot_id)
                                _kg_state[pid] = {
                                    "tier":        bi.tier,
                                    "pad_state":   pad["pad_state"],
                                    "descriptors": pad["descriptors"],
                                    "rapport":     r,
                                    "trust":       t,
                                }
                                if pid == last_person_id:
                                    self._last_pad_result = pad
                                    last_tier        = bi.tier
                                    last_pad_state   = pad["pad_state"]
                                    last_descriptors = pad["descriptors"]
                                    last_rapport     = r
                                    last_trust       = t
                            else:
                                # KG-only tick: ensure session, refresh overlay state.
                                tier, r, t = self._kg_tick(pid)
                                _kg_state[pid] = {
                                    "tier":        tier,
                                    "pad_state":   None,
                                    "descriptors": None,
                                    "rapport":     r,
                                    "trust":       t,
                                }
                                # Emotion drives the FAST MoodEdge only (no PAD).
                                if self._emotion_enabled and self._mood_tick(
                                        pid, d.get("emotion"), d.get("va")):
                                    mood_dirty = True
                                if pid == last_person_id:
                                    last_tier    = tier
                                    last_rapport = r
                                    last_trust   = t

                        # Persist after the tick so the live viz server
                        # (modules.graph_relationship.viz.server) can poll it within ~1s.
                        # PAD mode mutates every tick; the KG-only path otherwise saves
                        # on session creation + each chat turn, so we only add a per-tick
                        # save when the FAST mood actually changed (mood_dirty).
                        if ((self._pad_enabled or mood_dirty)
                                and any(d["person_id"] for d in raw_dets)):
                            self._mark_kg_dirty()

                # Clear stale chat
                if last_user_msg and time.time() > chat_expire_t:
                    last_user_msg = None
                    last_verbal   = None

                # ── Draw overlay ──────────────────────────────────────────────
                if self.show_window:
                    cursor_on = (time.time() % 1.0) < 0.5
                    display = draw_overlay(
                        frame,
                        person_id   = last_person_id,
                        sim         = last_sim,
                        box         = last_box,
                        emotion     = last_emotion,
                        e_conf      = last_e_conf,
                        tier        = last_tier,
                        pad_state   = last_pad_state,
                        descriptors = last_descriptors,
                        rapport     = last_rapport,
                        trust       = last_trust,
                        va          = last_va,
                        last_user_msg = last_user_msg,
                        last_verbal   = last_verbal,
                        robot_name  = self._robot_display,
                        tick        = tick_n,
                        fps         = fps_display,
                        input_mode      = input_mode,
                        input_text      = input_text,
                        input_error     = input_error,
                        cursor_on       = cursor_on,
                        enroll_capturing = enroll_capturing,
                        enroll_progress  = enroll_progress,
                        enroll_total     = enroll_total,
                        enroll_prompt    = enroll_prompt,
                        llm_on          = bool(self.llm and self.llm.available),
                        all_detections  = last_all_detections,
                    )
                    # _mute_stderr on first frame only — Qt finishes font init there
                    if tick_n <= 1:
                        with _mute_stderr():
                            cv2.imshow(WIN, display)
                    else:
                        cv2.imshow(WIN, display)

                # ── Key handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == 255:  # no key pressed
                    continue

                if input_mode != _MODE_IDLE:
                    # ── Text capture mode ─────────────────────────────────────
                    if key in (13, 10):  # Enter
                        if input_mode == _MODE_CHAT:
                            msg = input_text.strip()
                            if msg:
                                input_error = ""
                                last_user_msg  = msg
                                chat_expire_t  = time.time() + 45.0
                                print(f"\n  [you]  \"{msg}\"")

                                # ── First impression: meet a stranger ──────────
                                # (1) Unrecognised person → auto-enrol as guest_N so
                                #     THIS turn is attributed to them and the full
                                #     culture/interest pipeline builds their profile.
                                #     Uses the last frame that had a face (robust to
                                #     glancing at the keyboard while typing).
                                if last_person_id is None:
                                    if last_face_frame is not None:
                                        gid = self._auto_enroll(last_face_frame)
                                        if gid:
                                            last_person_id = gid
                                            last_tier      = "visitor"
                                    else:
                                        print("[FirstImpression] no face seen yet — "
                                              "look at the camera so I can meet you")
                                # (2) Did they introduce themselves? Re-key the
                                #     provisional guest id to their real name.
                                if last_person_id and last_person_id.startswith("guest_"):
                                    nm   = _extract_name(msg)
                                    slug = _slug_name(nm) if nm else ""
                                    if slug and slug != last_person_id and \
                                            self._learn_name(last_person_id, slug, nm):
                                        if last_person_id in _kg_state:
                                            _kg_state[slug] = _kg_state.pop(last_person_id)
                                        last_person_id = slug

                                if self.llm and self.llm.available:
                                    # Dispatch the whole chat turn (RAG + prompt build +
                                    # LLM reply) to the chat worker so the display never
                                    # freezes; the reply is applied on the main thread
                                    # when ready (see the _chat_results drain near the
                                    # top of the loop). History is snapshotted now — only
                                    # the main thread mutates self._chat_history.
                                    hist = list(self._chat_history.get(
                                        last_person_id or "", []))
                                    self._chat_requests.put({
                                        "msg": msg, "pid": last_person_id,
                                        "emotion": last_emotion, "history": hist,
                                    })
                                    last_verbal = "…"   # 'thinking' until the reply lands
                                else:
                                    last_verbal = "[LLM not enabled — run with --llm]"
                                    print("  [chat] LLM not connected. Run with --llm.\n")
                            input_mode = _MODE_IDLE
                            input_text = ""

                        elif input_mode == _MODE_ENROLL:
                            name = input_text.strip()
                            if name:
                                input_error      = ""
                                enroll_name      = name
                                enroll_progress  = 0
                                enroll_attempts  = 0
                                enroll_capturing = True
                                print(f"[Enroll] Capturing frames for '{name}' …")
                            else:
                                input_error = "Name cannot be empty"
                            input_mode = _MODE_IDLE
                            input_text = ""

                    elif key == 27:  # Escape → cancel
                        input_mode  = _MODE_IDLE
                        input_text  = ""
                        input_error = ""

                    elif key in (8, 127):  # Backspace
                        input_text = input_text[:-1]
                        input_error = ""

                    elif 32 <= key <= 126:  # Printable ASCII
                        input_text += chr(key)
                        input_error = ""

                else:
                    # ── Hotkeys (idle mode) ───────────────────────────────────
                    if key in (ord("q"), 27):   # Q or Esc
                        break
                    elif key == ord("t") or key == ord("T"):
                        input_mode  = _MODE_CHAT
                        input_text  = ""
                        input_error = ""
                    elif key == ord("e") or key == ord("E"):
                        input_mode  = _MODE_ENROLL
                        input_text  = ""
                        input_error = ""
                    elif key in (ord("k"), ord("K")):
                        with self._store_lock:
                            _dump_kg(self.store, self.robot_id)
                    elif key in (ord("x"), ord("X")):
                        with self._store_lock:
                            self._extract_session()   # run extraction mid-session (testing)
                    elif key in (ord("c"), ord("C")):
                        with self._store_lock:
                            self._consolidate_preview()   # dry-run: preview topic merges
                    elif key in (ord("b"), ord("B")) and last_person_id:
                        with self._store_lock:
                            _update_rapport_trust(
                                self.store, last_person_id, self.robot_id,
                                delta=0.15, verbose=True,
                            )
                            self._mark_kg_dirty()
                    elif key in (ord("s"), ord("S")):
                        self.face_id.save(self.faces_path)

        except KeyboardInterrupt:
            print("\n[WebcamLoop] Interrupted.")
        finally:
            worker.stop()
            worker.join(timeout=2.0)
            # Stop the chat worker before touching the store at shutdown, so
            # end-of-session extraction can't race a chat-worker prompt read.
            self._chat_requests.put(None)
            chat_thread.join(timeout=2.0)
            if self.face_id.known_people():
                self.face_id.save(self.faces_path)
            # End-of-session knowledge extraction → update the graph.
            try:
                with self._store_lock:
                    self._extract_session()
            except Exception as exc:  # noqa: BLE001 — never fail on shutdown
                print(f"[WebcamLoop] extraction failed: {exc}")
            self._flush_kg(force=True)   # final flush regardless of debounce
            self._session_store.close()
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone enroll mode (--mode enroll CLI shortcut)
# ─────────────────────────────────────────────────────────────────────────────

def run_enroll_mode(name: str, faces_path: str,
                    camera_index: int, n_captures: int, threshold: float,
                    detector: str = "opencv",
                    poses: Optional[str] = None) -> None:
    from modules.face_webcam.face_id import DEFAULT_POSES
    fi = FaceIdentifier(threshold=threshold, detector=detector)
    if os.path.exists(faces_path):
        fi.load(faces_path)
    pose_list = None
    if poses:
        wanted = [p.strip().lower() for p in poses.split(",") if p.strip()]
        pose_list = [(pr, tg) for pr, tg in DEFAULT_POSES if tg in wanted] or None
    ok = fi.enroll_from_camera(name, camera_index=camera_index,
                               n_captures=n_captures, poses=pose_list)
    if ok:
        fi.save(faces_path)
        na, nd = fi.gallery_size(name)
        print(f"Done. '{name}' now has {na} enrolled view(s). "
              f"Known people: {fi.known_people()}")
    else:
        print("Enrollment failed — no faces captured.")


def run_reset_adaptive(faces_path: str, name: Optional[str]) -> None:
    """Drop self-learned views, keeping deliberately enrolled ones. The undo for
    adaptive capture if it ever learns a bad view."""
    fi = FaceIdentifier()
    if not (os.path.exists(faces_path) and fi.load(faces_path)):
        print(f"[reset-adaptive] no face DB at '{faces_path}'")
        return
    who = name or "all people"
    n = fi.reset_adaptive(name)
    fi.save(faces_path)
    print(f"[reset-adaptive] removed {n} self-learned view(s) for {who}; "
          "enrolled views kept.")
    for p in fi.known_people():
        na, nd = fi.gallery_size(p)
        print(f"    {p}: {na} enrolled + {nd} learned")


def run_faces_info(faces_path: str) -> None:
    """Print the stored gallery per person — how many views, how distinct they are.
    Use it to answer 'why does it still fail at 60 degrees?'."""
    fi = FaceIdentifier()
    if not (os.path.exists(faces_path) and fi.load(faces_path)):
        print(f"[faces-info] no face DB at '{faces_path}'")
        return
    from modules.face_webcam.face_id import ANCHOR
    print(f"\nFace DB: {os.path.abspath(faces_path)}")
    for p in fi.known_people():
        na, nd = fi.gallery_size(p)
        P, M = fi._protos[p], fi._meta[p]
        print(f"\n  {p}:  {na} enrolled + {nd} learned "
              f"({fi._counts.get(p, 0)} frames total)")
        for i in range(len(P)):
            kind = "enrolled" if M[i, 1] == ANCHOR else "learned "
            print(f"      view {i}: {kind}  weight={M[i,0]:5.1f}")
        if len(P) > 1:
            S = P @ P.T
            np.fill_diagonal(S, -9.0)
            print(f"      most-similar pair = {S.max():.3f}  "
                  f"(closer to 1.0 = views are redundant; add more angles)")
        else:
            print("      only ONE view — recognition will be angle-brittle. "
                  "Re-enroll with the guided poses.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone topic consolidation (--mode consolidate) — Feature 2
# ─────────────────────────────────────────────────────────────────────────────

def run_consolidate_mode(kg_path: str, embed_model: str,
                         merge_floor: float, dry_run: bool) -> None:
    """Merge near-duplicate topics in an existing KG by embedding similarity.
    Reviewable + deterministic; run with --dry-run first to preview."""
    store = InMemoryGraphStore()
    if not (os.path.exists(kg_path) and store.load(kg_path)):
        print(f"[consolidate] no KG at '{kg_path}'")
        return
    try:
        from modules.graph_relationship.embedding import ollama_embed_fn
        embed_fn = ollama_embed_fn(model=embed_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[consolidate] embeddings unavailable ({exc}) — cannot consolidate")
        return
    from modules.kg_extraction import (
        consolidate_topics, consolidate_interests, link_related_topics,
        link_cross_namespace_bridges,
    )
    merges = (consolidate_topics(store, embed_fn, floor=merge_floor, dry_run=dry_run)["merges"]
              + consolidate_interests(store, embed_fn, floor=merge_floor, dry_run=dry_run)["merges"])
    links = link_related_topics(store, embed_fn, merge_floor=merge_floor, dry_run=dry_run)["links"]
    # Step 2: cross-namespace culture↔person bridges (exact-slug + embedding band).
    br = link_cross_namespace_bridges(store, embed_fn, merge_floor=merge_floor, dry_run=dry_run)
    bridges = br["exact"] + br["links"]
    if not merges and not links and not bridges:
        print(f"[consolidate] no near-duplicate/related topics or culture bridges (floor {merge_floor})")
        return
    tag = "DRY RUN — no changes written" if dry_run else "APPLIED"
    print(f"[consolidate] {len(merges)} merge(s), {len(links)} related-link(s), "
          f"{len(bridges)} culture-bridge(s) — {tag}:")
    for canon, dup in merges:
        print(f"    merge  '{dup}'  →  '{canon}'")
    for a, b, sim in links:
        print(f"    link   '{a}' ~ '{b}'  ({sim})")
    for a, b, w in bridges:
        print(f"    bridge '{a}' ~ '{b}'  ({w})")
    if not dry_run:
        store.save(kg_path)
        print(f"[consolidate] saved → {kg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Culture demo seed (--mode seed-culture-demo) + --assign-culture — Command A
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_culture_demo(kg_path: str, robot_id: str = _DEFAULT_ROBOT) -> None:
    """Seed ALL demo cultures (Korean + Māori) as `robot_id`'s prior knowledge (dummy
    priors, robot-owned culture-topic nodes) into an existing KG. Idempotent; does
    NOT touch any person or any shared person-interest topic. Which culture is ACTIVE
    for a person is resolved at prompt time (override → person's tag → generic)."""
    store = InMemoryGraphStore()
    if not (os.path.exists(kg_path) and store.load(kg_path)):
        print(f"[seed-culture] no KG at '{kg_path}' — starting fresh")
    from modules.culture_seed import seed_all_cultures
    infos = seed_all_cultures(store, robot_id=robot_id)
    store.save(kg_path)
    for label, info in infos.items():
        print(f"[seed-culture] culture={info['culture']}  robot={info['robot'] or '—'}  "
              f"topics={info['topics']}  priors={info['priors']}")
    print(f"[seed-culture] saved → {kg_path}")
    print("[seed-culture] tag someone with: --assign-culture <person> Korean|Maori")


def run_assign_culture(kg_path: str, person_id: str, culture_label: str) -> None:
    """Manually link a person to a culture (creating it if needed). Person must
    already exist in the KG (enroll/talk first). Idempotent."""
    store = InMemoryGraphStore()
    if not (os.path.exists(kg_path) and store.load(kg_path)):
        print(f"[assign-culture] no KG at '{kg_path}'")
        return
    if store.get_node(person_id) is None:
        print(f"[assign-culture] person '{person_id}' not in KG — "
              "enroll/talk to them first so the person node exists")
        return
    from modules.culture_seed import assign_person_culture
    cid = assign_person_culture(store, person_id, culture_label)
    store.save(kg_path)
    print(f"[assign-culture] {person_id} → {cid}  (saved → {kg_path})")


def run_migrate_sessions(kg_path: str, sessions_db: str) -> None:
    """One-off: move existing graph SessionNodes' transcripts into the SQLite store,
    then remove the SessionNodes (+ has_session edges) from the graph. Migrated turns
    are marked extracted (they already shaped the graph). interaction_count is
    refreshed from the store."""
    store = InMemoryGraphStore()
    if not (os.path.exists(kg_path) and store.load(kg_path)):
        print(f"[migrate] no KG at '{kg_path}'")
        return
    from modules.graph_relationship.interactions import set_interaction_count
    ss = SessionStore(sessions_db)
    sessions = [n for n in list(store._nodes.values()) if n.node_type == "session"]
    if not sessions:
        print("[migrate] no SessionNodes in the graph — nothing to move")
        ss.close()
        return
    moved_turns = 0
    pairs: set = set()
    for sess in sessions:
        # The interaction is the source of the has_session edge into this session.
        inter_id = None
        for edge, _nbr in store.query_neighbors(sess.id, "has_session"):
            if edge.target_id == sess.id:
                inter_id = edge.source_id
                break
        person = robot = None
        if inter_id and inter_id.startswith("interaction:"):
            parts = inter_id.split(":", 2)   # ['interaction', person, robot]
            if len(parts) == 3:
                person, robot = parts[1], parts[2]
        for t in (sess.turns or []):
            ss.append_turn(session_id=sess.id, person_id=person or "unknown",
                           robot_id=robot or "chatbox",
                           emotion=t.get("emotion"), child=t.get("child"),
                           reply=t.get("reply"))
            moved_turns += 1
        if person and robot:
            pairs.add((person, robot))
        store.delete_node(sess.id)   # also drops the has_session edge
    # Old history already shaped the graph → don't re-extract it.
    for person, _robot in pairs:
        ss.mark_extracted(person)
    for person, robot in pairs:
        set_interaction_count(store, person, robot,
                              ss.person_turn_count(person, robot), source="migrate")
    store.save(kg_path)
    ss.close()
    print(f"[migrate] moved {moved_turns} turn(s) from {len(sessions)} session(s) → "
          f"{sessions_db}; removed SessionNodes from the graph → {kg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Live webcam → KG relationship engine → PAD persona loop"
    )
    p.add_argument("--mode",
                   choices=["run", "enroll", "consolidate", "migrate-sessions",
                            "seed-culture-demo"],
                   default="run",
                   help="run | enroll | consolidate (merge near-dup topics) | "
                        "migrate-sessions (move graph SessionNodes → SQLite) | "
                        "seed-culture-demo (seed Korean culture demo priors)")
    p.add_argument("--assign-culture", nargs=2, metavar=("PERSON", "CULTURE"),
                   default=None,
                   help="Manually link a person to a culture, e.g. "
                        "--assign-culture jay Korean, then exit")
    p.add_argument("--sessions-db", default=_DEFAULT_SESSIONS_DB,
                   help=f"SQLite transcript DB path (default: {_DEFAULT_SESSIONS_DB})")
    p.add_argument("--name",       default=None,
                   help="Person name for enroll mode")
    p.add_argument("--faces",      default=_DEFAULT_FACES,
                   help=f"Face DB .npz path (default: {_DEFAULT_FACES})")
    p.add_argument("--kg",         default=_DEFAULT_KG,
                   help=f"KG state JSON path (default: {_DEFAULT_KG})")
    p.add_argument("--robot",      default=_DEFAULT_ROBOT,
                   choices=["chatbox", "ellebot"])
    p.add_argument("--camera",     type=int,   default=_DEFAULT_CAMERA)
    p.add_argument("--threshold",  type=float, default=_DEFAULT_THRESH,
                   help="Cosine similarity threshold (default: 0.75)")
    p.add_argument("--tick",       type=float, default=_DEFAULT_TICK,
                   help="Pipeline tick interval in seconds (default: 1.0)")
    p.add_argument("--n-captures", type=int,   default=15,
                   help="Frames to capture in enroll mode (default: 15)")
    p.add_argument("--llm",        action="store_true",
                   help="Enable LLM verbal responses via Ollama")
    p.add_argument("--model",      default=_DEFAULT_MODEL,
                   help=f"Ollama model (default: {_DEFAULT_MODEL})")
    p.add_argument("--emotion",    default="hsemotion",
                   choices=["hsemotion", "hsemotion-b2", "efficientnet"],
                   help="Emotion detection backend (default: hsemotion)")
    p.add_argument("--detector",   default="opencv",
                   choices=["opencv", "mtcnn"],
                   help="Face locator: 'opencv' (single Haar pass shared by "
                        "face-reco + emotion, default) or 'mtcnn' (landmark-aligned, "
                        "more accurate). Re-enroll people after switching.")
    p.add_argument("--id-interval", type=float, default=3.0,
                   help="Seconds to HOLD a confirmed identity before re-checking "
                        "(no re-identification during this window). Default 3.")
    p.add_argument("--id-sample",   type=float, default=1.0,
                   help="Seconds to SAMPLE/vote face recognition when re-checking. "
                        "Default 1.")
    p.add_argument("--id-confirm",  type=float, default=0.6,
                   help="Fraction of sampled frames a person must be recognised in "
                        "to confirm identity (0-1). Default 0.6.")
    p.add_argument("--id-miss-grace", type=int, default=2,
                   help="Vote windows a known person may be missed (hard pose) before "
                        "the label drops. A rival face or an empty frame drops it at "
                        "once. Default 2.")
    p.add_argument("--id-debug", action="store_true",
                   help="Print a per-frame identification trace (box, similarity, "
                        "votes) — use it to calibrate the thresholds below.")
    # ── Angle robustness: detection ───────────────────────────────────────────
    p.add_argument("--no-multi-pose", dest="multi_pose", action="store_false",
                   help="Only run the FRONTAL face cascade. By default the profile "
                        "cascades run too, so turned heads are detected at all.")
    p.set_defaults(multi_pose=True)
    # ── Angle robustness: multi-view gallery ──────────────────────────────────
    p.add_argument("--proto-max", type=int, default=12,
                   help="Max deliberately-enrolled views stored per person (default 12).")
    p.add_argument("--proto-adapt-max", type=int, default=6,
                   help="Max self-learned views stored per person (default 6).")
    p.add_argument("--proto-dup", type=float, default=0.92,
                   help="Similarity at/above which a face is the SAME view as one "
                        "already stored (refine it) rather than a new one. Default 0.92.")
    # ── Flicker: hysteresis ───────────────────────────────────────────────────
    p.add_argument("--retain-threshold", type=float, default=0.62,
                   help="Lower similarity bar to KEEP the identity already held "
                        "(vs --threshold to acquire one). Default 0.62.")
    p.add_argument("--switch-margin", type=float, default=0.08,
                   help="How far a rival must beat the held identity to take over "
                        "(default 0.08).")
    p.add_argument("--id-margin", type=float, default=0.05,
                   help="Runner-up margin required to accept a match; only applied "
                        "with 2+ people enrolled. Default 0.05.")
    # ── Adaptive capture ──────────────────────────────────────────────────────
    p.add_argument("--no-adapt", dest="adapt", action="store_false",
                   help="Disable learning new views of a person during conversation.")
    p.set_defaults(adapt=True)
    p.add_argument("--adapt-floor", type=float, default=None,
                   help="Never learn a view below this similarity. Defaults to "
                        "--threshold (conservative). Lowering it also learns from the "
                        "marginal band, which needs unanimous votes.")
    p.add_argument("--adapt-interval", type=float, default=20.0,
                   help="Minimum seconds between learned views, per person (default 20).")
    p.add_argument("--adapt-max", type=int, default=20,
                   help="Max views learned per session (default 20).")
    p.add_argument("--adapt-min-px", type=int, default=90,
                   help="Minimum face box side (px) to learn from (default 90).")
    p.add_argument("--reset-adaptive", nargs="?", const="__ALL__", default=None,
                   metavar="NAME",
                   help="Delete self-learned views (all people, or just NAME) from the "
                        "face DB and exit. Enrolled views are kept.")
    p.add_argument("--faces-info", action="store_true",
                   help="Print each enrolled person's stored views and exit.")
    p.add_argument("--enroll-poses", default=None,
                   help="Comma-separated poses for guided enrollment "
                        "(default: front,left,right,up,down).")
    p.add_argument("--no-window",  action="store_true",
                   help="Headless — terminal output only")
    p.add_argument("--esp32-host", default="",
                   help="ESP32 IP address for TCP expression dispatch (blank=disabled)")
    p.add_argument("--esp32-port", type=int, default=8888,
                   help="ESP32 TCP port (default: 8888)")
    # ── KG integration options ────────────────────────────────────────────────
    p.add_argument("--no-seed", action="store_true",
                   help="Do not seed the KG from spec files on startup")
    p.add_argument("--spec-dir", default=None,
                   help="Directory of KG spec YAML/JSON files "
                        "(default: modules/graph_relationship/specs)")
    p.add_argument("--no-embed", action="store_true",
                   help="Use the keyword matcher instead of embeddings for "
                        "topic↔capability linking during extraction")
    p.add_argument("--embed-model", default="nomic-embed-text",
                   help="Ollama embedding model (default: nomic-embed-text)")
    p.add_argument("--embed-floor", type=float, default=0.62,
                   help="Min cosine similarity to link topic↔capability (default: 0.62). "
                        "Higher = fewer, stricter links (avoids e.g. tennis↔'good at math')")
    p.add_argument("--enable-pad", action="store_true",
                   help="Enable the PAD persona engine (disabled by default this pass)")
    p.add_argument("--enable-emotion", action="store_true",
                   help="Enable emotion detection (disabled by default this pass)")
    p.add_argument("--culture", action="store_true",
                   help="Show a cultural-awareness estimate (region/heritage + age) "
                        "on the webcam view. Display-only — not fed to the KG/prompt.")
    p.add_argument("--culture-backend", default="fairface", choices=["fairface"],
                   help="Demographics model backend (default: fairface)")
    p.add_argument("--debug-prompt", action="store_true",
                   help="At each chat turn, print the 3 prompt-feeding modules "
                        "(KG retrieval / embedding RAG / BN overlay) + the final prompt")
    # ── Feature 2: topic consolidation (--mode consolidate) ────────────────────
    p.add_argument("--merge-floor", type=float, default=0.86,
                   help="Min cosine similarity to MERGE two near-duplicate topics "
                        "(default: 0.86; same-category only)")
    p.add_argument("--dry-run", action="store_true",
                   help="consolidate mode: preview merges without writing")
    args = p.parse_args()

    if args.reset_adaptive is not None:
        run_reset_adaptive(args.faces,
                           None if args.reset_adaptive == "__ALL__" else args.reset_adaptive)
        return

    if args.faces_info:
        run_faces_info(args.faces)
        return

    if args.mode == "enroll":
        if not args.name:
            p.error("--name is required for enroll mode")
        run_enroll_mode(args.name, args.faces, args.camera,
                        args.n_captures, args.threshold, detector=args.detector,
                        poses=args.enroll_poses)
        return

    if args.mode == "consolidate":
        run_consolidate_mode(args.kg, args.embed_model, args.merge_floor, args.dry_run)
        return

    if args.mode == "migrate-sessions":
        run_migrate_sessions(args.kg, args.sessions_db)
        return

    if args.mode == "seed-culture-demo":
        run_seed_culture_demo(args.kg, args.robot)
        return

    # --assign-culture is a standalone action (independent of --mode run).
    if args.assign_culture:
        run_assign_culture(args.kg, args.assign_culture[0], args.assign_culture[1])
        return

    llm = None
    if args.llm:
        llm = LLMClient(model=args.model)
        llm.connect()

    # Embedding matcher (default when --llm); degrades gracefully on failure.
    # embed_fn is also handed to the loop for the on-demand topic-merge preview (C).
    matcher = None
    embed_fn = None
    if not args.no_embed:
        try:
            from modules.graph_relationship.embedding import (
                make_embedding_matcher, ollama_embed_fn,
            )
            embed_fn = ollama_embed_fn(model=args.embed_model)
            matcher = make_embedding_matcher(embed_fn, floor=args.embed_floor)
            print(f"[WebcamLoop] topic matching via embeddings "
                  f"({args.embed_model}, floor {args.embed_floor})")
        except Exception as exc:  # noqa: BLE001
            print(f"[WebcamLoop] embedding matcher unavailable ({exc}) — "
                  "using keyword matcher")
            matcher = None
            embed_fn = None

    loop = WebcamKGLoop(
        robot_id         = args.robot,
        faces_path       = args.faces,
        kg_path          = args.kg,
        threshold        = args.threshold,
        tick_interval    = args.tick,
        llm_client       = llm,
        show_window      = not args.no_window,
        emotion_backend  = args.emotion,
        esp32_host       = args.esp32_host,
        esp32_port       = args.esp32_port,
        spec_dir         = args.spec_dir,
        seed             = not args.no_seed,
        matcher          = matcher,
        embed_fn         = embed_fn,
        sessions_db      = args.sessions_db,
        pad_enabled      = args.enable_pad,
        emotion_enabled  = args.enable_emotion,
        demographics_enabled = args.culture,
        demographics_backend = args.culture_backend,
        detector             = args.detector,
        id_check_interval    = args.id_interval,
        id_sample_window     = args.id_sample,
        id_confirm_ratio     = args.id_confirm,
        id_miss_grace        = args.id_miss_grace,
        id_debug             = args.id_debug,
        multi_pose           = args.multi_pose,
        retain_threshold     = args.retain_threshold,
        switch_margin        = args.switch_margin,
        id_margin            = args.id_margin,
        proto_dup            = args.proto_dup,
        proto_max            = args.proto_max,
        proto_adapt_max      = args.proto_adapt_max,
        adapt_enabled        = args.adapt,
        adapt_floor          = args.adapt_floor,
        adapt_interval       = args.adapt_interval,
        adapt_max            = args.adapt_max,
        adapt_min_px         = args.adapt_min_px,
        debug_prompt         = args.debug_prompt,
    )
    loop.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
