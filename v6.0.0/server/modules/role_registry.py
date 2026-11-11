"""Which robots the viz knows about, saved in robots.json so they survive a restart.

    {"id": "stuart", "name": "Stuart", "kind": "dial-out", "host": "192.168.0.11", "port": 8765,
     "tts": "navel", "color": "#9ccc65", "voice_adjust": {"rate": 0, "pitch": 0, "volume": 0}}

kind "connects-in": the robot connects to this viz's hub itself (the ChatBox Jetson does) and must call
                    itself `name` in client_init, which is how its messages are told apart.
kind "dial-out":    the robot runs its own WebSocket server and the viz dials host:port (Navel does).
tts "navel":        voice specs are sent as Navel TTS tags in front of its text; "none": plain text.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List, Optional

PALETTE = ["#ff8c42", "#4fc3f7", "#9ccc65", "#e6689c", "#b39ddb", "#ffd24a", "#26a69a", "#f4694c"]
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ]{0,19}$")
VOICE_LIMITS = {"rate": 30.0, "pitch": 6.0, "volume": 6.0}      # per side, same scale as persona_voice


def robot_id(name: str) -> str:
    """The id a robot is known by: its name lower-cased, without spaces (what the hub matches robot_name to)."""
    return name.lower().replace(" ", "")


def default_specs() -> List[dict]:
    return [
        {"id": "chatbox", "name": "ChatBox", "kind": "connects-in", "tts": "none", "color": PALETTE[0], "builtin": True},
        {"id": "navel", "name": "Navel", "kind": "dial-out", "host": "", "port": 8765, "tts": "navel",
         "color": PALETTE[1], "builtin": False},
    ]


class RobotRegistry:
    def __init__(self, path: Optional[str] = None, defaults: Optional[List[dict]] = None) -> None:
        self.path, self._lock = path, threading.Lock()
        self.robots: List[dict] = []
        loaded = []
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    loaded = json.load(fh).get("robots", [])
            except (OSError, ValueError):
                loaded = []
        for spec in loaded:
            self._normalise(spec)
        have = {r["id"] for r in loaded}
        self.robots = loaded + [self._normalise(dict(d)) for d in (defaults or default_specs()) if d["id"] not in have]
        self.save()

    # ---- read ----
    def get(self, rid: str) -> Optional[dict]:
        return next((r for r in self.robots if r["id"] == rid), None)

    def colors(self) -> Dict[str, str]:
        return {r["id"]: r["color"] for r in self.robots}

    def voice_adjust(self, rid: str) -> dict:
        return dict((self.get(rid) or {}).get("voice_adjust") or {})

    # ---- change ----
    def add(self, name: str, kind: str = "connects-in", host: str = "", port: int = 8765,
            tts: Optional[str] = None, color: Optional[str] = None) -> dict:
        name = (name or "").strip()
        if not NAME_RE.match(name):
            raise ValueError("name: a letter first, then letters, digits or spaces (max 20)")
        rid = robot_id(name)
        if self.get(rid):
            raise ValueError(f"a robot called {name!r} already exists")
        if kind not in ("connects-in", "dial-out"):
            raise ValueError("kind must be connects-in or dial-out")
        if kind == "dial-out" and not (host or "").strip():
            raise ValueError("a dial-out robot needs the host (IP) to dial")
        if color and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("colour must look like #rrggbb")
        used = set(self.colors().values())
        spec = {"id": rid, "name": name, "kind": kind, "host": (host or "").strip(), "port": int(port or 8765),
                "tts": tts or ("navel" if kind == "dial-out" else "none"),
                "color": color or next((c for c in PALETTE if c not in used), PALETTE[len(self.robots) % len(PALETTE)]),
                "builtin": False}
        with self._lock:
            self.robots.append(self._normalise(spec))
            self.save()
        return spec

    def remove(self, rid: str) -> None:
        r = self.get(rid)
        if not r:
            raise ValueError(f"unknown robot {rid!r}")
        if r.get("builtin"):
            raise ValueError(f"{r['name']} is built in and cannot be removed")
        with self._lock:
            self.robots = [x for x in self.robots if x["id"] != rid]
            self.save()

    def set_voice_adjust(self, rid: str, adjust: dict) -> dict:
        r = self.get(rid)
        if not r:
            raise ValueError(f"unknown robot {rid!r}")
        clean = {}
        for k, lim in VOICE_LIMITS.items():
            try:
                v = float(adjust.get(k) or 0)
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be a number")
            clean[k] = max(-lim, min(lim, v))
        with self._lock:
            r["voice_adjust"] = clean
            self.save()
        return clean

    def set_host(self, rid: str, host: str, port: Optional[int] = None) -> None:
        r = self.get(rid)
        if r and host:
            r["host"] = host
            if port:
                r["port"] = int(port)
            self.save()

    # ---- file ----
    @staticmethod
    def _normalise(spec: dict) -> dict:
        spec.setdefault("kind", "connects-in"); spec.setdefault("host", ""); spec.setdefault("port", 8765)
        spec.setdefault("tts", "navel" if spec["kind"] == "dial-out" else "none")
        spec.setdefault("builtin", False)
        spec.setdefault("voice_adjust", {"rate": 0, "pitch": 0, "volume": 0})
        spec.setdefault("color", PALETTE[0])
        return spec

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"robots": self.robots}, fh, indent=2)
        os.replace(tmp, self.path)
