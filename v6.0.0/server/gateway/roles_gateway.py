"""
gateway/roles_gateway.py
========================
The Roles page: assign a role + OCEAN personality to a robot, watch the role/persona graph, and talk to a
robot as that persona. The persona stack lives in modules/ (graph_relationship, persona_chat, persona_voice,
action_tag_generator); this file connects it to the rest of the server.

Robots are the ones the server already manages. `GatewayHub` gives the persona stack the small hub interface
it was written against (robots / say / say_steps / push_persona) on top of WebSocketGateway, so a role goes
to a robot through the same connection the demo uses and the same client code handles it, for the ChatBox
and for the Navel:

    persona_update  {persona_name, personality, [robot_role, voice_config]}
    demo_step       {step_id, text, require_ack}      text = "[TAG] words"; the robot acks when it has spoken

The page itself is served by a small HTTP server on its own port (`ROLES_PORT`, default 8780); the
dashboard's Roles tab embeds it.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Dict, Optional

from modules.persona_voice import navel_tts_tags, voice_for

# the ChatBox client replaces its whole voice_config on persona_update, so keep its base keys
BASE_VOICE = {"language": "en", "gender": "female", "provider": "gtts"}
# robots whose client knows the `[DEFAULT]` pose tag (the ChatBox's EMOTION_MAP); others just get speech
DEFAULT_POSE_ROBOTS = {"chatbox"}

# The robots of this deployment. A connected robot is matched to one by its robot_name (lower-case, no
# spaces), so the real ones (chatbox_jetson_001, navel_001) and the fake ones (chatbox_01, navel_01) both
# work. A spec may carry an optional "client_id" to pin it to one connection instead.
DEFAULT_ROBOTS = [
    {"id": "chatbox", "name": "ChatBox", "tts": "none", "kind": "connects-in",
     "color": "#ff8c42", "builtin": True},
    {"id": "navel", "name": "Navel", "tts": "navel", "kind": "dial-out",
     "color": "#4fc3f7"},
]


def _name_key(name) -> str:
    return "".join(str(name or "").lower().split())


def ocean_to_personality(ocean: Dict[str, float]) -> Dict[str, float]:
    """the page's OCEAN is -1..1; the robot's `personality` block is 0..1."""
    return {k: round((float(v) + 1) / 2, 3) for k, v in ocean.items()}


class GatewayHub:
    """modules' RobotHub interface over the server's WebSocketGateway (which dials out to every robot)."""

    port = None                 # no hub port of our own: robots already connect to the server
    on_speech = None            # the server's own turn handler transcribes and answers a robot's microphone

    def __init__(self, ws, specs, on_persona: Optional[Callable[[str, str, str], None]] = None,
                 registry=None) -> None:
        self.ws = ws
        self.registry = registry                # the server's robot registry: get(client_id).robot_name
        self.specs = {s["id"]: s for s in specs}
        self.on_persona = on_persona            # (client_id, persona_name, role_brief): e.g. save it as robots.robot_role
        self.links: dict = {}                   # RobotHub parity: the server owns the connections
        self.voices: Dict[str, dict] = {}       # robot id -> voice spec, applied as TTS tags where the robot takes them
        self._step, self._lock = 0, threading.Lock()
        self._acks: Dict[str, threading.Event] = {}
        self._persona_applied: set = set()      # robot ids a persona was pushed to, so the demo can undo it
        ws.ack_listeners.append(self._ack_step)
        ws.demo_start_listeners.append(self.restore_demo_voice)

    def voice_items(self) -> list:
        """What the robots' microphones heard and answered, for the page's /voice feed."""
        return self.ws.voice_items()

    # ---- who is there ----
    def _client_for(self, spec: dict) -> Optional[str]:
        """The connected client_id for a spec: its pinned client_id if that is connected, else the first
        connected robot whose robot_name matches the spec's name."""
        up = self.ws.get_connected_ids()
        if spec.get("client_id") in up:
            return spec["client_id"]
        want = _name_key(spec["name"])
        for cid in up:
            inst = self.registry.get(cid) if self.registry else None
            if inst is not None and _name_key(getattr(inst, "robot_name", "")) == want:
                return cid
        return None

    def _live(self, robot: Optional[str] = None) -> list:
        """the specs whose robot is connected, each with the client_id it is connected as"""
        out = []
        for rid, s in self.specs.items():
            if robot is None or rid == robot:
                cid = self._client_for(s)
                if cid:
                    out.append({**s, "client_id": cid})
        return out

    def robots(self, robot: Optional[str] = None) -> list:
        return [{"robot_name": s["name"], "client_id": s["client_id"], "via": "server"} for s in self._live(robot)]

    def describe(self, rid: str) -> str:
        s = self.specs.get(rid)
        return f"not connected: connect a robot named {s['name']} on the Robots tab" if s else "unknown robot"

    # ---- to the robot ----
    def set_voice(self, robot_id: str, voice: Optional[dict]) -> None:
        if voice:
            self.voices[robot_id] = voice
        else:
            self.voices.pop(robot_id, None)

    def _spoken(self, spec: dict, text: str) -> str:
        voice = self.voices.get(spec["id"])
        return (navel_tts_tags(voice) + text) if (voice and spec.get("tts") == "navel") else text

    def _step_id(self) -> str:
        with self._lock:
            self._step += 1
            return f"persona-{self._step}"             # the prefix routes the robot's ack here, not to the demo

    def _ack_step(self, step_id: str) -> None:
        ev = self._acks.get(step_id)
        if ev:
            ev.set()

    def restore_demo_voice(self) -> None:
        """The lab demo starts: robots a persona was assigned to go back to the base voice, not the persona's.
        A ChatBox gets the neutral voice spec (its client replaces voice_config on persona_update); a Navel
        keeps its own voice, so only the stored persona tags are dropped. Nothing is sent to a robot that no
        persona was assigned to."""
        applied, self._persona_applied = self._persona_applied, set()
        for s in self._live():
            if s["id"] not in applied:
                continue
            self.set_voice(s["id"], None)
            if s.get("tts") != "navel":
                self.ws.send_to_robot(s["client_id"], {
                    "event": "persona_update", "persona_name": "Lab demo",
                    "voice_config": {**BASE_VOICE, **voice_for("", {})}})

    def push_persona(self, persona_name: str, robot_role: str, ocean: Dict[str, float],
                     voice: Optional[Dict[str, str]] = None, robot: Optional[str] = None) -> dict:
        personality = ocean_to_personality(ocean)
        sent = 0
        for s in self._live(robot):
            if s.get("tts") == "navel":
                # a Navel keeps its own role, tags and voice in its client_config; it only takes the name and OCEAN
                payload = {"event": "persona_update", "persona_name": persona_name, "personality": personality}
                if voice:
                    self.set_voice(s["id"], voice)
            else:
                payload = {"event": "persona_update", "persona_name": persona_name, "robot_role": robot_role,
                           "personality": personality}
                if voice:
                    payload["voice_config"] = {**BASE_VOICE, **voice}
            self.ws.send_to_robot(s["client_id"], payload)
            self._persona_applied.add(s["id"])
            sent += 1
            inst = self.registry.get(s["client_id"]) if self.registry else None
            if inst is not None and robot_role:
                inst.assigned_role = robot_role         # what the server talks as outside the demo; the demo uses the DB role
        if self.on_persona:
            for rid, s in self.specs.items():
                if robot is None or rid == robot:
                    cid = self._client_for(s) or s.get("client_id")
                    if cid:
                        self.on_persona(cid, persona_name, robot_role)
        return {"robots": sent, "sent": bool(sent)}

    def say(self, text: str, robot: Optional[str] = None) -> dict:
        step, live = self._step_id(), self._live(robot)
        for s in live:
            self.ws.send_to_robot(s["client_id"], {"event": "demo_step", "step_id": step,
                                                   "text": self._spoken(s, text), "require_ack": False})
        return {"robots": len(live), "sent": bool(live)}

    def say_steps(self, texts, timeout: float = 30.0, robot: Optional[str] = None) -> dict:
        """Speak `texts` one after another, each sent once the robot acked the last (a generator is fine).
        Blocks: call from a thread. A robot that never acks costs `timeout` per step."""
        live, n = self._live(robot), 0
        for text in texts:
            n += 1
            step = self._step_id()
            ev = self._acks[step] = threading.Event()
            for s in live:
                self.ws.send_to_robot(s["client_id"], {"event": "demo_step", "step_id": step,
                                                       "text": self._spoken(s, text), "require_ack": True})
            if live:
                ev.wait(timeout)
            self._acks.pop(step, None)
        if n:       # speech is over: a tag-only step so a robot that knows the pose goes back to default
            for s in live:
                if s["id"] in DEFAULT_POSE_ROBOTS:
                    self.ws.send_to_robot(s["client_id"], {"event": "demo_step", "step_id": self._step_id(),
                                                           "text": "[DEFAULT]", "require_ack": False})
        return {"robots": len(live), "sent": bool(live), "steps": n}


def start_roles_service(ws, data_dir: str, port: int = 8780, host: str = "0.0.0.0",
                        ai_server: str = "", ollama_model: str = "qwen2.5:7b",
                        ollama_url: str = "http://127.0.0.1:11434",
                        action_tags: bool = True, on_persona=None, specs=None, llm=None,
                        server_registry=None):
    """Build the persona stack on `ws` (the WebSocketGateway) and serve the Roles page in a thread.
    `server_registry` is the server's robot registry, used to match connected robots by name.
    Returns (server, hub). Graph and persona files are kept in `data_dir`."""
    from http.server import ThreadingHTTPServer

    from modules.graph_relationship.viz.server import (
        GraphState, HistoryProvider, PRIMARY, make_handler, seed_robot_graph)
    from modules.persona_chat import OllamaChat, PersonaChat, ai_server_chat
    from modules.role_registry import RobotRegistry

    data_dir = os.path.abspath(data_dir)       # the persona store resolves a relative path against the graph's folder
    os.makedirs(data_dir, exist_ok=True)
    specs = specs or DEFAULT_ROBOTS
    registry = RobotRegistry(os.path.join(data_dir, "robots.json"), [dict(s) for s in specs])
    llm = llm or (ai_server_chat(ai_server) if ai_server else OllamaChat(ollama_model, ollama_url))
    hub = GatewayHub(ws, specs, on_persona, server_registry)

    def space(spec):
        rid = spec["id"]
        path = os.path.join(data_dir, "kg_culture.json" if rid == PRIMARY else f"kg_{rid}.json")
        seed_robot_graph(path, rid)             # the robot + the role library; a graph that exists is kept
        st = GraphState(path, os.path.join(data_dir, f"persona_assignment{'' if rid == PRIMARY else '_' + rid}.json"))
        return st, PersonaChat(path, st._persona_path(), llm)

    spaces = {s["id"]: space(s) for s in specs}
    state, chat = spaces.pop(PRIMARY)
    tagger = None
    if action_tags:
        from modules.action_tag_generator import ActionTagGenerator, ChatLLM, navel_tagger
        tagger = {PRIMARY: ActionTagGenerator(ChatLLM(llm))}
        if "navel" in spaces:
            tagger["navel"] = navel_tagger(ChatLLM(llm))
    handler = make_handler(state, HistoryProvider(os.path.join(data_dir, "sessions.db")), chat, hub, tagger,
                           None, spaces, registry, None, None)
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="roles-page").start()
    print(f"[Roles] role / persona page on http://{host}:{port}/")
    return server, hub
