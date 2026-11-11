"""What the Navel graph starts with: the robot and the topics it declares. Nothing else.

Facts come from the robot-robot-interaction repo (branch feat/navel-demo-client, commit dad21e4):
lab-demo-server/data/snapshot (navel_001's 7 declared topics); the gesture tags and connection facts
below come from navel-client/client_config.json and navel-client/client.py. They are kept here as plain
data (a robot client / the viz can read them) and are NOT graph nodes: the graph is the robot and its topics.

    robot --about--> topic        (the ChatBox graph shares none of it)
"""
from __future__ import annotations

from typing import Dict

from modules.graph_relationship.schema import AboutEdge, Provenance
from modules.graph_relationship.topics import resolve_topic

SOURCE = "spec:navel_profile"

TOPICS = ["co-speech gesture", "gaze and head coordination", "gesture selection",
          "speech and gesture timing", "non verbal interaction", "robot to robot communication",
          "speech segmentation"]

# Navel's own gesture tags (client_config.json allowed_tags / tag_descriptions): it performs these, not the ChatBox's
ACTIONS: Dict[str, str] = {
    "DEFAULT": "neutral pose, when no expressive action fits",
    "POINT": "point towards a person, object, display or direction",
    "EMPHASISE": "hand or arm gesture to reinforce an important statement",
    "AGREE": "affirmative gesture such as a nod, to confirm or agree",
    "LISTEN": "attentive pose that invites the person to keep talking",
    "UNCERTAIN": "hesitant gesture when unsure or unable to give a definite answer",
    "DENY": "negative gesture to refuse, disagree or correct",
    "GREET": "raise one hand in a friendly greeting: welcome, hello, goodbye",
    "HAPPY": "cheerful or celebratory gesture for praise, success or good news",
    "SURPRISE": "surprised reaction to unexpected information",
    "SAD": "subdued, sympathetic gesture for sadness or difficulty",
}

# How to reach it. NOTE the direction: Navel does NOT dial out; it runs a server and waits to be dialled.
CONNECTION: Dict[str, str] = {
    "websocket": "Navel runs a WebSocket server on port 8765 and waits; the server dials out to it (navel-client/client.py)",
    "register": "on start it POSTs {server_url}/robots/register (its IP and ws_port) as client_id navel_001, robot_name Navel",
    "server": "server_url in navel-client/client_config.json: the lab demo / central server, port 5000 (run one of them)",
    "planner": "Method B planner service, POST /method_b/plan on port 8001 (navel-server/tools/method_b_plan_service.py)",
    "messages": "JSON messages with an event field: persona_update, demo_step, chat_response, speech_response, tts_stop; it sends ack, speech, tts_state",
}


def seed_navel(store) -> None:
    """Idempotent: topic ids are derived from the label, so running it again changes nothing."""
    for label in TOPICS:
        topic = resolve_topic(store, label)
        store.upsert_edge(AboutEdge(source_id="navel", target_id=topic.id,
                                    provenance=Provenance(source=SOURCE, confidence=1.0)))
