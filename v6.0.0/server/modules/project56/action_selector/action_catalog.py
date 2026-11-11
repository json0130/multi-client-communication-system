"""
modules/action_selector/action_catalog.py
=========================================
Temporary server-side action description catalog.

Descriptions are grouped by robot client ID so different robots can assign
different physical and semantic meanings to the same action tag.

This file temporarily supplies configuration that will later be loaded from
the robot capability records in Supabase. The action selector depends only on
get_action_descriptions(), allowing the storage implementation to be replaced
without changing scoring logic.
"""

from __future__ import annotations


_DEFAULT_DESCRIPTION = (
    "Remain in a neutral pose. Use only when no expressive action "
    "fits the dialogue."
)


V2_ACTION_DESCRIPTIONS: dict[str, str] = {
    "DEFAULT": (
        "Baseline speaking gesture. Use for ordinary statements, descriptions "
        "and explanations when none of the specific triggers below is clearly present."
    ),
    "POINT": (
        "Use ONLY when the sentence refers to a specific physical thing, place, "
        "person or direction the listener could look at (e.g. \"this screen\", "
        "\"over there\", \"to your right\", \"that table\"). Do not use for "
        "abstract explanations, processes, data, goals or ideas."
    ),
    "EMPHASISE": (
        "Use when the sentence stresses that something is important, a key point, "
        "a warning or a constraint (e.g. \"importantly\", \"the key point is\", "
        "\"keep in mind\", \"be careful\")."
    ),
    "AGREE": "Use when the sentence confirms or answers positively (e.g. \"yes\", \"right\", \"exactly\").",
    "DENY": (
        "Use when the sentence says no, disagrees, corrects the listener or advises "
        "against something (e.g. \"no\", \"I don't think so\", \"that isn't right\", "
        "\"better not\")."
    ),
    "LISTEN": (
        "Use when the robot asks the listener a question or invites them to respond or "
        "ask something (e.g. \"do you want to...?\", \"any questions?\", \"tell me...\")."
    ),
    "UNCERTAIN": (
        "Use only when the speaker is unsure of the answer (e.g. \"I'm not sure\", "
        "\"maybe\", \"it depends\"). A problem or limitation stated as a fact is not uncertainty."
    ),
    "GREET": (
        "Use for greetings, welcomes, thanks for coming and goodbyes (e.g. \"hello\", "
        "\"welcome\", \"nice to meet you\", \"goodbye\")."
    ),
    "HAPPY": (
        "Use for good news, success, praise or excitement (e.g. \"great news\", "
        "\"it worked\", \"well done\")."
    ),
    "SURPRISE": (
        "Use for reacting to something unexpected or remarkable (e.g. \"wow\", "
        "\"surprisingly\", \"I didn't expect that\")."
    ),
    "SAD": (
        "Use for bad news, failure, problems or sympathy (e.g. \"sadly\", "
        "\"it failed\", \"I'm sorry to hear that\")."
    ),
}

# V2.1 is intentionally a one-variable experiment: only POINT differs.
V2_1_ACTION_DESCRIPTIONS: dict[str, str] = {
    **V2_ACTION_DESCRIPTIONS,
    "POINT": (
        "Use when the sentence refers to a specific physical thing, place, "
        "person or direction that is present around the speaker and the listener "
        "could look at (e.g. \"this screen\", \"that machine over there\", \"to your "
        "right\", \"the table behind you\"). Do not use for abstract explanations, "
        "processes, data, goals or ideas."
    ),
}

# TODO: Replace this in-memory catalog with descriptions loaded from
# robot capability records persisted during client registration.
ROBOT_ACTION_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Navel_001": {
        "DEFAULT": _DEFAULT_DESCRIPTION,
        "POINT": (
            "Point towards a person, object, display, or direction "
            "being discussed."
        ),
        "EMPHASISE": (
            "Use an expressive hand or arm gesture to reinforce an "
            "important statement."
        ),
        "AGREE": (
            "Use an affirmative gesture, such as nodding, when "
            "confirming information or expressing agreement."
        ),
        "LISTEN": (
            "Adopt an attentive pose when inviting the user to "
            "continue speaking."
        ),
        "UNCERTAIN": (
            "Use a hesitant gesture when unsure, considering "
            "possibilities, or unable to give a definite answer."
        ),
        "DENY": (
            "Use a negative gesture when refusing, disagreeing, or "
            "correcting information."
        ),
        "GREET": (
            "Raise one hand in a friendly greeting for a welcome, "
            "introduction, hello, or goodbye."
        ),
        "HAPPY": (
            "Use a cheerful or celebratory gesture for praise, "
            "success, excitement, or positive news."
        ),
        "SURPRISE": (
            "React with a surprised gesture when responding to "
            "unexpected information."
        ),
        "SAD": (
            "Use a subdued or sympathetic gesture when responding "
            "to sadness, disappointment, or difficulty."
        ),
    },

    # Example of another robot with different implementations:
    "Pepper_001": {
        "DEFAULT": _DEFAULT_DESCRIPTION,
        "GREET": (
            "Raise the left arm and wave while welcoming or saying "
            "goodbye to a user."
        ),
        "HAPPY": (
            "Raise both arms in a celebratory gesture for positive "
            "news, praise, or success."
        ),
    },
}


def get_action_descriptions(
    robot_id: str,
    tags: list[str],
) -> dict[str, str]:
    """
    Return descriptions for the supplied actions supported by one robot.

    Only requested tags are returned, preventing the scorer from considering
    actions outside the robot's configured allowed-tag list. Unknown tags use
    a generic description until a robot-specific entry is configured.
    """
    robot_descriptions = ROBOT_ACTION_DESCRIPTIONS.get(
        robot_id,
        {},
    )

    result: dict[str, str] = {}

    for raw_tag in tags:
        tag = _normalise(raw_tag)

        if not tag:
            continue

        if tag == "DEFAULT":
            fallback = _DEFAULT_DESCRIPTION
        else:
            fallback = (
                f"Perform the robot action represented by the {tag} tag."
            )

        result[tag] = robot_descriptions.get(tag, fallback)

    return result


def _normalise(tag: str) -> str:
    """Convert '[greet]' and similar values to the canonical 'GREET' form."""
    return tag.strip().strip("[]").upper()
