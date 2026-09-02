"""
App-layer seed for the culture DEMO layer (Command A).

Seeds ONE culture ("Korean") as the ROBOT's prior knowledge:
    robot --knows_culture--> Korean --culture_prior--> CultureTopic(...)

Culture topics are the robot's OWN nodes (`ck:korean:<slug>`), deliberately kept
SEPARATE from shared person-interest topics — so seeding the culture never touches
or couples any person. A person is tagged with the culture only via a separate,
manual `assign_person_culture` step.

Seeding is idempotent (deterministic ids).

NOTE: the prior values below are HAND-SET PLACEHOLDER DEMO DATA — a rough starting
guess for prompt/plumbing testing, NOT research claims about any group.
"""

from __future__ import annotations

from typing import List, Tuple

from modules.graph_relationship.cultures import (
    ensure_culture, ensure_culture_topic, knows_culture, assign_culture,
    set_culture_prior,
)
from modules.graph_relationship.store import GraphStore

_CULTURE_LABEL = "Korean"
_DEFAULT_ROBOT = "chatbox"

# HAND-WRITTEN DEMO SEED — the static "how to talk" manner hint for this culture, NOT
# a research claim. Same text for every interaction (no tier/affect/situation
# variation — that is Approach 2). Injected into HOW-TO-REPLY as soft, secondary
# guidance when a person is tagged to this culture.
_KOREAN_STYLE_HINT = (
    "Be polite, warm, and a little formal, especially early on. Addressing elders or "
    "new acquaintances respectfully is valued. Compliments are sometimes deflected out "
    "of modesty — offer them once and don't insist. Sharing food and small gestures of "
    "care read as friendly."
)

# (label, category, DUMMY prior in [0,1], [facts...]) — demo placeholders, not
# research claims. `facts` are short, shareable bits the robot can mention when it
# brings the topic up (kept general and light, never asserted about the person).
_KOREAN_DEMO: List[Tuple[str, str, float, List[str]]] = [
    ("kimchi",        "food",     0.60, [
        "Korea's iconic fermented vegetable dish, usually napa cabbage with chilli.",
        "There are hundreds of varieties and it's served with almost every meal."]),
    ("korean bbq",    "food",     0.50, [
        "Meat grilled right at your table — a social, shared way to eat.",
        "Often wrapped in lettuce with garlic and side dishes (banchan)."]),
    ("kpop",          "music",    0.50, [
        "Korean pop music known for polished songs and synchronised dance.",
        "Groups like BTS, BLACKPINK, TWICE and NewJeans have huge global fandoms."]),
    ("kdrama",        "media",    0.45, [
        "Korean TV dramas with big international followings via streaming.",
        "Genres span romance, thriller and historical (sageuk)."]),
    ("bibimbap",      "food",     0.45, [
        "A rice bowl topped with seasoned vegetables, egg and gochujang.",
        "You mix everything together before eating."]),
    ("hiking",        "activity", 0.40, [
        "Extremely popular in Korea — mountains are everywhere, even in Seoul.",
        "Weekend hiking clubs and well-marked trails are common."]),
    ("son heung-min", "person",   0.35, [
        "Korean footballer, a captain at Tottenham Hotspur and a national hero.",
        "One of the Premier League's standout forwards."]),
    ("noraebang",     "activity", 0.35, [
        "Korean karaoke — private singing rooms rented by the hour.",
        "A staple social outing with friends or after dinner."]),
    ("chuseok",       "activity", 0.30, [
        "Korea's harvest/thanksgiving holiday, a major family gathering.",
        "People share songpyeon (rice cakes) and honour ancestors."]),
    ("esports",       "activity", 0.30, [
        "Korea is the heart of competitive gaming — StarCraft and League of Legends.",
        "Pro leagues, star players and PC bangs (gaming cafés) are part of the culture."]),
    ("baseball",      "sport",    0.30, [
        "One of Korea's most popular sports, with the lively KBO league.",
        "Games are famous for organised cheering, chants and fan songs."]),
    ("taekwondo",     "sport",    0.25, [
        "A Korean martial art and the national sport, now an Olympic event.",
        "Known for fast, high kicks."]),
]

# HAND-WRITTEN DEMO SEED for Māori — placeholder data for prompt/plumbing testing,
# NOT research claims about any group. Written respectfully and kept general.
_MAORI_STYLE_HINT = (
    "Be warm and relational — connection and hospitality (manaakitanga) matter, and "
    "so does respect for elders (kaumātua). A friendly greeting like 'kia ora' is "
    "welcome, and acknowledging where someone is from helps build trust. Value "
    "humility over boasting, and share generously."
)

_MAORI_DEMO: List[Tuple[str, str, float, List[str]]] = [
    ("haka",          "activity", 0.55, [
        "A ceremonial group posture dance for welcome, respect, or challenge.",
        "Famously performed by the All Blacks before matches."]),
    ("kapa haka",     "activity", 0.50, [
        "Māori performing arts — group song, dance, and haka.",
        "Regional and national competitions are a big deal."]),
    ("waiata",        "music",    0.50, [
        "Māori song, often sung together to support a speaker or share feeling.",
        "Waiata frequently follow speeches on the marae."]),
    ("marae",         "place",    0.55, [
        "A communal meeting ground — the heart of a Māori community.",
        "Visitors are welcomed with a pōwhiri (welcome ceremony)."]),
    ("te reo maori",  "other",    0.55, [
        "The Māori language; everyday words like 'kia ora' are widely used.",
        "It's an official language of Aotearoa New Zealand."]),
    ("hangi",         "food",     0.50, [
        "A feast cooked in an earth oven with heated stones.",
        "Shared kai (food) is central to hospitality."]),
    ("kai",           "food",     0.45, [
        "Kai means food — sharing it is a core part of manaakitanga (care).",
        "Seafood (kaimoana) features a lot."]),
    ("rugby",         "sport",    0.55, [
        "Hugely popular in Aotearoa; the All Blacks are iconic.",
        "Many communities are built around local clubs."]),
    ("pounamu",       "other",    0.40, [
        "Greenstone/jade — a treasure (taonga), often worn or gifted.",
        "Different shapes carry different meanings."]),
    ("pepeha",        "other",    0.45, [
        "A way of introducing yourself through your mountain, river, and people.",
        "It places you in relationship to others and the land."]),
    ("kumara",        "food",     0.35, [
        "Sweet potato — a staple food.",
        "A traditional crop grown for centuries."]),
    ("matariki",      "activity", 0.35, [
        "The Māori new year, marked by the rising of the Matariki star cluster.",
        "A time to remember those who've passed and plan for the year ahead."]),
]

# Registry of demo cultures: label -> (topic rows, style hint).
_CULTURES: dict = {
    "Korean": (_KOREAN_DEMO, _KOREAN_STYLE_HINT),
    "Maori":  (_MAORI_DEMO,  _MAORI_STYLE_HINT),
}


def _seed_culture(store: GraphStore, label: str, demo, style_hint: str, *,
                  robot_id: str = _DEFAULT_ROBOT, source: str = "culture-seed") -> dict:
    """Generic culture seeder: seed `label` as `robot_id`'s prior knowledge + its demo
    topic priors + its static style_hint. Idempotent. Touches no person / no shared
    person-interest topic. Returns {'culture','robot','topics','priors'}."""
    cnode = ensure_culture(store, label)
    # Set/overwrite the static manner hint idempotently (same text every seed).
    if cnode.style_hint != style_hint:
        cnode = cnode.model_copy(update={"style_hint": style_hint})
        store.upsert_node(cnode)
    if store.get_node(robot_id) is not None:
        knows_culture(store, robot_id, cnode.id, source=source)
    for topic_label, category, prior, facts in demo:
        ct = ensure_culture_topic(store, cnode.id, topic_label, category=category,
                                  facts=facts)
        set_culture_prior(store, cnode.id, ct.id, prior, source=source)
    return {
        "culture": cnode.id,
        "robot":   robot_id if store.get_node(robot_id) is not None else None,
        "topics":  len(demo),
        "priors":  len(demo),
    }


def seed_korean_demo(store: GraphStore, *, robot_id: str = _DEFAULT_ROBOT,
                     source: str = "culture-seed") -> dict:
    """Seed the Korean culture demo (see _seed_culture). Idempotent."""
    return _seed_culture(store, "Korean", _KOREAN_DEMO, _KOREAN_STYLE_HINT,
                         robot_id=robot_id, source=source)


def seed_maori_demo(store: GraphStore, *, robot_id: str = _DEFAULT_ROBOT,
                    source: str = "culture-seed") -> dict:
    """Seed the Māori culture demo (see _seed_culture). Idempotent."""
    return _seed_culture(store, "Maori", _MAORI_DEMO, _MAORI_STYLE_HINT,
                         robot_id=robot_id, source=source)


def seed_all_cultures(store: GraphStore, *, robot_id: str = _DEFAULT_ROBOT,
                      source: str = "culture-seed") -> dict:
    """Seed EVERY demo culture the robot knows (Korean + Māori). Idempotent.
    Returns {label: seed_info}. The robot knows all of them; which one is ACTIVE for a
    given person is resolved at prompt time (override → person's tag → generic)."""
    return {label: _seed_culture(store, label, demo, hint,
                                 robot_id=robot_id, source=source)
            for label, (demo, hint) in _CULTURES.items()}


def assign_person_culture(store: GraphStore, person_id: str, culture_label: str,
                          *, source: str = "culture-seed") -> str:
    """Manually tag a person with a culture (creating the culture if needed).
    Returns the culture_id. The person node must already exist in the store.
    Does NOT link the person to any culture topics."""
    cnode = ensure_culture(store, culture_label)
    assign_culture(store, person_id, cnode.id, source=source)
    return cnode.id
