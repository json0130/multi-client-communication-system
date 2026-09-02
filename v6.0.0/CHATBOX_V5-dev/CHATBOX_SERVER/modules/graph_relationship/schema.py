"""
Dual-cluster relational knowledge graph schema for the CHATBOX multi-robot system.

Architecture overview
---------------------
The graph holds two entity clusters:

  Robot cluster  — one node per robot (CHATBOX cat, elephant).
                   Robots own *Relationship* edges that track the evolving bond
                   with each child (rapport, trust, disclosure depth, etc.).

  Person cluster — one node per child user plus Topic/Event nodes that
                   describe the child's current mental state and history.
                   *PersonAttribute* edges attach state (mood, attention, …)
                   and stable traits/preferences to the person.

Both robots share ONE graph instance so they read the same person/relationship
data. Every edge carries a Provenance record (source + timestamp + confidence)
so attribution and later reliability-weighting are always available.

Edge types are tagged FAST or SLOW (Timescale enum).  The update-policy module
(built separately) branches on this flag to decide decay cadence and promotion
thresholds.  Nothing in this module performs updates — schema only.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Embodiment(str, Enum):
    CAT = "CAT"
    ELEPHANT = "ELEPHANT"


class TopicCategory(str, Enum):
    """CLOSED taxonomy for TopicNode.category. 'other' is the explicit fallback —
    never invent a category outside this set. Defined ONCE here; the app-layer
    extractor validates LLM output against TOPIC_CATEGORIES (its value set)."""
    MUSIC = "music"
    SCIENCE = "science"
    ANIMALS = "animals"
    FOOD = "food"
    ACTIVITY = "activity"
    PLACE = "place"
    PERSON = "person"
    MEDIA = "media"
    SPORT = "sport"
    OTHER = "other"


# Value set for O(1) membership checks / enum-constrained validation.
TOPIC_CATEGORIES: frozenset = frozenset(c.value for c in TopicCategory)


class Timescale(str, Enum):
    """Decay cadence hint consumed by the update-policy module."""
    FAST = "FAST"   # mood, attention — decay within a session
    SLOW = "SLOW"   # traits, preferences — stable across sessions


# ---------------------------------------------------------------------------
# Provenance — required on every edge
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Who wrote this edge, when, and how confident."""
    source: str = Field(
        ...,
        description="Writer identifier, e.g. 'robot:cat', 'robot:elephant', 'sensor:emotion'",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class RobotNode(BaseModel):
    """One node per robot in the system (CAT / ELEPHANT)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    embodiment: Embodiment
    persona_traits: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    node_type: Literal["robot"] = "robot"


class PersonNode(BaseModel):
    """One node per child user."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    display_name: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_type: Literal["person"] = "person"


class TopicNode(BaseModel):
    """A concept or subject that a person engages with.

    `notes` accumulates short per-person conversation summaries about this topic
    (extracted from sessions), so the topic node holds a summary of interactions:
        {"person": str, "text": str, "ts": iso8601}
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    # Fine-grained type from the CLOSED TopicCategory taxonomy. Defaults to
    # "other" so pre-existing kg_state.json topics (no category field) still load.
    # NOTE: `category` is an ATTRIBUTE, not part of identity — the topic id is
    # derived from the normalized LABEL only (see topics.topic_id), so two
    # extractions disagreeing on category still resolve to the SAME node.
    category: TopicCategory = TopicCategory.OTHER
    notes: List[Dict[str, Any]] = Field(default_factory=list)
    node_type: Literal["topic"] = "topic"


class InteractionNode(BaseModel):
    """The SINGLE abstraction of a person↔robot relationship.

    One InteractionNode per (person, robot) pair; both link to it. It holds the
    aggregate relationship state — how close they are and how much they have
    interacted — and its Session subnodes hold the actual conversation history:

        person --has_interaction--> Interaction <--has_interaction-- robot
        Interaction --has_session--> Session {turns...}
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rapport: float = 0.0
    trust: float = 0.0
    interaction_count: int = 0        # total turns across all sessions
    node_type: Literal["interaction"] = "interaction"


class SessionNode(BaseModel):
    """A conversation SESSION (a meetup), hanging under an InteractionNode.

    Each turn of the meetup is appended to `turns`, so the node holds the
    session's conversation history. A later meetup creates a new SessionNode.

    `turns` entries look like:
        {"turn": int, "emotion": str|None, "child": str|None,
         "reply": str|None, "ts": iso8601}
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    turns: List[Dict[str, Any]] = Field(default_factory=list)
    # How many turns have already been fed to knowledge extraction, so a later
    # extract only processes NEW turns of this session (not the whole history).
    extracted_turns: int = 0
    node_type: Literal["session"] = "session"


# ---------------------------------------------------------------------------
# Authored-attribute nodes  (identity subnodes for Robot / Person anchors)
# ---------------------------------------------------------------------------
# These carry a single authored `descriptor` string and hang off an anchor
# (Robot or Person) via a Has* edge.  They describe stable identity — persona,
# role, conversational style, capabilities — seeded from spec files, not
# learned per turn.

class PersonaNode(BaseModel):
    """Authored persona/character descriptor, e.g. 'introverted, shy'."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    descriptor: str
    node_type: Literal["persona"] = "persona"


class RoleNode(BaseModel):
    """Authored role descriptor, e.g. 'companion'."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    descriptor: str
    node_type: Literal["role"] = "role"


class CapabilityNode(BaseModel):
    """A robot's capabilities as ONE node holding a list of items, e.g.
    ['tells stories', 'knows jazz', 'good at math'].

    When a capability item matches a Topic (keyword now, embedding later), the
    node gets an about-edge to that shared Topic whose `label` records the
    matching item ('knows jazz'):  robot --has_capability--> Capability
    --about[label='knows jazz']--> Topic."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: List[str] = Field(default_factory=list)
    node_type: Literal["capability"] = "capability"


class InterestNode(BaseModel):
    """A human's interest area, e.g. 'music'. Bridges a person to a shared
    TopicNode: person --has_interest--> Interest --about--> Topic."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    node_type: Literal["interest"] = "interest"


class ConversationNode(BaseModel):
    """Live conversation-status node — a rolling list of the most recent topic
    keywords plus the current mood/emotion.

    FAST/transient and, crucially, NOT a shared TopicNode: it is never matched to
    capabilities or interests, so it cannot pick up spurious about-edges. One per
    person↔robot:  person --has_conversation--> Conversation <--has_conversation-- robot.
    `topics` is updated in place (most-recent last, capped) rather than recreating
    a node per topic.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topics: List[str] = Field(default_factory=list)   # rolling, most-recent last
    mood: Optional[float] = None                       # current valence, -1..+1
    emotion: Optional[str] = None                      # current emotion label
    node_type: Literal["conversation"] = "conversation"


class CultureNode(BaseModel):
    """A cultural background, e.g. 'Korean'. This is the ROBOT's prior knowledge:
    a robot links to it via KnowsCultureEdge, and the culture carries soft priors
    over its OWN CultureTopicNodes (CulturePriorEdge). A person may (manually,
    never auto-detected) be tagged with ONE via BelongsToCultureEdge — a starting
    guess about their background, never a fact about them.

    Deterministic id `culture:<normalized-label>` (same slug as TopicNode) so
    re-seeding the same culture resolves to the SAME node.

    `style_hint` is a single short, STATIC "how to talk" paragraph (manner/politeness)
    for this culture — hand-written seed data, the same for every interaction. It is
    the manner half of cultural adaptation (the topic priors are the content half).
    Default "" → old graphs load unchanged and inject nothing. Deliberately dumb: no
    tier/affect/situation variation (that is Approach 2's policy vector).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    style_hint: str = ""
    node_type: Literal["culture"] = "culture"


class CultureTopicNode(BaseModel):
    """A topic that belongs to the ROBOT's cultural knowledge, e.g. 'kimchi' under
    Korean. Deliberately SEPARATE from the shared person-interest TopicNode so a
    culture's background knowledge never couples unrelated people together (person
    A's `topic:hiking` stays distinct from `ck:korean:hiking`). A person links to a
    real TopicNode only by actually discussing it; the culture layer never writes
    person→topic edges.

    Deterministic id `ck:<culture-slug>:<topic-slug>`. `category` mirrors
    TopicCategory for viz colouring. Not touched by topic consolidation / interest
    machinery (different node_type), so it can't merge into a person topic.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    category: TopicCategory = TopicCategory.OTHER
    # Short cultural facts the robot can share when it brings up this topic, e.g.
    # esports → ["Korea is the heart of competitive gaming (StarCraft, LoL)"].
    facts: List[str] = Field(default_factory=list)
    node_type: Literal["culture_topic"] = "culture_topic"


# Union of all node types (discriminated on node_type for (de)serialisation)
AnyNode = Union[
    RobotNode, PersonNode, TopicNode,
    PersonaNode, RoleNode, CapabilityNode, InterestNode,
    InteractionNode, SessionNode, ConversationNode,
    CultureNode, CultureTopicNode,
]


# ---------------------------------------------------------------------------
# Edge base — provenance required on all edges
# ---------------------------------------------------------------------------

class EdgeBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    provenance: Provenance


# ---------------------------------------------------------------------------
# Relationship edges  (Robot → Person)
# ---------------------------------------------------------------------------

class RapportEdge(EdgeBase):
    """Perceived warmth / positive affect between robot and child."""
    edge_type: Literal["rapport"] = "rapport"
    weight: float = Field(..., ge=0.0, le=1.0)


class TrustEdge(EdgeBase):
    """Child's willingness to rely on the robot."""
    edge_type: Literal["trust"] = "trust"
    weight: float = Field(..., ge=0.0, le=1.0)


class InteractionCountEdge(EdgeBase):
    """Running count of completed turns / exchanges."""
    edge_type: Literal["interaction_count"] = "interaction_count"
    count: int = Field(..., ge=0)


RelationshipEdge = Union[
    RapportEdge, TrustEdge, InteractionCountEdge
]


# ---------------------------------------------------------------------------
# Person-attribute edges  (Person → Topic | scalar state)
# ---------------------------------------------------------------------------

class MoodEdge(EdgeBase):
    """Current affective state — FAST, decays between sessions.

    `label` optionally carries the source emotion name (e.g. "happy") so the
    live visualizer can show the current emotion alongside the valence.
    """
    edge_type: Literal["mood"] = "mood"
    value: float = Field(..., ge=-1.0, le=1.0)  # valence: -1 sad … +1 happy
    label: Optional[str] = None                 # emotion label, e.g. "happy"
    timescale: Timescale = Timescale.FAST


class AttentionEdge(EdgeBase):
    """Estimated engagement level — FAST."""
    edge_type: Literal["attention"] = "attention"
    value: float = Field(..., ge=0.0, le=1.0)
    timescale: Timescale = Timescale.FAST


class TraitEdge(EdgeBase):
    """Stable personality/character attribute — SLOW."""
    edge_type: Literal["trait"] = "trait"
    value: Any
    timescale: Timescale = Timescale.SLOW


class PreferenceEdge(EdgeBase):
    """Person → Topic affinity that persists across sessions — SLOW."""
    edge_type: Literal["preference"] = "preference"
    weight: float = Field(..., ge=0.0, le=1.0)
    timescale: Timescale = Timescale.SLOW


PersonAttributeEdge = Union[
    MoodEdge, AttentionEdge, TraitEdge, PreferenceEdge
]


# ---------------------------------------------------------------------------
# Identity edges  (Robot | Person anchor → authored-attribute node)
# ---------------------------------------------------------------------------
# Authored, cross-session identity. SLOW timescale; they follow the standard
# replace-on-newer merge rule (NOT accumulate) — re-seeding a spec overwrites.

class HasPersonaEdge(EdgeBase):
    """anchor → PersonaNode."""
    edge_type: Literal["has_persona"] = "has_persona"
    timescale: Timescale = Timescale.SLOW


class HasRoleEdge(EdgeBase):
    """anchor → RoleNode."""
    edge_type: Literal["has_role"] = "has_role"
    timescale: Timescale = Timescale.SLOW


class HasCapabilityEdge(EdgeBase):
    """robot → CapabilityNode (the capability node holding the items list)."""
    edge_type: Literal["has_capability"] = "has_capability"
    timescale: Timescale = Timescale.SLOW


IdentityEdge = Union[
    HasPersonaEdge, HasRoleEdge, HasCapabilityEdge
]


# ---------------------------------------------------------------------------
# Topic / Interest edges  (shared-TopicNode layer)
# ---------------------------------------------------------------------------
# A single TopicNode is reached from both sides THROUGH a subnode:
#   robot  --has_capability--> Capability --about--> Topic
#   person --has_interest-->   Interest   --about--> Topic
# All authored, SLOW, replace-on-newer.

class HasInterestEdge(EdgeBase):
    """person → InterestNode."""
    edge_type: Literal["has_interest"] = "has_interest"
    timescale: Timescale = Timescale.SLOW


class AboutEdge(EdgeBase):
    """subnode (Interest | Capability) → TopicNode: about this shared topic.

    `label` records which capability produced a robot→topic link, e.g.
    'knows jazz'. Left None for a person Interest → Topic edge.

    On a person's Interest → Topic edge this is the "observed" evidence the
    preference BN and the person-memory prompt both read:
      * `affinity`   — how positively the person feels about the topic, stored
        internally in [0,1] (0.0 dislike / 0.5 neutral / 1.0 like) so it drops
        straight into the BN clamp. The human-facing scale is 0–10; convert ONLY
        at the boundary via scales.aff01_from_10 / aff10_from_01. Default 0.5
        (neutral) so pre-existing edges load as neutral.
      * `confidence` — how sure the reading is, in [0,1]. Feeds prompt hedging
        ("clearly" vs "possibly"); does NOT weight the BN clamp in this step.
        Default 1.0 (fully trusted) so pre-existing edges load unchanged.
    Robot Capability → Topic edges simply carry the neutral defaults (unused).
    """
    edge_type: Literal["about"] = "about"
    label: Optional[str] = None
    affinity: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timescale: Timescale = Timescale.SLOW


class RelatedTopicEdge(EdgeBase):
    """Topic ↔ Topic semantic relation — distinct but related (e.g. rap ~ hiphop,
    tennis ~ basketball). SLOW; conceptually undirected, stored once with the two
    endpoints sorted. `weight` is the embedding similarity."""
    edge_type: Literal["related_topic"] = "related_topic"
    weight: float = Field(..., ge=0.0, le=1.0)
    timescale: Timescale = Timescale.SLOW


TopicEdge = Union[HasInterestEdge, AboutEdge, RelatedTopicEdge]


# ---------------------------------------------------------------------------
# Interaction / Session edges
# ---------------------------------------------------------------------------
# Person and Robot connect through ONE InteractionNode (has_interaction from
# each). The interaction's Session subnodes (conversation history) hang off it
# via has_session. This abstracts the whole relationship into one node.

class HasInteractionEdge(EdgeBase):
    """participant (Person | Robot) → InteractionNode."""
    edge_type: Literal["has_interaction"] = "has_interaction"


class HasSessionEdge(EdgeBase):
    """InteractionNode → SessionNode (a meetup's conversation)."""
    edge_type: Literal["has_session"] = "has_session"


class HasConversationEdge(EdgeBase):
    """participant (Person | Robot) → ConversationNode — FAST live status link."""
    edge_type: Literal["has_conversation"] = "has_conversation"
    timescale: Timescale = Timescale.FAST


InteractionEdge = Union[HasInteractionEdge, HasSessionEdge, HasConversationEdge]


# ---------------------------------------------------------------------------
# Culture edges  (Robot → Culture → CultureTopic ;  Person → Culture)
# ---------------------------------------------------------------------------
# The culture layer is the ROBOT's prior knowledge:
#   robot --knows_culture--> Culture --culture_prior--> CultureTopic
# A person is only TAGGED with a culture (manual), never wired to its topics:
#   person --belongs_to_culture--> Culture
# All SLOW. Priors are authored/seeded starting guesses — NOT per-person state.

class KnowsCultureEdge(EdgeBase):
    """robot → CultureNode. The robot holds this culture as background knowledge.
    SLOW, idempotent — one per robot-culture pair (replace-on-newer)."""
    edge_type: Literal["knows_culture"] = "knows_culture"
    timescale: Timescale = Timescale.SLOW


class BelongsToCultureEdge(EdgeBase):
    """person → CultureNode. Manual assignment only (no auto-detection). SLOW,
    idempotent — one per person-culture pair (replace-on-newer). Tags a person
    with a background; does NOT link them to any of the culture's topics."""
    edge_type: Literal["belongs_to_culture"] = "belongs_to_culture"
    timescale: Timescale = Timescale.SLOW


class CulturePriorEdge(EdgeBase):
    """CultureNode → CultureTopicNode soft prior in [0,1] — how likely someone
    from this background engages with the topic (a starting guess, not a person's
    state). SLOW; one per culture-topic pair (upsert replaces the prior value)."""
    edge_type: Literal["culture_prior"] = "culture_prior"
    prior: float = Field(..., ge=0.0, le=1.0)
    timescale: Timescale = Timescale.SLOW


CultureEdge = Union[KnowsCultureEdge, BelongsToCultureEdge, CulturePriorEdge]


AnyEdge = Union[
    RelationshipEdge, PersonAttributeEdge, IdentityEdge, TopicEdge,
    InteractionEdge, CultureEdge,
]
