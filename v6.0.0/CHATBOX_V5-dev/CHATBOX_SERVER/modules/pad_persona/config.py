from dataclasses import dataclass, field


@dataclass
class OceanTraits:
    o: float  # Openness          [0, 1]
    c: float  # Conscientiousness [0, 1]
    e: float  # Extraversion      [0, 1]
    agreeableness: float           # [0, 1]
    n: float  # Neuroticism       [0, 1]


@dataclass
class PersonaConfig:
    robot_id: str
    ocean: OceanTraits


@dataclass
class PADWeights:
    w_user: float = 0.4       # weight of live affect-stream offset
    w_rel: float = 0.3        # weight of relationship-tier offset
    alpha_decay: float = 0.15  # per-turn decay back toward baseline


# OCEAN values on [0,1] scale — converted from the design table (−1→+1) via (v+1)/2.
# Design table (−1→+1):  CHATBOX  O=−0.3  C=+0.2  E=−0.5  A=+0.6  N=−0.5
#                         ELLEBOT  O=+0.4  C=+0.5  E=+0.6  A=+0.6  N=−0.2

# ChatBox: calm home companion — low extraversion, highly agreeable, low neuroticism
CHATBOX_PERSONA = PersonaConfig(
    robot_id="chatbox",
    ocean=OceanTraits(o=0.35, c=0.60, e=0.25, agreeableness=0.80, n=0.25),
)

# ElleBot: outgoing mobile helper — high extraversion, high conscientiousness
ELLEBOT_PERSONA = PersonaConfig(
    robot_id="ellebot",
    ocean=OceanTraits(o=0.70, c=0.75, e=0.80, agreeableness=0.80, n=0.40),
)
