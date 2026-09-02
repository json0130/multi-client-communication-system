"""
OCEAN → PAD baseline.

Equations (updated):
  P  =  0.21·E + 0.59·A + 0.19·N
  Ar =  0.15·O + 0.30·A − 0.57·N
  D  =  0.25·O + 0.17·C + 0.60·E − 0.32·A

O, C, E, A, N are on the [−1, +1] scale.
Config stores traits in [0, 1]; they are mapped to [−1, +1] via (v − 0.5) × 2
before the equations are applied.  PAD output is clamped to [−1, 1].
"""

from .config import OceanTraits


def ocean_to_baseline_pad(ocean: OceanTraits) -> tuple[float, float, float]:
    # Map [0, 1] → [−1, +1]
    o = (ocean.o             - 0.5) * 2.0
    c = (ocean.c             - 0.5) * 2.0
    e = (ocean.e             - 0.5) * 2.0
    a = (ocean.agreeableness - 0.5) * 2.0
    n = (ocean.n             - 0.5) * 2.0

    pleasure  =  0.21 * e + 0.59 * a + 0.19 * n
    arousal   =  0.15 * o + 0.30 * a - 0.57 * n
    dominance =  0.25 * o + 0.17 * c + 0.60 * e - 0.32 * a

    clamp = lambda v: max(-1.0, min(1.0, v))
    return clamp(pleasure), clamp(arousal), clamp(dominance)


if __name__ == "__main__":
    from .config import CHATBOX_PERSONA, ELLEBOT_PERSONA

    for persona in (CHATBOX_PERSONA, ELLEBOT_PERSONA):
        p, a, d = ocean_to_baseline_pad(persona.ocean)
        print(f"{persona.robot_id:10s}  P={p:+.3f}  A={a:+.3f}  D={d:+.3f}")
