_TIER_OFFSETS: dict[str, tuple[float, float, float]] = {
    "close":   ( 0.0, +0.10, +0.4),
    "family":  ( 0.0, +0.05, +0.2),
    "known":   ( 0.0,  0.0,   0.0),
    "visitor": ( 0.0,  0.0,  -0.2),
    "unknown": ( 0.0,  0.0,  -0.4),
}


def tier_to_pad_offset(tier: str) -> tuple[float, float, float]:
    """Return (dP, dA, dD) offset for a relationship tier.

    Tiers: "close", "family", "known", "visitor", "unknown".
    Falls back to "unknown" for unrecognised values.
    """
    return _TIER_OFFSETS.get(tier, _TIER_OFFSETS["unknown"])
