"""
core/profiles/
==============
Scenario Profile loading and validation.

See core/profiles/registry.py. Profiles live in v6.0.0/server/profiles/*.yaml
and are documented in v6.0.0/README.md.
"""

from core.profiles.registry import (
    ProfileError,
    ProfileRegistry,
    RobotProfileEntry,
    ScenarioProfile,
    parse_profile,
)

__all__ = [
    "ProfileError",
    "ProfileRegistry",
    "RobotProfileEntry",
    "ScenarioProfile",
    "parse_profile",
]
