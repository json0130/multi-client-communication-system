"""
core/profiles/registry.py
=========================
Scenario Profile loading and validation.

A Scenario Profile declares, per deployment, which robots take part and what
each one's Social Identity is — role, persona and, for RBAC, its access level.
The paper's point is that access hierarchies are adjustable at the database
level without reconfiguring individual robot nodes: the profile is the
declarative source, and ProfileRegistry.sync_to_db() reconciles the `robots`
table with it at boot.

Profiles are per-deployment and loaded once at boot. There is deliberately no
runtime hot-swapping of access levels — a level that can change mid-session is a
level you cannot audit.

Validation is fail-fast and happens at boot, not at first request:
  - unknown access level
  - unknown default visibility
  - duplicate robot IDs
  - a scenario with no global (Manager) robot
  - missing scenario_id or empty robot list

This module is application-agnostic: it parses and validates, and takes the DB
writer as a callable so it never imports the data layer.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from core.rbac import AccessLevel, InvalidAccessLevel, Visibility, parse_access_level


class ProfileError(ValueError):
    """A scenario profile is invalid. Always raised at boot, never at request time."""


@dataclass(frozen=True)
class RobotProfileEntry:
    """One robot's declared identity within a scenario."""

    id: str
    role: str
    access_level: AccessLevel
    default_visibility: str = Visibility.LOCAL.value
    persona: Optional[str] = None
    # How much this project matters to the LAB, independent of any visitor —
    # the planner's first importance layer, and the only thing that decides
    # which block is cut when nobody has stated an interest. None means
    # unset, which the planner reads as neutral.
    #
    # It has to be declared SOMEWHERE deliberate: production called
    # block_importance with defaults={}, so every block scored the same and
    # the skip order fell through to the alphabetical tie-break. Which
    # project a visitor loses was decided by robot_id spelling.
    importance: Optional[float] = None
    # The simulation twin this PHYSICAL robot was trained on (e.g. navel_001's
    # is navel_01). Its presence is what makes a robot physical; a robot named
    # as someone's twin is a simulation robot. A run is live only when every
    # robot in it is physical. Mirrored into robots.embodiment / sim_twin at boot.
    sim_twin: Optional[str] = None
    embodiment: Optional[str] = None      # 'physical' | 'sim' | None (undeclared)

    @property
    def is_manager(self) -> bool:
        return self.access_level is AccessLevel.GLOBAL

    @property
    def is_physical(self) -> bool:
        return self.embodiment == "physical"


@dataclass(frozen=True)
class ScenarioProfile:
    """A single deployment: one scenario, its robots, and their access levels."""

    scenario_id: str
    robots: tuple[RobotProfileEntry, ...]
    description: str = ""
    source_path: Optional[str] = None
    # Each project robot stands at its own station, out of reach of the
    # others. See PresenceTracker.set_separate_stations.
    separate_stations: bool = False

    def get(self, robot_id: str) -> Optional[RobotProfileEntry]:
        for r in self.robots:
            if r.id == robot_id:
                return r
        return None

    @property
    def managers(self) -> tuple[RobotProfileEntry, ...]:
        return tuple(r for r in self.robots if r.is_manager)


def parse_profile(raw: object, source_path: Optional[str] = None) -> ScenarioProfile:
    """
    Validate one parsed YAML document into a ScenarioProfile.

    Every failure mode raises ProfileError with the offending value and the file
    it came from, so a typo is a legible boot error rather than a silent deny at
    the first retrieval.
    """
    where = f" in {source_path}" if source_path else ""

    if not isinstance(raw, dict):
        raise ProfileError(f"Scenario profile{where} must be a YAML mapping, got {type(raw).__name__}.")

    scenario_id = raw.get("scenario_id")
    if not scenario_id or not isinstance(scenario_id, str):
        raise ProfileError(f"Scenario profile{where} is missing a string 'scenario_id'.")

    robots_raw = raw.get("robots")
    if not isinstance(robots_raw, list) or not robots_raw:
        raise ProfileError(
            f"Scenario '{scenario_id}'{where} must declare a non-empty 'robots' list."
        )

    entries: list[RobotProfileEntry] = []
    seen: set[str] = set()
    # A physical robot inherits whatever it does not set from its twin, so the
    # twin's entry must be read first wherever it appears in the list.
    by_id = {it.get("id"): it for it in robots_raw if isinstance(it, dict)}

    for i, item in enumerate(robots_raw):
        if not isinstance(item, dict):
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: robots[{i}] must be a mapping, "
                f"got {type(item).__name__}."
            )

        robot_id = item.get("id")
        if not robot_id or not isinstance(robot_id, str):
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: robots[{i}] is missing a string 'id'."
            )
        if robot_id in seen:
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: duplicate robot id '{robot_id}'. "
                f"Each robot may appear once per scenario."
            )
        seen.add(robot_id)

        twin_id = item.get("sim_twin")
        if twin_id is not None:
            if not isinstance(twin_id, str) or twin_id == robot_id:
                raise ProfileError(
                    f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has an invalid "
                    f"sim_twin {twin_id!r} (must name a different robot).")
            twin = by_id.get(twin_id)
            if twin is None:
                raise ProfileError(
                    f"Scenario '{scenario_id}'{where}: robot '{robot_id}' names sim_twin "
                    f"'{twin_id}', which this scenario does not declare.")
            if twin.get("sim_twin"):
                raise ProfileError(
                    f"Scenario '{scenario_id}'{where}: '{robot_id}' names '{twin_id}' as its "
                    f"sim_twin, but '{twin_id}' is itself physical (it has a sim_twin).")
            # Inherit the twin's identity; anything set here wins.
            item = {**{k: v for k, v in twin.items() if k not in ("id", "sim_twin", "embodiment")},
                    **item}

        raw_emb = item.get("embodiment")
        if raw_emb is not None and raw_emb not in ("physical", "sim"):
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has embodiment "
                f"{raw_emb!r}; valid values are 'physical' and 'sim'.")
        if twin_id is not None and raw_emb == "sim":
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has a sim_twin, so it "
                f"is physical; it cannot also declare embodiment: sim.")
        is_a_twin = any(isinstance(o, dict) and o.get("sim_twin") == robot_id
                        for o in robots_raw)
        embodiment = ("physical" if twin_id is not None
                      else "sim" if is_a_twin else raw_emb)

        raw_level = item.get("access_level")
        try:
            level = parse_access_level(raw_level)
        except InvalidAccessLevel as e:
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has an invalid "
                f"access_level. {e}"
            ) from e

        raw_vis = item.get("default_visibility", Visibility.LOCAL.value)
        try:
            visibility = Visibility(str(raw_vis).strip().lower())
        except ValueError as e:
            raise ProfileError(
                f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has an invalid "
                f"default_visibility {raw_vis!r}. Valid values: "
                f"{', '.join(v.value for v in Visibility)}."
            ) from e

        raw_imp = item.get("importance")
        importance = None
        if raw_imp is not None:
            try:
                importance = float(raw_imp)
            except (TypeError, ValueError):
                raise ProfileError(
                    f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has a "
                    f"non-numeric importance {raw_imp!r}."
                )
            if not 0.0 <= importance <= 1.0:
                raise ProfileError(
                    f"Scenario '{scenario_id}'{where}: robot '{robot_id}' has "
                    f"importance {importance}, which is outside [0, 1]."
                )

        entries.append(RobotProfileEntry(
            id=robot_id,
            role=str(item.get("role") or ""),
            access_level=level,
            default_visibility=visibility.value,
            persona=item.get("persona"),
            importance=importance,
            sim_twin=twin_id,
            embodiment=embodiment,
        ))

    profile = ScenarioProfile(
        scenario_id=scenario_id,
        robots=tuple(entries),
        description=str(raw.get("description") or ""),
        source_path=source_path,
        separate_stations=bool(raw.get("separate_stations", False)),
    )

    if not profile.managers:
        raise ProfileError(
            f"Scenario '{scenario_id}'{where} declares no robot with "
            f"access_level 'global'. Every scenario needs at least one Manager, "
            f"otherwise no robot can maintain a unified view of the user's history."
        )

    return profile


class ProfileRegistry:
    """
    Holds the scenario profiles for this deployment.

    Load once at boot. Nothing here mutates a profile after loading.
    """

    def __init__(self, profiles: Iterable[ScenarioProfile] = ()):
        self._by_scenario: dict[str, ScenarioProfile] = {}
        for p in profiles:
            self._add(p)

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, directory: str | os.PathLike) -> "ProfileRegistry":
        """
        Load every *.yaml / *.yml in a directory.

        A missing directory yields an empty registry — a deployment need not use
        profiles at all, in which case every robot stays at the fail-closed
        'local' default.
        """
        path = Path(directory)
        registry = cls()
        if not path.is_dir():
            print(f"[ProfileRegistry] No profile directory at {path} — using DB values only.")
            return registry

        files = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in (".yaml", ".yml")]
        )
        for f in files:
            registry._add(cls._load_file(f))

        if files:
            print(
                f"[ProfileRegistry] Loaded {len(files)} profile(s): "
                f"{', '.join(sorted(registry._by_scenario))}"
            )
        return registry

    @staticmethod
    def _load_file(path: Path) -> ScenarioProfile:
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise ProfileError(f"Could not parse {path}: {e}") from e
        return parse_profile(raw, source_path=str(path))

    def _add(self, profile: ScenarioProfile) -> None:
        existing = self._by_scenario.get(profile.scenario_id)
        if existing is not None:
            raise ProfileError(
                f"Duplicate scenario_id '{profile.scenario_id}' "
                f"({existing.source_path} and {profile.source_path})."
            )
        # A robot may only belong to one scenario in a deployment, otherwise its
        # DB row could not carry a single scenario_id.
        for robot in profile.robots:
            owner = self.find_scenario(robot.id)
            if owner is not None:
                raise ProfileError(
                    f"Robot '{robot.id}' appears in both '{owner.scenario_id}' and "
                    f"'{profile.scenario_id}'. A robot belongs to one scenario."
                )
        self._by_scenario[profile.scenario_id] = profile

    # ── Lookup ────────────────────────────────────────────────────────────────

    def separate_stations(self) -> bool:
        """True if any loaded scenario puts its robots at separate stations."""
        return any(p.separate_stations for p in self._by_scenario.values())

    def importance_defaults(self, scenario_id: str = "") -> dict:
        """{robot_id: importance} for every robot that declares one.

        Empty when nothing does, which the planner reads as neutral — and
        which is what production silently had: block_importance was called
        with defaults={}, so every block tied and the skip order fell to the
        alphabetical tie-break.
        """
        out = {}
        profiles = ([self._by_scenario[scenario_id]]
                    if scenario_id and scenario_id in self._by_scenario
                    else list(self._by_scenario.values()))
        for prof in profiles:
            for e in prof.robots:
                if e.importance is not None:
                    out[e.id] = e.importance
        return out

    def get_scenario(self, scenario_id: str) -> Optional[ScenarioProfile]:
        return self._by_scenario.get(scenario_id)

    def find_scenario(self, robot_id: str) -> Optional[ScenarioProfile]:
        for p in self._by_scenario.values():
            if p.get(robot_id) is not None:
                return p
        return None

    def get_robot(self, robot_id: str) -> Optional[RobotProfileEntry]:
        p = self.find_scenario(robot_id)
        return p.get(robot_id) if p else None

    @property
    def scenarios(self) -> tuple[ScenarioProfile, ...]:
        return tuple(self._by_scenario.values())

    def __len__(self) -> int:
        return len(self._by_scenario)

    # ── Reconciliation ────────────────────────────────────────────────────────

    def is_physical(self, robot_id: str) -> bool:
        """Declared physical in any loaded scenario (it names a sim_twin)."""
        entry = self.get_robot(robot_id)
        return bool(entry and entry.is_physical)

    def sync_embodiment_to_db(self, writer: Callable[[str, Optional[str], Optional[str]], bool]) -> int:
        """Mirror embodiment and sim_twin into robots, so SQL (the backfill's live
        rule, the paper stats) can tell a physical robot from its twin.
        `writer(robot_id, embodiment, sim_twin) -> bool`. Returns rows written."""
        written = 0
        for profile in self._by_scenario.values():
            for robot in profile.robots:
                try:
                    if writer(robot.id, robot.embodiment, robot.sim_twin):
                        written += 1
                except Exception as e:
                    print(f"[ProfileRegistry] embodiment sync error for '{robot.id}': {e}")
        return written

    def sync_to_db(self, writer: Callable[[str, str, Optional[str]], bool]) -> int:
        """
        Push declared access levels into the database.

        `writer(robot_id, access_level, scenario_id) -> bool` keeps this module
        free of any data-layer import. Returns how many rows were reconciled.

        A robot declared in a profile but absent from the DB is reported and
        skipped — the writer returns False and registration stays the operator's
        job via the web UI.
        """
        synced = 0
        failed: list[str] = []
        for profile in self._by_scenario.values():
            for robot in profile.robots:
                try:
                    if writer(robot.id, robot.access_level.value, profile.scenario_id):
                        synced += 1
                    else:
                        failed.append(robot.id)
                except Exception as e:
                    failed.append(robot.id)
                    print(f"[ProfileRegistry] sync error for '{robot.id}': {e}")

        if failed:
            # One line, not one per robot — the underlying cause is almost
            # always shared (missing migration, or robots not yet registered).
            print(
                f"[ProfileRegistry] Could not set access level for "
                f"{len(failed)} robot(s): {', '.join(failed)}. "
                f"They stay at the fail-closed 'local' default."
            )
        return synced
