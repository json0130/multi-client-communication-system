"""
decision/presence.py
====================
Where each robot physically is, and whether it is close enough to the tour
group to take a question.

WHY THIS IS NOT A ROOM ENUM
The obvious shape is `location: Literal["lab_a", "corridor", ...]`. It is
wrong for the thing that will actually populate this: a ROS2 pose topic
publishes continuous coordinates in a named frame, not a room label. Storing
a label would mean a lossy translation at the boundary and a migration the
day the first real pose arrives. `Pose` is pose-shaped from the start —
x/y/frame_id/timestamp — so a subscriber can write straight into it.

Runtime state, deliberately not persisted. A pose is only meaningful for as
long as the robot has not moved, so there is nothing worth surviving a
restart: a live deployment gets fresh poses from ROS2 within a tick, and a
sim/harness run sets what it needs at the top of the run. That is also why
this needs no migration now and none later.

IN_CONVERSATION IS DERIVED, NEVER STORED
There is one source of truth — location — and `in_conversation` is a read
over it: is this robot within `proximity_m` of the robot the group is
currently standing in front of. Storing both would let them disagree, and
the disagreement would be silent.

The one exception is an explicit override, which exists because there is no
pose source yet and a sim/harness run still needs to say "pretend Navel
stepped out". Overrides are logged with their source precisely so that a
value that did NOT come from a location is always attributable afterwards.

FAILING OPEN
Unknown location means present. Every fallback in the router already works
this way — an unseeded graph, an unresolvable topic, a disconnected peer all
degrade to "no opinion", never to "excluded". A robot that is standing right
there but whose pose has not arrived yet must not be silently dropped from
routing; the cost of wrongly including it is one question routed to someone
who has to say "let's cover that at my station", and the cost of wrongly
excluding it is a robot that never speaks with no visible reason.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROXIMITY_M = 3.0
"""How close a robot must be to the group to be considered part of the
conversation. A tour group clusters within a couple of metres of whoever is
presenting; 3m is that, plus room for a robot standing at the back of its own
station. Not tuned against real poses — there are none yet."""


@dataclass(frozen=True)
class Pose:
    """A robot's position. Shaped for a ROS2 geometry_msgs/PoseStamped, minus
    the parts routing has no use for (orientation, covariance)."""

    x: float
    y: float
    frame_id: str = "map"
    timestamp: Optional[float] = None

    def distance_to(self, other: Optional["Pose"]) -> Optional[float]:
        """Metres between two poses, or None when they cannot be compared.

        Different frame_ids are NOT comparable — 2m apart in `map` and 2m
        apart in `pepper/odom` are different claims, and silently treating
        them alike would produce confident nonsense. None propagates to the
        fail-open path, same as a missing pose.
        """
        if other is None or self.frame_id != other.frame_id:
            return None
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "frame_id": self.frame_id,
                "timestamp": self.timestamp}


class PresenceTracker:
    """
    Who is where, and who is close enough to answer.

    Thread-safe for the same reason DemoRunTracker is: a ROS2 callback (or a
    harness thread) writes while the gateway's decision path reads.
    """

    def __init__(self, proximity_m: float = DEFAULT_PROXIMITY_M):
        self._lock = threading.RLock()
        self._locations: dict = {}
        self._overrides: dict = {}
        self._proximity_m = proximity_m

    # ── Writes ────────────────────────────────────────────────────────────────

    def set_location(self, robot_id: str, pose: Optional[Pose],
                     source: str = "ros2") -> None:
        """Record where a robot is. `pose=None` clears it back to unknown."""
        with self._lock:
            if pose is None:
                self._locations.pop(robot_id, None)
            else:
                self._locations[robot_id] = pose
        logger.debug(f"[Presence] {robot_id} location <- {pose} (source={source})")

    def set_in_conversation(self, robot_id: str, value: Optional[bool],
                            source: str) -> None:
        """
        Override the DERIVED in_conversation for one robot.

        Only path by which in_conversation can disagree with location, and it
        logs at INFO for exactly that reason: any later question of "why was
        this robot excluded from routing" must be answerable from the log
        without guessing whether a pose or a human put it there. `source` is
        required, not defaulted — an unattributed override is the thing this
        is trying to prevent.

        `value=None` removes the override and returns the robot to derivation.
        """
        with self._lock:
            if value is None:
                had = self._overrides.pop(robot_id, None)
                if had is not None:
                    logger.info(f"[Presence] {robot_id} in_conversation override "
                                f"CLEARED (source={source}) — back to derived")
                return
            self._overrides[robot_id] = bool(value)
        logger.info(f"[Presence] {robot_id} in_conversation <- {bool(value)} "
                    f"OVERRIDE (source={source}) — not derived from location")

    def reset(self) -> None:
        with self._lock:
            self._locations.clear()
            self._overrides.clear()

    # ── Reads ─────────────────────────────────────────────────────────────────

    def location(self, robot_id: str) -> Optional[Pose]:
        with self._lock:
            return self._locations.get(robot_id)

    def has_override(self, robot_id: str) -> bool:
        with self._lock:
            return robot_id in self._overrides

    def in_conversation(self, robot_id: str,
                        reference_robot_id: Optional[str] = None) -> bool:
        """
        Is `robot_id` close enough to the group to take a question?

        The group's position is taken to be the presenting robot's — the tour
        stands in front of whoever is currently talking, so that robot's pose
        IS the group's pose, and tracking a separate group position would be
        a second thing to keep in sync for no gain.

        Fails open at every step: no override and no reference robot, no
        reference pose, no own pose, or two poses in different frames all
        return True. See the module docstring.
        """
        with self._lock:
            if robot_id in self._overrides:
                return self._overrides[robot_id]
            own = self._locations.get(robot_id)
            ref = (self._locations.get(reference_robot_id)
                   if reference_robot_id else None)
            proximity = self._proximity_m

        if own is None or ref is None:
            return True
        distance = own.distance_to(ref)
        if distance is None:
            return True
        return distance <= proximity

    def absent(self, robot_ids: Iterable[str],
               reference_robot_id: Optional[str] = None) -> set:
        """The subset of `robot_ids` that is NOT in the conversation.

        Returned as the absent set rather than the present set on purpose:
        the router filters by exclusion, and an empty set — the default state
        with no poses and no overrides — then means "exclude nobody", which
        is the fail-open behaviour by construction rather than by a check.
        """
        return {r for r in robot_ids
                if not self.in_conversation(r, reference_robot_id)}

    def snapshot(self, robot_ids: Optional[Iterable[str]] = None,
                 reference_robot_id: Optional[str] = None) -> dict:
        """{robot_id: {location, in_conversation, source}} for the dashboard
        and for the decision log's observation payload."""
        with self._lock:
            ids = list(robot_ids) if robot_ids is not None else list(
                set(self._locations) | set(self._overrides))
        out = {}
        for rid in ids:
            out[rid] = {
                "location": self.location(rid).as_dict() if self.location(rid) else None,
                "in_conversation": self.in_conversation(rid, reference_robot_id),
                "source": "override" if self.has_override(rid)
                          else ("derived" if self.location(rid) else "unknown"),
            }
        return out
