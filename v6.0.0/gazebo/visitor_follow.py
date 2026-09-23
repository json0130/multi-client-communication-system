"""
gazebo/visitor_follow.py
=========================
Moves the 3 visitor actors (visitor_1/2/3, defined in lab_world) to trail
the guide robot's LIVE odometry — actually reactive to however long the
real demo's dialogue/Q&A takes, unlike a fixed pre-scripted timeline (which
was tried first and correctly criticized: it drifts out of sync with, or
runs completely independently of, whatever the actual demo is doing).

How it works: keeps a short time-ordered history of the guide's odom
samples. Once the guide gets near the door (its first stop — see
gazebo/launch_demo_env.launch.py / demo_script.py's GAZEBO_ROUTES), each
visitor is repositioned every tick to the guide's position from N seconds
ago (a different N per visitor), via the /set_entity_state service. If the
guide stops moving (mid-dialogue), the history for that window is all the
same pose, so the visitors naturally stop too, right behind it — no manual
sync tuning needed.

Requires /set_entity_state, which is NOT loaded by gazebo_ros by default —
launch_demo_env.launch.py passes `-s libgazebo_ros_state.so` via
extra_gazebo_args to enable it.

Run after Gazebo is up (needs /set_entity_state to exist):
    source /opt/ros/humble/setup.bash
    python3 visitor_follow.py

Also runs a tiny local HTTP server (RESET_HTTP_PORT) exposing POST /reset,
which teleports every robot and visitor back to its launch-time spawn pose
and clears this node's own follow state (_active, _history). The server's
DemoOrchestrator.start() calls it (best-effort — a missing/dead Gazebo just
means the reset is skipped, never a reason to fail a real demo start) so a
new demo run doesn't require relaunching launch_demo_env.launch.py or this
script by hand every time.
"""
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

DOOR = (0.092, 6.818)
# Guide's own spawn pose (0.092, 5.528) is only 1.29m from the door — a
# radius above that would trigger the moment the world loads, before the
# guide has moved at all.
#
# Must also be >= gazebo_bridge.py's final_arrival_tolerance_m (defaults to
# 1.0m, not overridden in any config/*_sim.json here) — that is the radius
# the guide's OWN nav controller uses to decide "arrived, stop, send ack"
# for the approach_visitors step's single waypoint. The original 0.5m was
# tighter than that: the guide could legitimately finish approaching the
# door anywhere up to 1.0m away, send its ack, and the demo would move on
# to the greeting with follow-mode never having activated at all — visitors
# just never move. 1.15m clears the 1.0m tolerance with margin while still
# comfortably under the 1.29m spawn distance above.
ACTIVATE_RADIUS_M = 1.15
HISTORY_SECONDS = 30.0    # must exceed the largest TRAIL_DELAY_SEC with margin
STOPPED_EPS_M   = 0.08    # guide position change below this counts as "not moving"
STOPPED_HOLD_SEC = 1.0    # how long that has to hold before treating the guide as stopped
UPDATE_HZ = 5.0

VISITORS = [
    ("visitor_1", 3.0),
    ("visitor_2", 6.0),
    ("visitor_3", 9.0),
]

# Per-visitor offset in the GUIDE'S OWN local frame at the historical sample
# being used (x = forward, y = left) — NOT a world-frame offset, so it
# rotates with whichever way the guide was actually facing at that moment
# instead of staying a fixed compass direction. Without this, _pose_at()
# returns the guide's own (x, y) verbatim for every visitor: while walking
# that's fine (each visitor's delay already spaces them out along the
# path), but the moment the guide STOPS at a station, every visitor's
# sample converges on the exact same stopped pose and all three end up
# stacked directly on top of the guide and each other. Values spread the
# group out in front of and to both sides of wherever the guide ends up
# facing once everyone has caught up — positive (not negative) forward
# offset, deliberately: every station's final_yaw is chosen to face back
# toward the visitors, and speech never happens until the guide is
# stopped, so whatever the visitors converge in front of IS the audience
# the guide needs to be looking at while it talks. A negative (behind)
# offset here made the guide correctly turn to face its final_yaw and
# then talk to an empty spot, because the visitors it turned to face had
# just relocated to directly behind it.
VISITOR_OFFSETS = {
    "visitor_1": (0.6, 0.6),
    "visitor_2": (0.6, -0.6),
    "visitor_3": (1.2, 0.0),
}

RESET_HTTP_PORT = 8899

# Visitors are teleported, not physically simulated, so nothing stops one
# being placed inside a project robot. Any visitor pose closer than this to
# a project robot's live odom position is pushed straight out to this radius.
PROJECT_ROBOTS = ("robot2", "robot3", "robot4")  # chatbox_01, navel_01, silbot_01
KEEPOUT_M = 1.0

# Spawn poses (x, y, yaw) — kept in sync by hand with
# gazebo/launch_demo_env.launch.py's spawn_robot1-4 args and
# gazebo/lab_world's visitor_1/2/3 <pose> values. Robots have a real yaw;
# visitors don't (they get repositioned by _update_visitors anyway, and a
# waiting pose facing any particular way is fine).
ROBOT_SPAWNS = {
    "robot1": (0.092, 5.528, 0.0),      # pepper_01 (guide)
    "robot2": (9.695, 1.035, 3.140),    # chatbox_01
    "robot3": (0.25, -5.0, 0.0),         # navel_01
    "robot4": (8.128, -5.582, 1.570),   # silbot_01
}
VISITOR_SPAWNS = {
    "visitor_1": (-0.308, 7.2, 0.0),
    "visitor_2": (0.092, 7.2, 0.0),
    "visitor_3": (0.492, 7.2, 0.0),
}


class VisitorFollowNode(Node):
    def __init__(self):
        super().__init__("visitor_follow")
        self._history = deque()  # (wall_time, x, y, yaw)
        self._active = False
        # name -> True once frozen at the station (guide stopped translating).
        # Cleared back to False the moment the guide starts moving again, so
        # the next leg's trailing behavior resumes untouched.
        self._frozen = {}
        self._stopped_since = None  # wall time the guide last went still, or None while moving
        self.create_subscription(Odometry, "/robot1/odom", self._on_guide_odom, 10)
        self._robot_xy = {r: ROBOT_SPAWNS[r][:2] for r in PROJECT_ROBOTS}
        for r in PROJECT_ROBOTS:
            self.create_subscription(
                Odometry, f"/{r}/odom",
                lambda m, r=r: self._robot_xy.__setitem__(
                    r, (m.pose.pose.position.x, m.pose.pose.position.y)), 10)
        self._set_state = self.create_client(SetEntityState, "/set_entity_state")
        self.create_timer(1.0 / UPDATE_HZ, self._update_visitors)
        self.get_logger().info("[VisitorFollow] Waiting for guide to near the door...")

    def _on_guide_odom(self, msg: Odometry):
        now = time.time()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._history.append((now, p.x, p.y, yaw))
        while self._history and now - self._history[0][0] > HISTORY_SECONDS:
            self._history.popleft()

        if not self._active and math.hypot(p.x - DOOR[0], p.y - DOOR[1]) < ACTIVATE_RADIUS_M:
            self._active = True
            self.get_logger().info("[VisitorFollow] Guide reached the door — visitors now following.")

    def _pose_at(self, target_time: float):
        """Nearest history sample to target_time, or None if we have nothing
        old enough yet (guide hasn't been moving long enough for this
        visitor's own delay — that visitor just stays put until it does)."""
        if not self._history:
            return None
        if target_time <= self._history[0][0]:
            return self._history[0][1:]
        best = self._history[0]
        for sample in self._history:
            if sample[0] > target_time:
                break
            best = sample
        return best[1:]

    def _guide_is_stationary(self, now: float) -> bool:
        """True once the guide has stopped TRANSLATING — turning in place
        (the new face-the-robot / face-visitors turn steps) doesn't count
        as movement here, only a real position change does."""
        cur = self._pose_at(now)
        past = self._pose_at(now - STOPPED_HOLD_SEC)
        if cur is None or past is None:
            return False
        return math.hypot(cur[0] - past[0], cur[1] - past[1]) < STOPPED_EPS_M

    def _update_visitors(self):
        if not self._active:
            return
        now = time.time()
        if self._guide_is_stationary(now):
            if self._stopped_since is None:
                self._stopped_since = now - STOPPED_HOLD_SEC  # back-date to when it actually went still
        else:
            self._stopped_since = None
            for name, _ in VISITORS:
                self._frozen[name] = False

        for name, delay in VISITORS:
            # Once the guide has arrived and stopped, freeze this visitor at
            # its current spot and stop touching it — otherwise every turn
            # the guide makes to face whichever robot/visitors is currently
            # speaking (VISITOR_OFFSETS being guide-frame, not world-frame)
            # swings the visitors around it too, which reads as visitors
            # aimlessly shuffling in place rather than standing and
            # listening. Re-armed the instant the guide starts walking
            # again (above), so the next leg's trailing-behind-at-a-delay
            # look is completely unaffected.
            #
            # Not frozen the instant the guide stops: this visitor's own
            # `delay`-second-old sample is still catching up from mid-walk
            # for up to `delay` more seconds after the guide itself has
            # already stopped (visitor_3's 9s delay is the worst case) —
            # freezing before that catch-up finishes would lock it in
            # wherever it happened to be mid-transit, not where it was
            # actually walking to. Wait until this visitor's own delay
            # window has fully elapsed since the guide went still.
            if self._frozen.get(name):
                continue

            pose = self._pose_at(now - delay)
            if pose is None:
                continue
            x, y, yaw = pose
            off_x, off_y = VISITOR_OFFSETS.get(name, (0.0, 0.0))
            # Local (forward, left) -> world, using THIS sample's own yaw so
            # the offset stays behind/beside the guide regardless of which
            # way it was actually facing when this historical pose was
            # recorded (not a fixed world-frame nudge, which would put a
            # visitor "behind" only when the guide happened to face one way).
            wx = x + off_x * math.cos(yaw) - off_y * math.sin(yaw)
            wy = y + off_x * math.sin(yaw) + off_y * math.cos(yaw)
            wx, wy = self._keep_clear(wx, wy)
            self._set_entity_state(name, wx, wy, yaw)

            if self._stopped_since is not None and (now - self._stopped_since) >= delay:
                self._frozen[name] = True

    def _keep_clear(self, x: float, y: float):
        for rx, ry in self._robot_xy.values():
            d = math.hypot(x - rx, y - ry)
            if d < KEEPOUT_M:
                if d < 1e-6:
                    x, d = x + 1e-3, 1e-3
                x = rx + (x - rx) * KEEPOUT_M / d
                y = ry + (y - ry) * KEEPOUT_M / d
        return x, y

    def _set_entity_state(self, name: str, x: float, y: float, yaw: float):
        if not self._set_state.service_is_ready():
            return
        req = SetEntityState.Request()
        state = EntityState()
        state.name = name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.0
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        # Zeroed explicitly (not just left as the message default) so a
        # teleport during reset_positions() — which can land mid-drive, with
        # real wheel velocity — doesn't keep sliding for a moment after the
        # jump. EntityState's fields already default-zero for the follow-only
        # path (visitors are never mid-drive), so this is a no-op there.
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        req.state = state
        # Fire-and-forget — waiting on every call at UPDATE_HZ * 3 visitors
        # would stall the timer. A dropped update just means that visitor
        # skips one tick, not a lasting error.
        self._set_state.call_async(req)

    def reset_positions(self):
        """Teleport every robot and visitor back to its spawn pose, and
        clear follow state so the next run starts fresh (not mid-trail from
        wherever the guide last was). Called from the HTTP reset handler,
        itself called by the server at the start of every demo run."""
        self._active = False
        self._history.clear()
        for name, (x, y, yaw) in {**ROBOT_SPAWNS, **VISITOR_SPAWNS}.items():
            self._set_entity_state(name, x, y, yaw)
        self.get_logger().info("[VisitorFollow] Reset — all robots and visitors back to spawn.")


class _ResetHandler(BaseHTTPRequestHandler):
    node: "VisitorFollowNode" = None  # set by _start_reset_server before use

    def do_POST(self):
        if self.path != "/reset":
            self.send_response(404)
            self.end_headers()
            return
        self.node.reset_positions()
        body = json.dumps({"success": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # BaseHTTPRequestHandler logs every request to stderr by default — noisy at 1 req/demo-start


def _start_reset_server(node: "VisitorFollowNode"):
    _ResetHandler.node = node
    server = ThreadingHTTPServer(("127.0.0.1", RESET_HTTP_PORT), _ResetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="reset-http")
    thread.start()
    node.get_logger().info(f"[VisitorFollow] Reset endpoint on http://127.0.0.1:{RESET_HTTP_PORT}/reset")


def main():
    rclpy.init()
    node = VisitorFollowNode()
    _start_reset_server(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
