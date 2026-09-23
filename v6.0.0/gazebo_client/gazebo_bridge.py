"""
gazebo_client/gazebo_bridge.py
===============================
Bridge adapter: v6.0.0 WebSocket protocol  <->  a simulated robot in Gazebo.

From the central server's perspective: a normal v6.0.0 robot client — it
imports and subclasses the same BasicClient every real robot (Pepper,
ChatBox, Navel, Silbot) uses, so no server-side code needs to know this
robot is simulated.

From Gazebo's perspective: an rclpy node that reports odometry as presence
and, for navigation demo steps only, drives the robot toward a target pose
using closed-loop odom feedback (turn to face it, then drive, then stop).
This is separate from gazebo/approach_demo.py's open-loop scripted tour,
which remains a standalone manually-run script for ad hoc testing — the
two are not meant to run against the same robot at the same time.

Run from this directory, one process per simulated robot:
    python3 gazebo_bridge.py configs/pepper_01_sim.json

Requires ROS2 sourced first (e.g. `source /opt/ros/humble/setup.bash`) and
a "gazebo_config" block in the given client_config.json.
"""

import logging
import math
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "client"))

from client import BasicClient  # noqa: E402
from InputModules.text_input import TextInputModule  # noqa: E402
from OutputModules.edge_tts_output import EdgeTTSOutputModule  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
import requests  # noqa: E402

from sim_tts_output import SimTTSOutputModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class _OdomPresenceNode(Node):
    """
    Subscribes to a robot's odom (position -> presence, position+yaw -> nav
    control) and publishes Twist commands for navigation steps.
    """

    def __init__(self, node_name: str, odom_topic: str, cmd_vel_topic: str, on_pose):
        super().__init__(node_name)
        self._on_pose = on_pose
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self._cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

    def _on_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # Yaw from a planar quaternion (roll/pitch are ~0 for a ground robot).
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._on_pose(pos.x, pos.y, yaw)

    def publish_twist(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_vel_pub.publish(msg)


class GazeboBridge(BasicClient):
    """
    Bridges the v6.0.0 WebSocket protocol and a simulated Gazebo robot.

    Inherits from BasicClient to get:
      - WebSocket server (central server connects to us)
      - Persona update handling
      - The default demo_step / chat_response / chat_sentence handlers —
        unlike PepperBridge, these are NOT overridden here. BasicClient's
        _on_demo_step already does the right thing (find the first output
        module with speak_with_callback, block on it, then ACK) once a TTS
        output module is registered below, so there is nothing Gazebo-
        specific to reroute.

    Adds:
      - An rclpy node (run in a background thread) that subscribes to this
        robot's odometry topic and periodically POSTs its x/y to the
        server's /robots/<id>/presence endpoint.
      - A "speech" output module — EdgeTTSOutputModule (real audio, the
        same module every real robot uses) by default, or SimTTSOutputModule
        (silent, console-only) when gazebo_config.tts_mode is "console".
      - TextInputModule (unmodified, from client/InputModules) as simulated
        chat input — type at the prompt to simulate a spoken question.
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)

        gz_cfg = self.config.get("gazebo_config", {})
        self._namespace = gz_cfg.get("ros_namespace", "robot1")
        self._odom_topic = gz_cfg.get("odom_topic", f"/{self._namespace}/odom")
        self._cmd_vel_topic = gz_cfg.get("cmd_vel_topic", f"/{self._namespace}/cmd_vel")
        self._frame_id = gz_cfg.get("frame_id", "map")
        self._presence_interval = gz_cfg.get("presence_interval_sec", 1.0)
        words_per_second = gz_cfg.get("words_per_second", 2.3)

        # Navigation tuning — generous defaults for a TurtleBot3 in an
        # open-plan lab world, not tuned against any specific run.
        # nav_goal_tolerance_m gates INTERMEDIATE waypoints (a doorway, a
        # corridor turn) — those should be hit fairly precisely so the next
        # leg starts from where it expects. final_arrival_tolerance_m gates
        # only the LAST waypoint of a path — the actual station — and is
        # deliberately much looser: stopping "on top of" the robot being
        # visited reads as a collision, not an arrival.
        self._nav_goal_tolerance_m      = gz_cfg.get("nav_goal_tolerance_m", 0.3)
        self._final_arrival_tolerance_m = gz_cfg.get("final_arrival_tolerance_m", 1.0)
        self._nav_heading_tolerance  = gz_cfg.get("nav_heading_tolerance_rad", 0.15)
        self._nav_linear_speed       = gz_cfg.get("nav_linear_speed", 0.3)
        self._nav_angular_speed      = gz_cfg.get("nav_angular_speed", 0.6)
        self._nav_control_hz         = gz_cfg.get("nav_control_hz", 10.0)

        self._server_url = self.config.get("server_url", "http://localhost:5000").rstrip("/")
        self._client_id = self.config.get("client_id")

        self._latest_pose = None  # (x, y, yaw), written by ROS callback thread
        self._pose_lock = threading.Lock()
        self._ros_node = None
        self._ros_thread = None
        self._ros_stop = threading.Event()
        self._presence_thread = None
        self._presence_stop = threading.Event()
        self._last_presence_warn = 0.0

        # tts_mode: "audio" (default) speaks for real through this machine's
        # speakers via edge-tts/gTTS, the same module and voice every real
        # robot uses — the sim configs already carry tts_voice/voice_config
        # copied from the real robot configs. "console" falls back to the
        # silent, word-count-timed SimTTSOutputModule for headless/CI runs
        # with no audio device.
        tts_mode = gz_cfg.get("tts_mode", "audio")
        if tts_mode == "console":
            tts = SimTTSOutputModule("sim_tts_output", {"words_per_second": words_per_second})
        else:
            # Same construction robot.py's SimpleConcurrentClient uses —
            # built from the robot's own top-level config, not a separate
            # edge_tts_config block most configs never set.
            edge_cfg = dict(self.config.get("edge_tts_config") or {})
            edge_cfg.setdefault("rate", "+0%")
            edge_cfg.setdefault("pitch", "+0Hz")
            edge_cfg.setdefault("remove_emotion_tags", True)
            for key in ("tts_voice", "audio_device", "audio_cmd", "sim_speed"):
                if self.config.get(key) is not None:
                    edge_cfg.setdefault(key, self.config[key])
            tts = EdgeTTSOutputModule("edge_tts_output", edge_cfg)
        self.register_output_module(tts)

        text_input = TextInputModule("text_input")
        self.register_input_module(text_input)

        # Overrides BasicClient's safe default (_default_nav_handler, which
        # just acks immediately) with real closed-loop movement.
        self.server_connection.register_handler("demo_nav", self._on_nav_goal)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start ROS2 (odom subscription + presence-posting timer thread) before
        the usual BasicClient startup (server registration, WS listener,
        modules) — presence can start flowing independently of whether the
        dashboard has connected the WebSocket yet.
        """
        self._start_ros()
        return super().start()

    def stop(self):
        super().stop()
        self._stop_ros()

    # ── ROS2 ──────────────────────────────────────────────────────────────

    def _start_ros(self):
        rclpy.init(args=None)
        node_name = f"gazebo_bridge_{self._namespace}"
        self._ros_node = _OdomPresenceNode(
            node_name, self._odom_topic, self._cmd_vel_topic, self._on_pose
        )

        self._ros_stop.clear()
        self._ros_thread = threading.Thread(
            target=self._ros_spin_loop, daemon=True, name="gazebo-ros-spin"
        )
        self._ros_thread.start()
        logger.info(f"[ROS] Node '{node_name}' spinning, subscribed to {self._odom_topic}")

        self._presence_stop.clear()
        self._presence_thread = threading.Thread(
            target=self._presence_loop, daemon=True, name="gazebo-presence"
        )
        self._presence_thread.start()

    def _ros_spin_loop(self):
        # rclpy.spin(node) blocks indefinitely and does not reliably return
        # when rclpy.shutdown() is called concurrently from another thread —
        # racing node.destroy_node() against a still-spinning executor
        # crashes the process on interpreter exit. Polling spin_once() with
        # a short timeout, gated on an explicit stop event, means the spin
        # thread has always fully returned (join() below) before teardown
        # touches the node or the context.
        while not self._ros_stop.is_set():
            try:
                rclpy.spin_once(self._ros_node, timeout_sec=0.2)
            except ExternalShutdownException:
                # rclpy installs its own SIGINT handler, which can shut down
                # the global context out from under this polling loop when
                # the process itself receives Ctrl+C — that's just another,
                # asynchronous way this loop is told to stop, not an error.
                break

    def _stop_ros(self):
        self._presence_stop.set()
        if self._presence_thread:
            self._presence_thread.join(timeout=2)

        self._ros_stop.set()
        if self._ros_thread:
            self._ros_thread.join(timeout=2)

        if self._ros_node:
            self._ros_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _on_pose(self, x: float, y: float, yaw: float):
        with self._pose_lock:
            self._latest_pose = (x, y, yaw)

    def _presence_loop(self):
        while not self._presence_stop.is_set():
            with self._pose_lock:
                pose = self._latest_pose
            if pose is not None:
                self._post_presence(pose[0], pose[1])
            self._presence_stop.wait(timeout=self._presence_interval)

    def _post_presence(self, x: float, y: float):
        url = f"{self._server_url}/robots/{self._client_id}/presence"
        try:
            requests.post(
                url,
                json={"x": x, "y": y, "frame_id": self._frame_id, "source": "gazebo"},
                timeout=3,
            )
        except requests.RequestException as e:
            now = time.time()
            if now - self._last_presence_warn >= 10:
                logger.warning(f"[Presence] Could not reach server at {url}: {e}")
                self._last_presence_warn = now

    # ── Navigation ────────────────────────────────────────────────────────

    def _on_nav_goal(self, data: dict):
        """
        Handle a demo_nav event: speak a short cue, drive through the
        target's waypoints in order, and report arrival with the SAME ack
        mechanism a demo_step uses after TTS finishes — the orchestrator's
        WAITING_ACK wait doesn't care what produced the ack.

        `target` is a list of {x, y, frame_id} waypoints, driven through in
        order — a straight line between two points in lab_world usually
        crosses a wall, so a single-point target isn't enough; this walks
        the same door -> corridor -> station route gazebo/approach_demo.py
        already proved works, just as live waypoints instead of open-loop
        timed legs. A single {x, y, frame_id} dict (no waypoints) is also
        accepted as a 1-waypoint path, for a target with open space in
        front of it the whole way.

        Only the FINAL waypoint (the actual station) uses
        final_arrival_tolerance_m — stopping "on top of" the robot being
        visited reads as a collision, not an arrival. Every waypoint before
        it uses the tighter nav_goal_tolerance_m, since those are just
        via-points (a doorway, a corridor turn) the next leg's start
        assumes was hit fairly precisely.
        """
        step_id   = data.get("step_id", "")
        target    = data.get("target")
        text      = data.get("text", "")
        need_ack  = data.get("require_ack", True)
        final_yaw = data.get("final_yaw")

        waypoints = target.get("waypoints") if isinstance(target, dict) else None
        if waypoints is None and isinstance(target, list):
            waypoints = target
        elif waypoints is None and isinstance(target, dict) and "x" in target:
            waypoints = [target]
        waypoints = waypoints or []

        if not waypoints:
            logger.warning(f"[Nav] '{step_id}': no target given — ACK now")
            if need_ack:
                self.send_ack(step_id)
            return

        if text:
            self._speak_blocking(text)

        logger.info(f"[Nav] '{step_id}': navigating through {len(waypoints)} waypoint(s)")
        # One shared deadline across every leg — a multi-leg path shouldn't
        # get penalized for legs already driven. Derived from the step's
        # OWN timeout_sec (sent by the orchestrator) rather than a fixed
        # constant: a 6-waypoint return-to-start route covering ~36m + 5
        # turns needs far more time than a single short hop, and a
        # hardcoded 100s silently stopped a real run mid-corridor, well
        # short of its target, with no collision at all — just ran out of
        # its budget. 10s margin so a genuinely stuck run still reports
        # "gave up" locally before the orchestrator's own ack-wait times out.
        step_timeout = float(data.get("timeout_sec", 100.0))
        total_budget = max(10.0, step_timeout - 10.0)
        # The final turn-to-face-visitors gets its OWN reserved slice, taken
        # off the driving budget up front, rather than sharing one deadline
        # with the waypoint driving. A shared deadline starved it on longer
        # routes: navel_01's is 4 waypoints (the collision-avoidance step-
        # aside added one more), and driving those routinely ran the clock
        # down to almost nothing before the turn ever got a chance — it
        # would start, immediately hit the same deadline, and stop partway
        # through (~30° off target live). 15s is generous against the
        # ~0.6 rad/s nav_angular_speed default (a full 180° turn is ~5s);
        # reserved only when a final_yaw was actually requested.
        TURN_RESERVE_SEC = 15.0
        turn_reserve = TURN_RESERVE_SEC if final_yaw is not None else 0.0
        drive_deadline = time.time() + max(5.0, total_budget - turn_reserve)
        arrived = True
        for i, wp in enumerate(waypoints):
            tx, ty = wp.get("x"), wp.get("y")
            if tx is None or ty is None:
                continue
            is_final = (i == len(waypoints) - 1)
            tolerance = self._final_arrival_tolerance_m if is_final else self._nav_goal_tolerance_m
            arrived = self._drive_to(float(tx), float(ty), drive_deadline, tolerance)
            if not arrived:
                break
        self._ros_node.publish_twist(0.0, 0.0)  # always stop, arrived or not

        # Turning to face visitors, not whatever the last leg's direction of
        # travel happened to leave it facing — usually straight at the
        # project robot just walked up to. Only after a successful arrival;
        # a step that timed out has bigger problems than its final heading.
        if arrived and final_yaw is not None:
            turn_deadline = time.time() + turn_reserve
            self._turn_to_yaw(float(final_yaw), turn_deadline)
            self._ros_node.publish_twist(0.0, 0.0)

        if arrived:
            logger.info(f"[Nav] '{step_id}': arrived")
            if need_ack:
                self.send_ack(step_id)
        else:
            # Deliberately no ack — same as a stuck TTS today, this lets the
            # orchestrator's own timeout_sec -> ERROR -> manual "next"
            # recovery handle it rather than inventing new error handling.
            logger.warning(f"[Nav] '{step_id}': did not reach target in time — "
                            "not ACKing; use the dashboard's manual 'next' to recover.")

    def _speak_blocking(self, text: str, timeout: float = 10.0):
        """
        Speak `text` via the first registered output module and block until
        it finishes (or timeout) — same "find the module, block on it"
        pattern BasicClient._on_demo_step uses, just invoked directly here
        since a nav cue isn't itself a DemoStep with its own ack.
        """
        for module in self.output_modules.values():
            if hasattr(module, "speak_with_callback"):
                done = threading.Event()
                module.speak_with_callback(text, callback=done.set)
                done.wait(timeout=timeout)
                return
        logger.warning(f"[Nav] No TTS module to speak cue: {text[:40]}")

    def _drive_to(self, tx: float, ty: float, deadline: float, tolerance: float) -> bool:
        """
        Closed-loop turn-then-drive controller for ONE waypoint, using live
        odom feedback. Returns True once within `tolerance` metres of
        (tx, ty), False if `deadline` (an absolute time.time() value) passes
        first.
        """
        dt = 1.0 / self._nav_control_hz

        while time.time() < deadline:
            with self._pose_lock:
                pose = self._latest_pose
            if pose is None:
                time.sleep(dt)
                continue

            x, y, yaw = pose
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)
            if distance <= tolerance:
                return True

            heading_error = math.atan2(dy, dx) - yaw
            # Normalize to [-pi, pi] so the turn always takes the short way.
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            if abs(heading_error) > self._nav_heading_tolerance:
                turn = self._nav_angular_speed if heading_error > 0 else -self._nav_angular_speed
                self._ros_node.publish_twist(0.0, turn)
            else:
                # Small proportional correction while driving so it doesn't
                # need to stop and re-turn every time it drifts off heading.
                self._ros_node.publish_twist(self._nav_linear_speed, 0.5 * heading_error)

            time.sleep(dt)

        return False

    def _turn_to_yaw(self, target_yaw: float, deadline: float) -> bool:
        """
        Turn in place (no forward motion) to face `target_yaw` (radians,
        world frame). Returns True once within `_nav_heading_tolerance` for
        SETTLE_CHECKS consecutive control cycles, False if `deadline`
        passes first. Called with its own reserved deadline (see
        _on_nav_goal) — separate from the waypoint-driving budget, so a
        long multi-waypoint route can't starve this of any time at all.

        Consecutive-checks, not a single instant, on the theory that
        momentum was carrying it past a momentary tolerance-band pass
        before it actually stopped — it wasn't: this made no measurable
        difference live. What DOES fully explain it: `_latest_pose`'s yaw
        comes from /robotN/odom (integrated wheel odometry, exactly like
        the waypoint-driving loop above uses), not Gazebo's own ground-
        truth pose. Differential-drive odometry drifts from ground truth
        during IN-PLACE ROTATION specifically (wheel-slip integrates
        error much faster turning than translating) — lab_world's own
        wheel friction values are flagged in a pre-existing comment as
        "don't contain reliable data". The robot genuinely believes it
        reached target_yaw; live checks against Gazebo's ground-truth pose
        (/model_states, not available to a real robot and not used here on
        purpose — this loop has to work the same way real hardware would)
        showed it stopped ~20-25° off, consistently. The consecutive-check
        gate is kept anyway (harmless, and it's still the correct defense
        against a genuine momentary-pass/momentum case), but it will NOT
        fully close this gap — a full fix would need a slip-corrected
        odometry source or an external pose reference, out of scope here.
        """
        dt = 1.0 / self._nav_control_hz
        settle_needed = max(1, round(0.3 / dt))  # ~0.3s of genuinely holding still
        settled_for = 0
        while time.time() < deadline:
            with self._pose_lock:
                pose = self._latest_pose
            if pose is None:
                time.sleep(dt)
                continue

            _, _, yaw = pose
            heading_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            if abs(heading_error) <= self._nav_heading_tolerance:
                self._ros_node.publish_twist(0.0, 0.0)  # stop commanding torque while it settles
                settled_for += 1
                if settled_for >= settle_needed:
                    return True
                time.sleep(dt)
                continue
            settled_for = 0

            turn = self._nav_angular_speed if heading_error > 0 else -self._nav_angular_speed
            self._ros_node.publish_twist(0.0, turn)
            time.sleep(dt)

        return False


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "client_config.json"
    try:
        bridge = GazeboBridge(config_file)

        print("\n" + "=" * 60)
        print("  Gazebo Bridge")
        print(f"  Robot     : {bridge.config.get('robot_name', 'Robot')}")
        print(f"  ID        : {bridge.config.get('client_id')}")
        print(f"  Server    : {bridge.config.get('server_url')}")
        print(f"  WS port   : {bridge.config.get('ws_port')}")
        gz = bridge.config.get("gazebo_config", {})
        print(f"  ROS ns    : {gz.get('ros_namespace')}")
        print(f"  Odom topic: {gz.get('odom_topic')}")
        print("=" * 60 + "\n")

        bridge.run()
        return 0

    except FileNotFoundError:
        print(f"Error: config file not found: {config_file}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    code = main()
    # os._exit(), not sys.exit(): rclpy installs its own process-level
    # signal handlers, and observed in testing that after our own stop()
    # sequence completes cleanly (logged "[Client] Stopped.") the process
    # can still fail to actually terminate on SIGTERM — something in
    # rclpy's C-level state outlives our teardown and blocks normal
    # interpreter shutdown. sys.exit() only raises SystemExit, which that
    # state can swallow; os._exit() ends the process immediately at the OS
    # level regardless, which is safe here since stop() has already run.
    os._exit(code)
