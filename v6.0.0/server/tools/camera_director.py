#!/usr/bin/env python3
"""
tools/camera_director.py
=========================
Points Gazebo's GUI camera at whatever's relevant in the live /demo/status,
so a recorded run isn't a static view the whole time: follows the guide
while it's talking or walking solo, and follows whichever project robot is
presenting once one is.

Implementation note: this previously tried driving the camera through
direct `gz topic -p .../user_camera/pose` publishes, computed to frame BOTH
the guide and the presenter at once (a real midpoint, not just one named
model). That looked right in isolated testing right after a fresh
`gzserver`/`gzclient` restart, but silently did nothing once a real demo
was actually running: `gz topic -i /gazebo/default/user_camera/pose` showed
zero subscribers, meaning nothing was ever listening for a published pose —
whatever made the earlier tests appear to work was not this mechanism
actually taking effect under load. `gz camera -c gzclient_camera -f
<model>` (Classic's built-in follow mode), by contrast, has been reliable
and near-instant every time it's been tried this session. Its only real
downside — once engaged it permanently overrides the camera every frame,
with no CLI "unfollow" — only matters if something else also needs to set
the camera, which nothing here does anymore. Dual-robot framing is dropped
for now: pick this back up only with a verified mechanism, not by
resurrecting the pose-topic approach.

Run alongside tools/video_scenarios.py, pointed at the same server.

    python3 tools/camera_director.py
    python3 tools/camera_director.py --server http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import requests
from Xlib import X, display
from Xlib.ext import xtest

# Center of the Gazebo viewport in the established recording layout
# (Gazebo window at (0,0), 1152x1163 — a 60/40 split against the dashboard
# on a 1920-wide capture; see the video-recording setup this tool is meant
# to run alongside). Only used to aim synthetic scroll events for zoom; if
# that layout changes, this needs to change with it.
VIEWPORT_CENTER = (576, 580)
ZOOM_CLICKS = 15  # scroll-wheel "ticks" to zoom in after each follow switch

# demo_orchestrator robot_id -> Gazebo world model name (see
# gazebo/launch_demo_env.launch.py spawn_robot1..4 -entity args).
WORLD_MODEL = {
    "pepper_01": "robot1",
    "chatbox_01": "robot2",
    "navel_01": "robot3",
    "silbot_01": "robot4",
}
GUIDE = "pepper_01"
PROJECT_ROBOTS = ["chatbox_01", "navel_01", "silbot_01"]

POLL_SEC = 0.5


_xdisplay = None


def _zoom_in(clicks: int = ZOOM_CLICKS) -> None:
    """Scroll-wheel zoom, synthesized via XTest. `gz camera -f` has no
    distance/zoom flag — this is the only way found to close the follow
    distance in, and it composes fine with follow mode: scrolling while
    already following just tightens the same tracked offset rather than
    fighting it (confirmed live)."""
    global _xdisplay
    if _xdisplay is None:
        _xdisplay = display.Display(':0')  # not inherited reliably from a
                                            # background/nohup'd process
    root = _xdisplay.screen().root
    x, y = VIEWPORT_CENTER
    root.warp_pointer(x, y)
    _xdisplay.sync()
    for _ in range(clicks):
        xtest.fake_input(_xdisplay, X.ButtonPress, 4)   # button 4 = scroll up = zoom in
        _xdisplay.sync()
        xtest.fake_input(_xdisplay, X.ButtonRelease, 4)
        _xdisplay.sync()
        time.sleep(0.03)


def _follow(model: str) -> None:
    subprocess.run(
        ["bash", "-c", f"source /opt/ros/humble/setup.bash && "
                        f"gz camera -c gzclient_camera -f {model}"],
        timeout=5, capture_output=True,
    )
    time.sleep(0.3)  # let the follow-lock settle before scrolling, or the
                      # zoom can land before the camera re-centers on the
                      # new target.
    _zoom_in()
    print(f"  [camera] following {model}, zoomed in")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    args = ap.parse_args()

    current = None
    while True:
        try:
            s = requests.get(f"{args.server}/demo/status", timeout=5).json()
        except requests.RequestException:
            time.sleep(POLL_SEC)
            continue

        if s["state"] in ("completed", "idle", "error"):
            print(f"Demo {s['state']} — camera director stopping.")
            return 0

        robot = s.get("robot_id")
        # Prefer the presenting project robot over the guide when both are
        # "active" at once (e.g. a Q&A window nominally belongs to the
        # guide's step, but the interesting subject is still whichever
        # robot just presented).
        target = WORLD_MODEL.get(robot) if robot in (PROJECT_ROBOTS + [GUIDE]) else None

        if target is not None and target != current:
            _follow(target)
            current = target

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
