#!/usr/bin/env python3
"""
run_gazebo_demo.py — Start all 3 simulated robot bridges and connect them to
the server. The Gazebo equivalent of client/run_lab_test.py.

Requires Gazebo already running (gazebo/launch_demo_env.launch.py) and ROS2
sourced in THIS shell before running this script — child processes inherit
this process's environment, so `source /opt/ros/humble/setup.bash` once here
is enough for all three bridges.

Usage:  cd v6.0.0/gazebo_client && python3 run_gazebo_demo.py
"""
import json
import os
import sys
import time
import threading
import subprocess
import requests

CONFIGS = [
    "configs/pepper_01_sim.json",
    "configs/chatbox_01_sim.json",
    "configs/navel_01_sim.json",
    "configs/silbot_01_sim.json",
]
STARTUP_WAIT = 5   # seconds — a touch longer than run_lab_test.py's, since
                    # rclpy.init() + node/subscription setup adds a little
                    # startup time on top of the usual WS-server + register.


def _stream(proc, prefix):
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="", flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # Read metadata from each config
    robots = []
    server_url = "http://127.0.0.1:5000"
    for rel in CONFIGS:
        path = os.path.join(here, rel)
        with open(path) as f:
            cfg = json.load(f)
        robots.append({"id": cfg["client_id"], "cfg": rel})
        server_url = cfg.get("server_url", server_url)

    # Spawn all bridge processes
    procs = []
    print("Starting Gazebo-simulated robot bridges...\n")
    for r in robots:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(here, "gazebo_bridge.py"), r["cfg"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=here,
        )
        threading.Thread(target=_stream, args=(proc, r["id"]), daemon=True).start()
        procs.append(proc)
        print(f"  [+] {r['id']}")

    print(f"\nWaiting {STARTUP_WAIT}s for bridges to start and register...")
    time.sleep(STARTUP_WAIT)

    # Auto-connect all robots via server HTTP API
    print(f"\nConnecting to server at {server_url} ...")
    for r in robots:
        try:
            resp = requests.post(f"{server_url}/robots/{r['id']}/connect", timeout=5)
            status = "OK" if resp.ok else f"FAILED — {resp.text}"
        except Exception as e:
            status = f"ERROR — {e}"
        print(f"  {r['id']}: {status}")

    print("\nAll bridges ready. Press Ctrl+C to stop.\n")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all bridges...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
