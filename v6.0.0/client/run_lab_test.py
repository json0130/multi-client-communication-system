#!/usr/bin/env python3
"""
run_lab_test.py — Start all 4 test robot instances and connect them to the server.
Usage:  cd v6.0.0/client && python run_lab_test.py
"""
import json
import os
import sys
import time
import threading
import subprocess
import requests

CONFIGS = [
    "test_configs/pepper_01.json",
    "test_configs/chatbox_01.json",
    "test_configs/navel_01.json",
    "test_configs/silbot_01.json",
]
STARTUP_WAIT = 4   # seconds to let robots start their WS servers and self-register


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

    # Spawn all robot processes
    procs = []
    print("Starting lab demo test robots...\n")
    for r in robots:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(here, "test_robot.py"), r["cfg"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=here,
        )
        threading.Thread(target=_stream, args=(proc, r["id"]), daemon=True).start()
        procs.append(proc)
        print(f"  [+] {r['id']}")

    print(f"\nWaiting {STARTUP_WAIT}s for robots to start and register...")
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

    print("\nAll robots ready. Press Ctrl+C to stop.\n")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all robots...")
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
