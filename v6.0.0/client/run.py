#!/usr/bin/env python3
"""
run.py — start a robot client
=============================
    python3 run.py navel                                  # robots/navel + configs/navel.json
    python3 run.py irobi                                  # no robots/irobi: robots/generic + configs/irobi.json
    python3 run.py gazebo --config configs/sim/navel_01_sim.json

The robot name picks robots/<name> (or robots/generic when there is none) and,
unless --config is given, configs/<name>.json. A new robot platform: copy
robots/_template.py.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name: str):
    """robots.<name>.Robot, or robots.generic.Robot when there is no robots/<name>.
    A robot module that exists but fails to import (its SDK is missing) is an
    error, not a reason to fall back to a robot that cannot drive this body."""
    try:
        module = importlib.import_module(f"robots.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"robots.{name}":
            raise
        module = importlib.import_module("robots.generic")
    return module.Robot


def main() -> int:
    parser = argparse.ArgumentParser(description="Start a robot client.")
    parser.add_argument("robot", help="robots/<robot> to run, e.g. navel, chatbox, pepper, gazebo, irobi")
    parser.add_argument("--config", help="config file (default: configs/<robot>.json)")
    args = parser.parse_args()
    config = args.config or os.path.join(HERE, "configs", f"{args.robot}.json")
    if not os.path.isfile(config):
        print(f"Config not found: {config}")
        return 1
    load(args.robot)(config).run()
    return 0


if __name__ == "__main__":
    code = main()
    # os._exit, not sys.exit: rclpy (gazebo) can keep the process alive after
    # stop() has run cleanly, and nothing else needs the interpreter's teardown.
    sys.stdout.flush()
    os._exit(code)
