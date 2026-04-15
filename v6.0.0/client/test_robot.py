#!/usr/bin/env python3
"""
test_robot.py
=============
Thin launcher that accepts a config file path as a CLI argument.
Allows running multiple robot instances from separate terminals.

Usage:
    python test_robot.py test_configs/pepper_01.json
    python test_robot.py test_configs/chatbox_01.json
    python test_robot.py test_configs/navel_01.json
    python test_robot.py test_configs/silbot_01.json
"""

import sys
import logging
import argparse

from robot import SimpleConcurrentClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Start a robot client with a custom config.")
    parser.add_argument("config", help="Path to client config JSON file")
    args = parser.parse_args()

    try:
        client = SimpleConcurrentClient(args.config)
        client.print_startup_info()
        client.run()
    except FileNotFoundError:
        print(f"Error: config file not found — {args.config}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
