#!/bin/bash
# setup.sh — install the robot client on any Linux robot or laptop (once per machine).
#   ./setup.sh
# Makes client/.venv (Python >= 3.8) and installs requirements.txt into it. Then run
#   .venv/bin/python run.py <robot>
# or point robot-client.service's ExecStart at that python.
#
# The venv sees the system's packages (--system-site-packages), so a robot SDK that
# lives in the system python (Navel's navel, ROS's rclpy) still imports.
# Jetson Nano (JetPack 4 = Ubuntu 18.04) ships Python 3.6, too old; this installs
# python3.8 from apt beside it.

set -e
cd "$(dirname "$0")"

sudo apt-get update -qq
sudo apt-get install -y ffmpeg alsa-utils portaudio19-dev python3-pip python3-dev python3-venv

PY=python3
if ! $PY -c 'import sys; sys.exit(sys.version_info < (3, 8))'; then
    sudo apt-get install -y python3.8 python3.8-dev python3.8-venv
    PY=python3.8
fi

$PY -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Done. Start a robot:  $(pwd)/.venv/bin/python run.py <robot>"
