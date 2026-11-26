#!/bin/bash
# export_kit.sh — copy the robot client, without this lab's robots, into another repo.
#   ./export_kit.sh <dest>          e.g. ./export_kit.sh ~/stuart/robot_client
# Copies the shared parts only: the protocol (client.py), run.py, ai_server.py, Input/OutputModules,
# robots/_template.py + generic.py, configs/generic.json, setup, service, tests, README.
# Re-run it to update a copy: files that came from here are replaced, the other
# repo's own robots/<name>.py and configs/<name>.json are left alone.

set -e
[ -n "$1" ] || { echo "usage: $0 <dest>"; exit 1; }
cd "$(dirname "$0")"
mkdir -p "$1/robots" "$1/configs"

rsync -a --exclude __pycache__ \
    client.py run.py ai_server.py README.md setup.sh requirements.txt robot-client.service \
    InputModules OutputModules \
    test_protocol.py test_dial_out.py test_tts_pipeline.py test_template.py test_ai_server.py \
    "$1/"
cp robots/__init__.py robots/_template.py robots/generic.py "$1/robots/"
cp configs/generic.json "$1/configs/"
echo "$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+uncommitted')" > "$1/KIT_VERSION"

echo "Robot client kit copied to $1 (version $(cat "$1/KIT_VERSION"))"
