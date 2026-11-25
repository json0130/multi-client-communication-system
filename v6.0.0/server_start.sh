#!/bin/bash
# server_start.sh — start the server and the dashboard. Ctrl+C stops everything.
#   ./server_start.sh            server + dashboard (http://127.0.0.1:3000)
#   ./server_start.sh --robots   ...plus the four fake lab robots (client/run_lab_test.py)
# Python: $PYTHON, else the repo's robot/ venv, else python3.

set -e
cd "$(dirname "$0")"

ROBOTS=
for arg in "$@"; do
    case $arg in
        --robots) ROBOTS=1 ;;
        -h|--help) sed -n 2,5p "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

PY=${PYTHON:-../robot/bin/python}
[ -x "$PY" ] || PY=python3
PY=$(realpath -s "$(command -v "$PY")")   # absolute (the parts cd first), venv symlink kept

# Any exit (Ctrl+C, or one part dying) takes the other parts down with it.
trap 'trap - EXIT INT TERM; kill 0 2>/dev/null' EXIT INT TERM

[ -d server/robot-dashboard/node_modules ] || (cd server/robot-dashboard && npm install)

(cd server && exec "$PY" app.py) &
SERVER=$!
(cd server/robot-dashboard && exec npm run dev) &

if [ -n "$ROBOTS" ]; then
    # The fake robots register with the server_url in their configs, so wait for that.
    URL=$("$PY" -c 'import json; print(json.load(open("client/configs/lab/silbot_01.json"))["server_url"])')
    echo "Waiting for the server at $URL ..."
    until curl -sf "$URL/health" >/dev/null; do
        kill -0 $SERVER 2>/dev/null || { echo "The server exited before it came up."; exit 1; }
        sleep 1
    done
    (cd client && exec "$PY" run_lab_test.py) &
fi

wait -n
