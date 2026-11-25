# Robot client

Connects a robot to the CARES multi-robot orchestration server. The server
holds the conversation, personas and roles; this client turns what it sends
into speech and gestures on your robot, and sends back what the robot hears.

You write one file for your robot (the adapter) and one config. Everything
else (registering, the WebSocket, sentence streaming, demo steps and their
ACKs, barge-in, speaking state, persona changes) is already in `client.py`.

Needs Linux and Python 3.8 or newer.

## Add a robot

1. **Install.** `./setup.sh` makes `.venv` and installs `requirements.txt`.
   On a Jetson Nano (Python 3.6) it installs python3.8 alongside.

2. **Write the adapter.** Copy `robots/_template.py` to `robots/<name>.py`
   (or `robots/<name>/__init__.py` if it needs more files) and fill in:

   | Method | What it does |
   |---|---|
   | `Speech.say(sentence)` | Speak one sentence with your robot's TTS; return once it has been said. Tags are already removed. |
   | `Robot.on_emotion_detected(tag)` | Perform a gesture, e.g. `WAVE`. Ignore tags you don't know. |
   | `Speech.update_voice_config(cfg)` | Optional. Change voice when the server assigns a persona. |
   | `Speech.initialize()` | Optional. Connect to your SDK; return `False` if it isn't there. |

   If `say()` can't block until speech ends, start it, then wait with
   `self._interrupt_event.wait(self.estimated_duration(sentence))` so
   barge-in still works.

   A robot with its own mic array: write an InputModule that calls
   `self.client.send_to_server("speech", wav_bytes)` and register it instead
   of `VoiceInputModule`.

   No body to drive, just the machine's speakers and mic? Skip this step:
   `run.py` uses `robots/generic.py` for any robot without a module.

3. **Write the config.** Copy `configs/generic.json` to `configs/<name>.json`:

   | Key | |
   |---|---|
   | `client_id` | Unique per robot, e.g. `stuart_001` |
   | `robot_name` | Shown in the dashboard |
   | `server_url` | e.g. `http://<server-ip>:5000` |
   | `robot_role` | Default role until the dashboard assigns one |
   | `allowed_tags` | Gestures this body can do, e.g. `"[WAVE]"`. The server only asks for these. |
   | `modules` | Include `"speech"` to use the mic |
   | `connect_to_server` | `true` if the server can't reach the robot (NAT, Tailscale): the robot dials out instead |

   If the server uses a token, set `API_TOKEN` in the environment. Never put
   it in the config.

4. **Run.** `.venv/bin/python run.py <name>`. The banner shows the server it
   registered with. To start at every boot, fill in `robot-client.service`
   and install it (instructions inside).

## Check it

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest test_*.py
```

`test_template.py` checks the contract an adapter inherits: tags stripped,
sentences in order, `interrupt()` returns the unspoken rest, the demo-step
ACK still fires, speaking state is reported.

## Copying this to another repo

From the server repo's `v6.0.0/client`: `./export_kit.sh <dest>`. That copies
only the shared parts (no lab robots) and writes `KIT_VERSION`, the commit it
came from. Re-run it to update: shared files are replaced, your own
`robots/<name>.py` and `configs/<name>.json` are kept. Don't edit the shared
files in the copy; changes belong in the server repo.
