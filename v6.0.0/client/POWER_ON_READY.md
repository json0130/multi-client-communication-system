# "Power-on and ready" — deploying a robot client

This is the reference pattern so any robot (Navel, ChatBox, Pepper, Silbot, …) becomes
available to the central server automatically whenever it is switched on — no manual IP
editing, no manual launch.

It has three parts, all built into the shared `client/` framework:

## 1. Auto-IP detection (in `client.py`)
The central server **dials out** to each robot, so it needs the robot's LAN IP. Instead
of hand-editing `ip_address` in every `configs/<robot>.json`, `BasicClient` detects it:

- `register_with_server()` checks the configured `ip_address`.
- If it is missing or not a valid IPv4 (e.g. the `REPLACE_WITH_..._IP` placeholder), it
  calls `detect_local_ip(server_url)` and registers that instead.
- A real configured IP is always respected (override wins).

`detect_local_ip()` opens a UDP socket toward the server and reads the local end of the
route — picking the correct interface without sending any packet, so it works even with
no internet. Every robot built on `BasicClient` gets this for free; you can leave
`ip_address` as a placeholder.

## 2. Headless-safe input (in `InputModules/voice_input.py`)
Under systemd there is no terminal, so the interactive `input()` prompt is skipped when
`stdin` is not a TTY (and any `EOFError` is handled). Voice/VAD input still runs normally.

## 3. Boot autostart (systemd)
Use `robot-client.service` as a template. It:
- waits for the network at boot,
- runs the client as the robot user with audio access,
- `Restart=always` so it relaunches on crash.

Install (on the robot, once):
```bash
sudo cp robot-client.service /etc/systemd/system/<robot>-client.service
# edit User / WorkingDirectory (this client/ folder) / ExecStart (run.py <robot>) in that file
sudo systemctl daemon-reload
sudo systemctl enable <robot>-client.service   # start on every boot
sudo systemctl start  <robot>-client.service   # start now
journalctl -u <robot>-client.service -f        # live logs
```

## Adding a new robot — checklist
1. Write `configs/<name>.json` (copy `configs/generic.json`): set `client_id`, `robot_name`,
   `robot_role`, `allowed_tags`, `modules`. You can leave `ip_address` as a placeholder — it
   auto-detects.
2. Speakers and mic on the machine are all it needs? Stop here: `python3 run.py <name>` runs
   it as `robots/generic.py`. Otherwise copy `robots/_template.py` to `robots/<name>.py` and
   fill in how it speaks (`say()`) and moves (`on_emotion_detected()`).
3. Deploy the whole `client/` folder; nothing else is needed beside it.
4. Copy `robot-client.service`, fill in the placeholders, enable it.
5. Power on → it auto-registers and waits for the server. In the dashboard/demo modal the
   robot appears by its `client_id`; connect/select it there.
