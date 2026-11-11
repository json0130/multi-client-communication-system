# Roles: role + persona assignment on this server

The Roles page lets you give a robot a **role** (companion, storyteller, tutor, receptionist, reading buddy…)
and an **OCEAN personality**, see the role/persona graph, and talk to the robot as that persona. It was built
on the ChatBox project (graph_relationship, persona_chat, persona_voice, Method B action tags) and is moved
here so it runs on the same robot connections and the same robot client code as the lab demo.

## How it fits

```
dashboard "Roles" tab ──iframe──▶ Roles page (modules/graph_relationship/viz, port ROLES_PORT, default 8780)
                                          │  persona stack: graph + role library, persona_chat, persona_voice,
                                          │  action_tag_generator (Method B)
                                          ▼
                             gateway/roles_gateway.py  GatewayHub
                                          ▼
                             WebSocketGateway.send_to_robot   (the pool that dials out to every robot)
                                          ▼
                       shared client (client/): ChatBox, Navel — same code, same two events
```

The robots are the ones the server already manages (`chatbox_jetson_001`, `navel_001`); the Roles page does
not keep its own connections. A robot must be connected (Robots tab) before the page can reach it.

## What goes over the wire

| Event | Payload | Notes |
|---|---|---|
| `persona_update` | `persona_name`, `personality` (OCEAN 0..1), and for the ChatBox `robot_role`, `voice_config` | A Navel only gets name + personality: it keeps its own role, tags and voice in its client_config. |
| `demo_step` | `step_id` (`persona-N`), `text` = `[TAG] words`, `require_ack` | Method B: one gesture tag per sentence, each sent once the robot acked the last. Navel's tags are its own 11; the ChatBox's are its EMOTION_MAP keys. |

`ack` for a `persona-N` step is routed to the Roles page (`WebSocketGateway.ack_listeners`); every other ack
still goes to the demo orchestrator. A Navel also gets its voice spec as TTS tags (`<vol,N><rpit,N><rspd,N>`)
in front of each line.

## Run

```bash
cd v6.0.0/server && python3 app.py          # the Roles page starts with the server, on ROLES_PORT
cd robot-dashboard && npm run dev           # dashboard; the Roles tab embeds http://<host>:8780/
```

| Env var | Default | Meaning |
|---|---|---|
| `ROLES_PORT` | `8780` | Roles page port; `0` turns it off |
| `ROLES_DATA_DIR` | `./roles_data` | its graph / persona-assignment / registry files (gitignored) |
| `ROLES_AI_SERVER` | `` | `HOST[:PORT]` AI server for the persona chat; empty = the server's Ollama (`OLLAMA_*`) |
| `ROLES_SYNC_DB` | `0` | `1`: an assigned role is also written to `robots.robot_role`, which is what the server's own turns (demo Q&A) use. Leave it off while running the lab demo: it would replace the robot's demo persona. |
| `VITE_ROLES_URL` | `http://<host>:8780/` | where the dashboard's Roles tab points |

## Limits

- The persona chat on this page is the persona stack's own (graph memory of the person), not the demo's
  grounded Q&A. With `ROLES_SYNC_DB=1` the demo's turns use the assigned role too.
- Do not drive a robot from the Roles page and a demo run at the same time: both speak through the same connection.
- A robot that connects after a role was assigned is not re-sent it (the server dials the robot; there is no
  connect hook yet). Assign again, or use `ROLES_SYNC_DB=1` so the server holds the role.
- Robots are added on the server's own Robots tab; the Roles page's own add-robot form is disabled here.
- The Roles tab embeds the page as it is. Rewriting it as React components is possible later.
- Tests: `tests/test_roles_gateway.py` (adapter, ack routing, page API with a stub LLM) and `tests/persona/`.
