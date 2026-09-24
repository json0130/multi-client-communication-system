# Multi-robot communication system

Role-adaptive LLM orchestration server that drives several robots at once (Pepper,
SilBot, iRobi, Navel, ChatBox, plus a Gazebo sim client). Part IV research project,
University of Auckland.

**Work happens in `v6.0.0/`.** `v1-5/` is an archived tree — don't edit it, and don't
let tooling collect from it. `v6.0.0/README.md` is the detailed reference (schema,
env vars, HTTP API, profile format); this file only covers what isn't in there.

## Layout (`v6.0.0/`)

| Path | What |
|---|---|
| `server/app.py` | entry point — wires modules, starts Flask |
| `server/core/` | `config.py` (all env vars), `profiles/`, `rbac/` |
| `server/data/` | the only place that touches Supabase; one repo per table, `migrations/` |
| `server/modules/` | llm / speech / emotion / rag, each a `BaseModule` |
| `server/robot/` | per-robot business logic, prompt building, registry |
| `server/gateway/` | Flask routes + outbound WebSocket pool (server dials **out** to robots) |
| `server/decision/` | flow planner, grounding, KG, presence, visitor profile, style fit |
| `server/demo/` | lab demo orchestrator + script |
| `server/tools/` | `check_*.py` layer checkpoints, eval/harness/stats scripts |
| `client/`, `*_client/` | per-robot clients; each runs a WebSocket server locally |
| `gazebo/`, `gazebo_client/` | simulated lab world + bridge (current branch) |

## Commands

```bash
cd v6.0.0/server && python3 -m pytest tests/      # ~1s, hermetic
python3 app.py                                    # from v6.0.0/server
```

- Run pytest **from `v6.0.0/server`** — `pytest.ini` scopes collection there on purpose.
- The suite is hermetic: no network, DB, Ollama or `.env`. A guard fixture fails any
  test that reaches a real Supabase client; opt out with `@pytest.mark.allow_db` only
  if there's no other way.
- `tools/check_*.py` are manual checkpoints that need live infra, and `check_data.py`
  writes real rows to the database. Don't run them to "verify" a change; add a test.
  `check_rbac.py` is the exception — pure logic, no infra.

## Conventions

- Env vars are read in `core/config.py` and nowhere else; import from there.
- Supabase only via `data/` repos. No ad-hoc queries in gateways or modules.
- Scenario profiles (`server/profiles/*.yaml`) are validated at boot and the server
  **refuses to start** if one is invalid — that strictness is deliberate, keep it.
- Schema changes go in `data/migrations/NNN_*.sql` with a matching `*_down.sql`.
- Demo bugs: fix the reported symptom minimally; flag any deeper redesign as an option
  rather than doing it.

# Code graph (graphify)

A code-structure graph is kept up to date by git hooks at `graphify-out/graph.json`.

- **Before opening a code file over ~500 lines, run `graphify explain` on the target and
  read only the line ranges it lists** (Read with offset/limit), not the whole file.
- For how code connects (callers, dependencies, where something is defined), query the
  graph before grepping or reading many files:
  - `graphify explain "<function, class or file>"` for a node and its neighbors
  - `graphify path "<A>" "<B>"` for the shortest path between two nodes
  - `graphify affected "<X>"` before changing X, to see what depends on it
- Skip the graph for small files, a file the user named, or text searches (log strings,
  env vars, prompt text); grep/read directly is cheaper there.
- If a name is ambiguous, the command lists candidate node ids; re-run with a full id.
- Do not read `graphify-out/GRAPH_REPORT.md` or `graph.json` in full; they are large.
- The graph covers code only, not docs or SQL, and can lag by a commit. If results look stale or wrong, fall back to grep/read and say so.
