#!/usr/bin/env python3
"""
tools/model_transfer.py
=======================
Sim-to-real transfer: copy what a simulation twin knows to its physical robot,
as a numbered, recorded version (migration 017).

    python3 tools/model_transfer.py --pair navel_01:navel_001            # dry run
    python3 tools/model_transfer.py --pair navel_01:navel_001 \\
        --pair pepper_01:pepper_001 --pair chatbox_01:chatbox_jetson_001  # dry run
    python3 tools/model_transfer.py --pair navel_01:navel_001 --apply --notes "..."
    # rename to a held-back twin's physical id without transferring it:
    python3 tools/model_transfer.py --pair pepper_01:pepper_001 \
        --twin chatbox_01:chatbox_jetson_001

A dry run (the default) reads the live database and prints exactly what would be
copied; nothing is written. --apply writes it, after the operator has approved
the dry run, and records the transfer both in model_transfers and as a file in
data/transfers/ (commit that file).

What is copied, per pair:
  facts             as-is, in their original order (grounding reads them in order)
  declared scope    eligible / specialised, for edges that differ from the
                    default; the physical robot's own live weight and counts are
                    never touched
  KG / style prior  prior_weight + prior_n, only for edges with sim evidence
                    (the router caps sim evidence at SIM_PRIOR_CAP observations)
  duration priors   median per step and censored median per Q&A block, from
                    full-tour runs in PRIOR_MODES; movement steps excluded (a
                    physical robot that can't move spends 0 s on them)
Log rows (decisions, corrections, durations, chat) are never copied.

Step ids that name a simulation twin are renamed to its physical twin using
EVERY --pair and --twin given (introduce_navel_01, spoken by the guide, becomes
introduce_navel_001). --twin only renames; nothing of that robot is copied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRIOR_MODES = ("mock", "gazebo", "scripted_visitor")
"""Simulated runs whose timings are evidence. Never 'live' (that is the robot's
own data), never 'headless_sim' (no robots or speech) or 'sim_unknown' (client
type unrecoverable)."""

from decision.kg import SIM_PRIOR_CAP  # noqa: E402  applied when routing reads a prior

TRANSFERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "transfers")


def _is_movement(step_id: str) -> bool:
    return step_id.startswith(("navigate_", "face_")) or step_id == "approach_visitors"


def rename_step(step_id: str, twin_map: dict) -> str:
    """Replace every simulation id in a step id with its physical twin.

    Whole-id matches only, so navel_01 never rewrites part of navel_011.
    """
    for sim in sorted(twin_map, key=len, reverse=True):
        step_id = re.sub(rf"(?<![A-Za-z0-9]){re.escape(sim)}(?![A-Za-z0-9])",
                         twin_map[sim], step_id)
    return step_id


def expected_speakers(twin_map: dict) -> dict:
    """{step_id: robot_id} for the current full-tour script, physical lineup.

    The guide and the three project robots, each replaced by its physical twin
    where one is being transferred to. A step prior is only worth copying if
    the physical robot will actually speak that step: old script names, and
    guide lines a project robot spoke in a one-robot test run, are dropped.
    """
    from demo.demo_script import CHATBOX, NAVEL, PEPPER, SILBOT, build_script
    phys = lambda r: twin_map.get(r, r)   # noqa: E731
    return {s.step_id: s.robot_id
            for s in build_script(phys(PEPPER), [phys(CHATBOX), phys(NAVEL), phys(SILBOT)])}


def plan_transfer(pair: tuple, twin_map: dict, snap: dict, version: int,
                  speakers: dict = None) -> dict:
    """What transfer `version` of `pair` would write, from a source snapshot.

    Pure: no database access. `snap` is data.model_transfer_repo.source_snapshot;
    `speakers` is expected_speakers() — when given, a step prior is kept only if
    the physical robot speaks that step in the current script.
    """
    from data.demo_duration_repo import MIN_BLOCK_WINDOWS, censored_median

    src, dst = pair
    runs = snap["runs"]

    def prior_run(run_id):
        r = runs.get(run_id)
        return r is not None and r["mode"] in PRIOR_MODES and r.get("format") == "full_tour"

    facts = [{"robot_id": dst, "topic_id": f["topic_id"], "kind": f["kind"],
              "fact": f["fact"], "verified": f["verified"], "source": f["source"]}
             for f in sorted(snap["facts"], key=lambda f: f["id"])]

    scope, kg_priors = [], []
    for e in sorted(snap["edges"], key=lambda e: e["topic_id"]):
        n_obs = (e.get("n_supervisor") or 0) + (e.get("n_outcome") or 0) + (e.get("n_displaced") or 0)
        declared = e.get("specialised") or e.get("eligible") is False
        if not (declared or n_obs):
            continue            # the default state: nothing to carry across
        row = {"robot_id": dst, "topic_id": e["topic_id"],
               "eligible": e.get("eligible", True), "specialised": e.get("specialised", False)}
        if n_obs:
            row.update(prior_weight=e["weight"], prior_n=n_obs)
            kg_priors.append(e["topic_id"])
        scope.append(row)

    style_priors = [{"robot_id": dst, "style": s["style"],
                     "prior_weight": s["weight"], "prior_n": s["n_supervisor"]}
                    for s in snap["styles"] if (s.get("n_supervisor") or 0) > 0]

    excluded = {"movement_steps": 0, "non_prior_runs": 0, "not_its_step": 0}
    by_step: dict = {}
    modes_used: set = set()
    for d in snap["step_rows"]:
        if _is_movement(d["step_id"]):
            excluded["movement_steps"] += 1
            continue
        if not prior_run(d["run_id"]):
            excluded["non_prior_runs"] += 1
            continue
        sid = rename_step(d["step_id"], twin_map)
        if speakers is not None and speakers.get(sid) != dst:
            excluded["not_its_step"] += 1
            continue
        by_step.setdefault(sid, []).append(float(d["seconds"]))
        modes_used.add(runs[d["run_id"]]["mode"])

    step_modes = "+".join(sorted(modes_used))
    duration_priors = [{"robot_id": dst, "step_id": sid, "kind": "step",
                        "median_sec": round(statistics.median(v), 2), "n": len(v),
                        "source_mode": step_modes}
                       for sid, v in sorted(by_step.items())]

    qa_obs, qa_modes, qa_excluded = [], set(), 0
    for q in snap["qa_rows"]:
        if not prior_run(q["run_id"]):
            qa_excluded += 1
            continue
        qa_obs.append((float(q["seconds"]), (q.get("closed_by") or "") != "timeout"))
        qa_modes.add(runs[q["run_id"]]["mode"])
    qa = None
    ended = sum(1 for _t, e in qa_obs if e)
    if qa_obs:
        median, lower = censored_median(qa_obs)
        qa = {"median_sec": round(median, 1), "lower_bound": lower,
              "windows": len(qa_obs), "visitor_ended": ended,
              "copied": ended >= MIN_BLOCK_WINDOWS}
    # The same bar the planner sets for a block's own median: fewer
    # visitor-ended windows than that is not a prior worth carrying.
    if qa and qa["copied"]:
        duration_priors.append({"robot_id": dst, "step_id": f"qa_invite_{dst}", "kind": "qa",
                                "median_sec": round(median, 1), "n": ended,
                                "source_mode": "+".join(sorted(qa_modes))})

    items = {
        "facts": len(facts),
        "scope_rows": len(scope),
        "kg_priors": len(kg_priors),
        "style_priors": len(style_priors),
        "step_priors": len(by_step),
        "qa_prior": qa,
        "prior_modes": list(PRIOR_MODES),
        "step_source_modes": step_modes,
        "excluded": {**excluded, "qa_non_prior_runs": qa_excluded},
        "sim_prior_cap": SIM_PRIOR_CAP,
        "twin_map": twin_map,
    }
    return {"from": src, "to": dst, "version": version, "facts": facts, "scope": scope,
            "kg_prior_topics": kg_priors, "style_priors": style_priors,
            "duration_priors": duration_priors, "items": items,
            "target_edges_before": snap.get("target_edges", []),
            "replaces_transferred_facts": len(snap.get("target_transferred_facts", []))}


def _print_plan(p: dict) -> None:
    it = p["items"]
    before = {e["topic_id"]: e for e in p["target_edges_before"]}
    print(f"\n══ Transfer #{p['version']}: {p['from']} → {p['to']} ══")
    print(f"  Facts: {it['facts']} copied (replacing {p['replaces_transferred_facts']} "
          f"copied by an earlier transfer)")
    kinds: dict = {}
    for f in p["facts"]:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    if kinds:
        print("    by kind: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))
    for f in p["facts"][:3]:
        print(f"    e.g. [{f['kind']}] {f['topic_id']}: {f['fact'][:90]}…")
    print(f"  Declared scope: {it['scope_rows']} edges")
    for e in p["scope"]:
        state = "update" if e["topic_id"] in before else "new"
        prior = (f", prior {e['prior_weight']:.3f} over {e['prior_n']} obs"
                 if "prior_weight" in e else "")
        print(f"    {state:6} {e['topic_id']:<36} specialised={e['specialised']} "
              f"eligible={e['eligible']}{prior}")
    print(f"  KG priors (edges with sim evidence): {it['kg_priors']}"
          f"{'  — none: the twin has no learned competence yet' if not it['kg_priors'] else ''}")
    print(f"  Style-fit priors: {it['style_priors']}")
    steps = [d for d in p["duration_priors"] if d["kind"] == "step"]
    print(f"  Step-duration priors: {len(steps)} steps, from {it['step_source_modes'] or '—'} runs")
    for d in steps:
        print(f"    {d['step_id']:<40} median {d['median_sec']:>6.1f}s  n={d['n']}")
    qa = it["qa_prior"]
    if qa and qa["copied"]:
        print(f"  Q&A prior: qa_invite_{p['to']} censored median {qa['median_sec']}s"
              f"{' (lower bound)' if qa['lower_bound'] else ''}, {qa['visitor_ended']} "
              f"visitor-ended of {qa['windows']} windows")
    elif qa:
        print(f"  Q&A prior: NOT copied — {qa['visitor_ended']} visitor-ended window(s), "
              f"below the planner's own bar of 5")
    else:
        print("  Q&A prior: none (no windows in prior modes)")
    ex = it["excluded"]
    print(f"  Excluded: {ex['movement_steps']} movement-step rows; {ex['not_its_step']} rows for "
          f"steps {p['to']} does not speak in the current script (old names, one-robot tests); "
          f"{ex['non_prior_runs']} step rows and {ex['qa_non_prior_runs']} Q&A windows from runs "
          f"not in {list(PRIOR_MODES)} or not full_tour")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", action="append", required=True, metavar="SIM:PHYSICAL",
                    help="transfer this twin")
    ap.add_argument("--twin", action="append", default=[], metavar="SIM:PHYSICAL",
                    help="name a twin for step-id renaming and the lineup only; nothing "
                         "of it is transferred (e.g. a pair held back for later)")
    ap.add_argument("--apply", action="store_true",
                    help="write the transfer (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true", help="the default; nothing is written")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()
    if args.apply and args.dry_run:
        ap.error("--apply and --dry-run are exclusive")

    def parse(values, flag):
        out = []
        for p in values:
            if p.count(":") != 1 or not all(p.split(":")):
                ap.error(f"{flag} must be SIM:PHYSICAL, got {p!r}")
            out.append(tuple(p.split(":")))
        return out
    pairs = parse(args.pair, "--pair")
    twin_map = {**dict(parse(args.twin, "--twin")), **dict(pairs)}

    from data import model_transfer_repo as repo

    speakers = expected_speakers(twin_map)
    plans = []
    for src, dst in pairs:
        snap = repo.source_snapshot(src, dst)
        if not snap["runs"]:
            # demo_runs has RLS and no policies: under the anon key it reads as
            # empty, every timing would be excluded, and the plan would look
            # valid. Refuse instead of planning from nothing.
            print("demo_runs is empty or unreadable. SUPABASE_KEY must be the "
                  "service_role key (the new tables have RLS and no policies).")
            return 1
        missing = [r for r in (src, dst) if r not in snap["robots"]]
        if missing:
            print(f"Unknown robot id(s) {missing} — not in the robots table.")
            return 1
        plans.append(plan_transfer((src, dst), twin_map, snap, repo.next_version(src, dst),
                                   speakers=speakers))

    for p in plans:
        _print_plan(p)

    if not args.apply:
        print("\nDRY RUN — nothing was written. Re-run with --apply once approved.")
        return 0

    os.makedirs(TRANSFERS_DIR, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=TRANSFERS_DIR).stdout.strip()
    except Exception:
        commit = ""
    for p in plans:
        name = f"{p['from']}__{p['to']}__v{p['version']}.json"
        rel = os.path.join("v6.0.0", "server", "data", "transfers", name)
        tid = repo.apply_transfer(p, repo_record=rel, notes=args.notes)
        record = {"transfer_id": tid, "applied_at": datetime.now(timezone.utc).isoformat(),
                  "code_commit": commit, "notes": args.notes,
                  **{k: p[k] for k in ("from", "to", "version", "items", "facts", "scope",
                                       "style_priors", "duration_priors")}}
        with open(os.path.join(TRANSFERS_DIR, name), "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"\nApplied transfer #{p['version']} ({p['from']} → {p['to']}) as id {tid}; "
              f"recorded in {rel} — commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
