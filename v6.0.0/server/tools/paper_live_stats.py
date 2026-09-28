"""
tools/paper_live_stats.py  (v2)
===============================
The live-data numbers for the paper, estimated the way the SYSTEM estimates
them rather than the way that is convenient.

    python3 tools/paper_live_stats.py
    python3 tools/paper_live_stats.py --runs     # also list run ids

v1 of this script took a plain median over visitor-ended windows only. That is
precisely the treatment demo_duration_repo.censored_median exists to avoid: the
windows that hit their limit are the LONG ones, for the projects visitors want
more of, so dropping them biases those projects down. This version reports the
Kaplan-Meier median the planner actually consumes, and prints the naive figure
beside it so the size of that bias is visible.

Three exclusions, all reported rather than silent:
  * every run that is not mode 'live', format 'full_tour' (demo_runs), counted
    per mode/format — a window whose run has no demo_runs row is 'unknown'
  * the guide robot's own windows      (it hosts, it does not present)
  * windows with no block              (the open-floor round at the end)
"""
import argparse, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GUIDE_IDS = {"pepper_01", "pepper_001"}
PAPER_KIND = ("live", "full_tour")


def split_by_kind(rows, runs) -> tuple:
    """(paper_rows, {(mode, format): n}) — the windows of live full-tour runs,
    and how many windows every kind of run contributed. `runs` is
    demo_run_repo.run_modes(): {run_id: {"mode", "format"}}."""
    kept, counts = [], {}
    for r in rows:
        run = runs.get(r.get("run_id")) or {}
        kind = (run.get("mode") or "unknown", run.get("format") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == PAPER_KIND:
            kept.append(r)
    return kept, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="store_true", help="list run ids and exit")
    args = ap.parse_args()

    from data.demo_duration_repo import (censored_median, qa_median_by_block,
                                         qa_windows, MIN_BLOCK_WINDOWS)
    from data.demo_run_repo import run_modes
    rows = qa_windows()
    runs = run_modes()
    if not rows:
        print("No rows. Check the connection and that migration 008 is applied.")
        return 1

    if args.runs:
        seen: dict = {}
        for r in rows:
            seen.setdefault(str(r.get("run_id")), 0)
            seen[str(r.get("run_id"))] += 1
        print(f"{len(seen)} distinct run ids\n")
        for rid, n in sorted(seen.items()):
            run = runs.get(rid) or {}
            print(f"  {rid:40} {run.get('mode') or 'unknown':>16} "
                  f"{run.get('format') or 'unknown':>9} {n:>3} windows")
        return 0

    def visitor_ended(r) -> bool:
        """The repo's rule, copied from demo_duration_repo._windows_by_block:
        only a window that ran out its own allocation is censored. An earlier
        version of this script used an allow-list of closers, which silently
        counted 'auto' and 'unknown' windows as censored and disagreed with the
        planner's own medians."""
        return (r.get("closed_by") or "") != "timeout"

    total = len(rows)
    rows, by_kind = split_by_kind(rows, runs)

    project, guide, openfloor = [], [], []
    for r in rows:
        b = r.get("block_robot_id")
        (guide if b in GUIDE_IDS else openfloor if not b else project).append(r)

    print(f"{total} windows in the table, by run mode / format:")
    for (mode, fmt), n in sorted(by_kind.items()):
        mark = "  <-- analysed" if (mode, fmt) == PAPER_KIND else "  excluded"
        print(f"  {n:>4} {mode}/{fmt}{mark}")
    print(f"  {len(guide):>4} excluded as guide-robot windows")
    print(f"  {len(openfloor):>4} excluded as open-floor (no project block)")
    print(f"  {len(project):>4} project windows analysed\n")

    by_block: dict = {}
    for r in project:
        by_block.setdefault(r["block_robot_id"], []).append(r)

    km = qa_median_by_block(rows)
    print(f"{'project':14} {'n':>4} {'vis':>4} {'cens':>5} "
          f"{'KM median':>11} {'naive':>7} {'bias':>7} {'IQR (vis)':>14}")
    for block, rs in sorted(by_block.items()):
        vis = [float(x["seconds"]) for x in rs if visitor_ended(x)]
        cens = len(rs) - len(vis)
        pairs = [(float(x["seconds"]), visitor_ended(x)) for x in rs]
        kmv, lower = censored_median(pairs)
        naive = statistics.median(vis) if vis else float("nan")
        iqr = "-"
        if len(vis) >= 4:
            q = statistics.quantiles(vis, n=4, method="inclusive")
            iqr = f"{q[0]:.0f}-{q[2]:.0f}s"
        trusted = "" if len(vis) >= MIN_BLOCK_WINDOWS else "  (below trust threshold)"
        star = ">=" if lower else ""
        print(f"{block:14} {len(rs):>4} {len(vis):>4} {cens:>5} "
              f"{star}{kmv:>10.0f}s {naive:>6.0f}s {kmv-naive:>+6.0f}s {iqr:>14}"
              f"{trusted}")

    print(f"\nplanner-visible medians (qa_median_by_block): {km}")
    tours = len({r.get("run_id") for r in project})
    vis_n = sum(1 for r in project if visitor_ended(r))
    print(f"pooled: {tours} tours, {len(project)} project windows, "
          f"{vis_n} visitor-ended, {len(project)-vis_n} censored")
    print("\n'>=' marks a median that is only a lower bound: the longest windows "
          "were all cut off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())