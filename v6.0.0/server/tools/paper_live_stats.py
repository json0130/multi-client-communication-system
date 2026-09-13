"""
tools/paper_live_stats.py
=========================
The three numbers the paper's live-data subsection is missing.

    python3 tools/paper_live_stats.py

Reads demo_qa_durations directly (not the aggregate view, which groups away
closed_by and run_id). Prints, per project block and pooled:

    tours            distinct run_id
    n                windows recorded
    visitor-ended    usable observations
    censored         windows that hit their own limit
    median, IQR      over visitor-ended windows only

Paste the table straight into the paper. Simulated runs are excluded by
run_id prefix; check SIM_PREFIXES matches how your simulated runs are tagged.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIM_PREFIXES = ("sim", "lab_demo_sim", "harness")

def main() -> int:
    from data.demo_duration_repo import qa_windows
    rows = qa_windows()
    if not rows:
        print("No rows. Check the database connection and that 008 is applied.")
        return 1

    kept = [r for r in rows
            if not str(r.get("run_id", "")).lower().startswith(SIM_PREFIXES)]
    print(f"{len(rows)} windows total, {len(rows) - len(kept)} excluded as simulated\n")

    by_block: dict = {}
    for r in kept:
        by_block.setdefault(r.get("block_robot_id") or "(none)", []).append(r)

    def visitor_ended(r) -> bool:
        return str(r.get("closed_by", "")).lower() in ("operator", "policy")

    print(f"{'block':22} {'tours':>6} {'n':>5} {'vis':>5} {'cens':>5} "
          f"{'median':>8} {'IQR':>16}")
    for block, rs in sorted(by_block.items()):
        tours = len({r.get("run_id") for r in rs})
        vis = [float(r["seconds"]) for r in rs if visitor_ended(r)]
        cens = len(rs) - len(vis)
        if len(vis) >= 4:
            qs = statistics.quantiles(vis, n=4, method="inclusive")
            med, iqr = statistics.median(vis), f"{qs[0]:.0f}-{qs[2]:.0f}s"
        elif vis:
            med, iqr = statistics.median(vis), "n too small"
        else:
            med, iqr = float("nan"), "no usable windows"
        print(f"{block:22} {tours:>6} {len(rs):>5} {len(vis):>5} {cens:>5} "
              f"{med:>7.0f}s {iqr:>16}")

    all_tours = len({r.get("run_id") for r in kept})
    all_vis = sum(1 for r in kept if visitor_ended(r))
    print(f"\npooled: {all_tours} tours, {len(kept)} windows, "
          f"{all_vis} visitor-ended, {len(kept) - all_vis} censored")
    return 0

if __name__ == "__main__":
    sys.exit(main())