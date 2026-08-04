"""
tools/check_rbac.py
===================
Checkpoint script for the RBAC layer, in the same style as the other
tools/check_*.py scripts.

Unlike most of them this needs no database, no Ollama and no robot hardware —
core.rbac.policy is pure and the scenario profiles are local files. Run it any
time to confirm the access matrix and the shipped profiles are sane:

    python3 tools/check_rbac.py

For the full suite (including the FAISS top-k regression), use pytest:

    python3 -m pytest tests/ -v
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.profiles import ProfileError, ProfileRegistry          # noqa: E402
from core.rbac import (                                          # noqa: E402
    AccessLevel,
    ClearanceError,
    ClearedRecord,
    Decision,
    GrantStore,
    MemoryAuditSink,
    MemoryRecord,
    RBACFilter,
    Reason,
    RobotIdentity,
    Visibility,
    assert_cleared,
    can_read,
    new_grant,
)

PASSED = 0
TOTAL = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global PASSED, TOTAL
    TOTAL += 1
    if condition:
        PASSED += 1
    icon = "  PASS" if condition else "  FAIL"
    suffix = f"   ({detail})" if detail and not condition else ""
    print(f"{icon}  {label}{suffix}")
    return condition


SCENARIO = "lab_demo"
MANAGER = "pepper_01"
WORKER = "silbot_01"
PEER = "navel_001"
NOW = datetime.now(timezone.utc)


def _rec(rid="faiss:1#0", src=WORKER, vis=Visibility.LOCAL, scen=SCENARIO):
    return MemoryRecord(record_id=rid, content="text", source_robot_id=src,
                        scenario_id=scen, visibility=vis)


def check_policy_matrix():
    print("\nAccess matrix")
    mgr = RobotIdentity(MANAGER, SCENARIO, "s", AccessLevel.GLOBAL)
    wrk = RobotIdentity(WORKER, SCENARIO, "s", AccessLevel.LOCAL)

    d = can_read(mgr, _rec(src=MANAGER), (), NOW)
    check("Manager reads its own record", d.allowed and d.reason == Reason.OWN_RECORD, d.reason)

    d = can_read(mgr, _rec(src=WORKER, vis=Visibility.GLOBAL), (), NOW)
    check("Manager reads a Worker's global record", d.allowed, d.reason)

    d = can_read(mgr, _rec(src=WORKER, vis=Visibility.LOCAL), (), NOW)
    check("Manager denied a Worker's local record", not d.allowed, d.reason)

    d = can_read(mgr, _rec(src=WORKER, vis=Visibility.GLOBAL, scen="hospital"), (), NOW)
    check("Manager denied across scenarios",
          not d.allowed and d.reason == Reason.SCENARIO_MISMATCH, d.reason)

    d = can_read(wrk, _rec(src=WORKER), (), NOW)
    check("Worker reads its own record", d.allowed, d.reason)

    d = can_read(wrk, _rec(src=MANAGER, vis=Visibility.GLOBAL), (), NOW)
    check("Worker denied the Manager's record",
          not d.allowed and d.reason == Reason.LOCAL_ISOLATION, d.reason)

    d = can_read(wrk, _rec(src=PEER, vis=Visibility.GLOBAL), (), NOW)
    check("Worker denied a peer Worker's record", not d.allowed, d.reason)


def check_default_deny():
    print("\nDefault deny")
    mgr = RobotIdentity(MANAGER, SCENARIO, "s", AccessLevel.GLOBAL)

    d = can_read(mgr, _rec(src=MANAGER, vis=None), (), NOW)
    check("Missing visibility denies",
          not d.allowed and d.reason == Reason.INVALID_VISIBILITY, d.reason)

    d = can_read(mgr, _rec(src=MANAGER, vis="public"), (), NOW)
    check("Unrecognised visibility denies", not d.allowed, d.reason)

    bad = RobotIdentity("x", SCENARIO, "s", "superuser")
    d = can_read(bad, _rec(src="x"), (), NOW)
    check("Unknown access level denies",
          not d.allowed and d.reason == Reason.UNKNOWN_ACCESS_LEVEL, d.reason)

    d = can_read(mgr, _rec(src=None, vis=Visibility.GLOBAL), (), NOW)
    check("Missing provenance denies",
          not d.allowed and d.reason == Reason.MISSING_PROVENANCE, d.reason)

    nobody = RobotIdentity("", SCENARIO, "s", AccessLevel.GLOBAL)
    d = can_read(nobody, _rec(vis=Visibility.GLOBAL), (), NOW)
    check("Unidentified requester denies", not d.allowed, d.reason)


def check_grants():
    print("\nDelegation grants")
    wrk = RobotIdentity(WORKER, SCENARIO, "s", AccessLevel.LOCAL)
    rec = _rec(rid="faiss:1#7", src=MANAGER, vis=Visibility.GLOBAL)

    g = new_grant(["faiss:1#7"], WORKER, MANAGER, "task-1", "s", ttl_sec=180, now=NOW)
    d = can_read(wrk, rec, [g], NOW)
    check("Active grant admits the named snippet",
          d.allowed and d.reason == Reason.DELEGATION_GRANT, d.reason)

    d = can_read(wrk, rec, [g], NOW + timedelta(seconds=181))
    check("Expired grant denies",
          not d.allowed and d.reason == Reason.GRANT_EXPIRED, d.reason)

    other = _rec(rid="faiss:1#8", src=MANAGER, vis=Visibility.GLOBAL)
    check("Grant does not cover unnamed records",
          not can_read(wrk, other, [g], NOW).allowed)

    g2 = new_grant(["faiss:1#7"], PEER, MANAGER, "task-2", "s", ttl_sec=180, now=NOW)
    check("Grant issued to another robot is ignored",
          not can_read(wrk, rec, [g2], NOW).allowed)

    store = GrantStore()
    store.issue(g)
    ok = len(store.active_for(WORKER, now=NOW)) == 1
    store.revoke_task("task-1")
    check("Revoke on task completion clears the grant",
          ok and store.active_for(WORKER, now=NOW) == [])

    check("A grant carries only explicit IDs, never a query",
          set(vars(g)) == {"grant_id", "snippet_ids", "granted_to", "granted_by",
                           "session_id", "task_id", "expires_at"})


def check_clearance_stamp():
    print("\nClearance stamp")
    try:
        ClearedRecord(record=_rec(), decision=Decision(True, "own_record"))
        check("ClearedRecord cannot be forged", False, "construction succeeded")
    except ClearanceError:
        check("ClearedRecord cannot be forged", True)

    try:
        assert_cleared(["a raw retrieved string"], "check_rbac")
        check("Prompt assembly refuses unstamped records", False, "accepted")
    except ClearanceError:
        check("Prompt assembly refuses unstamped records", True)

    rbac = RBACFilter()
    wrk = RobotIdentity(WORKER, SCENARIO, "s", AccessLevel.LOCAL)
    cleared = rbac.filter_records(wrk, [_rec(src=WORKER)], (), NOW)
    check("Filtered records are accepted", assert_cleared(cleared, "check_rbac") == cleared)


def check_audit():
    print("\nAudit")
    audit = MemoryAuditSink()
    rbac = RBACFilter(audit_sink=audit)
    wrk = RobotIdentity(WORKER, SCENARIO, "s", AccessLevel.LOCAL)
    rbac.filter_records(
        wrk, [_rec(src=WORKER), _rec(rid="r2", src=MANAGER)], (), NOW, store="faiss"
    )
    check("Every decision is recorded", len(audit.events) == 2, str(len(audit.events)))
    check("Denials countable by reason",
          audit.denials_by_reason() == {Reason.LOCAL_ISOLATION: 1},
          str(audit.denials_by_reason()))
    check("Denials countable by robot",
          audit.denials_by_robot() == {WORKER: 1}, str(audit.denials_by_robot()))

    class Exploding:
        def record(self, event): raise RuntimeError("audit sink down")
        def flush(self): pass

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = RBACFilter(audit_sink=Exploding())
        got = broken.filter_records(wrk, [_rec(src=WORKER)], (), NOW)
    check("A failing audit sink does not break retrieval", len(got) == 1)


def check_profiles():
    print("\nScenario profiles")
    profile_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles"
    )
    try:
        registry = ProfileRegistry.from_directory(profile_dir)
        check("Shipped profiles load", len(registry) > 0, f"{len(registry)} loaded")
    except ProfileError as e:
        check("Shipped profiles load", False, str(e))
        return

    lab = registry.get_scenario("lab_demo")
    check("lab_demo present", lab is not None)
    if lab:
        check("lab_demo has exactly one Manager", len(lab.managers) == 1,
              str([m.id for m in lab.managers]))
        check("pepper_01 is the Manager",
              lab.get("pepper_01") is not None and lab.get("pepper_01").is_manager)

    from core.profiles import parse_profile
    for label, doc, needle in [
        ("unknown access level",
         {"scenario_id": "s", "robots": [{"id": "a", "access_level": "wizard"}]},
         "access_level"),
        ("duplicate robot ids",
         {"scenario_id": "s", "robots": [{"id": "a", "access_level": "global"},
                                         {"id": "a", "access_level": "local"}]},
         "duplicate"),
        ("no global robot",
         {"scenario_id": "s", "robots": [{"id": "a", "access_level": "local"}]},
         "global"),
    ]:
        try:
            parse_profile(doc, "check_rbac")
            check(f"Rejects {label}", False, "accepted")
        except ProfileError as e:
            check(f"Rejects {label}", needle in str(e), str(e)[:60])


def run():
    print("=" * 55)
    print("  RBAC checkpoint")
    print("=" * 55)
    print("  No database, Ollama or robot hardware required.")

    check_policy_matrix()
    check_default_deny()
    check_grants()
    check_clearance_stamp()
    check_audit()
    check_profiles()

    print("\n" + "=" * 55)
    print(f"  {PASSED}/{TOTAL} checks passed")
    print("=" * 55)
    return 0 if PASSED == TOTAL else 1


if __name__ == "__main__":
    sys.exit(run())
