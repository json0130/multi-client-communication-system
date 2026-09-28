"""
tests/test_model_transfer.py
============================
The sim-to-real transfer plan (tools/model_transfer.py::plan_transfer) — pure,
no database.
"""

from __future__ import annotations

from tools.model_transfer import (PRIOR_MODES, expected_speakers, plan_transfer,
                                  rename_step)

TWINS = {"navel_01": "navel_001", "pepper_01": "pepper_001",
         "chatbox_01": "chatbox_jetson_001"}


def _snap(**over):
    snap = {
        "facts": [
            {"id": 7, "topic_id": "topic:b", "kind": "method", "fact": "second",
             "verified": True, "source": "doc"},
            {"id": 3, "topic_id": "topic:a", "kind": "method", "fact": "first",
             "verified": True, "source": "doc"},
        ],
        "edges": [
            {"topic_id": "topic:a", "weight": 0.5, "n_supervisor": 0, "n_outcome": 0,
             "n_displaced": 0, "eligible": True, "specialised": True},
            {"topic_id": "topic:old", "weight": 0.5, "n_supervisor": 0, "n_outcome": 0,
             "n_displaced": 0, "eligible": True, "specialised": False},
            {"topic_id": "topic:learned", "weight": 0.8, "n_supervisor": 2, "n_outcome": 1,
             "n_displaced": 0, "eligible": True, "specialised": False},
            {"topic_id": "topic:banned", "weight": 0.5, "n_supervisor": 0, "n_outcome": 0,
             "n_displaced": 0, "eligible": False, "specialised": False},
        ],
        "styles": [{"style": "technical", "weight": 0.7, "n_supervisor": 4},
                   {"style": "casual", "weight": 0.5, "n_supervisor": 0}],
        "runs": {
            "r-mock": {"mode": "mock", "format": "full_tour"},
            "r-gz": {"mode": "gazebo", "format": "full_tour"},
            "r-unknown": {"mode": "sim_unknown", "format": "full_tour"},
            "r-solo": {"mode": "gazebo", "format": "solo"},
            "r-live": {"mode": "live", "format": "full_tour"},
        },
        "step_rows": [
            {"run_id": "r-mock", "step_id": "navel_01_greeting", "seconds": 10},
            {"run_id": "r-gz", "step_id": "navel_01_greeting", "seconds": 14},
            {"run_id": "r-gz", "step_id": "navel_01_greeting", "seconds": 12},
            {"run_id": "r-gz", "step_id": "navigate_to_navel_01", "seconds": 40},
            {"run_id": "r-unknown", "step_id": "navel_01_greeting", "seconds": 99},
            {"run_id": "r-solo", "step_id": "navel_01_greeting", "seconds": 99},
            {"run_id": "r-live", "step_id": "navel_01_greeting", "seconds": 99},
        ],
        "qa_rows": [
            {"run_id": "r-mock", "seconds": 20, "closed_by": "policy"},
            {"run_id": "r-gz", "seconds": 40, "closed_by": "policy"},
            {"run_id": "r-gz", "seconds": 60, "closed_by": "timeout"},
            {"run_id": "r-unknown", "seconds": 500, "closed_by": "policy"},
        ] + [{"run_id": "r-mock", "seconds": 40, "closed_by": "policy"}] * 3,
        "target_edges": [],
        "target_transferred_facts": [],
    }
    snap.update(over)
    return snap


def _plan(**over):
    return plan_transfer(("navel_01", "navel_001"), TWINS, _snap(**over), version=1)


def test_step_ids_are_renamed_for_every_twin():
    assert rename_step("introduce_navel_01", TWINS) == "introduce_navel_001"
    assert rename_step("transition_to_chatbox_01", TWINS) == "transition_to_chatbox_jetson_001"
    assert rename_step("navel_011_x", TWINS) == "navel_011_x"      # whole ids only
    assert rename_step("wrap_up", TWINS) == "wrap_up"


def test_facts_are_copied_in_their_original_order():
    p = _plan()
    assert [f["fact"] for f in p["facts"]] == ["first", "second"]
    assert all(f["robot_id"] == "navel_001" for f in p["facts"])


def test_only_edges_that_differ_from_the_default_carry_scope():
    topics = {e["topic_id"] for e in _plan()["scope"]}
    assert topics == {"topic:a", "topic:learned", "topic:banned"}   # not topic:old


def test_learned_edges_become_a_prior_not_a_live_weight():
    p = _plan()
    learned = next(e for e in p["scope"] if e["topic_id"] == "topic:learned")
    assert learned["prior_weight"] == 0.8 and learned["prior_n"] == 3
    assert "weight" not in learned and "n_supervisor" not in learned
    assert p["kg_prior_topics"] == ["topic:learned"]


def test_style_priors_only_where_sim_has_evidence():
    assert [s["style"] for s in _plan()["style_priors"]] == ["technical"]


def test_step_priors_use_only_full_tour_prior_mode_runs_and_skip_movement():
    p = _plan()
    steps = [d for d in p["duration_priors"] if d["kind"] == "step"]
    assert steps == [{"robot_id": "navel_001", "step_id": "navel_001_greeting", "kind": "step",
                      "median_sec": 12.0, "n": 3, "source_mode": "gazebo+mock"}]
    assert p["items"]["excluded"]["movement_steps"] == 1
    assert p["items"]["excluded"]["non_prior_runs"] == 3      # sim_unknown, solo, live
    assert "live" not in PRIOR_MODES


def test_qa_prior_is_the_censored_median_of_prior_mode_windows():
    p = _plan()
    qa = next(d for d in p["duration_priors"] if d["kind"] == "qa")
    assert qa["step_id"] == "qa_invite_navel_001"
    assert qa["n"] == 5                                     # visitor-ended windows
    assert p["items"]["qa_prior"]["windows"] == 6           # the 500 s sim_unknown one is out
    assert qa["median_sec"] == 40.0


def test_nothing_to_learn_from_yields_an_empty_but_valid_plan():
    p = _plan(facts=[], edges=[], styles=[], step_rows=[], qa_rows=[])
    assert p["facts"] == p["scope"] == p["duration_priors"] == []
    assert p["items"]["qa_prior"] is None


def test_only_steps_the_physical_robot_speaks_are_kept():
    rows = [{"run_id": "r-gz", "step_id": "navel_01_greeting", "seconds": 8},
            {"run_id": "r-gz", "step_id": "navel_greeting", "seconds": 9},    # old name
            {"run_id": "r-gz", "step_id": "lab_intro", "seconds": 15}]        # guide's line
    p = plan_transfer(("navel_01", "navel_001"), TWINS, _snap(step_rows=rows), 1,
                      speakers={"navel_001_greeting": "navel_001", "lab_intro": "pepper_001"})
    assert [d["step_id"] for d in p["duration_priors"] if d["kind"] == "step"] == \
        ["navel_001_greeting"]
    assert p["items"]["excluded"]["not_its_step"] == 2


def test_expected_speakers_follow_the_physical_lineup():
    sp = expected_speakers(TWINS)
    assert sp["navel_001_greeting"] == "navel_001"
    assert sp["lab_intro"] == "pepper_001"
    assert sp["introduce_silbot_01"] == "pepper_001"       # no physical Silbot: sim id kept


def test_a_qa_prior_below_the_planners_own_bar_is_not_copied():
    rows = [{"run_id": "r-gz", "seconds": 11, "closed_by": "policy"}]
    p = _plan(qa_rows=rows)
    assert not any(d["kind"] == "qa" for d in p["duration_priors"])
    assert p["items"]["qa_prior"]["copied"] is False


def test_a_run_flagged_exclude_from_priors_is_left_out():
    runs = _snap()["runs"]
    runs["r-gz"] = {**runs["r-gz"], "exclude_from_priors": True}
    p = _plan(runs=runs)
    step = next(d for d in p["duration_priors"] if d["kind"] == "step")
    assert step["median_sec"] == 10.0 and step["n"] == 1          # the mock run alone
    assert p["items"]["excluded"]["non_prior_runs"] == 5
