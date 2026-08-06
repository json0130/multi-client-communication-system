"""
tests/test_profile_registry.py
==============================
Scenario profile parsing, validation and reconciliation.

The point of these is that an invalid profile stops the boot with a legible
error, rather than turning into a silent deny at the first retrieval.
"""

from __future__ import annotations

import pytest
import yaml

from core.profiles import ProfileError, ProfileRegistry, parse_profile
from core.rbac import AccessLevel, Visibility


VALID = {
    "scenario_id": "lab_demo",
    "robots": [
        {"id": "pepper_01", "role": "Guide", "access_level": "global"},
        {"id": "silbot_01", "role": "Navigation", "access_level": "local",
         "default_visibility": "global"},
    ],
}


# ── Happy path ────────────────────────────────────────────────────────────────

class TestParsing:

    def test_parses_a_valid_profile(self):
        p = parse_profile(VALID)
        assert p.scenario_id == "lab_demo"
        assert [r.id for r in p.robots] == ["pepper_01", "silbot_01"]
        assert p.get("pepper_01").access_level is AccessLevel.GLOBAL
        assert p.get("silbot_01").access_level is AccessLevel.LOCAL

    def test_default_visibility_defaults_to_local(self):
        """Fail closed — a robot added without thought stays isolated."""
        p = parse_profile(VALID)
        assert p.get("pepper_01").default_visibility == Visibility.LOCAL.value
        assert p.get("silbot_01").default_visibility == Visibility.GLOBAL.value

    def test_managers_are_identified(self):
        p = parse_profile(VALID)
        assert [m.id for m in p.managers] == ["pepper_01"]
        assert p.get("pepper_01").is_manager
        assert not p.get("silbot_01").is_manager

    def test_access_level_is_case_and_space_insensitive(self):
        doc = {"scenario_id": "s", "robots": [{"id": "a", "access_level": " GLOBAL "}]}
        assert parse_profile(doc).get("a").access_level is AccessLevel.GLOBAL

    def test_role_is_not_an_access_decision(self):
        """robot_role is a persona string; only access_level governs access."""
        doc = {"scenario_id": "s", "robots": [
            {"id": "a", "role": "Administrator", "access_level": "local"},
            {"id": "b", "role": "Intern", "access_level": "global"},
        ]}
        p = parse_profile(doc)
        assert p.get("a").access_level is AccessLevel.LOCAL
        assert p.get("b").access_level is AccessLevel.GLOBAL


# ── Boot validation ───────────────────────────────────────────────────────────

class TestValidationFailsFast:

    @pytest.mark.parametrize("bad", ["superuser", "admin", "GLOBAL_", "", None, 1, []])
    def test_unknown_access_level(self, bad):
        doc = {"scenario_id": "s", "robots": [{"id": "a", "access_level": bad}]}
        with pytest.raises(ProfileError, match="access_level"):
            parse_profile(doc)

    def test_missing_access_level(self):
        doc = {"scenario_id": "s", "robots": [{"id": "a"}]}
        with pytest.raises(ProfileError, match="access_level"):
            parse_profile(doc)

    def test_unknown_default_visibility(self):
        doc = {"scenario_id": "s", "robots": [
            {"id": "a", "access_level": "global", "default_visibility": "public"}
        ]}
        with pytest.raises(ProfileError, match="default_visibility"):
            parse_profile(doc)

    def test_duplicate_robot_ids(self):
        doc = {"scenario_id": "s", "robots": [
            {"id": "a", "access_level": "global"},
            {"id": "a", "access_level": "local"},
        ]}
        with pytest.raises(ProfileError, match="duplicate robot id"):
            parse_profile(doc)

    def test_scenario_with_no_global_robot(self):
        doc = {"scenario_id": "s", "robots": [
            {"id": "a", "access_level": "local"},
            {"id": "b", "access_level": "local"},
        ]}
        with pytest.raises(ProfileError, match="no robot with access_level 'global'"):
            parse_profile(doc)

    def test_missing_scenario_id(self):
        with pytest.raises(ProfileError, match="scenario_id"):
            parse_profile({"robots": [{"id": "a", "access_level": "global"}]})

    def test_empty_robot_list(self):
        with pytest.raises(ProfileError, match="non-empty 'robots'"):
            parse_profile({"scenario_id": "s", "robots": []})

    def test_missing_robot_id(self):
        doc = {"scenario_id": "s", "robots": [{"access_level": "global"}]}
        with pytest.raises(ProfileError, match="missing a string 'id'"):
            parse_profile(doc)

    def test_profile_is_not_a_mapping(self):
        with pytest.raises(ProfileError, match="must be a YAML mapping"):
            parse_profile(["not", "a", "mapping"])

    def test_error_names_the_source_file(self):
        doc = {"scenario_id": "s", "robots": [{"id": "a", "access_level": "nope"}]}
        with pytest.raises(ProfileError, match="hospital.yaml"):
            parse_profile(doc, source_path="hospital.yaml")


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_lookup_by_robot(self):
        r = ProfileRegistry([parse_profile(VALID)])
        assert r.get_robot("silbot_01").access_level is AccessLevel.LOCAL
        assert r.find_scenario("silbot_01").scenario_id == "lab_demo"
        assert r.get_robot("unknown") is None

    def test_duplicate_scenario_id_is_rejected(self):
        with pytest.raises(ProfileError, match="Duplicate scenario_id"):
            ProfileRegistry([parse_profile(VALID), parse_profile(VALID)])

    def test_a_robot_may_not_span_two_scenarios(self):
        """Its DB row carries one scenario_id, so two would be unrepresentable."""
        other = {
            "scenario_id": "hospital",
            "robots": [{"id": "pepper_01", "access_level": "global"}],
        }
        with pytest.raises(ProfileError, match="appears in both"):
            ProfileRegistry([parse_profile(VALID), parse_profile(other)])

    def test_missing_directory_yields_an_empty_registry(self, tmp_path):
        """A deployment need not use profiles; everything then stays 'local'."""
        r = ProfileRegistry.from_directory(tmp_path / "nope")
        assert len(r) == 0
        assert r.get_robot("anything") is None

    def test_loads_from_directory(self, tmp_path):
        (tmp_path / "lab.yaml").write_text(yaml.safe_dump(VALID))
        r = ProfileRegistry.from_directory(tmp_path)
        assert len(r) == 1
        assert r.get_robot("pepper_01").is_manager

    def test_an_invalid_file_stops_the_load(self, tmp_path):
        (tmp_path / "bad.yaml").write_text(yaml.safe_dump(
            {"scenario_id": "s", "robots": [{"id": "a", "access_level": "wizard"}]}
        ))
        with pytest.raises(ProfileError):
            ProfileRegistry.from_directory(tmp_path)

    def test_malformed_yaml_stops_the_load(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("robots: [unclosed\n  - :::")
        with pytest.raises(ProfileError, match="Could not parse"):
            ProfileRegistry.from_directory(tmp_path)


class TestSyncToDb:

    def test_writes_declared_levels(self):
        written = []
        r = ProfileRegistry([parse_profile(VALID)])
        n = r.sync_to_db(lambda rid, lvl, sid: written.append((rid, lvl, sid)) or True)
        assert n == 2
        assert written == [
            ("pepper_01", "global", "lab_demo"),
            ("silbot_01", "local", "lab_demo"),
        ]

    def test_an_unregistered_robot_is_skipped_not_fatal(self, capsys):
        r = ProfileRegistry([parse_profile(VALID)])
        assert r.sync_to_db(lambda rid, lvl, sid: False) == 0

        out = capsys.readouterr().out
        assert "Could not set access level for 2 robot(s)" in out
        assert "pepper_01" in out and "silbot_01" in out
        # The failure must state what happens next, not just that it failed.
        assert "local" in out

    def test_sync_failures_are_reported_once_not_per_robot(self, capsys):
        r = ProfileRegistry([parse_profile(VALID)])
        r.sync_to_db(lambda rid, lvl, sid: False)
        out = capsys.readouterr().out
        assert out.count("Could not set access level") == 1

    def test_a_writer_error_does_not_abort_the_rest(self):
        seen = []

        def flaky(rid, lvl, sid):
            seen.append(rid)
            if rid == "pepper_01":
                raise RuntimeError("connection reset")
            return True

        r = ProfileRegistry([parse_profile(VALID)])
        assert r.sync_to_db(flaky) == 1
        assert seen == ["pepper_01", "silbot_01"]


# ── The shipped profile ───────────────────────────────────────────────────────

def test_the_lab_demo_profile_is_valid():
    """The profile that actually ships must load and satisfy every rule."""
    import os
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = ProfileRegistry.from_directory(os.path.join(server_dir, "profiles"))

    p = r.get_scenario("lab_demo")
    assert p is not None
    assert p.get("pepper_01").access_level is AccessLevel.GLOBAL
    assert len(p.managers) == 1
    for worker in ("chatbox_jetson_001", "navel_001", "silbot_01"):
        assert p.get(worker).access_level is AccessLevel.LOCAL


def test_shipped_profile_ids_match_the_demo_script():
    """A profile naming robots the demo never uses would silently do nothing."""
    import os
    from demo import demo_script

    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = ProfileRegistry.from_directory(os.path.join(server_dir, "profiles"))
    declared = {e.id for e in r.get_scenario("lab_demo").robots}

    script_ids = {
        demo_script.PEPPER, demo_script.CHATBOX,
        demo_script.NAVEL, demo_script.SILBOT,
    }
    assert declared == script_ids
