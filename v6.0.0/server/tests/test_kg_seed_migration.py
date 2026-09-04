"""
tests/test_kg_seed_migration.py
================================
data/migrations/009_kg_role_seed.sql — the starter robot->topic set.

Static checks against the SQL text itself, not a live database: this repo has
no migration runner and no test database (see tools/check_migrations.py, which
probes a live one instead), so "does the migration do what it claims" is
verified here the same way a reviewer would check it — by parsing the file.

The guard that matters most: decision/kg.py and tools/seed_kg.py both state
that a robot->topic edge must start at the neutral prior and move only on
observation. A seed migration that quietly wrote a non-neutral weight would
be a capability fact wearing a competence score, silently — this test is what
makes that a caught mistake instead of a numeric typo nobody notices.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MIGRATION = Path(__file__).resolve().parent.parent / "data/migrations/009_kg_role_seed.sql"
DOWN_MIGRATION = Path(__file__).resolve().parent.parent / "data/migrations/009_kg_role_seed_down.sql"


@pytest.fixture(scope="module")
def migration_text() -> str:
    assert MIGRATION.exists(), f"expected {MIGRATION} to exist"
    return MIGRATION.read_text()


def _robot_topic_rows(text: str) -> list[tuple]:
    match = re.search(
        r"INSERT INTO demo_robot_topic\s*\([^)]*\)\s*VALUES\s*(.*?)ON CONFLICT",
        text, re.DOTALL,
    )
    assert match, "expected an INSERT INTO demo_robot_topic ... VALUES ... ON CONFLICT block"
    body = match.group(1)
    rows = re.findall(
        r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*([\d.]+)\s*,\s*(true|false)\s*\)",
        body,
    )
    assert rows, "found no parseable (robot_id, topic_id, weight, eligible) rows"
    return [(robot, topic, float(weight), elig == "true")
            for robot, topic, weight, elig in rows]


class TestSeedWeightGuard:
    """Same discipline as tools/seed_kg.py::seed_from_db: a seeded robot->topic
    row must never carry competence — weight stays exactly the neutral prior."""

    def test_every_seeded_row_writes_weight_0_5(self, migration_text):
        rows = _robot_topic_rows(migration_text)
        offenders = [(r, t, w) for r, t, w, _ in rows if w != 0.5]
        assert offenders == [], f"seeded row(s) with weight != 0.5: {offenders}"

    def test_every_seeded_row_is_eligible(self, migration_text):
        # This migration only ADDS capability — nothing is excluded yet.
        rows = _robot_topic_rows(migration_text)
        offenders = [(r, t) for r, t, _, eligible in rows if not eligible]
        assert offenders == [], f"seed row(s) marked ineligible: {offenders}"

    def test_the_starter_set_covers_exactly_the_five_specified_pairs(self, migration_text):
        pairs = {(r, t) for r, t, _, _ in _robot_topic_rows(migration_text)}
        assert pairs == {
            ("pepper_01", "topic:human-pose-estimation"),
            ("chatbox_01", "topic:retrieval-augmented-generation"),
            ("chatbox_01", "topic:long-term-interaction"),
            ("navel_001", "topic:non-verbal-interaction"),
            ("silbot_01", "topic:social-robot-navigation"),
        }

    def test_seed_inserts_never_overwrite_on_conflict(self, migration_text):
        # A real observation may already have moved a row's weight (true today
        # for chatbox_01/retrieval-augmented-generation) — the seed must never
        # be able to clobber it back to the prior.
        assert "ON CONFLICT (robot_id, topic_id) DO UPDATE" not in migration_text
        assert "ON CONFLICT (robot_id, topic_id) DO NOTHING" in migration_text
        assert "ON CONFLICT (id) DO UPDATE" not in migration_text
        assert "ON CONFLICT (id) DO NOTHING" in migration_text

    def test_eligible_column_defaults_to_true(self, migration_text):
        assert re.search(
            r"ADD COLUMN IF NOT EXISTS eligible BOOLEAN NOT NULL DEFAULT true",
            migration_text,
        )

    def test_reused_topics_are_not_redefined(self, migration_text):
        # retrieval-augmented-generation and social-robot-navigation already
        # exist (see the existing 14-topic vocabulary) — this migration must
        # reference them by id, not attempt to redeclare them.
        insert_topics = re.search(
            r"INSERT INTO demo_topics.*?ON CONFLICT", migration_text, re.DOTALL,
        ).group(0)
        assert "retrieval-augmented-generation" not in insert_topics
        assert "social-robot-navigation" not in insert_topics


class TestDownMigration:
    def test_down_migration_exists(self):
        assert DOWN_MIGRATION.exists()

    def test_down_migration_drops_the_eligible_column(self):
        text = DOWN_MIGRATION.read_text()
        assert "DROP COLUMN IF EXISTS eligible" in text

    def test_down_migration_deletes_the_exact_seeded_pairs(self):
        # By identity — exact (robot_id, topic_id) tuples — not a
        # "WHERE source = 'role-seed'" filter, which would also remove any
        # row a real observation touched after 009 ran. A wider blast radius
        # than reversing what this migration specifically added.
        text = DOWN_MIGRATION.read_text()
        delete_block = re.search(
            r"DELETE FROM demo_robot_topic.*?;", text, re.DOTALL,
        ).group(0)
        assert "source" not in delete_block
        assert "'pepper_01'" in delete_block
        assert "'topic:human-pose-estimation'" in delete_block
