-- 009_kg_role_seed.sql
-- Run in Supabase SQL editor (or psql) after 008_demo_durations.sql.
--
-- Two independent things, in one migration because both touch demo_robot_topic:
--
--   1. eligible: a HARD routing filter, orthogonal to weight.
--   2. A starter set of robot->topic rows — one obvious topic per robot's
--      stated role, NOT an exhaustive breakdown. Additive: existing edges are
--      never touched (ON CONFLICT DO NOTHING everywhere), so any weight
--      already learned from real observations survives this untouched.
--
-- WHY ELIGIBLE IS A SEPARATE COLUMN, NOT A WEIGHT OF 0.0
-- tools/seed_kg.py::seed_from_db already states the rule this column exists to
-- satisfy: "robot->topic edges must start at the 0.5 prior and move only on
-- observation... if a capability genuinely constrains routing, filter the
-- candidate list before route() sees it rather than encoding it here." A weight
-- of 0.0 is a competence CLAIM ("this robot is bad at this") that the graph
-- should only ever reach through observation. eligible=false is a capability
-- FACT ("this robot must never be routed here") that a human or a system
-- constraint asserts directly and that no accumulation of evidence should be
-- able to override by drifting a weight upward. Keeping them separate means a
-- future exclusion never has to be laundered through fake bad observations.
--
-- Default true, so this is opt-out: absence of a row, or a row nobody has
-- restricted, keeps a robot in the running. The starter set below writes
-- eligible=true everywhere — nothing is excluded yet, this migration only adds
-- the capability, and decision/kg_infer.py::route filters on it either way.
--
-- WHY THESE FIVE ROBOT->TOPIC ROWS AND NO OTHERS
-- One topic per robot's stated role, not a full breakdown of what each project
-- covers — see tools/seed_kg.py's own SEED_TOPICS for the fuller vocabulary.
-- Two of the five reuse an EXISTING topic verbatim (retrieval-augmented-
-- generation, social-robot-navigation) rather than redefining it. Three are
-- genuinely new topics (human-pose-estimation, long-term-interaction,
-- non-verbal-interaction) because nothing in the existing 14 covers them
-- closely enough to reuse — though two of those three were close enough to
-- flag rather than dismiss outright; see below.
--
-- FLAGGED FOR HUMAN REVIEW, NOT AUTO-MERGED
--   long-term-interaction  vs  conversational-memory (existing, category "ai")
--   non-verbal-interaction vs  social-signals         (existing, category "hri")
-- Both pairs are related, arguably overlapping, and NOT identical: conversational
-- memory is one specific mechanism for sustaining a long-term interaction, and
-- social signals is one specific channel of non-verbal interaction, but each new
-- topic is broader than the existing one it resembles. Added as separate topics
-- per the instruction not to auto-merge; revisit if that reads as one topic
-- wearing two labels once there is more usage to judge it by.
--
-- Reversible: see 009_kg_role_seed_down.sql

BEGIN;

-- ── Eligibility: a hard filter, orthogonal to weight ─────────────────────────

ALTER TABLE demo_robot_topic
    ADD COLUMN IF NOT EXISTS eligible BOOLEAN NOT NULL DEFAULT true;

COMMENT ON COLUMN demo_robot_topic.eligible IS
    'Hard routing filter, never a competence seed. False structurally excludes '
    'this robot from being routed to on this topic regardless of weight. '
    'Filtered BEFORE ranking in decision/kg_infer.py::route, not after — an '
    'ineligible robot never enters rank_robots/infer and cannot win by score.';

-- ── New topics (only what genuinely has no existing match) ───────────────────

INSERT INTO demo_topics (id, label, category, source) VALUES
    ('topic:human-pose-estimation', 'human pose estimation', 'hri', 'role-seed'),
    ('topic:long-term-interaction', 'long term interaction', 'hri', 'role-seed'),
    ('topic:non-verbal-interaction', 'non verbal interaction', 'hri', 'role-seed')
ON CONFLICT (id) DO NOTHING;

-- ── Starter robot->topic set: one obvious topic per stated role ─────────────
-- weight is 0.5 (PRIOR) on every row, unconditionally — see decision/kg.py's
-- NEVER SEED robot->topic FROM PROJECT ASSIGNMENTS. ON CONFLICT DO NOTHING
-- everywhere: if a row already exists (real observations have already moved
-- its weight — true today for chatbox_01/retrieval-augmented-generation, at
-- 0.575 after one observation), this migration must not touch it.

INSERT INTO demo_robot_topic (robot_id, topic_id, weight, eligible) VALUES
    ('pepper_01',  'topic:human-pose-estimation',              0.5, true),
    ('chatbox_01', 'topic:retrieval-augmented-generation',     0.5, true),
    ('chatbox_01', 'topic:long-term-interaction',               0.5, true),
    ('navel_001',  'topic:non-verbal-interaction',              0.5, true),
    ('silbot_01',  'topic:social-robot-navigation',             0.5, true)
ON CONFLICT (robot_id, topic_id) DO NOTHING;

-- ── View: expose eligible alongside the rest ─────────────────────────────────
-- DROP then CREATE, not CREATE OR REPLACE — same reason as 007: replace can
-- only append columns, and eligible belongs next to weight rather than tacked
-- on at the end. A view holds no data, so dropping it here is safe.

DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id,
       rt.topic_id,
       t.label                                   AS topic_label,
       t.category                                AS topic_category,
       rt.weight,
       rt.eligible,
       rt.n_supervisor,
       rt.n_outcome,
       rt.n_displaced,
       rt.n_supervisor + rt.n_outcome + rt.n_displaced        AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                 / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0)
       END                                       AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                    / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0))
       END                                       AS clamped,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_displaced)::real
                 / (rt.n_supervisor + rt.n_outcome + rt.n_displaced)
       END                                       AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;

COMMIT;
