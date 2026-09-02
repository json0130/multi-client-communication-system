-- apply_all.sql
-- Combined 002 - 008, in dependency order. Paste into the Supabase SQL editor
-- and run once. Every statement is idempotent, so re-running is safe.
--
-- Verify afterwards with:  python3 tools/check_migrations.py

-- ===================================================================
-- 002_rbac.sql
-- ===================================================================
-- 002_rbac.sql
-- Run in Supabase SQL editor (or psql) to add RBAC provenance + visibility.
--
-- Implements the data model from Song et al., "Orchestrating Role-Based Social
-- Continuity for Heterogeneous Multi-Robot Teams":
--
--   robots.access_level   global (Manager) | local (Worker)
--   <record>.visibility   global | local | restricted
--   <record>.source_robot_id / scenario_id / session_id / subject_user_id
--
-- Access hierarchies are adjustable at the database level without reconfiguring
-- individual robot nodes — the level is data, not code.
--
-- Reversible: see 002_rbac_down.sql
-- Depends on: 001_projects.sql (only for ordering; no shared objects)

BEGIN;

-- ── Robots: declare the access level ──────────────────────────────────────────
-- Defaults to 'local' so any robot not covered by a scenario profile is a
-- Worker. Fail closed: a new robot never silently gets cross-client visibility.

ALTER TABLE robots
    ADD COLUMN IF NOT EXISTS access_level TEXT NOT NULL DEFAULT 'local',
    ADD COLUMN IF NOT EXISTS scenario_id  TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'robots_access_level_check'
    ) THEN
        ALTER TABLE robots
            ADD CONSTRAINT robots_access_level_check
            CHECK (access_level IN ('global', 'local'));
    END IF;
END $$;

COMMENT ON COLUMN robots.access_level IS
    'RBAC level: global = Manager (cross-client visibility within its scenario), '
    'local = Worker (local isolation, own records only). Seeded from the scenario '
    'profile YAML at boot by core.profiles.ProfileRegistry.';
COMMENT ON COLUMN robots.scenario_id IS
    'Deployment this robot belongs to. Cross-robot visibility never crosses scenarios.';


-- ── Interaction log: provenance + visibility ─────────────────────────────────

ALTER TABLE chat_logs
    ADD COLUMN IF NOT EXISTS source_robot_id TEXT,
    ADD COLUMN IF NOT EXISTS scenario_id     TEXT,
    ADD COLUMN IF NOT EXISTS session_id      TEXT,
    ADD COLUMN IF NOT EXISTS subject_user_id INTEGER,
    ADD COLUMN IF NOT EXISTS visibility      TEXT NOT NULL DEFAULT 'local';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chat_logs_visibility_check'
    ) THEN
        ALTER TABLE chat_logs
            ADD CONSTRAINT chat_logs_visibility_check
            CHECK (visibility IN ('global', 'local', 'restricted'));
    END IF;
END $$;

COMMENT ON COLUMN chat_logs.source_robot_id IS
    'client_id of the robot that generated this record. NULL means unattributed '
    'legacy data, readable only via an explicit delegation grant.';
COMMENT ON COLUMN chat_logs.visibility IS
    'global = visible to Manager robots in the same scenario; local = source robot '
    'only; restricted = source robot plus an explicit delegation grant only.';
COMMENT ON COLUMN chat_logs.subject_user_id IS
    'The human this record is about (nullable). Distinct from user_id, which is the '
    'per-robot-session row created by RobotRegistry._ensure_user().';

CREATE INDEX IF NOT EXISTS idx_chat_logs_source_robot ON chat_logs (source_robot_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_scenario     ON chat_logs (scenario_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_visibility   ON chat_logs (visibility);


-- ── Backfill ─────────────────────────────────────────────────────────────────
-- chat_logs has never carried a robot column. The only recoverable provenance is
-- RobotRegistry._ensure_user(), which sets users.name = robot_name. Map back
-- through that. DISTINCT ON keeps the join deterministic if two robots somehow
-- share a robot_name; rows that cannot be attributed keep source_robot_id NULL.
--
-- Every existing row gets visibility='local' (the column DEFAULT), which
-- preserves current behaviour exactly: today each robot has its own user_id and
-- therefore its own FAISS index, so nothing is cross-visible either.
--
-- scenario_id is deliberately left NULL on legacy rows. Policy treats an
-- unscoped record as never satisfying cross-robot visibility, while still being
-- readable by its own source robot — so no existing scenario changes behaviour.

WITH robot_by_name AS (
    SELECT DISTINCT ON (robot_name) robot_name, client_id
    FROM robots
    ORDER BY robot_name, client_id
)
UPDATE chat_logs cl
SET    source_robot_id = rbn.client_id
FROM   users u
JOIN   robot_by_name rbn ON rbn.robot_name = u.name
WHERE  cl.user_id = u.user_id
  AND  cl.source_robot_id IS NULL;

-- The interaction log's user_id is the subject of the record.
UPDATE chat_logs
SET    subject_user_id = user_id
WHERE  subject_user_id IS NULL
  AND  user_id IS NOT NULL;

COMMIT;

-- ===================================================================
-- 003_rbac_audit.sql
-- ===================================================================
-- 003_rbac_audit.sql
-- Run in Supabase SQL editor (or psql) after 002_rbac.sql.
--
-- Every RBAC access decision is recorded here — allow and deny alike. This is a
-- separate stream from chat_logs on purpose: the interaction log records what
-- was said, this records what was permitted.
--
-- The paper lists "enforcement effectiveness under real-world conditions" as
-- future work. This table is what makes that measurable, so the two aggregate
-- views below are part of the schema rather than left to ad-hoc SQL.
--
-- Reversible: see 003_rbac_audit_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS rbac_audit_log (
    id                  BIGSERIAL   PRIMARY KEY,
    requester_robot_id  TEXT        NOT NULL,
    record_id           TEXT        NOT NULL,   -- store-scoped, e.g. 'faiss:1113#4'
    allowed             BOOLEAN     NOT NULL,
    reason              TEXT        NOT NULL,   -- see core.rbac.policy.Reason
    matched_grant_id    TEXT,                   -- set when a delegation grant allowed it
    scenario_id         TEXT,
    session_id          TEXT,
    store               TEXT        NOT NULL,   -- 'faiss' | 'chat_logs' | 'projects' | 'delegation'
    decided_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE rbac_audit_log IS
    'Every RBAC access decision. Written asynchronously in batches by '
    'core.rbac.audit.BatchingAuditSink; a write failure degrades to a warning '
    'and never blocks retrieval, so absence of rows is not proof of no access.';

-- Denials are the interesting rows and the ones most often filtered on.
CREATE INDEX IF NOT EXISTS idx_rbac_audit_robot   ON rbac_audit_log (requester_robot_id);
CREATE INDEX IF NOT EXISTS idx_rbac_audit_reason  ON rbac_audit_log (reason);
CREATE INDEX IF NOT EXISTS idx_rbac_audit_time    ON rbac_audit_log (decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_rbac_audit_denials ON rbac_audit_log (requester_robot_id, reason)
    WHERE allowed = FALSE;


-- ── Aggregates: denial counts by robot and by reason ─────────────────────────

CREATE OR REPLACE VIEW rbac_denials_by_robot AS
SELECT requester_robot_id,
       scenario_id,
       COUNT(*)                    AS denials,
       COUNT(DISTINCT reason)      AS distinct_reasons,
       MAX(decided_at)             AS last_denied_at
FROM   rbac_audit_log
WHERE  allowed = FALSE
GROUP BY requester_robot_id, scenario_id;

CREATE OR REPLACE VIEW rbac_denials_by_reason AS
SELECT reason,
       scenario_id,
       COUNT(*)                          AS denials,
       COUNT(DISTINCT requester_robot_id) AS distinct_robots,
       MAX(decided_at)                   AS last_denied_at
FROM   rbac_audit_log
WHERE  allowed = FALSE
GROUP BY reason, scenario_id;

COMMIT;

-- ===================================================================
-- 004_demo_decisions.sql
-- ===================================================================
-- 004_demo_decisions.sql
-- Run in Supabase SQL editor (or psql) after 003_rbac_audit.sql.
--
-- Two streams, deliberately separate from rbac_audit_log:
--
--   demo_decision_log     what the orchestration layer decided, and which rule
--                         decided it
--   demo_correction_log   what a human supervisor changed it to
--
-- rbac_audit_log records what a robot was *permitted* to read. This records
-- what the demo chose to *do*. They share scenario_id and session_id so the two
-- can be joined, which is what makes "did this policy widen context exposure
-- relative to the baseline?" a query rather than an argument.
--
-- Every column here exists to answer a specific question about a demo run:
--   mechanism    which rule fired  → is the phrase list or the LLM doing the work?
--   observation  the full state    → replay and offline policy evaluation
--   decision_id  the join key      → correction rate per mechanism
--
-- Reversible: see 004_demo_decisions_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS demo_decision_log (
    id                    BIGSERIAL   PRIMARY KEY,
    decision_id           UUID        NOT NULL UNIQUE,
    decision_point        TEXT        NOT NULL,   -- see decision.models.DecisionPoint
    action_kind           TEXT        NOT NULL,   -- see decision.models.ActionKind
    action_payload        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    mechanism             TEXT        NOT NULL,   -- see decision.policy.Mechanism
    observation           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    decider_robot_id      TEXT,
    decider_access_level  TEXT,                   -- 'global' | 'local', or the raw
                                                  -- value when it failed to parse
    matched_grant_id      TEXT,                   -- joins to rbac_audit_log
    scenario_id           TEXT,
    session_id            TEXT,
    step_id               TEXT,
    step_idx              INTEGER     NOT NULL DEFAULT 0,
    decided_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE demo_decision_log IS
    'Every orchestration decision made during a lab demo. Written asynchronously '
    'in batches by decision.recorder.BatchingDecisionSink; a write failure '
    'degrades to a warning and never blocks the demo, so absence of rows is not '
    'proof that no decision was made.';

COMMENT ON COLUMN demo_decision_log.mechanism IS
    'Which rule produced the action. Contract, not a debug string — every '
    'baseline comparison groups by this column.';

CREATE INDEX IF NOT EXISTS idx_demo_dec_session   ON demo_decision_log (session_id);
CREATE INDEX IF NOT EXISTS idx_demo_dec_point     ON demo_decision_log (decision_point);
CREATE INDEX IF NOT EXISTS idx_demo_dec_mechanism ON demo_decision_log (mechanism);
CREATE INDEX IF NOT EXISTS idx_demo_dec_time      ON demo_decision_log (decided_at DESC);


CREATE TABLE IF NOT EXISTS demo_correction_log (
    id                    BIGSERIAL   PRIMARY KEY,
    correction_id         UUID        NOT NULL UNIQUE,
    -- Nullable on purpose. An operator clicking "Move On" when no decision was
    -- logged is still a label: it says the window should have closed then.
    -- Requiring a parent row would discard the most informative corrections.
    decision_id           UUID        REFERENCES demo_decision_log (decision_id)
                                      ON DELETE SET NULL,
    decision_point        TEXT        NOT NULL,
    corrected_to_kind     TEXT        NOT NULL,
    corrected_to_payload  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    source                TEXT        NOT NULL,   -- 'operator' | 'auto' | 'policy'
    reason                TEXT        NOT NULL DEFAULT '',
    supervisor_id         TEXT,                   -- needed for inter-rater agreement
    step_id               TEXT,
    step_idx              INTEGER     NOT NULL DEFAULT 0,
    scenario_id           TEXT,
    session_id            TEXT,
    corrected_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE demo_correction_log IS
    'Supervisor overrides of demo decisions — the training signal. supervisor_id '
    'is required for inter-rater agreement between two or more supervisors.';

CREATE INDEX IF NOT EXISTS idx_demo_corr_decision ON demo_correction_log (decision_id);
CREATE INDEX IF NOT EXISTS idx_demo_corr_session  ON demo_correction_log (session_id);
CREATE INDEX IF NOT EXISTS idx_demo_corr_point    ON demo_correction_log (decision_point);
CREATE INDEX IF NOT EXISTS idx_demo_corr_time     ON demo_correction_log (corrected_at DESC);


-- ── Aggregates ───────────────────────────────────────────────────────────────
-- Part of the schema rather than ad-hoc SQL, for the same reason as the RBAC
-- denial views: these are the numbers the work is evaluated on.

-- Correction rate per mechanism. A mechanism corrected often is one the learned
-- policy should replace first; one never corrected is already good enough.
CREATE OR REPLACE VIEW demo_corrections_by_mechanism AS
SELECT d.mechanism,
       d.decision_point,
       d.scenario_id,
       COUNT(*)                                   AS decisions,
       COUNT(c.correction_id)                     AS corrections,
       ROUND(
           COUNT(c.correction_id)::numeric
           / NULLIF(COUNT(*), 0), 4
       )                                          AS correction_rate,
       MAX(d.decided_at)                          AS last_decided_at
FROM   demo_decision_log d
LEFT   JOIN demo_correction_log c ON c.decision_id = d.decision_id
GROUP  BY d.mechanism, d.decision_point, d.scenario_id;

-- Per-step breakdown — which points in the tour are contentious.
CREATE OR REPLACE VIEW demo_decisions_by_step AS
SELECT session_id,
       scenario_id,
       step_idx,
       step_id,
       decision_point,
       COUNT(*)                        AS decisions,
       COUNT(DISTINCT mechanism)       AS distinct_mechanisms,
       MIN(decided_at)                 AS first_decided_at,
       MAX(decided_at)                 AS last_decided_at
FROM   demo_decision_log
GROUP  BY session_id, scenario_id, step_idx, step_id, decision_point;

-- Plan revisions only, with their ops unpacked — how often the tour actually
-- changed shape, and why.
CREATE OR REPLACE VIEW demo_plan_revisions AS
SELECT d.session_id,
       d.scenario_id,
       d.step_id,
       d.step_idx,
       d.mechanism,
       op ->> 'kind'      AS op_kind,
       op ->> 'robot_id'  AS op_robot_id,
       d.decided_at
FROM   demo_decision_log d
CROSS  JOIN LATERAL jsonb_array_elements(
           COALESCE(d.action_payload -> 'ops', '[]'::jsonb)
       ) AS op
WHERE  d.decision_point = 'plan_revise';

COMMIT;

-- ===================================================================
-- 005_log_rls.sql
-- ===================================================================
-- 005_log_rls.sql
-- Run in Supabase SQL editor (or psql) after 004_demo_decisions.sql.
--
-- Grants the three log tables the write access the server actually needs.
--
-- WHY THIS EXISTS
-- Supabase enables row-level security on new public tables. A table with RLS on
-- and no policy denies everything, so both 003 and 004 landed as write-only-in-
-- theory: the server connects with the anon key, every INSERT was rejected, and
-- because both sinks degrade a write failure to a warning, a whole demo could
-- run and record nothing while reporting success. Confirmed empirically —
-- `rbac_audit_log` and `demo_decision_log` both returned 42501 on insert.
--
-- SCOPE OF WHAT THIS OPENS
-- These policies give the anon role INSERT and SELECT on three log tables. That
-- is the same access the anon role already has to `chat_logs`, which holds the
-- actual conversation transcripts — so this widens nothing that was previously
-- closed. It does NOT make the tables safe to expose publicly: the anon key
-- ships in the dashboard bundle, so anyone holding it can read and write these
-- rows. Treat the decision and audit logs as no more private than chat_logs.
--
-- THE ALTERNATIVE, IF YOU WANT THESE CLOSED
-- Point the server at SUPABASE_SERVICE_ROLE_KEY (already in .env) instead of
-- SUPABASE_KEY, drop these policies, and leave RLS denying anon. service_role
-- bypasses RLS entirely, so the server keeps working and the anon key loses all
-- access to the logs. That is the stricter setup and it is a one-line change in
-- core/config.py — but it also gives the server unrestricted access to every
-- other table, so it is a deliberate trade, not a default.
--
-- Reversible: see 005_log_rls_down.sql

BEGIN;

-- ── RBAC audit log (migration 003) ───────────────────────────────────────────

ALTER TABLE rbac_audit_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rbac_audit_log_insert ON rbac_audit_log;
CREATE POLICY rbac_audit_log_insert ON rbac_audit_log
    FOR INSERT TO anon, authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS rbac_audit_log_select ON rbac_audit_log;
CREATE POLICY rbac_audit_log_select ON rbac_audit_log
    FOR SELECT TO anon, authenticated
    USING (true);


-- ── Demo decision log (migration 004) ────────────────────────────────────────

ALTER TABLE demo_decision_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_decision_log_insert ON demo_decision_log;
CREATE POLICY demo_decision_log_insert ON demo_decision_log
    FOR INSERT TO anon, authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS demo_decision_log_select ON demo_decision_log;
CREATE POLICY demo_decision_log_select ON demo_decision_log
    FOR SELECT TO anon, authenticated
    USING (true);


-- ── Demo correction log (migration 004) ──────────────────────────────────────
-- No UPDATE policy anywhere here, deliberately. Corrections are the training
-- signal; an append-only log is the point, and a supervisor changing their mind
-- should be a new row, not an edit to an old one.
--
-- DELETE is granted, but ONLY for the probe rows tools/check_migrations.py
-- writes to prove the insert path works. Without it that probe cannot clean up
-- after itself: the delete is silently denied (PostgREST reports success with
-- zero rows affected), and every run leaves a fake row behind that then shows up
-- in demo_corrections_by_mechanism as its own mechanism. Real rows stay
-- undeletable — the predicate matches only the literal probe marker.

ALTER TABLE demo_correction_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_correction_log_insert ON demo_correction_log;
CREATE POLICY demo_correction_log_insert ON demo_correction_log
    FOR INSERT TO anon, authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS demo_correction_log_select ON demo_correction_log;
CREATE POLICY demo_correction_log_select ON demo_correction_log
    FOR SELECT TO anon, authenticated
    USING (true);


-- ── Probe cleanup ────────────────────────────────────────────────────────────
-- Scoped to the marker check_migrations.py writes and nothing else.

DROP POLICY IF EXISTS rbac_audit_log_delete_probe ON rbac_audit_log;
CREATE POLICY rbac_audit_log_delete_probe ON rbac_audit_log
    FOR DELETE TO anon, authenticated
    USING (reason = '_probe' AND requester_robot_id = '_probe');

DROP POLICY IF EXISTS demo_decision_log_delete_probe ON demo_decision_log;
CREATE POLICY demo_decision_log_delete_probe ON demo_decision_log
    FOR DELETE TO anon, authenticated
    USING (mechanism = '_probe' AND decision_point = '_probe');

DROP POLICY IF EXISTS demo_correction_log_delete_probe ON demo_correction_log;
CREATE POLICY demo_correction_log_delete_probe ON demo_correction_log
    FOR DELETE TO anon, authenticated
    USING (source = '_probe' AND decision_point = '_probe');


-- Remove probe rows left behind by runs made before these policies existed.
DELETE FROM demo_correction_log WHERE source = '_probe' AND decision_point = '_probe';
DELETE FROM demo_decision_log   WHERE mechanism = '_probe' AND decision_point = '_probe';
DELETE FROM rbac_audit_log      WHERE reason = '_probe' AND requester_robot_id = '_probe';

COMMIT;

-- ===================================================================
-- 006_demo_kg.sql
-- ===================================================================
-- 006_demo_kg.sql
-- Run in Supabase SQL editor (or psql) after 005_log_rls.sql.
--
-- The robot→topic competence graph. Three tables:
--
--   demo_topics        the vocabulary: one row per thing a visitor can ask about
--   demo_topic_links   topic ↔ topic semantic edges (undirected, stored once)
--   demo_robot_topic   the LEARNED edge: how good is this robot at this topic
--
-- Only demo_robot_topic is learned. The other two are structure: a vocabulary
-- and a similarity graph over it, both seeded rather than trained.
--
-- WHY THE EDGE CARRIES A COUNT
-- A supervisor correction is one noisy observation, not a declaration. Storing
-- only a weight and overwriting it means the last correction wins, the weight
-- oscillates between disagreeing supervisors, and running the demo fifty times
-- teaches nothing. n_supervisor/n_outcome make the update additive and let the
-- read path hedge an edge that has barely been seen. See decision/kg.py.
--
-- WHY PROVENANCE IS SPLIT
-- A human pressing a button and a segment finishing inside its budget are
-- different strengths of evidence. Separate counts, different learning rates,
-- and "how much of this graph came from people?" stays a query.
--
-- Reversible: see 006_demo_kg_down.sql

BEGIN;

-- ── Vocabulary ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_topics (
    id          TEXT        PRIMARY KEY,   -- slug, e.g. 'topic:rag'
    label       TEXT        NOT NULL,      -- human-facing, e.g. 'retrieval augmented generation'
    category    TEXT        NOT NULL DEFAULT 'other',
    source      TEXT        NOT NULL DEFAULT '',   -- how it entered the vocabulary
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE demo_topics IS
    'Topic vocabulary for the lab demo. Seeded from project keywords — this '
    'system has no shared namespace with the CHATBOX KG, whose topics are a '
    'child''s interests (jazz, baseball) and transfer nothing.';


-- ── Topic ↔ topic similarity ─────────────────────────────────────────────────
-- Undirected, stored once with endpoints sorted, matching the convention in
-- graph_relationship.topics.link_related_topic.

CREATE TABLE IF NOT EXISTS demo_topic_links (
    topic_a     TEXT        NOT NULL REFERENCES demo_topics(id) ON DELETE CASCADE,
    topic_b     TEXT        NOT NULL REFERENCES demo_topics(id) ON DELETE CASCADE,
    weight      REAL        NOT NULL CHECK (weight >= 0 AND weight <= 1),
    source      TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (topic_a, topic_b),
    CONSTRAINT demo_topic_links_sorted CHECK (topic_a < topic_b)
);

COMMENT ON TABLE demo_topic_links IS
    'Semantic neighbours. This is the ONLY thing that makes a correction '
    'generalise beyond the exact topic it was made on, so a sparse link set '
    'means sparse generalisation — worth measuring before relying on it.';


-- ── The learned edge ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_robot_topic (
    robot_id      TEXT        NOT NULL,
    topic_id      TEXT        NOT NULL REFERENCES demo_topics(id) ON DELETE CASCADE,
    weight        REAL        NOT NULL DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    n_supervisor  INTEGER     NOT NULL DEFAULT 0 CHECK (n_supervisor >= 0),
    n_outcome     INTEGER     NOT NULL DEFAULT 0 CHECK (n_outcome >= 0),
    last_updated  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (robot_id, topic_id)
);

COMMENT ON COLUMN demo_robot_topic.weight IS
    'Learned competence in [0,1]. 0.5 = no evidence. Updated additively: '
    'w += lr(kind, n) * (target - w). Never overwritten.';
COMMENT ON COLUMN demo_robot_topic.n_supervisor IS
    'Human corrections folded into this edge. With n_outcome, drives both the '
    'learning rate and the read-time confidence.';

CREATE INDEX IF NOT EXISTS idx_demo_rt_robot ON demo_robot_topic (robot_id);
CREATE INDEX IF NOT EXISTS idx_demo_rt_topic ON demo_robot_topic (topic_id);


-- ── Read view: the graph as the dashboard wants it ───────────────────────────
-- confidence and clamped are computed here so the UI, the policy and any
-- analysis all read the same numbers rather than each reimplementing them.
-- Constants mirror decision/kg.py (CONFIDENCE_HALFLIFE = 3, NEUTRAL = 0.5).

-- DROP then CREATE, so this stays re-runnable after 007 has added n_displaced.
-- CREATE OR REPLACE cannot remove a column from an existing view, so replacing
-- the post-007 view with this narrower definition fails with 42P16 — meaning a
-- second run of apply_all.sql would error out partway through.
DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id,
       rt.topic_id,
       t.label                                   AS topic_label,
       t.category                                AS topic_category,
       rt.weight,
       rt.n_supervisor,
       rt.n_outcome,
       rt.n_supervisor + rt.n_outcome            AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome)::real
                 / ((rt.n_supervisor + rt.n_outcome) + 3.0)
       END                                       AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome)::real
                    / ((rt.n_supervisor + rt.n_outcome) + 3.0))
       END                                       AS clamped,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.0
            ELSE rt.n_supervisor::real / (rt.n_supervisor + rt.n_outcome)
       END                                       AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;


-- ── RLS, matching 005 ────────────────────────────────────────────────────────
-- The graph IS editable, unlike the append-only logs: an update rewrites the
-- weight in place, so UPDATE is granted here where it is withheld there.

ALTER TABLE demo_topics       ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_topic_links  ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_robot_topic  ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_topics_all ON demo_topics;
CREATE POLICY demo_topics_all ON demo_topics
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS demo_topic_links_all ON demo_topic_links;
CREATE POLICY demo_topic_links_all ON demo_topic_links
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS demo_robot_topic_all ON demo_robot_topic;
CREATE POLICY demo_robot_topic_all ON demo_robot_topic
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

COMMIT;

-- ===================================================================
-- 007_kg_displaced.sql
-- ===================================================================
-- 007_kg_displaced.sql
-- Run in Supabase SQL editor (or psql) after 006_demo_kg.sql.
--
-- Adds the third evidence count to demo_robot_topic.
--
-- WHY A THIRD COUNT
-- When an operator reroutes a question from A to B, that is one action carrying
-- two different claims of very different strength. It is strong evidence FOR B —
-- a person deliberately picked them. It is weak evidence against A: the operator
-- wanted B, which is not the same as judging A incompetent. They may have known
-- B had a better answer ready, or simply that A had already spoken twice.
--
-- Folding both into n_supervisor would make one preference for B erode A across
-- every topic B is good at, and would make the human/automatic decomposition
-- meaningless. Separate count, separate learning rate (0.08 vs 0.50).
--
-- Reversible: see 007_kg_displaced_down.sql

BEGIN;

ALTER TABLE demo_robot_topic
    ADD COLUMN IF NOT EXISTS n_displaced INTEGER NOT NULL DEFAULT 0
    CHECK (n_displaced >= 0);

COMMENT ON COLUMN demo_robot_topic.n_displaced IS
    'Times this robot was the one routed AWAY from by an operator. Weakest '
    'evidence class — see decision/kg.py BASE_LR. Counts as human evidence in '
    'human_share: the operator did act, they just acted by choosing someone else.';

-- The view recomputes confidence/clamped/human_share, so it has to learn about
-- the new count or every displaced observation would be invisible to the
-- dashboard and to any analysis reading the view rather than the table.
--
-- DROP then CREATE, not CREATE OR REPLACE. Replace can only APPEND columns to a
-- view; it cannot insert one, and n_displaced belongs next to the other two
-- counts rather than tacked on after last_updated. Postgres rejects the shift
-- with 42P16 ("cannot change name of view column"). Dropping is safe here — a
-- view holds no data, and it is recreated in the same transaction.
DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id,
       rt.topic_id,
       t.label                                   AS topic_label,
       t.category                                AS topic_category,
       rt.weight,
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
       -- Displacements count as human: a person acted, by choosing someone else.
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_displaced)::real
                 / (rt.n_supervisor + rt.n_outcome + rt.n_displaced)
       END                                       AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;

COMMIT;

-- ===================================================================
-- 008_demo_durations.sql
-- ===================================================================
-- 008_demo_durations.sql
-- Run in Supabase SQL editor (or psql) after 007_kg_displaced.sql.
--
-- How long the tour actually takes. TWO TABLES, deliberately.
--
-- WHY NOT ONE
-- Scripted step duration is a property of the CONTENT: a robot explaining RAG
-- takes about as long every time, so averaging across runs converges on
-- something useful. Q&A length is a property of the OPERATOR and the GROUP —
-- a chatty school group and a silent industry delegation produce wildly
-- different numbers for the identical step. Folding them into one average gives
-- every step a variance that has nothing to do with the step, and the estimate
-- gets WORSE the more data you collect.
--
-- WHAT CHANGES BECAUSE OF THAT
-- Q&A stops being something the planner predicts and becomes something it SETS.
-- Faced with 15 minutes for a 25-minute tour, the planner does not ask "how long
-- will this Q&A run" — it says "this Q&A gets 90 seconds". That is also the
-- better lever: visitors notice a missing robot, they do not notice a Q&A that
-- ran a little short.
--
-- Q&A durations are still recorded here, because you need them to know what a
-- SENSIBLE budget is and how often an operator overruns it. That is the
-- difference between allocating 90 seconds because it is a round number and
-- allocating it because most windows land there.
--
-- Reversible: see 008_demo_durations_down.sql

BEGIN;

-- ── Scripted steps: the predictable part ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_step_durations (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    step_id         TEXT        NOT NULL,
    robot_id        TEXT,
    block_robot_id  TEXT,
    role            TEXT        NOT NULL DEFAULT '',
    seconds         REAL        NOT NULL CHECK (seconds >= 0),
    -- Generated speech varies in length run to run, so the same step_id can
    -- legitimately differ. Stored to separate "this step is slow" from "the
    -- model was verbose today".
    text_chars      INTEGER     NOT NULL DEFAULT 0,
    generated       BOOLEAN     NOT NULL DEFAULT FALSE,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_step_dur_step ON demo_step_durations (step_id);
CREATE INDEX IF NOT EXISTS idx_step_dur_run  ON demo_step_durations (run_id);


-- ── Q&A windows: the operator-dependent part ─────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_qa_durations (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    step_id         TEXT        NOT NULL,
    block_robot_id  TEXT,
    seconds         REAL        NOT NULL CHECK (seconds >= 0),
    turns           INTEGER     NOT NULL DEFAULT 0,
    -- NULL when the window was manual-advance only, which is the default for an
    -- unhurried tour. A number means the planner allocated a budget.
    budget_sec      REAL,
    -- 'operator' | 'policy' | 'timeout' | 'auto'. With budget_sec, this is what
    -- makes "how often does the operator overrun the allocation" a query.
    closed_by       TEXT        NOT NULL DEFAULT 'unknown',
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qa_dur_step ON demo_qa_durations (step_id);
CREATE INDEX IF NOT EXISTS idx_qa_dur_run  ON demo_qa_durations (run_id);


-- ── Aggregates ───────────────────────────────────────────────────────────────
-- Two views, never joined. The planner reads step_stats to predict and
-- qa_stats to choose a default budget.

DROP VIEW IF EXISTS demo_step_duration_stats;
CREATE VIEW demo_step_duration_stats AS
SELECT step_id,
       role,
       block_robot_id,
       COUNT(*)                          AS runs,
       ROUND(AVG(seconds)::numeric, 2)   AS mean_sec,
       ROUND(STDDEV_SAMP(seconds)::numeric, 2) AS sd_sec,
       ROUND(MIN(seconds)::numeric, 2)   AS min_sec,
       ROUND(MAX(seconds)::numeric, 2)   AS max_sec
FROM   demo_step_durations
GROUP  BY step_id, role, block_robot_id;

DROP VIEW IF EXISTS demo_qa_duration_stats;
CREATE VIEW demo_qa_duration_stats AS
SELECT step_id,
       block_robot_id,
       COUNT(*)                          AS windows,
       ROUND(AVG(seconds)::numeric, 2)   AS mean_sec,
       ROUND(STDDEV_SAMP(seconds)::numeric, 2) AS sd_sec,
       ROUND(MIN(seconds)::numeric, 2)   AS min_sec,
       ROUND(MAX(seconds)::numeric, 2)   AS max_sec,
       ROUND(AVG(turns)::numeric, 2)     AS mean_turns,
       -- How often a budgeted window ran past its allocation. The number that
       -- says whether the planner's budgets are realistic.
       COUNT(*) FILTER (WHERE budget_sec IS NOT NULL AND seconds > budget_sec)
                                         AS overruns,
       COUNT(*) FILTER (WHERE budget_sec IS NOT NULL) AS budgeted
FROM   demo_qa_durations
GROUP  BY step_id, block_robot_id;


-- ── RLS, matching the other log tables ───────────────────────────────────────

ALTER TABLE demo_step_durations ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_qa_durations   ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_step_durations_rw ON demo_step_durations;
CREATE POLICY demo_step_durations_rw ON demo_step_durations
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS demo_qa_durations_rw ON demo_qa_durations;
CREATE POLICY demo_qa_durations_rw ON demo_qa_durations
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

COMMIT;

