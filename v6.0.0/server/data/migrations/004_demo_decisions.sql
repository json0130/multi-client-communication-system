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
