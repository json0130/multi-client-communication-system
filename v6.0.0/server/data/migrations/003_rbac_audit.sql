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
