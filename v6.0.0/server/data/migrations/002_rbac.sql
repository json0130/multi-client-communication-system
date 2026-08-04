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
