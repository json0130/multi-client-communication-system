-- 002_rbac_down.sql
-- Reverses 002_rbac.sql. Run in Supabase SQL editor (or psql).
--
-- WARNING: this drops the provenance and visibility columns. Any RBAC metadata
-- written since 002_rbac.sql was applied is lost and cannot be reconstructed —
-- the backfill in 002_rbac.sql only recovers what users.name happened to encode.
--
-- The FAISS sidecars on disk are versioned separately and are NOT touched here.
-- modules/rag/rag_module.py reads both the v1 (bare list) and v2 (envelope)
-- formats, so rolling this migration back leaves the sidecars readable.

BEGIN;

DROP INDEX IF EXISTS idx_chat_logs_visibility;
DROP INDEX IF EXISTS idx_chat_logs_scenario;
DROP INDEX IF EXISTS idx_chat_logs_source_robot;

ALTER TABLE chat_logs DROP CONSTRAINT IF EXISTS chat_logs_visibility_check;

ALTER TABLE chat_logs
    DROP COLUMN IF EXISTS visibility,
    DROP COLUMN IF EXISTS subject_user_id,
    DROP COLUMN IF EXISTS session_id,
    DROP COLUMN IF EXISTS scenario_id,
    DROP COLUMN IF EXISTS source_robot_id;

ALTER TABLE robots DROP CONSTRAINT IF EXISTS robots_access_level_check;

ALTER TABLE robots
    DROP COLUMN IF EXISTS scenario_id,
    DROP COLUMN IF EXISTS access_level;

COMMIT;
