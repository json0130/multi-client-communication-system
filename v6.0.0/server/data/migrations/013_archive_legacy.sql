-- 013_archive_legacy.sql
-- Moves legacy rows out of public into an `archive` schema. Nothing is dropped:
-- every row removed from public is copied first, in the same transaction.
--
--   chat_logs : created before 2026-03-01 (pre-v6), or from robots removed below
--   robots    : Navel_001, jay_mock_001 (not in lab_demo.yaml, no/near-zero logs)
--   users     : session rows no remaining chat_log refers to (user_id or subject_user_id)
--
-- `archive` is not exposed through PostgREST, and anon/authenticated get no grants.
-- Applied live 2026-09-24 04:08 UTC (supabase_migrations version 20260924040832).
-- Primary keys on the archive tables were added separately: 013b_archive_primary_keys.
-- Reversible: see 013_archive_legacy_down.sql (run 013b_..._down.sql first)

BEGIN;

CREATE SCHEMA IF NOT EXISTS archive;
REVOKE ALL ON SCHEMA archive FROM anon, authenticated, public;

CREATE TABLE IF NOT EXISTS archive.chat_logs (LIKE public.chat_logs);
CREATE TABLE IF NOT EXISTS archive.users     (LIKE public.users);
CREATE TABLE IF NOT EXISTS archive.robots    (LIKE public.robots);
ALTER TABLE archive.chat_logs ADD COLUMN IF NOT EXISTS archived_at timestamptz DEFAULT now(),
                              ADD COLUMN IF NOT EXISTS archive_reason text;
ALTER TABLE archive.users     ADD COLUMN IF NOT EXISTS archived_at timestamptz DEFAULT now(),
                              ADD COLUMN IF NOT EXISTS archive_reason text;
ALTER TABLE archive.robots    ADD COLUMN IF NOT EXISTS archived_at timestamptz DEFAULT now(),
                              ADD COLUMN IF NOT EXISTS archive_reason text;

-- 1. chat logs
WITH moved AS (
    DELETE FROM public.chat_logs
    WHERE created_at < '2026-03-01'
       OR source_robot_id IN ('Navel_001', 'jay_mock_001')
    RETURNING *
)
INSERT INTO archive.chat_logs
SELECT m.*, now(),
       CASE WHEN m.created_at < '2026-03-01' THEN 'pre_v6' ELSE 'removed_robot' END
FROM moved m;

-- 2. robots
WITH moved AS (
    DELETE FROM public.robots
    WHERE client_id IN ('Navel_001', 'jay_mock_001')
    RETURNING *
)
INSERT INTO archive.robots SELECT m.*, now(), 'not_in_profile_unused' FROM moved m;

-- 3. session users nothing in public.chat_logs points at any more
WITH moved AS (
    DELETE FROM public.users u
    WHERE NOT EXISTS (
        SELECT 1 FROM public.chat_logs c
        WHERE c.user_id = u.user_id OR c.subject_user_id = u.user_id
    )
    RETURNING *
)
INSERT INTO archive.users
SELECT m.*, now(),
       CASE WHEN EXISTS (SELECT 1 FROM archive.chat_logs a WHERE a.user_id = m.user_id)
            THEN 'pre_v6_session' ELSE 'orphan_session' END
FROM moved m;

REVOKE ALL ON ALL TABLES IN SCHEMA archive FROM anon, authenticated, public;

COMMIT;
