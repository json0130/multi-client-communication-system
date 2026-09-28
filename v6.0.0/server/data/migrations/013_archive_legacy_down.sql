-- 013_archive_legacy_down.sql
-- Restores everything 013_archive_legacy.sql moved, then drops the archive tables.
-- Order matters: users before chat_logs (FK chat_logs.user_id -> users.user_id).

BEGIN;

INSERT INTO public.users
SELECT user_id, name, created_at, updated_at, interests, health_conditions
FROM archive.users
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO public.robots
SELECT id, client_id, robot_name, robot_role, modules, last_seen, created_at, character,
       ocean_traits, allowed_tags, is_active, ip_address, ws_port, persona_id,
       access_level, scenario_id
FROM archive.robots
ON CONFLICT (client_id) DO NOTHING;

INSERT INTO public.chat_logs
SELECT id, user_id, message, response, created_at, source_robot_id, scenario_id,
       session_id, subject_user_id, visibility
FROM archive.chat_logs
ON CONFLICT (id) DO NOTHING;

DROP TABLE archive.chat_logs, archive.users, archive.robots;
DROP SCHEMA IF EXISTS archive;

COMMIT;
