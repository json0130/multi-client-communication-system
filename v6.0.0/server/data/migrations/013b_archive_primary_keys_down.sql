-- 013b_archive_primary_keys_down.sql
-- Reverses 013b. Run before 013_archive_legacy_down.sql.

BEGIN;

ALTER TABLE archive.chat_logs DROP CONSTRAINT IF EXISTS chat_logs_pkey;
ALTER TABLE archive.users     DROP CONSTRAINT IF EXISTS users_pkey;
ALTER TABLE archive.robots    DROP CONSTRAINT IF EXISTS robots_pkey;

COMMIT;
