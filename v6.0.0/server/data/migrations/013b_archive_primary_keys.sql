-- 013b_archive_primary_keys.sql
-- Applied live 2026-09-24 04:10 UTC (supabase_migrations version 20260924041036).
-- Adds primary keys to the archive tables created by 013 (performance advisor: no_primary_key).
-- Reversible: see 013b_archive_primary_keys_down.sql

BEGIN;

ALTER TABLE archive.chat_logs ADD PRIMARY KEY (id);
ALTER TABLE archive.users     ADD PRIMARY KEY (user_id);
ALTER TABLE archive.robots    ADD PRIMARY KEY (id);

COMMIT;
