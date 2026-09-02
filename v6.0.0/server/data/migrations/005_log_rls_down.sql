-- 005_log_rls_down.sql
-- Reverses 005_log_rls.sql. Run in Supabase SQL editor (or psql).
--
-- Drops the policies and leaves RLS ENABLED, which means the anon role can no
-- longer read or write the logs. The server will keep running — both sinks
-- degrade a write failure to a warning — so the symptom of running this without
-- switching to SUPABASE_SERVICE_ROLE_KEY is silent data loss, not an outage.
--
-- Run tools/check_migrations.py afterwards; it probes an actual insert.

BEGIN;

DROP POLICY IF EXISTS demo_correction_log_delete_probe ON demo_correction_log;
DROP POLICY IF EXISTS demo_decision_log_delete_probe ON demo_decision_log;
DROP POLICY IF EXISTS rbac_audit_log_delete_probe ON rbac_audit_log;

DROP POLICY IF EXISTS demo_correction_log_select ON demo_correction_log;
DROP POLICY IF EXISTS demo_correction_log_insert ON demo_correction_log;

DROP POLICY IF EXISTS demo_decision_log_select ON demo_decision_log;
DROP POLICY IF EXISTS demo_decision_log_insert ON demo_decision_log;

DROP POLICY IF EXISTS rbac_audit_log_select ON rbac_audit_log;
DROP POLICY IF EXISTS rbac_audit_log_insert ON rbac_audit_log;

COMMIT;
