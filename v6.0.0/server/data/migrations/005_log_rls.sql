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
