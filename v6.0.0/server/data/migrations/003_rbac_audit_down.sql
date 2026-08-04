-- 003_rbac_audit_down.sql
-- Reverses 003_rbac_audit.sql. Run in Supabase SQL editor (or psql).
--
-- WARNING: drops the audit history. Decisions already recorded are lost.
-- Retrieval is unaffected — a missing audit sink degrades to a warning.

BEGIN;

DROP VIEW  IF EXISTS rbac_denials_by_reason;
DROP VIEW  IF EXISTS rbac_denials_by_robot;

DROP INDEX IF EXISTS idx_rbac_audit_denials;
DROP INDEX IF EXISTS idx_rbac_audit_time;
DROP INDEX IF EXISTS idx_rbac_audit_reason;
DROP INDEX IF EXISTS idx_rbac_audit_robot;

DROP TABLE IF EXISTS rbac_audit_log;

COMMIT;
