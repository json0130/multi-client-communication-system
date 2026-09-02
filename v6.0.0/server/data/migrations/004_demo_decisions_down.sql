-- 004_demo_decisions_down.sql
-- Reverses 004_demo_decisions.sql. Run in Supabase SQL editor (or psql).
--
-- WARNING: drops every recorded decision AND every supervisor correction.
-- The corrections are the training signal and are not reconstructable from the
-- chat logs — export before running this.
--
-- The demo itself is unaffected: a missing decision sink degrades to a warning.

BEGIN;

DROP VIEW  IF EXISTS demo_plan_revisions;
DROP VIEW  IF EXISTS demo_decisions_by_step;
DROP VIEW  IF EXISTS demo_corrections_by_mechanism;

DROP INDEX IF EXISTS idx_demo_corr_time;
DROP INDEX IF EXISTS idx_demo_corr_point;
DROP INDEX IF EXISTS idx_demo_corr_session;
DROP INDEX IF EXISTS idx_demo_corr_decision;

-- Corrections first — they carry the FK to demo_decision_log.
DROP TABLE IF EXISTS demo_correction_log;

DROP INDEX IF EXISTS idx_demo_dec_time;
DROP INDEX IF EXISTS idx_demo_dec_mechanism;
DROP INDEX IF EXISTS idx_demo_dec_point;
DROP INDEX IF EXISTS idx_demo_dec_session;

DROP TABLE IF EXISTS demo_decision_log;

COMMIT;
