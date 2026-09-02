-- 008_demo_durations_down.sql
-- Reverses 008_demo_durations.sql.
--
-- WARNING: duration data only accumulates with time and cannot be backfilled —
-- it is a record of runs that already happened. Export before dropping.

BEGIN;

DROP VIEW  IF EXISTS demo_qa_duration_stats;
DROP VIEW  IF EXISTS demo_step_duration_stats;

DROP POLICY IF EXISTS demo_qa_durations_rw   ON demo_qa_durations;
DROP POLICY IF EXISTS demo_step_durations_rw ON demo_step_durations;

DROP INDEX IF EXISTS idx_qa_dur_run;
DROP INDEX IF EXISTS idx_qa_dur_step;
DROP INDEX IF EXISTS idx_step_dur_run;
DROP INDEX IF EXISTS idx_step_dur_step;

DROP TABLE IF EXISTS demo_qa_durations;
DROP TABLE IF EXISTS demo_step_durations;

COMMIT;
