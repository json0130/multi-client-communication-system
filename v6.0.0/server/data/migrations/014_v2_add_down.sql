-- 014_v2_add_down.sql — reverses 014_v2_add.sql.
--
-- Run only before phase 2 code has written to the new tables and columns:
-- dropping them discards whatever they hold (demo_runs, sessions, run_id,
-- origin). The restored constraints are added NOT VALID, so rows written while
-- they were absent can't make the rollback fail; VALIDATE them afterwards if
-- you need the old guarantee back.

BEGIN;

DROP VIEW IF EXISTS public.log_integrity;
DROP VIEW IF EXISTS public.demo_qa_duration_stats_by_mode;
DROP VIEW IF EXISTS public.demo_step_duration_stats_by_mode;

DROP INDEX IF EXISTS public.idx_robots_persona;
DROP INDEX IF EXISTS public.idx_demo_topic_links_topic_b;
DROP INDEX IF EXISTS public.idx_correction_log_run;
DROP INDEX IF EXISTS public.idx_decision_log_run;
DROP INDEX IF EXISTS public.idx_chat_logs_run;
DROP INDEX IF EXISTS public.idx_chat_logs_session_created;
-- (idx_sessions_robot and idx_demo_runs_mode go with their tables.)

ALTER TABLE public.chat_logs ALTER COLUMN subject_user_id TYPE integer;

ALTER TABLE public.users
    ADD CONSTRAINT "Users_user_id_key" UNIQUE (user_id);
ALTER TABLE public.chat_logs
    ADD CONSTRAINT chat_logs_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES public.users (user_id) ON DELETE CASCADE NOT VALID;
ALTER TABLE public.demo_correction_log
    ADD CONSTRAINT demo_correction_log_decision_id_fkey
    FOREIGN KEY (decision_id) REFERENCES public.demo_decision_log (decision_id)
    ON DELETE SET NULL NOT VALID;

ALTER TABLE public.chat_logs           DROP COLUMN IF EXISTS run_id;
ALTER TABLE public.demo_correction_log DROP COLUMN IF EXISTS origin,
                                       DROP COLUMN IF EXISTS run_id_source,
                                       DROP COLUMN IF EXISTS run_id;
ALTER TABLE public.demo_decision_log   DROP COLUMN IF EXISTS run_id_source,
                                       DROP COLUMN IF EXISTS run_id;

DROP TABLE IF EXISTS public.sessions;
DROP TABLE IF EXISTS public.demo_runs;
DROP TABLE IF EXISTS public.supervisors;
DROP TABLE IF EXISTS public.scenarios;

COMMIT;
