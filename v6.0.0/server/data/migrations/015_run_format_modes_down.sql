-- 015_run_format_modes_down.sql — reverses 015_run_format_modes.sql.
--
-- Runs labelled 'mock' or 'sim_unknown' cannot satisfy the old mode check, so
-- they are relabelled 'headless_sim' (the nearest old value: simulated, not
-- live) before it is restored. The 'live' default comes back. The views are
-- dropped and recreated: CREATE OR REPLACE cannot remove a column.

BEGIN;

DROP VIEW IF EXISTS public.demo_step_duration_stats_by_mode;
DROP VIEW IF EXISTS public.demo_qa_duration_stats_by_mode;

CREATE VIEW public.demo_step_duration_stats_by_mode
WITH (security_invoker = true) AS
SELECT COALESCE(r.mode, 'unknown') AS mode, d.step_id, d.role, d.block_robot_id,
       count(*) AS runs,
       round(avg(d.seconds)::numeric, 2) AS mean_sec,
       round(stddev_samp(d.seconds)::numeric, 2) AS sd_sec,
       round(min(d.seconds)::numeric, 2) AS min_sec,
       round(max(d.seconds)::numeric, 2) AS max_sec
FROM public.demo_step_durations d
LEFT JOIN public.demo_runs r ON r.id = d.run_id
GROUP BY 1, d.step_id, d.role, d.block_robot_id;

CREATE VIEW public.demo_qa_duration_stats_by_mode
WITH (security_invoker = true) AS
SELECT COALESCE(r.mode, 'unknown') AS mode, q.step_id, q.block_robot_id,
       count(*) AS windows,
       round(avg(q.seconds)::numeric, 2) AS mean_sec,
       round(stddev_samp(q.seconds)::numeric, 2) AS sd_sec,
       round(min(q.seconds)::numeric, 2) AS min_sec,
       round(max(q.seconds)::numeric, 2) AS max_sec,
       round(avg(q.turns), 2) AS mean_turns,
       count(*) FILTER (WHERE q.budget_sec IS NOT NULL AND q.seconds > q.budget_sec) AS overruns,
       count(*) FILTER (WHERE q.budget_sec IS NOT NULL) AS budgeted
FROM public.demo_qa_durations q
LEFT JOIN public.demo_runs r ON r.id = q.run_id
GROUP BY 1, q.step_id, q.block_robot_id;

DROP INDEX IF EXISTS public.idx_demo_runs_format;

ALTER TABLE public.robots
    DROP COLUMN IF EXISTS retired_at,
    DROP COLUMN IF EXISTS sim_twin,
    DROP COLUMN IF EXISTS embodiment;

UPDATE public.demo_runs SET mode = 'headless_sim' WHERE mode IN ('mock', 'sim_unknown');
ALTER TABLE public.demo_runs DROP CONSTRAINT IF EXISTS demo_runs_mode_check;
ALTER TABLE public.demo_runs
    ADD CONSTRAINT demo_runs_mode_check
    CHECK (mode IN ('live', 'scripted_visitor', 'gazebo', 'headless_sim'));
ALTER TABLE public.demo_runs ALTER COLUMN mode SET DEFAULT 'live';

ALTER TABLE public.demo_runs DROP COLUMN IF EXISTS format;

COMMIT;
