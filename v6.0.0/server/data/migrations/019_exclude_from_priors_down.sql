-- 019_exclude_from_priors_down.sql — reverses 019_exclude_from_priors.sql.
--
-- The views go back to 015's definitions (every run counted) before the
-- column they read is dropped. run-1790597934 keeps its explanatory note.

BEGIN;

CREATE OR REPLACE VIEW public.demo_step_duration_stats_by_mode
WITH (security_invoker = true) AS
SELECT COALESCE(r.mode, 'unknown')             AS mode,
       d.step_id,
       d.role,
       d.block_robot_id,
       count(*)                                AS runs,
       round(avg(d.seconds)::numeric, 2)       AS mean_sec,
       round(stddev_samp(d.seconds)::numeric, 2) AS sd_sec,
       round(min(d.seconds)::numeric, 2)       AS min_sec,
       round(max(d.seconds)::numeric, 2)       AS max_sec,
       COALESCE(r.format, 'unknown')           AS format
FROM public.demo_step_durations d
LEFT JOIN public.demo_runs r ON r.id = d.run_id
GROUP BY 1, d.step_id, d.role, d.block_robot_id, COALESCE(r.format, 'unknown');

CREATE OR REPLACE VIEW public.demo_qa_duration_stats_by_mode
WITH (security_invoker = true) AS
SELECT COALESCE(r.mode, 'unknown')             AS mode,
       q.step_id,
       q.block_robot_id,
       count(*)                                AS windows,
       round(avg(q.seconds)::numeric, 2)       AS mean_sec,
       round(stddev_samp(q.seconds)::numeric, 2) AS sd_sec,
       round(min(q.seconds)::numeric, 2)       AS min_sec,
       round(max(q.seconds)::numeric, 2)       AS max_sec,
       round(avg(q.turns), 2)                  AS mean_turns,
       count(*) FILTER (WHERE q.budget_sec IS NOT NULL AND q.seconds > q.budget_sec) AS overruns,
       count(*) FILTER (WHERE q.budget_sec IS NOT NULL) AS budgeted,
       COALESCE(r.format, 'unknown')           AS format
FROM public.demo_qa_durations q
LEFT JOIN public.demo_runs r ON r.id = q.run_id
GROUP BY 1, q.step_id, q.block_robot_id, COALESCE(r.format, 'unknown');

ALTER TABLE public.demo_runs DROP COLUMN IF EXISTS exclude_from_priors;

COMMIT;
