-- 019_exclude_from_priors.sql — a run can be kept out of duration estimates.
--
-- Mode and format say what a run WAS; they are paper data and stay true. A
-- run can still be unfit to learn timings from — stopped mid-tour, a crash, a
-- test — so this is a separate, explicit flag rather than a relabelled mode:
--   * demo_runs.exclude_from_priors — false by default.
--   * the *_by_mode views skip flagged runs. They are what the planner costs a
--     tour with; model_transfer.py and app._qa_basis skip them in code.
--   * run-1790597934 is flagged: an e2e smoke test stopped mid-tour after the
--     planner crash fixed in 0038fa6, so its step timings are partial.
-- Its log rows are untouched; nothing is deleted.
--
-- Re-runnable. Reversible: 019_exclude_from_priors_down.sql

BEGIN;

ALTER TABLE public.demo_runs
    ADD COLUMN IF NOT EXISTS exclude_from_priors boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN public.demo_runs.exclude_from_priors IS
    'Keep this run out of duration estimates and sim-to-real priors. Mode and format still say what the run was; say why in notes.';

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
WHERE r.exclude_from_priors IS NOT TRUE
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
WHERE r.exclude_from_priors IS NOT TRUE
GROUP BY 1, q.step_id, q.block_robot_id, COALESCE(r.format, 'unknown');

UPDATE public.demo_runs
   SET exclude_from_priors = true,
       notes = 'e2e smoke test; stopped mid-tour after a planner crash (fixed in 0038fa6) — partial timings, excluded from priors'
 WHERE id = 'run-1790597934';

COMMIT;
