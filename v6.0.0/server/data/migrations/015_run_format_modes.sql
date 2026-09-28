-- 015_run_format_modes.sql — Lab Demo Schema v2: run format, sim modes, embodiment.
--
-- Additive for old code (nothing it reads or writes changes shape):
--   * demo_runs.format — 'full_tour' | 'solo'. A solo run (one robot, a human
--     guide) reuses the full tour's step ids, so without this its timings and
--     Q&A windows would feed the full-tour planner priors and the paper stats.
--   * demo_runs.mode gains 'mock' (real server, mock robot clients that cannot
--     move) and 'sim_unknown' (simulated, client type not recoverable), and
--     loses its 'live' default: every writer must now say what it recorded.
--     Nothing recorded so far is live.
--   * robots.embodiment ('physical' | 'sim') and robots.sim_twin, mirrored at
--     boot from the scenario profile (sim_twin: key), so SQL, the backfill and
--     the paper stats can tell a physical robot from its simulation twin.
--     A run is live only when every robot in it is physical.
--   * robots.retired_at — "delete robot" retires instead: is_active is the
--     connection flag and is overwritten on every connect.
--   * the *_by_mode views gain `format` (appended, as CREATE OR REPLACE requires).
--
-- Reversible: 015_run_format_modes_down.sql

BEGIN;

ALTER TABLE public.demo_runs
    ADD COLUMN IF NOT EXISTS format text NOT NULL DEFAULT 'full_tour'
        CHECK (format IN ('full_tour', 'solo'));

ALTER TABLE public.demo_runs DROP CONSTRAINT IF EXISTS demo_runs_mode_check;
ALTER TABLE public.demo_runs
    ADD CONSTRAINT demo_runs_mode_check
    CHECK (mode IN ('live', 'scripted_visitor', 'gazebo', 'headless_sim', 'mock', 'sim_unknown'));
ALTER TABLE public.demo_runs ALTER COLUMN mode DROP DEFAULT;

ALTER TABLE public.robots
    ADD COLUMN IF NOT EXISTS embodiment text CHECK (embodiment IN ('physical', 'sim')),
    ADD COLUMN IF NOT EXISTS sim_twin   text,       -- a physical robot's simulation twin (client_id)
    ADD COLUMN IF NOT EXISTS retired_at timestamptz;

CREATE INDEX IF NOT EXISTS idx_demo_runs_format ON public.demo_runs (format);

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

COMMIT;
