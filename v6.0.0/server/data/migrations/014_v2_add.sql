-- 014_v2_add.sql — Lab Demo Schema v2, phase 1 ("add").
--
-- Additive only: every statement is safe while the demo server on feat/navel-demo
-- (old code) is still writing to this database. New tables, new NULLABLE columns,
-- new views and indexes; the old views are untouched (the old server reads them
-- with select *). Nothing is backfilled here — that is phase 3.
--
-- The three removals are deliberate and safe for old code:
--   * demo_correction_log_decision_id_fkey — the sinks insert a whole batch per
--     statement and treat failure as a warning, so a correction naming a decision
--     whose batch was lost took its own batch down with it (the docstring in
--     data/demo_decision_repo.py saying otherwise was wrong). Log tables carry no
--     foreign keys under v2; log_integrity reports dangling ids instead.
--   * chat_logs_user_id_fkey — same rule for chat_logs. Old code still writes
--     user_id; it just is not enforced.
--   * "Users_user_id_key" — a duplicate of "Users_pkey"; it could only go once
--     the FK above no longer depended on it.
--
-- Knowledge-graph FKs to robots are NOT here: ON DELETE RESTRICT applies to new
-- deletes even when added NOT VALID, which would break the old dashboard's
-- "delete robot". They land in phase 4, after delete becomes deactivate.
--
-- New tables get RLS with no policies, so anon and authenticated cannot touch
-- them. The server must use the service_role key before phase-2 code writes
-- to them.
--
-- Reversible: 014_v2_add_down.sql

BEGIN;

-- ── Configuration ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.scenarios (
    id            text PRIMARY KEY,          -- profiles/*.yaml scenario id
    name          text,
    profile_path  text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.supervisors (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO public.supervisors (id, name)
VALUES ('legacy_unattributed', 'Unattributed (before supervisor tracking)')
ON CONFLICT (id) DO NOTHING;

-- ── Runtime ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.demo_runs (
    id                text PRIMARY KEY,      -- 'run-{epoch}'; 2-7 Sep runs are 12-char hex
    scenario_id       text REFERENCES public.scenarios (id) ON UPDATE CASCADE,
    mode              text NOT NULL DEFAULT 'live'
                      CHECK (mode IN ('live', 'scripted_visitor', 'gazebo', 'headless_sim')),
    started_at        timestamptz,
    ended_at          timestamptz,
    time_budget_sec   real,
    visitor_style     text,
    visitor_interest  text,
    supervisor_id     text REFERENCES public.supervisors (id) ON UPDATE CASCADE,
    notes             text,
    created_at        timestamptz NOT NULL DEFAULT now()
);

-- A robot connection. Spans many runs, so it has no run_id.
CREATE TABLE IF NOT EXISTS public.sessions (
    id              text PRIMARY KEY,        -- '{client_id}:{epoch}', 'sim-…', 'legacy-user-{user_id}'
    robot_id        text,                    -- no FK: sim sessions name robots that never existed
    scenario_id     text REFERENCES public.scenarios (id) ON UPDATE CASCADE,
    legacy_user_id  bigint,                  -- the users row this replaces, while users exists
    started_at      timestamptz,
    ended_at        timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.scenarios   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.supervisors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.demo_runs   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions    ENABLE ROW LEVEL SECURITY;

-- ── Log columns (nullable; filled by phase 2 code and the phase 3 backfill) ──

ALTER TABLE public.demo_decision_log
    ADD COLUMN IF NOT EXISTS run_id        text,
    ADD COLUMN IF NOT EXISTS run_id_source text
        CHECK (run_id_source IN ('recorded', 'inferred'));

ALTER TABLE public.demo_correction_log
    ADD COLUMN IF NOT EXISTS run_id        text,
    ADD COLUMN IF NOT EXISTS run_id_source text
        CHECK (run_id_source IN ('recorded', 'inferred')),
    -- Who produced the override. A supervisor id can't say that a script
    -- (tools/video_scenarios.py) clicked Move On, or that a test did.
    ADD COLUMN IF NOT EXISTS origin        text
        CHECK (origin IN ('human', 'scripted', 'test', 'synthetic'));

ALTER TABLE public.chat_logs
    ADD COLUMN IF NOT EXISTS run_id text;

-- ── Removals (see header) ────────────────────────────────────────────────────

ALTER TABLE public.demo_correction_log DROP CONSTRAINT IF EXISTS demo_correction_log_decision_id_fkey;
ALTER TABLE public.chat_logs           DROP CONSTRAINT IF EXISTS chat_logs_user_id_fkey;
ALTER TABLE public.users               DROP CONSTRAINT IF EXISTS "Users_user_id_key";

-- ── Housekeeping ─────────────────────────────────────────────────────────────

ALTER TABLE public.chat_logs ALTER COLUMN subject_user_id TYPE bigint;

CREATE INDEX IF NOT EXISTS idx_chat_logs_session_created ON public.chat_logs (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_logs_run             ON public.chat_logs (run_id);
CREATE INDEX IF NOT EXISTS idx_decision_log_run          ON public.demo_decision_log (run_id);
CREATE INDEX IF NOT EXISTS idx_correction_log_run        ON public.demo_correction_log (run_id);
CREATE INDEX IF NOT EXISTS idx_sessions_robot            ON public.sessions (robot_id);
CREATE INDEX IF NOT EXISTS idx_demo_runs_mode            ON public.demo_runs (mode);
CREATE INDEX IF NOT EXISTS idx_demo_topic_links_topic_b  ON public.demo_topic_links (topic_b);
CREATE INDEX IF NOT EXISTS idx_robots_persona            ON public.robots (persona_id);

-- ── Views ────────────────────────────────────────────────────────────────────
-- New views only. demo_step_duration_stats / demo_qa_duration_stats stay as they
-- are until phase 4: grouping them by mode would split the rows old code reads.
-- A duration whose run isn't in demo_runs yet reports mode 'unknown'.

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
       round(max(d.seconds)::numeric, 2)       AS max_sec
FROM public.demo_step_durations d
LEFT JOIN public.demo_runs r ON r.id = d.run_id
GROUP BY 1, d.step_id, d.role, d.block_robot_id;

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
       count(*) FILTER (WHERE q.budget_sec IS NOT NULL) AS budgeted
FROM public.demo_qa_durations q
LEFT JOIN public.demo_runs r ON r.id = q.run_id
GROUP BY 1, q.step_id, q.block_robot_id;

-- Ids in log tables with no parent row. Stands in for foreign keys (see header).
-- Expect large counts until the phase 3 backfill; after it, rows written by
-- phase 2 code should never appear here.
CREATE OR REPLACE VIEW public.log_integrity
WITH (security_invoker = true) AS
SELECT 'demo_decision_log' AS log_table, 'run_id' AS ref, count(*) AS dangling
  FROM public.demo_decision_log l
 WHERE l.run_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.demo_runs p WHERE p.id = l.run_id)
UNION ALL
SELECT 'demo_decision_log', 'session_id', count(*)
  FROM public.demo_decision_log l
 WHERE l.session_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.sessions p WHERE p.id = l.session_id)
UNION ALL
SELECT 'demo_correction_log', 'run_id', count(*)
  FROM public.demo_correction_log l
 WHERE l.run_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.demo_runs p WHERE p.id = l.run_id)
UNION ALL
SELECT 'demo_correction_log', 'session_id', count(*)
  FROM public.demo_correction_log l
 WHERE l.session_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.sessions p WHERE p.id = l.session_id)
UNION ALL
SELECT 'demo_correction_log', 'decision_id', count(*)
  FROM public.demo_correction_log l
 WHERE l.decision_id IS NOT NULL
   AND NOT EXISTS (SELECT 1 FROM public.demo_decision_log p WHERE p.decision_id = l.decision_id)
UNION ALL
SELECT 'demo_correction_log', 'supervisor_id', count(*)
  FROM public.demo_correction_log l
 WHERE l.supervisor_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.supervisors p WHERE p.id = l.supervisor_id)
UNION ALL
SELECT 'chat_logs', 'session_id', count(*)
  FROM public.chat_logs l
 WHERE l.session_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.sessions p WHERE p.id = l.session_id)
UNION ALL
SELECT 'chat_logs', 'run_id', count(*)
  FROM public.chat_logs l
 WHERE l.run_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.demo_runs p WHERE p.id = l.run_id)
UNION ALL
SELECT 'demo_step_durations', 'run_id', count(*)
  FROM public.demo_step_durations l
 WHERE NOT EXISTS (SELECT 1 FROM public.demo_runs p WHERE p.id = l.run_id)
UNION ALL
SELECT 'demo_qa_durations', 'run_id', count(*)
  FROM public.demo_qa_durations l
 WHERE NOT EXISTS (SELECT 1 FROM public.demo_runs p WHERE p.id = l.run_id)
UNION ALL
SELECT 'rbac_audit_log', 'session_id', count(*)
  FROM public.rbac_audit_log l
 WHERE l.session_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.sessions p WHERE p.id = l.session_id);

COMMIT;
