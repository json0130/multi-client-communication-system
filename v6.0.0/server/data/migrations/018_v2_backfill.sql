-- 018_v2_backfill.sql — Lab Demo Schema v2, phase 3 ("backfill").
--
-- Fills the tables and columns 014_v2_add created, from data already logged.
-- RE-RUNNABLE, and meant to be run again once phase-2 code is merged into the
-- branch that runs demos (to pick up rows the old server wrote in between):
--   * inserts are ON CONFLICT upserts that only widen time windows and fill NULLs;
--   * updates only touch rows whose target column is still NULL;
--   * nothing written by phase-2 code (mode, origin, supervisor, recorded
--     run_id) is ever overwritten.
-- Exact or NULL: a run_id is inferred only when exactly one run's time window
-- (±120 s) contains the row, and is marked run_id_source = 'inferred'.
--
-- Decisions and corrections are LABELLED here, not archived: nothing trains on
-- them yet, and analyses filter by demo_runs.mode and demo_correction_log.origin.
--
-- Modes are set from evidence, never guessed; nothing recorded before this
-- was live (every robot id is a *_01 simulation twin). See step 3.
--
-- Requires 015 (format, mock / sim_unknown, embodiment) and 017.
-- Reversible: 018_v2_backfill_down.sql (clears what this fills).

BEGIN;

-- ── 1. Scenarios every log refers to ─────────────────────────────────────────
INSERT INTO public.scenarios (id, name)
SELECT DISTINCT s, s FROM (
    SELECT scenario_id AS s FROM public.demo_decision_log
    UNION SELECT scenario_id FROM public.demo_correction_log
    UNION SELECT scenario_id FROM public.chat_logs
    UNION SELECT scenario_id FROM public.rbac_audit_log
    UNION SELECT scenario_id FROM public.robots
) x
WHERE s IS NOT NULL
ON CONFLICT (id) DO NOTHING;

-- ── 2. Runs, from every row that names one ───────────────────────────────────
-- Durations have always carried run_id; decisions since 11 Sep carry it in
-- observation (and, after phase 2, in the run_id column). 'unknown' is the
-- orchestrator's placeholder when no run was active, not a run.
WITH ev AS (
    SELECT run_id, recorded_at AS t FROM public.demo_step_durations
    UNION ALL SELECT run_id, recorded_at FROM public.demo_qa_durations
    UNION ALL SELECT observation->>'run_id', decided_at FROM public.demo_decision_log
              WHERE observation->>'run_id' IS NOT NULL
    UNION ALL SELECT run_id, decided_at FROM public.demo_decision_log
              WHERE run_id_source = 'recorded'
),
win AS (
    SELECT run_id, min(t) AS s, max(t) AS e
    FROM ev WHERE run_id IS NOT NULL AND run_id <> 'unknown'
    GROUP BY run_id
),
obs AS (      -- the run's last recorded observation
    SELECT DISTINCT ON (observation->>'run_id')
           observation->>'run_id'                        AS run_id,
           scenario_id,
           NULLIF(observation->>'time_budget_sec', '')::real AS time_budget_sec,
           NULLIF(observation->>'visitor_style', '')     AS visitor_style,
           NULLIF(observation->>'visitor_interest', '')  AS visitor_interest
    FROM public.demo_decision_log
    WHERE observation->>'run_id' IS NOT NULL
    ORDER BY observation->>'run_id', decided_at DESC
)
-- New rows start as 'sim_unknown' (mode has no default): step 3 below labels
-- them from evidence, and only ever rows still 'sim_unknown', so a mode the
-- server recorded itself is never overwritten.
INSERT INTO public.demo_runs AS r
       (id, mode, scenario_id, started_at, ended_at, time_budget_sec, visitor_style, visitor_interest)
SELECT w.run_id, 'sim_unknown', o.scenario_id, w.s, w.e, o.time_budget_sec, o.visitor_style, o.visitor_interest
FROM win w LEFT JOIN obs o USING (run_id)
ON CONFLICT (id) DO UPDATE SET
    started_at       = LEAST(r.started_at, EXCLUDED.started_at),
    ended_at         = GREATEST(r.ended_at, EXCLUDED.ended_at),
    scenario_id      = COALESCE(r.scenario_id, EXCLUDED.scenario_id),
    time_budget_sec  = COALESCE(r.time_budget_sec, EXCLUDED.time_budget_sec),
    visitor_style    = COALESCE(r.visitor_style, EXCLUDED.visitor_style),
    visitor_interest = COALESCE(r.visitor_interest, EXCLUDED.visitor_interest);

-- ── 3. Sessions ──────────────────────────────────────────────────────────────
-- Every session id any log mentions: '{client_id}:{epoch}', and 'sim-{name}'
-- from tools/demo_sim.py (robot_a..c are not real robots; hence no FK).
WITH ev AS (
    SELECT session_id, decided_at AS t, scenario_id FROM public.demo_decision_log
    UNION ALL SELECT session_id, corrected_at, scenario_id FROM public.demo_correction_log
    UNION ALL SELECT session_id, created_at, scenario_id FROM public.chat_logs
              WHERE session_id NOT LIKE 'legacy-user-%'
    UNION ALL SELECT session_id, decided_at, scenario_id FROM public.rbac_audit_log
)
INSERT INTO public.sessions AS s (id, robot_id, scenario_id, started_at, ended_at)
SELECT session_id,
       CASE WHEN session_id LIKE 'sim-%'        THEN substr(session_id, 5)
            WHEN session_id ~ ':[0-9]+$'        THEN regexp_replace(session_id, ':[0-9]+$', '')
       END,
       mode() WITHIN GROUP (ORDER BY scenario_id),
       min(t), max(t)
FROM ev WHERE session_id IS NOT NULL
GROUP BY session_id
ON CONFLICT (id) DO UPDATE SET
    started_at  = LEAST(s.started_at, EXCLUDED.started_at),
    ended_at    = GREATEST(s.ended_at, EXCLUDED.ended_at),
    robot_id    = COALESCE(s.robot_id, EXCLUDED.robot_id),
    scenario_id = COALESCE(s.scenario_id, EXCLUDED.scenario_id);

-- Chat logs written before sessions had ids: one synthetic session per users
-- row, the grouping RAG memory already used.
INSERT INTO public.sessions AS s (id, robot_id, scenario_id, legacy_user_id, started_at, ended_at)
SELECT 'legacy-user-' || user_id,
       mode() WITHIN GROUP (ORDER BY source_robot_id),
       mode() WITHIN GROUP (ORDER BY scenario_id),
       user_id, min(created_at), max(created_at)
FROM public.chat_logs
WHERE session_id IS NULL AND user_id IS NOT NULL
GROUP BY user_id
ON CONFLICT (id) DO UPDATE SET
    started_at  = LEAST(s.started_at, EXCLUDED.started_at),
    ended_at    = GREATEST(s.ended_at, EXCLUDED.ended_at),
    robot_id    = COALESCE(s.robot_id, EXCLUDED.robot_id),
    scenario_id = COALESCE(s.scenario_id, EXCLUDED.scenario_id);

UPDATE public.chat_logs SET session_id = 'legacy-user-' || user_id
 WHERE session_id IS NULL AND user_id IS NOT NULL;

-- ── 3b. Mode and format, from evidence ───────────────────────────────────────
-- Only runs still 'sim_unknown' (i.e. not labelled by the server). In order:
--   scripted_visitor  the three 11 Sep runs driven by visitor_sim.py
--   headless_sim      decisions of the run came from tools/demo_sim.py sessions
--   gazebo            movement steps took real time (median >= 2 s): only
--                     Gazebo moves a robot
--   mock              movement steps ACKed instantly, or the run is from an
--                     earlier day than the first run with movement steps (the
--                     script had none before the Gazebo work; mock clients from
--                     client/test_configs)
--   gazebo            no movement rows, but it ran on a robot connection whose
--                     runs with movement timing all moved in real time (evidence
--                     is only ever a run's own timing, so re-running can't chain)
--   live              every robot that spoke is physical (robots.embodiment,
--                     mirrored from the profile at boot) — none so far
-- Anything left stays 'sim_unknown' (and is excluded from duration priors).
UPDATE public.demo_runs SET mode = 'scripted_visitor'
 WHERE mode = 'sim_unknown'
   AND id IN ('run-1789100159', 'run-1789100793', 'run-1789101459');

UPDATE public.demo_runs r SET mode = 'headless_sim'
 WHERE r.mode = 'sim_unknown'
   AND EXISTS (SELECT 1 FROM public.demo_decision_log d
               WHERE COALESCE(d.run_id, d.observation->>'run_id') = r.id
                 AND d.session_id LIKE 'sim-%');

WITH nav AS (
    SELECT run_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY seconds) AS med
    FROM public.demo_step_durations
    WHERE step_id LIKE 'navigate\_%' OR step_id LIKE 'face\_%' OR step_id = 'approach_visitors'
    GROUP BY run_id
)
UPDATE public.demo_runs r
   SET mode = CASE WHEN nav.med >= 2 THEN 'gazebo' ELSE 'mock' END
  FROM nav
 WHERE r.mode = 'sim_unknown' AND nav.run_id = r.id;

UPDATE public.demo_runs r SET mode = 'mock'
 WHERE r.mode = 'sim_unknown'
   AND NOT EXISTS (SELECT 1 FROM public.demo_step_durations d
                   WHERE d.run_id = r.id
                     AND (d.step_id LIKE 'navigate\_%' OR d.step_id LIKE 'face\_%'
                          OR d.step_id = 'approach_visitors'))
   AND (r.started_at AT TIME ZONE 'Pacific/Auckland')::date
       < (SELECT (min(r2.started_at) AT TIME ZONE 'Pacific/Auckland')::date
          FROM public.demo_runs r2
                       WHERE EXISTS (SELECT 1 FROM public.demo_step_durations d2
                                     WHERE d2.run_id = r2.id
                                       AND (d2.step_id LIKE 'navigate\_%' OR d2.step_id LIKE 'face\_%'
                                            OR d2.step_id = 'approach_visitors')));

WITH nav AS (          -- each run's own movement timing: the only evidence used here
    SELECT run_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY seconds) AS med
    FROM public.demo_step_durations
    WHERE step_id LIKE 'navigate\_%' OR step_id LIKE 'face\_%' OR step_id = 'approach_visitors'
    GROUP BY run_id
),
conn AS (               -- runs whose start falls inside a robot connection
    SELECT r.id AS run_id, s.id AS session_id
    FROM public.demo_runs r
    JOIN public.sessions s
      ON s.id NOT LIKE 'legacy-user-%'
     AND r.started_at BETWEEN s.started_at - interval '5 minutes'
                          AND s.ended_at   + interval '5 minutes'
),
per_conn AS (           -- per connection: did every run with timing evidence move?
    SELECT c.session_id, bool_and(n.med >= 2) AS all_gazebo
    FROM conn c JOIN nav n ON n.run_id = c.run_id
    GROUP BY c.session_id
)
UPDATE public.demo_runs r SET mode = 'gazebo'
 WHERE r.mode = 'sim_unknown'
   AND NOT EXISTS (SELECT 1 FROM nav WHERE nav.run_id = r.id)
   AND EXISTS     (SELECT 1 FROM conn c JOIN per_conn p USING (session_id)
                   WHERE c.run_id = r.id AND p.all_gazebo)
   AND NOT EXISTS (SELECT 1 FROM conn c JOIN per_conn p USING (session_id)
                   WHERE c.run_id = r.id AND NOT p.all_gazebo);

UPDATE public.demo_runs r SET mode = 'live'
 WHERE r.mode = 'sim_unknown'
   AND EXISTS (SELECT 1 FROM public.demo_step_durations d WHERE d.run_id = r.id)
   AND NOT EXISTS (SELECT 1 FROM public.demo_step_durations d
                   LEFT JOIN public.robots rb ON rb.client_id = d.robot_id
                   WHERE d.run_id = r.id AND d.robot_id IS NOT NULL
                     AND COALESCE(rb.embodiment, '') <> 'physical');

UPDATE public.demo_runs
   SET notes = COALESCE(notes, 'ICRA video; visitor and Move On scripted by tools/video_scenarios.py')
 WHERE id IN ('run-1790139360', 'run-1790140549');

-- Solo: one robot spoke, and every step it spoke outside the closing belongs
-- to its own block. A full tour always opens with block-less guide steps.
WITH one AS (
    SELECT run_id, min(robot_id) AS robot
    FROM public.demo_step_durations
    WHERE robot_id IS NOT NULL
    GROUP BY run_id HAVING count(DISTINCT robot_id) = 1
)
UPDATE public.demo_runs r SET format = 'solo'
  FROM one
 WHERE r.id = one.run_id AND r.format = 'full_tour'
   AND NOT EXISTS (SELECT 1 FROM public.demo_step_durations d
                   WHERE d.run_id = r.id AND d.role <> 'closing'
                     AND d.block_robot_id IS DISTINCT FROM one.robot);

-- ── 4. run_id on decisions ───────────────────────────────────────────────────
UPDATE public.demo_decision_log
   SET run_id = observation->>'run_id', run_id_source = 'recorded'
 WHERE run_id IS NULL AND observation->>'run_id' IS NOT NULL;

-- Before 11 Sep the observation had no run_id: infer only when exactly one run
-- fits. Headless-sim sessions never had a run to fit.
WITH cand AS (
    SELECT d.id, min(r.id) AS run_id
    FROM public.demo_decision_log d
    JOIN public.demo_runs r
      ON d.decided_at BETWEEN r.started_at - interval '120 seconds'
                          AND r.ended_at   + interval '120 seconds'
    WHERE d.run_id IS NULL AND COALESCE(d.session_id, '') NOT LIKE 'sim-%'
    GROUP BY d.id HAVING count(*) = 1
)
UPDATE public.demo_decision_log d
   SET run_id = c.run_id, run_id_source = 'inferred'
  FROM cand c WHERE d.id = c.id;

-- ── 5. run_id on corrections ─────────────────────────────────────────────────
-- Through the linked decision first (as exact as that decision's run_id) ...
UPDATE public.demo_correction_log c
   SET run_id = d.run_id, run_id_source = d.run_id_source
  FROM public.demo_decision_log d
 WHERE c.run_id IS NULL AND c.decision_id = d.decision_id AND d.run_id IS NOT NULL;

-- ... then the same one-run-only time window.
WITH cand AS (
    SELECT c.id, min(r.id) AS run_id
    FROM public.demo_correction_log c
    JOIN public.demo_runs r
      ON c.corrected_at BETWEEN r.started_at - interval '120 seconds'
                            AND r.ended_at   + interval '120 seconds'
    WHERE c.run_id IS NULL AND COALESCE(c.session_id, '') NOT LIKE 'sim-%'
    GROUP BY c.id HAVING count(*) = 1
)
UPDATE public.demo_correction_log c
   SET run_id = cand.run_id, run_id_source = 'inferred'
  FROM cand WHERE c.id = cand.id;

-- ── 6. Who made each correction ──────────────────────────────────────────────
-- Only rows phase-2 code hasn't labelled (origin IS NULL). Order matters: the
-- first matching rule wins.
UPDATE public.demo_correction_log SET origin = 'synthetic'      -- tools/demo_sim.py
 WHERE origin IS NULL AND session_id LIKE 'sim-%';
UPDATE public.demo_correction_log SET origin = 'test'           -- labelled by the tester
 WHERE origin IS NULL AND reason IN ('auto-test-advance', 'testing');
UPDATE public.demo_correction_log SET origin = 'scripted'       -- tools/video_scenarios.py's default reason
 WHERE origin IS NULL AND reason = 'operator move-on';
-- The rest are unlabelled: a person or an unmarked script, unknown.
-- They keep origin NULL and are attributed to nobody in particular.
UPDATE public.demo_correction_log SET supervisor_id = 'legacy_unattributed'
 WHERE origin IS NULL AND supervisor_id IS NULL;

-- Next/Skip pressed while a solo robot waited for its human guide was recorded
-- as a QA_ADVANCE correction; it corrects the hand-off, not a Q&A window.
UPDATE public.demo_correction_log SET decision_point = 'solo_cue'
 WHERE step_id = 'wait_for_handoff' AND decision_point = 'qa_advance';

COMMIT;
