-- 008_demo_durations.sql
-- Run in Supabase SQL editor (or psql) after 007_kg_displaced.sql.
--
-- How long the tour actually takes. TWO TABLES, deliberately.
--
-- WHY NOT ONE
-- Scripted step duration is a property of the CONTENT: a robot explaining RAG
-- takes about as long every time, so averaging across runs converges on
-- something useful. Q&A length is a property of the OPERATOR and the GROUP —
-- a chatty school group and a silent industry delegation produce wildly
-- different numbers for the identical step. Folding them into one average gives
-- every step a variance that has nothing to do with the step, and the estimate
-- gets WORSE the more data you collect.
--
-- WHAT CHANGES BECAUSE OF THAT
-- Q&A stops being something the planner predicts and becomes something it SETS.
-- Faced with 15 minutes for a 25-minute tour, the planner does not ask "how long
-- will this Q&A run" — it says "this Q&A gets 90 seconds". That is also the
-- better lever: visitors notice a missing robot, they do not notice a Q&A that
-- ran a little short.
--
-- Q&A durations are still recorded here, because you need them to know what a
-- SENSIBLE budget is and how often an operator overruns it. That is the
-- difference between allocating 90 seconds because it is a round number and
-- allocating it because most windows land there.
--
-- Reversible: see 008_demo_durations_down.sql

BEGIN;

-- ── Scripted steps: the predictable part ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_step_durations (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    step_id         TEXT        NOT NULL,
    robot_id        TEXT,
    block_robot_id  TEXT,
    role            TEXT        NOT NULL DEFAULT '',
    seconds         REAL        NOT NULL CHECK (seconds >= 0),
    -- Generated speech varies in length run to run, so the same step_id can
    -- legitimately differ. Stored to separate "this step is slow" from "the
    -- model was verbose today".
    text_chars      INTEGER     NOT NULL DEFAULT 0,
    generated       BOOLEAN     NOT NULL DEFAULT FALSE,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_step_dur_step ON demo_step_durations (step_id);
CREATE INDEX IF NOT EXISTS idx_step_dur_run  ON demo_step_durations (run_id);


-- ── Q&A windows: the operator-dependent part ─────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_qa_durations (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    step_id         TEXT        NOT NULL,
    block_robot_id  TEXT,
    seconds         REAL        NOT NULL CHECK (seconds >= 0),
    turns           INTEGER     NOT NULL DEFAULT 0,
    -- NULL when the window was manual-advance only, which is the default for an
    -- unhurried tour. A number means the planner allocated a budget.
    budget_sec      REAL,
    -- 'operator' | 'policy' | 'timeout' | 'auto'. With budget_sec, this is what
    -- makes "how often does the operator overrun the allocation" a query.
    closed_by       TEXT        NOT NULL DEFAULT 'unknown',
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qa_dur_step ON demo_qa_durations (step_id);
CREATE INDEX IF NOT EXISTS idx_qa_dur_run  ON demo_qa_durations (run_id);


-- ── Aggregates ───────────────────────────────────────────────────────────────
-- Two views, never joined. The planner reads step_stats to predict and
-- qa_stats to choose a default budget.

DROP VIEW IF EXISTS demo_step_duration_stats;
CREATE VIEW demo_step_duration_stats AS
SELECT step_id,
       role,
       block_robot_id,
       COUNT(*)                          AS runs,
       ROUND(AVG(seconds)::numeric, 2)   AS mean_sec,
       ROUND(STDDEV_SAMP(seconds)::numeric, 2) AS sd_sec,
       ROUND(MIN(seconds)::numeric, 2)   AS min_sec,
       ROUND(MAX(seconds)::numeric, 2)   AS max_sec
FROM   demo_step_durations
GROUP  BY step_id, role, block_robot_id;

DROP VIEW IF EXISTS demo_qa_duration_stats;
CREATE VIEW demo_qa_duration_stats AS
SELECT step_id,
       block_robot_id,
       COUNT(*)                          AS windows,
       ROUND(AVG(seconds)::numeric, 2)   AS mean_sec,
       ROUND(STDDEV_SAMP(seconds)::numeric, 2) AS sd_sec,
       ROUND(MIN(seconds)::numeric, 2)   AS min_sec,
       ROUND(MAX(seconds)::numeric, 2)   AS max_sec,
       ROUND(AVG(turns)::numeric, 2)     AS mean_turns,
       -- How often a budgeted window ran past its allocation. The number that
       -- says whether the planner's budgets are realistic.
       COUNT(*) FILTER (WHERE budget_sec IS NOT NULL AND seconds > budget_sec)
                                         AS overruns,
       COUNT(*) FILTER (WHERE budget_sec IS NOT NULL) AS budgeted
FROM   demo_qa_durations
GROUP  BY step_id, block_robot_id;


-- ── RLS, matching the other log tables ───────────────────────────────────────

ALTER TABLE demo_step_durations ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_qa_durations   ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_step_durations_rw ON demo_step_durations;
CREATE POLICY demo_step_durations_rw ON demo_step_durations
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS demo_qa_durations_rw ON demo_qa_durations;
CREATE POLICY demo_qa_durations_rw ON demo_qa_durations
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

COMMIT;
