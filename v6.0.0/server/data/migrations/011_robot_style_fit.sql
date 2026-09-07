-- 011_robot_style_fit.sql
-- Run in Supabase SQL editor (or psql) after 010_kg_declared_scope.sql.
--
-- How well each robot pitches to each kind of audience.
--
-- WHY A SEPARATE TABLE AND NOT A COLUMN ON demo_robot_topic
-- Different key. demo_robot_topic is (robot, topic) and answers "who should
-- take this question". This is (robot, style) and answers "how should the
-- answer be framed". Keying style fit by topic as well would be the honest
-- model — a robot might be fine explaining navigation to a business visitor
-- and poor on mapping — but 4 robots x 17 topics x 4 styles is 272 cells fed
-- by supervision that arrives a handful of judgements per tour, and almost
-- all of them would stay empty forever. Style habits generalise across a
-- robot's own subjects far better than competence does, so (robot, style) is
-- the level where the data actually accumulates.
--
-- WHY IT DOES NOT AFFECT ROUTING
-- A robot that explains navigation badly to a business audience is still the
-- robot that knows navigation. This value changes the framing directive that
-- goes into generation and nothing else; decision/kg_infer.py never reads it.
-- Keeping them apart is the same discipline as splitting `specialised` out of
-- `weight` in 010 — one number, one claim.
--
-- SUPERVISOR EVIDENCE ONLY
-- No n_outcome column, deliberately. A Q&A window closing cleanly means
-- nobody objected to the routing; it says nothing about whether the answer
-- was pitched well. Recording one would manufacture a judgement nobody made.
--
-- Reversible: see 011_robot_style_fit_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS demo_robot_style (
    robot_id      TEXT        NOT NULL,
    style         TEXT        NOT NULL,
    weight        REAL        NOT NULL DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    n_supervisor  INTEGER     NOT NULL DEFAULT 0 CHECK (n_supervisor >= 0),
    last_updated  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (robot_id, style),
    CONSTRAINT demo_robot_style_known_style
        CHECK (style IN ('technical', 'business', 'interactive', 'general'))
);

COMMENT ON TABLE demo_robot_style IS
    'How well a robot pitches to one kind of audience, learned from operator '
    'judgements only. Feeds the framing directive at generation time '
    '(decision/style_fit.py); never read by routing.';
COMMENT ON COLUMN demo_robot_style.weight IS
    'Learned audience fit in [0,1]. 0.5 = no evidence. Clamped by observation '
    'count on read, so one poor rating does not change how a robot is briefed.';

CREATE INDEX IF NOT EXISTS idx_demo_robot_style_robot ON demo_robot_style (robot_id);

-- Read view: confidence and clamped computed here so the dashboard, the
-- generation path and any analysis all read the same numbers. Constants
-- mirror decision/kg.py (CONFIDENCE_HALFLIFE = 3, NEUTRAL = 0.5), which is
-- also where the update arithmetic lives.
CREATE OR REPLACE VIEW demo_style_fit AS
SELECT rs.robot_id,
       rs.style,
       rs.weight,
       rs.n_supervisor,
       CASE WHEN rs.n_supervisor = 0 THEN 0.0
            ELSE rs.n_supervisor::real / (rs.n_supervisor + 3.0)
       END                                        AS confidence,
       CASE WHEN rs.n_supervisor = 0 THEN 0.5
            ELSE 0.5 + (rs.weight - 0.5)
                 * (rs.n_supervisor::real / (rs.n_supervisor + 3.0))
       END                                        AS clamped,
       rs.last_updated
FROM   demo_robot_style rs;

ALTER TABLE demo_robot_style ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_robot_style_all ON demo_robot_style;
CREATE POLICY demo_robot_style_all ON demo_robot_style
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

COMMIT;
