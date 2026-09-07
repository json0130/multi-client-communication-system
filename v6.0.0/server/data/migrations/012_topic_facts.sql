-- 012_topic_facts.sql
-- Run in Supabase SQL editor (or psql) after 011_robot_style_fit.sql.
--
-- Grounded detail for the topics the robots present.
--
-- WHY
-- A robot had NO source of technical detail. demo_topics carries a label and
-- a category; robots.robot_role is one paragraph; the RAG index is episodic
-- visitor chat. So when a visitor asked "which technique do you use", the
-- model filled the gap: a live run had Silbot answer "Extended Kalman
-- Filters for state estimation and particle filters for tracking multiple
-- people" and ChatBox answer "a retrieval model like BM25". Neither string
-- appears anywhere in this repository or database. They were invented,
-- stated confidently, to visitors.
--
-- The technical presentation style makes this worse, not better: it asks for
-- "a specific method, model or number", which is an instruction to invent
-- when nothing real is available.
--
-- A topic node with no detail beneath it cannot ground an answer. This adds
-- that detail as rows hanging off demo_topics — leaf attributes of a node
-- that already exists, not a new graph.
--
-- VERIFIED IS FALSE BY DEFAULT
-- A fact nobody has checked must be distinguishable from one a researcher
-- confirmed. Seeded rows are drafted from robots.robot_role and are marked
-- unverified; tools/check_facts.py reports what is still unreviewed, and the
-- server warns at boot. Nothing is silently promoted to fact by being typed.
--
-- Reversible: see 012_topic_facts_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS demo_topic_facts (
    id           BIGSERIAL   PRIMARY KEY,
    topic_id     TEXT        NOT NULL,
    -- NULL means the fact is true of the topic generally rather than of one
    -- robot's work on it. Kept nullable so shared background (what SLAM is)
    -- does not have to be duplicated per robot.
    robot_id     TEXT,
    kind         TEXT        NOT NULL,
    fact         TEXT        NOT NULL,
    verified     BOOLEAN     NOT NULL DEFAULT false,
    source       TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT demo_topic_facts_kind_check CHECK (kind IN (
        'method',      -- how the work is done
        'model',       -- a named model or algorithm actually used
        'dataset',     -- what it was trained or evaluated on
        'metric',      -- a measured result
        'hardware',    -- the platform it runs on
        'limitation',  -- what it does NOT do — the most useful kind in a demo
        'publication'  -- a paper a visitor could be pointed at
    )),
    CONSTRAINT demo_topic_facts_fact_len CHECK (char_length(fact) BETWEEN 3 AND 600)
);

COMMENT ON TABLE demo_topic_facts IS
    'Grounded detail beneath each topic node. Injected into a robot''s prompt '
    'when it answers about that topic, together with an instruction to decline '
    'specifics it does not have here rather than invent them.';
COMMENT ON COLUMN demo_topic_facts.verified IS
    'False until a researcher has confirmed the claim. Seeded rows are drafted '
    'from robots.robot_role and start false.';
COMMENT ON COLUMN demo_topic_facts.robot_id IS
    'NULL = true of the topic generally. Otherwise this robot''s own work on it.';

CREATE INDEX IF NOT EXISTS idx_demo_topic_facts_topic ON demo_topic_facts (topic_id);
CREATE INDEX IF NOT EXISTS idx_demo_topic_facts_robot ON demo_topic_facts (robot_id);

-- Coverage view: which topics can actually ground an answer, and how much of
-- what they hold has been reviewed.
CREATE OR REPLACE VIEW demo_topic_fact_coverage AS
SELECT t.id                                              AS topic_id,
       t.label,
       t.category,
       COUNT(f.id)                                       AS facts,
       COUNT(f.id) FILTER (WHERE f.verified)             AS verified_facts,
       COUNT(DISTINCT f.kind)                            AS kinds,
       BOOL_OR(f.robot_id IS NOT NULL)                   AS has_robot_specific
FROM   demo_topics t
LEFT   JOIN demo_topic_facts f ON f.topic_id = t.id
GROUP  BY t.id, t.label, t.category;

ALTER TABLE demo_topic_facts ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS demo_topic_facts_all ON demo_topic_facts;
CREATE POLICY demo_topic_facts_all ON demo_topic_facts
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

COMMIT;
