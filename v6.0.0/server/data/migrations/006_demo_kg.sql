-- 006_demo_kg.sql
-- Run in Supabase SQL editor (or psql) after 005_log_rls.sql.
--
-- The robot→topic competence graph. Three tables:
--
--   demo_topics        the vocabulary: one row per thing a visitor can ask about
--   demo_topic_links   topic ↔ topic semantic edges (undirected, stored once)
--   demo_robot_topic   the LEARNED edge: how good is this robot at this topic
--
-- Only demo_robot_topic is learned. The other two are structure: a vocabulary
-- and a similarity graph over it, both seeded rather than trained.
--
-- WHY THE EDGE CARRIES A COUNT
-- A supervisor correction is one noisy observation, not a declaration. Storing
-- only a weight and overwriting it means the last correction wins, the weight
-- oscillates between disagreeing supervisors, and running the demo fifty times
-- teaches nothing. n_supervisor/n_outcome make the update additive and let the
-- read path hedge an edge that has barely been seen. See decision/kg.py.
--
-- WHY PROVENANCE IS SPLIT
-- A human pressing a button and a segment finishing inside its budget are
-- different strengths of evidence. Separate counts, different learning rates,
-- and "how much of this graph came from people?" stays a query.
--
-- Reversible: see 006_demo_kg_down.sql

BEGIN;

-- ── Vocabulary ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_topics (
    id          TEXT        PRIMARY KEY,   -- slug, e.g. 'topic:rag'
    label       TEXT        NOT NULL,      -- human-facing, e.g. 'retrieval augmented generation'
    category    TEXT        NOT NULL DEFAULT 'other',
    source      TEXT        NOT NULL DEFAULT '',   -- how it entered the vocabulary
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE demo_topics IS
    'Topic vocabulary for the lab demo. Seeded from project keywords — this '
    'system has no shared namespace with the CHATBOX KG, whose topics are a '
    'child''s interests (jazz, baseball) and transfer nothing.';


-- ── Topic ↔ topic similarity ─────────────────────────────────────────────────
-- Undirected, stored once with endpoints sorted, matching the convention in
-- graph_relationship.topics.link_related_topic.

CREATE TABLE IF NOT EXISTS demo_topic_links (
    topic_a     TEXT        NOT NULL REFERENCES demo_topics(id) ON DELETE CASCADE,
    topic_b     TEXT        NOT NULL REFERENCES demo_topics(id) ON DELETE CASCADE,
    weight      REAL        NOT NULL CHECK (weight >= 0 AND weight <= 1),
    source      TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (topic_a, topic_b),
    CONSTRAINT demo_topic_links_sorted CHECK (topic_a < topic_b)
);

COMMENT ON TABLE demo_topic_links IS
    'Semantic neighbours. This is the ONLY thing that makes a correction '
    'generalise beyond the exact topic it was made on, so a sparse link set '
    'means sparse generalisation — worth measuring before relying on it.';


-- ── The learned edge ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_robot_topic (
    robot_id      TEXT        NOT NULL,
    topic_id      TEXT        NOT NULL REFERENCES demo_topics(id) ON DELETE CASCADE,
    weight        REAL        NOT NULL DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    n_supervisor  INTEGER     NOT NULL DEFAULT 0 CHECK (n_supervisor >= 0),
    n_outcome     INTEGER     NOT NULL DEFAULT 0 CHECK (n_outcome >= 0),
    last_updated  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (robot_id, topic_id)
);

COMMENT ON COLUMN demo_robot_topic.weight IS
    'Learned competence in [0,1]. 0.5 = no evidence. Updated additively: '
    'w += lr(kind, n) * (target - w). Never overwritten.';
COMMENT ON COLUMN demo_robot_topic.n_supervisor IS
    'Human corrections folded into this edge. With n_outcome, drives both the '
    'learning rate and the read-time confidence.';

CREATE INDEX IF NOT EXISTS idx_demo_rt_robot ON demo_robot_topic (robot_id);
CREATE INDEX IF NOT EXISTS idx_demo_rt_topic ON demo_robot_topic (topic_id);


-- ── Read view: the graph as the dashboard wants it ───────────────────────────
-- confidence and clamped are computed here so the UI, the policy and any
-- analysis all read the same numbers rather than each reimplementing them.
-- Constants mirror decision/kg.py (CONFIDENCE_HALFLIFE = 3, NEUTRAL = 0.5).

-- DROP then CREATE, so this stays re-runnable after 007 has added n_displaced.
-- CREATE OR REPLACE cannot remove a column from an existing view, so replacing
-- the post-007 view with this narrower definition fails with 42P16 — meaning a
-- second run of apply_all.sql would error out partway through.
DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id,
       rt.topic_id,
       t.label                                   AS topic_label,
       t.category                                AS topic_category,
       rt.weight,
       rt.n_supervisor,
       rt.n_outcome,
       rt.n_supervisor + rt.n_outcome            AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome)::real
                 / ((rt.n_supervisor + rt.n_outcome) + 3.0)
       END                                       AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome)::real
                    / ((rt.n_supervisor + rt.n_outcome) + 3.0))
       END                                       AS clamped,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.0
            ELSE rt.n_supervisor::real / (rt.n_supervisor + rt.n_outcome)
       END                                       AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;


-- ── RLS, matching 005 ────────────────────────────────────────────────────────
-- The graph IS editable, unlike the append-only logs: an update rewrites the
-- weight in place, so UPDATE is granted here where it is withheld there.

ALTER TABLE demo_topics       ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_topic_links  ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_robot_topic  ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS demo_topics_all ON demo_topics;
CREATE POLICY demo_topics_all ON demo_topics
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS demo_topic_links_all ON demo_topic_links;
CREATE POLICY demo_topic_links_all ON demo_topic_links
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS demo_robot_topic_all ON demo_robot_topic;
CREATE POLICY demo_robot_topic_all ON demo_robot_topic
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

COMMIT;
