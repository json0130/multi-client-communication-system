-- 007_kg_displaced.sql
-- Run in Supabase SQL editor (or psql) after 006_demo_kg.sql.
--
-- Adds the third evidence count to demo_robot_topic.
--
-- WHY A THIRD COUNT
-- When an operator reroutes a question from A to B, that is one action carrying
-- two different claims of very different strength. It is strong evidence FOR B —
-- a person deliberately picked them. It is weak evidence against A: the operator
-- wanted B, which is not the same as judging A incompetent. They may have known
-- B had a better answer ready, or simply that A had already spoken twice.
--
-- Folding both into n_supervisor would make one preference for B erode A across
-- every topic B is good at, and would make the human/automatic decomposition
-- meaningless. Separate count, separate learning rate (0.08 vs 0.50).
--
-- Reversible: see 007_kg_displaced_down.sql

BEGIN;

ALTER TABLE demo_robot_topic
    ADD COLUMN IF NOT EXISTS n_displaced INTEGER NOT NULL DEFAULT 0
    CHECK (n_displaced >= 0);

COMMENT ON COLUMN demo_robot_topic.n_displaced IS
    'Times this robot was the one routed AWAY from by an operator. Weakest '
    'evidence class — see decision/kg.py BASE_LR. Counts as human evidence in '
    'human_share: the operator did act, they just acted by choosing someone else.';

-- The view recomputes confidence/clamped/human_share, so it has to learn about
-- the new count or every displaced observation would be invisible to the
-- dashboard and to any analysis reading the view rather than the table.
--
-- DROP then CREATE, not CREATE OR REPLACE. Replace can only APPEND columns to a
-- view; it cannot insert one, and n_displaced belongs next to the other two
-- counts rather than tacked on after last_updated. Postgres rejects the shift
-- with 42P16 ("cannot change name of view column"). Dropping is safe here — a
-- view holds no data, and it is recreated in the same transaction.
DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id,
       rt.topic_id,
       t.label                                   AS topic_label,
       t.category                                AS topic_category,
       rt.weight,
       rt.n_supervisor,
       rt.n_outcome,
       rt.n_displaced,
       rt.n_supervisor + rt.n_outcome + rt.n_displaced        AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                 / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0)
       END                                       AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                    / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0))
       END                                       AS clamped,
       -- Displacements count as human: a person acted, by choosing someone else.
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_displaced)::real
                 / (rt.n_supervisor + rt.n_outcome + rt.n_displaced)
       END                                       AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;

COMMIT;
