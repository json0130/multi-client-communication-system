-- 007_kg_displaced_down.sql
-- Reverses 007_kg_displaced.sql.
--
-- WARNING: drops n_displaced, losing the record of which observations were
-- displacements. The weights those observations produced REMAIN — the update is
-- applied in place — so the graph keeps their effect while losing the ability to
-- attribute it. The evidence-source decomposition becomes unreproducible.

BEGIN;

DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id, rt.topic_id, t.label AS topic_label, t.category AS topic_category,
       rt.weight, rt.n_supervisor, rt.n_outcome,
       rt.n_supervisor + rt.n_outcome AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome)::real
                 / ((rt.n_supervisor + rt.n_outcome) + 3.0) END AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome)::real
                    / ((rt.n_supervisor + rt.n_outcome) + 3.0)) END AS clamped,
       CASE WHEN rt.n_supervisor + rt.n_outcome = 0 THEN 0.0
            ELSE rt.n_supervisor::real / (rt.n_supervisor + rt.n_outcome) END AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;

ALTER TABLE demo_robot_topic DROP COLUMN IF EXISTS n_displaced;

COMMIT;
