-- 010_kg_declared_scope_down.sql
-- Reverses 010_kg_declared_scope.sql.
--
-- Dropping the column loses every declared project area, and routing returns
-- to whatever the learned weights say — which on a young graph means no
-- opinion at all and an alphabetical tie-break. The weights themselves are
-- untouched: this migration never wrote one, and its INSERT updated only
-- `specialised` on conflict.
--
-- The rows it INSERTED for pairs that did not previously exist are left in
-- place, at weight 0.5 with no observations. They are indistinguishable from
-- an unobserved edge and carry no competence claim, so removing them would
-- be churn rather than a reversal.

BEGIN;

DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id, rt.topic_id, t.label AS topic_label, t.category AS topic_category,
       rt.weight, rt.eligible,
       rt.n_supervisor, rt.n_outcome, rt.n_displaced,
       rt.n_supervisor + rt.n_outcome + rt.n_displaced AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                 / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0) END AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                    / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0)) END AS clamped,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_displaced)::real
                 / (rt.n_supervisor + rt.n_outcome + rt.n_displaced) END AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;

ALTER TABLE demo_robot_topic DROP COLUMN IF EXISTS specialised;

COMMIT;
