-- 009_kg_role_seed_down.sql
-- Reverses 009_kg_role_seed.sql.
--
-- WARNING: deletes the five seed rows and the three new topics BY IDENTITY —
-- exact (robot_id, topic_id) pairs and exact topic ids, not "everything with
-- source='role-seed'" filtered at run time, so this only ever removes what
-- this migration added. But if real observations landed on any of these five
-- edges since 009 ran, that learned weight is deleted along with the seed row
-- — the two are not distinguished once the row has been updated in place. And
-- deleting the three new topics CASCADEs to demo_robot_topic (see 006's FK),
-- so any OTHER edge against long-term-interaction, non-verbal-interaction, or
-- human-pose-estimation created after 009 ran — by real routing, not by this
-- migration — is lost too. Export first if any of that matters.
--
-- The two reused topics (retrieval-augmented-generation, social-robot-
-- navigation) are untouched — they existed before 009 and this migration only
-- referenced them, never created them.

BEGIN;

DELETE FROM demo_robot_topic WHERE (robot_id, topic_id) IN (
    ('pepper_01',  'topic:human-pose-estimation'),
    ('chatbox_01', 'topic:retrieval-augmented-generation'),
    ('chatbox_01', 'topic:long-term-interaction'),
    ('navel_001',  'topic:non-verbal-interaction'),
    ('silbot_01',  'topic:social-robot-navigation')
);

DELETE FROM demo_topics WHERE id IN (
    'topic:human-pose-estimation',
    'topic:long-term-interaction',
    'topic:non-verbal-interaction'
);

DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id, rt.topic_id, t.label AS topic_label, t.category AS topic_category,
       rt.weight, rt.n_supervisor, rt.n_outcome, rt.n_displaced,
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

ALTER TABLE demo_robot_topic DROP COLUMN IF EXISTS eligible;

COMMIT;
