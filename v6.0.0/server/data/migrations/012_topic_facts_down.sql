-- 012_topic_facts_down.sql
-- Reverses 012_topic_facts.sql.
--
-- WARNING: these rows are hand-written project knowledge — what a robot is
-- allowed to say about its own research. They cannot be recomputed from any
-- other table. Export before dropping.

BEGIN;
DROP VIEW   IF EXISTS demo_topic_fact_coverage;
DROP POLICY IF EXISTS demo_topic_facts_all ON demo_topic_facts;
DROP INDEX  IF EXISTS idx_demo_topic_facts_topic;
DROP INDEX  IF EXISTS idx_demo_topic_facts_robot;
DROP TABLE  IF EXISTS demo_topic_facts;
COMMIT;
