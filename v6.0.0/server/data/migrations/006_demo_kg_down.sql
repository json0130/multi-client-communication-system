-- 006_demo_kg_down.sql
-- Reverses 006_demo_kg.sql. Run in Supabase SQL editor (or psql).
--
-- WARNING: drops the learned graph. Every weight and observation count is lost,
-- and they are NOT reconstructable from demo_correction_log alone — the update
-- is order-dependent, so replaying corrections only reproduces the graph if they
-- are replayed in their original order against the same seed vocabulary.
-- Export demo_robot_topic first if the weights matter.

BEGIN;

DROP VIEW IF EXISTS demo_kg_edges;

DROP POLICY IF EXISTS demo_robot_topic_all ON demo_robot_topic;
DROP POLICY IF EXISTS demo_topic_links_all ON demo_topic_links;
DROP POLICY IF EXISTS demo_topics_all      ON demo_topics;

DROP INDEX IF EXISTS idx_demo_rt_topic;
DROP INDEX IF EXISTS idx_demo_rt_robot;

DROP TABLE IF EXISTS demo_robot_topic;
DROP TABLE IF EXISTS demo_topic_links;
DROP TABLE IF EXISTS demo_topics;

COMMIT;
