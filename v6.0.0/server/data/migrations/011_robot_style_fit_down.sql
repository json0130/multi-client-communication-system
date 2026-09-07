-- 011_robot_style_fit_down.sql
-- Reverses 011_robot_style_fit.sql.
--
-- WARNING: style-fit judgements only accumulate by someone watching a tour
-- and rating how an answer landed. They cannot be recomputed from any other
-- table. Export before dropping.

BEGIN;

DROP VIEW  IF EXISTS demo_style_fit;
DROP POLICY IF EXISTS demo_robot_style_all ON demo_robot_style;
DROP INDEX IF EXISTS idx_demo_robot_style_robot;
DROP TABLE IF EXISTS demo_robot_style;

COMMIT;
