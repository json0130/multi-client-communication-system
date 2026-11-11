-- 020_chatbox_navel_scope_down.sql: remove what 020 added (and nothing else).
BEGIN;

DELETE FROM demo_robot_topic
WHERE  topic_id IN (SELECT id FROM demo_topics WHERE source IN ('chatbox-role', 'navel-role'));

DELETE FROM demo_topic_links WHERE source IN ('chatbox-role', 'navel-role');

DELETE FROM demo_topics WHERE source IN ('chatbox-role', 'navel-role');

COMMIT;
