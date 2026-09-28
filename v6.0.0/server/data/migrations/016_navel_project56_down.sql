-- 016_navel_project56_down.sql
-- Reverses 016_navel_project56.sql: removes the Project 56 topics, links,
-- scope and facts, and restores Navel's emotion-aware scope and the facts
-- tools/seed_topic_facts.py drafted for it (text copied verbatim).
--
-- Undo any model transfer that copied these rows (tools/model_transfer.py)
-- FIRST: deleting the project-56 topics below cascades to every robot's
-- demo_robot_topic rows on them, navel_001's included.

BEGIN;

DELETE FROM demo_topic_facts
WHERE  robot_id = 'navel_01'
AND    source IN ('Project 56 Display Day Project Knowledge (2026-09-28)',
                  'this repository: Method B lives in navel-server');

DELETE FROM demo_robot_topic
WHERE  robot_id = 'navel_01'
AND    topic_id IN ('topic:co-speech-gesture', 'topic:robot-to-robot-communication',
                    'topic:speech-segmentation', 'topic:gesture-timing',
                    'topic:gesture-selection', 'topic:gaze-coordination');
-- navel_01's non-verbal-interaction row predates this migration (010 declared
-- it). navel_001 rows come from model transfers, which are undone separately.

UPDATE demo_robot_topic SET specialised = true
WHERE  robot_id = 'navel_01'
AND    topic_id IN ('topic:emotion-recognition', 'topic:facial-expression-analysis',
                    'topic:social-signals', 'topic:human-robot-trust');

DELETE FROM demo_topic_links WHERE source = 'project-56';
DELETE FROM demo_topics      WHERE source = 'project-56';

INSERT INTO demo_topic_facts (topic_id, robot_id, kind, fact, verified, source) VALUES
    ('topic:facial-expression-analysis', 'navel_01', 'model',
     'EfficientNet-B0 served as ONNX (the hsemotion backend), with a larger B2 variant available; roughly 5 ms per frame on CPU.',
     true, 'verified: modules/face_webcam/emotion_detector.py'),
    ('topic:facial-expression-analysis', 'navel_01', 'dataset',
     'Trained on AffectNet and VGAF. An older in-house EfficientNet-B0 trained on HQRAF is kept as a fallback backend.',
     true, 'verified: modules/face_webcam/emotion_detector.py'),
    ('topic:facial-expression-analysis', 'navel_01', 'method',
     'Seven expression classes — angry, disgust, fear, happy, neutral, sad, surprise. The 8-class variant maps contempt onto disgust. Face detection for the PyTorch backend is a Haar cascade.',
     true, 'verified: modules/face_webcam/emotion_detector.py'),
    ('topic:facial-expression-analysis', 'navel_01', 'limitation',
     'Affected by lighting, camera angle, occlusion, individual differences and genuinely ambiguous expressions.',
     false, 'supplied by the researcher — not independently checkable'),
    ('topic:emotion-recognition', 'navel_01', 'method',
     'Combines facial cues with conversational context to estimate how a visitor is responding, and adapts communication style to match. Scores are placed on Russell''s (1980) valence-arousal circumplex.',
     true, 'verified: emotion_detector.py _VA_TABLE'),
    ('topic:emotion-recognition', 'navel_01', 'limitation',
     'Emotion recognition is inherently uncertain, and accuracy on real visitors differs substantially from controlled datasets because of lighting, pose, cultural difference and natural variation in how people express themselves.',
     false, 'supplied by the researcher — not independently checkable'),
    ('topic:non-verbal-interaction', 'navel_01', 'method',
     'Communication carried by something other than the words themselves.',
     false, 'drafted from robots.robot_role'),
    ('topic:social-signals', 'navel_01', 'method',
     'Cues a person gives off during interaction, used to adapt the robot''s behaviour.',
     false, 'drafted from robots.robot_role'),
    ('topic:human-robot-trust', 'navel_01', 'method',
     'How a visitor''s confidence in the robot develops over an interaction.',
     false, 'drafted from robots.robot_role');

COMMIT;
