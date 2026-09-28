-- 016_navel_project56.sql
-- =========================
-- Navel's project is Project 56 — Human-Comprehensible Robot-to-Robot
-- Communication via Co-Speech Motion Generation — not emotion-aware
-- interaction, which is what 010's declared scope and the drafted facts in
-- tools/seed_topic_facts.py said. Source: "Project 56: Display Day Project
-- Knowledge" (researcher-supplied, 2026-09-28), so its facts ship verified.
--
-- Data only, no schema change. Written for navel_01, the simulation twin, only.
-- The physical navel_001 receives these facts and this declared scope through
-- a recorded sim-to-real transfer (tools/model_transfer.py, 017's
-- model_transfers), never by writing the same rows twice here. The
-- IN ('navel_01', 'navel_001') clauses below only retire old rows, and match
-- nothing for navel_001 until a transfer has run.
--
-- Originally 013_navel_project56 on feat/navel-demo (unapplied); renumbered.
--
-- Re-runnable: this migration's own facts are deleted by `source` before they
-- are inserted, and every other write is an upsert or a no-op on conflict.
-- Reversible: see 016_navel_project56_down.sql.

BEGIN;

-- ── Topics ───────────────────────────────────────────────────────────────────
INSERT INTO demo_topics (id, label, category, source) VALUES
    ('topic:co-speech-gesture',            'co-speech gesture',            'hri',    'project-56'),
    ('topic:robot-to-robot-communication', 'robot to robot communication', 'hri',    'project-56'),
    ('topic:speech-segmentation',          'speech segmentation',          'speech', 'project-56'),
    ('topic:gesture-timing',               'speech and gesture timing',    'speech', 'project-56'),
    ('topic:gesture-selection',            'gesture selection',            'ai',     'project-56'),
    ('topic:gaze-coordination',            'gaze and head coordination',   'hri',    'project-56')
ON CONFLICT (id) DO NOTHING;

-- topic_a < topic_b (demo_topic_links_sorted).
INSERT INTO demo_topic_links (topic_a, topic_b, weight, source) VALUES
    ('topic:co-speech-gesture', 'topic:gesture-selection',            0.80, 'project-56'),
    ('topic:co-speech-gesture', 'topic:gesture-timing',               0.75, 'project-56'),
    ('topic:co-speech-gesture', 'topic:non-verbal-interaction',       0.80, 'project-56'),
    ('topic:co-speech-gesture', 'topic:robot-to-robot-communication', 0.70, 'project-56'),
    ('topic:gaze-coordination', 'topic:non-verbal-interaction',       0.65, 'project-56'),
    ('topic:gesture-selection', 'topic:gesture-timing',               0.60, 'project-56'),
    ('topic:gesture-timing',    'topic:speech-segmentation',          0.70, 'project-56')
ON CONFLICT (topic_a, topic_b) DO NOTHING;

-- ── Declared scope ───────────────────────────────────────────────────────────
INSERT INTO demo_robot_topic (robot_id, topic_id, weight, specialised)
SELECT r.robot_id, t.topic_id, 0.5, true
FROM   (VALUES ('navel_01')) AS r(robot_id)
CROSS  JOIN (VALUES
    ('topic:co-speech-gesture'),
    ('topic:robot-to-robot-communication'),
    ('topic:speech-segmentation'),
    ('topic:gesture-timing'),
    ('topic:gesture-selection'),
    ('topic:gaze-coordination'),
    ('topic:non-verbal-interaction')
) AS t(topic_id)
ON CONFLICT (robot_id, topic_id) DO UPDATE SET specialised = true;

-- The old emotion-aware area stops being Navel's declared scope. Rows are
-- kept (their learned weights are history), only the declaration goes.
UPDATE demo_robot_topic SET specialised = false
WHERE  robot_id IN ('navel_01', 'navel_001')
AND    topic_id IN ('topic:emotion-recognition', 'topic:facial-expression-analysis',
                    'topic:social-signals', 'topic:human-robot-trust');

-- ── Facts ────────────────────────────────────────────────────────────────────
-- Retire the facts drafted for the old role — they describe a different
-- project and would be injected into every answer Navel gives.
DELETE FROM demo_topic_facts
WHERE  robot_id IN ('navel_01', 'navel_001')
AND    topic_id IN ('topic:emotion-recognition', 'topic:facial-expression-analysis',
                    'topic:social-signals', 'topic:human-robot-trust',
                    'topic:non-verbal-interaction');

DELETE FROM demo_topic_facts
WHERE  robot_id = 'navel_01'          -- not navel_001: its copies belong to a transfer
AND    source IN ('Project 56 Display Day Project Knowledge (2026-09-28)',
                  'this repository: Method B lives in navel-server');

-- ORDER MATTERS. facts_for_robot() spreads by kind in rounds and, within a
-- kind, keeps insertion order — so the first row of each kind is the one a
-- robot with no resolved topic is grounded on first. Lead each kind with the
-- fact the project talk most needs.
INSERT INTO demo_topic_facts (topic_id, robot_id, kind, fact, verified, source)
SELECT f.topic_id, r.robot_id, f.kind, f.fact, true,
       'Project 56 Display Day Project Knowledge (2026-09-28)'
FROM   (VALUES ('navel_01')) AS r(robot_id)
CROSS  JOIN (VALUES
    -- method
    ('topic:robot-to-robot-communication', 'method',
     'The problem: robots can exchange information successfully while people nearby cannot tell what they mean or why they are moving. Project 56 asks how non-verbal behaviour can be generated so that robot-to-robot communication becomes understandable to human observers.'),
    ('topic:co-speech-gesture', 'method',
     'Method B, the main contribution, splits a generated reply into communicative segments, estimates how long each takes to say, removes gestures that do not fit, avoids repeating the previous expressive gesture, scores the remaining gestures against what the segment means, and performs one gesture with each segment.'),
    ('topic:co-speech-gesture', 'method',
     'Method A, the baseline, attaches one action to the whole response. Method B can choose a different gesture for each part of one reply, such as greeting, then pointing, then showing uncertainty, and checks that each gesture fits its own speaking time.'),
    ('topic:gesture-selection', 'method',
     'The language model never controls the motors. It only scores high-level action tags from a fixed list of what Navel can do, using the segment text, the action descriptions, the conversation so far and optionally the visitor''s detected emotion. Navel maps the chosen tag to predefined, tested motion code.'),
    ('topic:robot-to-robot-communication', 'method',
     'People read speech together with gesture, gaze, posture, facial expression and timing, so the project treats robot movement as part of the message rather than decoration. The main scenario is a lab tour with social robots, visitors and question answering.'),
    ('topic:speech-segmentation', 'method',
     'A deterministic rule-based segmenter divides each reply into ordered communicative units, so different parts of one reply can get different gestures.'),
    ('topic:gesture-timing', 'method',
     'The Duration Filter keeps a gesture only if its measured duration fits within the estimated speaking time of its segment; missing, unprofiled or overlong gestures are rejected. The timing margin is currently zero seconds.'),
    ('topic:gesture-timing', 'method',
     'One-segment lookahead: while Navel speaks and gestures for the current segment, the server prepares the next segment''s gesture. It plans only one ahead because each choice depends on the one before it.'),
    ('topic:gesture-selection', 'method',
     'An expressive gesture is not repeated in the next segment. DEFAULT, a real generic speaking gesture, stays available and alternates between the left arm, right arm and both arms using a shuffled bag. DEFAULT is different from None, which means speech with no gesture.'),
    ('topic:gaze-coordination', 'method',
     'The Navel client keeps face perception active during speech, coordinates head movement with gestures, finds the visitor''s face again after a gesture, and can add small head movements while speaking.'),
    -- model
    ('topic:gesture-timing', 'model',
     'Speaking time per segment is estimated with a calibrated linear model: about 0.54 seconds plus 0.32 seconds per word.'),
    ('topic:gesture-selection', 'model',
     'Reply generation and gesture scoring use the qwen2.5:7b language model served through Ollama.'),
    ('topic:speech-segmentation', 'model',
     'An LLM segmenter using qwen2.5:7b was tested as a comparison, but the rule-based segmenter was selected for the running system.'),
    -- metric
    ('topic:co-speech-gesture', 'metric',
     'Main result so far: end-to-end feasibility. Generated speech has been segmented, timed, filtered, matched to gestures, planned one segment ahead and performed as coordinated speech and movement on the physical Navel robot.'),
    ('topic:speech-segmentation', 'metric',
     'On a frozen 30-sample held-out set, the rule-based segmenter scored 96.7% exact match and 0.987 boundary F1 with 0.03 ms median latency; the LLM segmenter scored 73.3% and 0.887 with about 221 ms latency.'),
    ('topic:gesture-timing', 'metric',
     'The duration estimator is off by 0.447 seconds on average for lab-tour speech (0.199 s for utterances of four words or fewer, 0.289 s on Harvard sentences). Its gesture-fits-or-not decisions were 94.7% correct: 142 of 150, with 2 false accepts and 6 false rejects.'),
    ('topic:co-speech-gesture', 'metric',
     'A recorded eight-segment run on the physical robot performed DEFAULT, DEFAULT, AGREE, POINT, UNCERTAIN, DEFAULT, DEFAULT, DEFAULT in order, with speech and gesture starting about 1.35 ms apart and all seven lookahead steps ready in time. This is one development run, not a general performance claim.'),
    ('topic:gesture-selection', 'metric',
     'In a diagnostic of 25 segments run 5 times each, DEFAULT was chosen 76% of the time and POINT every time it fitted; GREET, HAPPY, SAD, SURPRISE, LISTEN and EMPHASISE were never chosen, and "That is correct" varied between AGREE and DENY.'),
    -- dataset
    ('topic:gesture-timing', 'dataset',
     'Speech timing was calibrated by measuring how long Navel actually took to speak: 15 lab-tour sentences with three trials each (45 measurements, median about 2.56 words per second), validated on 10 Harvard sentences with three trials each (median about 2.77 words per second).'),
    -- hardware
    ('topic:co-speech-gesture', 'hardware',
     'Navel is the physical robot platform, and the only robot physically integrated with Method B so far. Its action library has eleven gestures: DEFAULT, POINT, EMPHASISE, AGREE, LISTEN, UNCERTAIN, DENY, GREET, HAPPY, SURPRISE and SAD, each with a measured duration from about 1 second (HAPPY, SAD) to about 6 seconds (EMPHASISE).'),
    -- limitation
    ('topic:robot-to-robot-communication', 'limitation',
     'No human study has been completed, so it has NOT been shown that people understand robot-to-robot communication better. The system is designed to improve comprehensibility; formal human evaluation is still pending.'),
    ('topic:co-speech-gesture', 'limitation',
     'The project does not generate new motor movements, full-body motion or facial-expression animation, and the language model does not generate movement. It selects and coordinates gestures from a fixed, tested library for each robot.'),
    ('topic:gesture-selection', 'limitation',
     'The gesture scorer has bias and consistency problems, and there are no human-labelled correct answers yet, so selection quality has not been measured. The next step is improving it with human-reviewed labels and comparing against the frozen V1 baseline.'),
    ('topic:gesture-timing', 'limitation',
     'The speaking-time estimate is based on word count and is not exact for every utterance. The first arm command can take about 3.2 seconds while the arm controller warms up; that happens once, not before every gesture.'),
    ('topic:co-speech-gesture', 'limitation',
     'Navel''s action and timing profile is still a static placeholder rather than being loaded from the database, and another robot would need its own profile and tested motion mappings before it could use Method B.'),
    ('topic:gaze-coordination', 'limitation',
     'Gaze and head coordination has been implemented and exercised on the physical robot, but there is no formal gaze-accuracy or naturalness evaluation yet.')
) AS f(topic_id, kind, fact);

-- Not from the document: true of THIS demo, and exactly what a visitor
-- watching Navel's gestures here would ask about.
INSERT INTO demo_topic_facts (topic_id, robot_id, kind, fact, verified, source)
SELECT 'topic:co-speech-gesture', r.robot_id, 'limitation',
       'Method B runs in its own navel-server codebase, not in this lab demo server, so the gestures in this demo come from simple per-reply emotion tags rather than Method B''s per-segment planning.',
       true, 'this repository: Method B lives in navel-server'
FROM   (VALUES ('navel_01')) AS r(robot_id);

COMMIT;
