-- 020_chatbox_navel_scope.sql
-- ===========================
-- Who answers what, by the visitor's own words.
--
-- ChatBox is the companion / home-assistant robot: retrieval-augmented generation and contextual,
-- long-term conversation, for personal experiences. Navel is contextual co-speech motion generation:
-- non-verbal interaction.
--
-- Why: a ChatBox-led pair run was asked "what is co-speech motion generation?" and ChatBox answered it
-- itself. resolve_topic() matches a question to a topic by the words of the topic's LABEL, and no label
-- held "motion" or "generation", so the question matched nothing and was never routed. The new topics
-- are phrased the way a visitor says it (and share no word with an existing label, because a shared word
-- turns a question that matched one topic into a tie, which routes nowhere: "contextual conversation"
-- did that to "conversational memory" and was dropped), are linked to the existing topics so the facts already held
-- for them still ground the answer, and are declared as specialised for one robot each. No facts are
-- invented here: a new topic is grounded through its links.
--
-- Data only. Written for the simulation twins (chatbox_01, navel_01); the physical robots get it
-- through a new sim-to-real transfer (tools/model_transfer.py), never by writing the rows twice.
-- Re-runnable, and reversible: see 020_chatbox_navel_scope_down.sql.

BEGIN;

INSERT INTO demo_topics (id, label, category, source) VALUES
    ('topic:companion-robot', 'companion robot', 'hri', 'chatbox-role'),
    ('topic:home-assistant-robot', 'home assistant robot', 'hri', 'chatbox-role'),
    ('topic:personal-experiences', 'personal experiences', 'hri', 'chatbox-role'),
    ('topic:co-speech-motion-generation', 'co-speech motion generation', 'hri', 'navel-role'),
    ('topic:gesture-generation', 'gesture generation', 'hri', 'navel-role')
ON CONFLICT (id) DO NOTHING;

-- topic_a < topic_b (demo_topic_links_sorted).
INSERT INTO demo_topic_links (topic_a, topic_b, weight, source) VALUES
    ('topic:companion-robot', 'topic:long-term-interaction', 0.70, 'chatbox-role'),
    ('topic:companion-robot', 'topic:conversational-memory', 0.70, 'chatbox-role'),
    ('topic:companion-robot', 'topic:retrieval-augmented-generation', 0.60, 'chatbox-role'),
    ('topic:companion-robot', 'topic:home-assistant-robot', 0.80, 'chatbox-role'),
    ('topic:conversational-memory', 'topic:home-assistant-robot', 0.60, 'chatbox-role'),
    ('topic:home-assistant-robot', 'topic:long-term-interaction', 0.60, 'chatbox-role'),
    ('topic:conversational-memory', 'topic:personal-experiences', 0.75, 'chatbox-role'),
    ('topic:long-term-interaction', 'topic:personal-experiences', 0.75, 'chatbox-role'),
    ('topic:personal-experiences', 'topic:retrieval-augmented-generation', 0.60, 'chatbox-role'),
    ('topic:co-speech-gesture', 'topic:co-speech-motion-generation', 0.90, 'navel-role'),
    ('topic:co-speech-motion-generation', 'topic:gesture-selection', 0.70, 'navel-role'),
    ('topic:co-speech-motion-generation', 'topic:non-verbal-interaction', 0.80, 'navel-role'),
    ('topic:co-speech-motion-generation', 'topic:gesture-timing', 0.65, 'navel-role'),
    ('topic:co-speech-gesture', 'topic:gesture-generation', 0.85, 'navel-role'),
    ('topic:gesture-generation', 'topic:gesture-selection', 0.75, 'navel-role'),
    ('topic:co-speech-motion-generation', 'topic:gesture-generation', 0.90, 'navel-role')
ON CONFLICT (topic_a, topic_b) DO NOTHING;

-- Declared scope: one robot owns each new subject.
INSERT INTO demo_robot_topic (robot_id, topic_id, weight, specialised) VALUES
    ('chatbox_01', 'topic:companion-robot', 0.5, true),
    ('chatbox_01', 'topic:home-assistant-robot', 0.5, true),
    ('chatbox_01', 'topic:personal-experiences', 0.5, true),
    ('navel_01', 'topic:co-speech-motion-generation', 0.5, true),
    ('navel_01', 'topic:gesture-generation', 0.5, true)
ON CONFLICT (robot_id, topic_id) DO UPDATE SET specialised = true;

COMMIT;
