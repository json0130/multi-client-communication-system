-- 010_kg_declared_scope.sql
-- Run in Supabase SQL editor (or psql) after 009_kg_role_seed.sql.
--
-- Splits "whose subject is this" apart from "how well is it handled".
--
-- WHY
-- demo_robot_topic.weight was carrying both claims at once, and that is what
-- made seeding it impossible. tools/seed_kg.py states the problem exactly:
-- seeding competence from project assignments "would make propagation
-- re-derive a partition that was typed in by hand, and every generalisation
-- number becomes circular: the graph would be confirming its own input."
--
-- So the graph could not be told the one thing it obviously knows — that
-- Silbot's project is navigation — and started every deployment unable to
-- route at all. Measured before this migration: every topic resolved
-- correctly and then routed to chatbox_01, because all three robots sat at
-- the 0.5 prior and argmax fell through to the alphabetical tie-break.
--
-- Separating the two dissolves it. `specialised` is configuration, known
-- before any evidence exists and never learned. `weight` becomes what
-- supervision can actually tell you: how well the subject was handled. The
-- generalisation claim improves as a result — it stops being "the graph
-- re-derived who owns what" and becomes "a correction about one topic's
-- handling generalised to a neighbouring topic", which is not circular.
--
-- HOW IT ROUTES
-- decision/kg_infer.py::route narrows candidates to declared specialists
-- BEFORE ranking, alongside the eligibility and presence filters. It is a
-- PREFERENCE, not an exclusion: if every specialist is ineligible or away,
-- the field opens back up rather than stranding the question, and an absent
-- specialist still produces a defer via ABSENT_ROBOT_POLICY.
--
-- WHAT IS DELIBERATELY LEFT UNDECLARED
-- speech-recognition, text-to-speech and robot-hardware have no specialist.
-- No project owns them, and inventing an owner would be exactly the
-- hand-typed partition this migration is trying not to smuggle in. They stay
-- open to the learned weight and to exploration — which also leaves the
-- evaluation somewhere to measure learned routing that declared scope has
-- not already decided.
--
-- Opt-in and defaulting false, so a database that has not run this behaves
-- exactly as before.
--
-- Reversible: see 010_kg_declared_scope_down.sql

BEGIN;

ALTER TABLE demo_robot_topic
    ADD COLUMN IF NOT EXISTS specialised BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN demo_robot_topic.specialised IS
    'DECLARED scope: this topic is in the robot''s stated project area. '
    'Configuration, never learned — the deployment asserts it the way it '
    'asserts role and access_level. Narrows routing candidates before '
    'ranking (decision/kg_infer.py::route). Distinct from weight, which is '
    'the learned quality of handling; conflating them is what made seeding '
    'competence circular.';

-- ── Declared project areas ───────────────────────────────────────────────────
-- Derived from each robot's role in profiles/lab_demo.yaml and its
-- client_config.json robot_role. ON CONFLICT touches ONLY specialised, so a
-- weight already moved by real observations is never reset.

INSERT INTO demo_robot_topic (robot_id, topic_id, weight, specialised) VALUES
    -- ChatBox — conversational AI: RAG and long-context memory
    ('chatbox_01', 'topic:retrieval-augmented-generation', 0.5, true),
    ('chatbox_01', 'topic:large-language-models',          0.5, true),
    ('chatbox_01', 'topic:conversational-memory',          0.5, true),
    ('chatbox_01', 'topic:long-term-interaction',          0.5, true),
    ('chatbox_01', 'topic:knowledge-graphs',               0.5, true),

    -- Navel — emotion-aware interaction
    ('navel_01',   'topic:emotion-recognition',            0.5, true),
    ('navel_01',   'topic:facial-expression-analysis',     0.5, true),
    ('navel_01',   'topic:social-signals',                 0.5, true),
    ('navel_01',   'topic:non-verbal-interaction',         0.5, true),
    ('navel_01',   'topic:human-robot-trust',              0.5, true),

    -- Silbot — human-aware navigation
    ('silbot_01',  'topic:social-robot-navigation',        0.5, true),
    ('silbot_01',  'topic:mapping-and-localisation',       0.5, true),
    ('silbot_01',  'topic:multi-robot-coordination',       0.5, true),

    -- Pepper — the guide. Declared for completeness; it is excluded from the
    -- peer candidate list before routing ever runs, so this changes nothing
    -- today and exists so the profile is not silently incomplete.
    ('pepper_01',  'topic:human-pose-estimation',          0.5, true)
ON CONFLICT (robot_id, topic_id) DO UPDATE
    SET specialised = EXCLUDED.specialised;

-- ── View: expose it alongside eligible ───────────────────────────────────────
-- DROP then CREATE, same reason as 007 and 009: replace can only append
-- columns, and specialised belongs beside eligible rather than after
-- last_updated.

DROP VIEW IF EXISTS demo_kg_edges;

CREATE VIEW demo_kg_edges AS
SELECT rt.robot_id,
       rt.topic_id,
       t.label                                   AS topic_label,
       t.category                                AS topic_category,
       rt.weight,
       rt.eligible,
       rt.specialised,
       rt.n_supervisor,
       rt.n_outcome,
       rt.n_displaced,
       rt.n_supervisor + rt.n_outcome + rt.n_displaced        AS n_obs,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                 / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0)
       END                                       AS confidence,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.5
            ELSE 0.5 + (rt.weight - 0.5)
                 * ((rt.n_supervisor + rt.n_outcome + rt.n_displaced)::real
                    / ((rt.n_supervisor + rt.n_outcome + rt.n_displaced) + 3.0))
       END                                       AS clamped,
       CASE WHEN rt.n_supervisor + rt.n_outcome + rt.n_displaced = 0 THEN 0.0
            ELSE (rt.n_supervisor + rt.n_displaced)::real
                 / (rt.n_supervisor + rt.n_outcome + rt.n_displaced)
       END                                       AS human_share,
       rt.last_updated
FROM   demo_robot_topic rt
JOIN   demo_topics t ON t.id = rt.topic_id;

COMMIT;
