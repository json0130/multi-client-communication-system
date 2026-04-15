-- 001_projects.sql
-- Run in Supabase SQL editor (or psql) to create the projects tables.
--
-- projects          : one row per research project
-- robot_project_access : RDAC junction table — which robots may read which projects

-- ── Projects ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS projects (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT        NOT NULL,
    description TEXT,
    researcher  TEXT,
    robot_id    TEXT,           -- primary assigned robot (client_id)
    keywords    TEXT[],         -- keyword list used for RAG retrieval
    details     TEXT,           -- rich text injected into system prompt / RAG index
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── RDAC: robot → project access ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS robot_project_access (
    robot_id    TEXT NOT NULL,
    project_id  UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    PRIMARY KEY (robot_id, project_id)
);

-- ── Access notes ──────────────────────────────────────────────────────────────
-- Pepper (pepper_01) = full access → grant a row for every project
-- Each specialist robot = one row for its assigned project only
--
-- Example seed (edit robot_ids to match your client_config.json):
--
-- INSERT INTO projects (name, description, researcher, robot_id, keywords, details)
-- VALUES
--   ('Conversational AI',     'RAG-based long-context conversations',  'Dr. Smith',  'chatbox_01', ARRAY['RAG','NLP','conversation'], 'ChatBox researches...'),
--   ('Emotion-Aware Interaction', 'Facial expression + tone detection', 'Dr. Lee',    'navel_01',   ARRAY['emotion','affect','HRI'],   'Navel researches...'),
--   ('Human-Aware Navigation','Socially-compliant robot navigation',   'Dr. Kim',    'silbot_01',  ARRAY['navigation','SLAM','social'],'Silbot researches...');
--
-- -- Grant Pepper full access
-- INSERT INTO robot_project_access (robot_id, project_id)
-- SELECT 'pepper_01', id FROM projects;
--
-- -- Grant each robot its own project
-- INSERT INTO robot_project_access (robot_id, project_id)
-- SELECT robot_id, id FROM projects;
