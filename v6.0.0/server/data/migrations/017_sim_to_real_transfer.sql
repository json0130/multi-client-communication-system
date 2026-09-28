-- 017_sim_to_real_transfer.sql — versioned sim-to-real transfer (schema only).
--
-- A physical robot (e.g. navel_001) starts with no evidence of its own. Its
-- simulation twin (navel_01) has facts, declared scope, learned competence and
-- timings. A TRANSFER copies them across deliberately, as a numbered version
-- (#1, #2, ... per pair), so sim can keep learning and be redeployed on purpose:
--   * facts and declared scope (eligible / specialised): copied as-is;
--   * KG and style-fit weights: copied as a PRIOR (prior_weight, prior_n), with
--     the physical robot's live counts left at 0 — the router uses the prior
--     until live evidence reaches N, and sim evidence never counts for more
--     than 3 observations;
--   * durations: summarised into duration_priors (median, n, source modes);
--     log rows (decisions, corrections, durations, chat) are NEVER copied.
-- The transfer itself is tools/model_transfer.py (it needs the censored Q&A
-- median in data/demo_duration_repo.py), run with --dry-run first; each one is
-- also recorded as a file in the repo (data/transfers/).
--
-- Additive: new tables and nullable columns only. Reversible:
-- 017_sim_to_real_transfer_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS public.model_transfers (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version         integer NOT NULL,                 -- per (from, to): 1, 2, 3, ...
    from_robot_id   text NOT NULL,
    to_robot_id     text NOT NULL,
    transferred_at  timestamptz NOT NULL DEFAULT now(),
    items           jsonb NOT NULL DEFAULT '{}'::jsonb, -- what was copied, with counts
    notes           text,
    repo_record     text,                             -- data/transfers/<file> for this transfer
    UNIQUE (from_robot_id, to_robot_id, version),
    CHECK (from_robot_id <> to_robot_id)
);

-- KG competence and style fit: the prior next to the live estimate.
ALTER TABLE public.demo_robot_topic
    ADD COLUMN IF NOT EXISTS prior_weight real,
    ADD COLUMN IF NOT EXISTS prior_n      integer,
    ADD COLUMN IF NOT EXISTS transfer_id  bigint
        REFERENCES public.model_transfers (id) ON DELETE SET NULL;
ALTER TABLE public.demo_robot_style
    ADD COLUMN IF NOT EXISTS prior_weight real,
    ADD COLUMN IF NOT EXISTS prior_n      integer,
    ADD COLUMN IF NOT EXISTS transfer_id  bigint
        REFERENCES public.model_transfers (id) ON DELETE SET NULL;

-- Facts copied by a transfer carry its id, so the next version replaces them
-- without touching facts written for the physical robot directly.
ALTER TABLE public.demo_topic_facts
    ADD COLUMN IF NOT EXISTS transfer_id bigint
        REFERENCES public.model_transfers (id) ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS public.duration_priors (
    transfer_id   bigint NOT NULL REFERENCES public.model_transfers (id) ON DELETE CASCADE,
    robot_id      text   NOT NULL,        -- the physical robot the prior is for
    step_id       text   NOT NULL,        -- already renamed to the physical robot's step id
    kind          text   NOT NULL CHECK (kind IN ('step', 'qa')),
    median_sec    real   NOT NULL,
    n             integer NOT NULL,
    source_mode   text   NOT NULL,        -- e.g. 'gazebo+mock'; never 'live'
    PRIMARY KEY (transfer_id, robot_id, step_id, kind)
);

-- The prior the planner uses: the newest transfer's, per robot and step.
CREATE OR REPLACE VIEW public.current_duration_priors
WITH (security_invoker = true) AS
SELECT DISTINCT ON (p.robot_id, p.step_id, p.kind)
       p.robot_id, p.step_id, p.kind, p.median_sec, p.n, p.source_mode,
       p.transfer_id, t.version, t.from_robot_id
FROM public.duration_priors p
JOIN public.model_transfers t ON t.id = p.transfer_id
ORDER BY p.robot_id, p.step_id, p.kind, t.transferred_at DESC, t.id DESC;

CREATE INDEX IF NOT EXISTS idx_topic_facts_transfer  ON public.demo_topic_facts (transfer_id);
CREATE INDEX IF NOT EXISTS idx_robot_topic_transfer  ON public.demo_robot_topic (transfer_id);
CREATE INDEX IF NOT EXISTS idx_robot_style_transfer  ON public.demo_robot_style (transfer_id);

ALTER TABLE public.model_transfers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.duration_priors ENABLE ROW LEVEL SECURITY;

COMMIT;
