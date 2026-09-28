-- 017_sim_to_real_transfer_down.sql — reverses 017_sim_to_real_transfer.sql.
--
-- Also removes what transfers copied: facts carrying a transfer_id, and KG /
-- style rows that exist only because of a transfer (no live evidence of their
-- own). Rows with live counts keep their live weight; only the prior goes.

BEGIN;

DELETE FROM public.demo_topic_facts WHERE transfer_id IS NOT NULL;
DELETE FROM public.demo_robot_topic
 WHERE transfer_id IS NOT NULL
   AND COALESCE(n_supervisor, 0) + COALESCE(n_outcome, 0) + COALESCE(n_displaced, 0) = 0;
DELETE FROM public.demo_robot_style
 WHERE transfer_id IS NOT NULL AND COALESCE(n_supervisor, 0) = 0;

DROP VIEW  IF EXISTS public.current_duration_priors;
DROP TABLE IF EXISTS public.duration_priors;

DROP INDEX IF EXISTS public.idx_robot_style_transfer;
DROP INDEX IF EXISTS public.idx_robot_topic_transfer;
DROP INDEX IF EXISTS public.idx_topic_facts_transfer;

ALTER TABLE public.demo_topic_facts DROP COLUMN IF EXISTS transfer_id;
ALTER TABLE public.demo_robot_style DROP COLUMN IF EXISTS transfer_id,
                                    DROP COLUMN IF EXISTS prior_n,
                                    DROP COLUMN IF EXISTS prior_weight;
ALTER TABLE public.demo_robot_topic DROP COLUMN IF EXISTS transfer_id,
                                    DROP COLUMN IF EXISTS prior_n,
                                    DROP COLUMN IF EXISTS prior_weight;

DROP TABLE IF EXISTS public.model_transfers;

COMMIT;
