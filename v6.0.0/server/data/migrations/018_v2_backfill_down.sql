-- 018_v2_backfill_down.sql — clears what 018_v2_backfill.sql filled.
--
-- Only safe before phase-2 code has written its own rows: this cannot tell a
-- backfilled demo_runs/sessions row or label from one the server wrote, so it
-- clears every legacy-shaped one. Nothing here touches the log rows themselves.

BEGIN;

UPDATE public.demo_correction_log SET supervisor_id = NULL
 WHERE supervisor_id = 'legacy_unattributed';
UPDATE public.demo_correction_log SET origin = NULL
 WHERE origin IN ('synthetic', 'test', 'scripted');
UPDATE public.demo_correction_log SET run_id = NULL, run_id_source = NULL;
UPDATE public.demo_decision_log   SET run_id = NULL, run_id_source = NULL;

UPDATE public.chat_logs SET session_id = NULL WHERE session_id LIKE 'legacy-user-%';
UPDATE public.demo_correction_log SET decision_point = 'qa_advance'
 WHERE step_id = 'wait_for_handoff' AND decision_point = 'solo_cue';

DELETE FROM public.sessions;
DELETE FROM public.demo_runs;
DELETE FROM public.scenarios;

COMMIT;
