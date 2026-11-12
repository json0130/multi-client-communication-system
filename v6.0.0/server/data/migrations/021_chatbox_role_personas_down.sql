-- 021 down: remove the role-library personas (an edited one is removed too; a renamed one is kept).
BEGIN;

DELETE FROM public.personas
WHERE name IN ('companion', 'tutor', 'storyteller', 'reading buddy', 'receptionist');

COMMIT;
