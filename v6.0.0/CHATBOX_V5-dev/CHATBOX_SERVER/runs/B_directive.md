# PAD prompt grid

_2026-08-14T02:53:05+00:00_  
message: `hey, what have you been up to?`  
model: `qwen2.5:7b`  temperature: 0.0  repeats: 2


## chatbox

### Descriptor triplets — the necessary condition

If two cells share a triplet their prompts are identical apart from the tier sentence, so any reply difference between them is noise.

| tier | neutral |
|---|---|
| **unknown** | even/calm/reserved |
| **known** | even/calm/reserved |
| **close** | even/calm/even-handed |

**2 distinct triplet(s)** across 3 cells.

Cells that CANNOT differ except by noise:

- `even/calm/reserved` — unknown/neutral, known/neutral

### Style / servo ladder

| tier | amplitude | tempo | posture | droop | wire |
|---|---|---|---|---|---|
| unknown | 0.300 | 0.698 | -0.668 | -0.149 | `STYLE 0.30 0.70 -0.67 -0.15 0.45` |
| known | 0.425 | 0.759 | -0.418 | -0.149 | `STYLE 0.42 0.76 -0.42 -0.15 0.45` |
| close | 0.565 | 0.865 | -0.138 | -0.149 | `STYLE 0.56 0.87 -0.14 -0.15 0.49` |

### What PAD actually contributed to the prompt

`unknown` vs `close` at emotion `neutral`:

```diff
-• Your manner right now is even, calm, reserved. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
-• With this person, stay quiet and responsive — answer what they ask, ask a little back, and do not propose topics of your own.
+• Your manner right now is even, calm, even-handed. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
+• With this person, mostly follow their lead — answer directly, and suggest something only now and then.
-━━━ WHO YOU'RE TALKING TO: p_chatbox_unknown ━━━
-You are meeting this person for the first time; be friendly and open.
+━━━ WHO YOU'RE TALKING TO: p_chatbox_close ━━━
+You know this person well and feel very comfortable with them.
```

Prompt length differs by **29 characters** (of 2067).
