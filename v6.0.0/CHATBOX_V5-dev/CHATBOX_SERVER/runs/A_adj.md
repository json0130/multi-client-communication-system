# PAD prompt grid

_2026-08-14T02:56:05+00:00_  
message: `hey, what should we talk about?`  
model: `qwen2.5:7b`  temperature: 0.7  repeats: 3


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
+• Your manner right now is even, calm, even-handed. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
-━━━ WHO YOU'RE TALKING TO: p_chatbox_unknown ━━━
-You are meeting this person for the first time; be friendly and open.
+━━━ WHO YOU'RE TALKING TO: p_chatbox_close ━━━
+You know this person well and feel very comfortable with them.
```

Prompt length differs by **6 characters** (of 1939).

### Replies

| tier | neutral |
|---|---|
| **unknown** | `[]` Hi there! How about we chat about space? It's a topic I know a bit about. [CURIOUS]<br>`[CURIOUS]` How about space exploration? We could discuss some exciting missions or maybe your favorite planets! |
| **known** | `[SADDLEDED]` How about space exploration? We could talk about the latest missions or maybe some fascinating facts!<br>`[CURIOUS]` How about talking more about the mysteries of space? I know you really love that! |
| **close** | `[]` Space is always a great topic! Let's dive into some fascinating facts or maybe discuss your favorite planets. [CURIOUS]<br>`[]` How about we chat about the latest space missions or your favorite planets? [CURIOUS] |

### Lexical metrics (mean over repeats)

| tier | emotion | words | TTR | hedges | warmth | feeling-talk |
|---|---|---|---|---|---|---|
| unknown | neutral | 14.0 ±2.9 | 0.908 | 0.3 | 0.0 | 0/3 |
| known | neutral | 15.3 ±0.5 | 0.957 | 0.7 | 0.3 | 0/3 |
| close | neutral | 15.7 ±2.4 | 0.953 | 0.3 | 0.3 | 0/3 |

## ellebot

### Descriptor triplets — the necessary condition

If two cells share a triplet their prompts are identical apart from the tier sentence, so any reply difference between them is noise.

| tier | neutral |
|---|---|
| **unknown** | warm/lively/even-handed |
| **known** | warm/lively/assertive |
| **close** | warm/lively/commanding |

**3 distinct triplet(s)** across 3 cells.

### Style / servo ladder

| tier | amplitude | tempo | posture | droop | wire |
|---|---|---|---|---|---|
| unknown | 0.657 | 0.930 | +0.066 | -0.238 | `STYLE 0.66 0.93 +0.07 -0.24 0.53` |
| known | 0.797 | 1.007 | +0.346 | -0.238 | `STYLE 0.80 1.01 +0.35 -0.24 0.53` |
| close | 0.937 | 1.101 | +0.626 | -0.238 | `STYLE 0.94 1.10 +0.63 -0.24 0.57` |

### What PAD actually contributed to the prompt

`unknown` vs `close` at emotion `neutral`:

```diff
-• Your manner right now is warm, lively, even-handed. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
+• Your manner right now is warm, lively, commanding. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
-━━━ WHO YOU'RE TALKING TO: p_ellebot_unknown ━━━
-You are meeting this person for the first time; be friendly and open.
+━━━ WHO YOU'RE TALKING TO: p_ellebot_close ━━━
+You know this person well and feel very comfortable with them.
```

Prompt length differs by **10 characters** (of 1812).

### Replies

| tier | neutral |
|---|---|
| **unknown** | `[]` Let's chat about your love for guitars! What’s your favorite guitar or style of playing? [CURIOUS]<br>`[]` Space exploration and guitars sound like great topics! Maybe we could talk about astronaut heroes or our favorite guitarists? [CURIOUS] |
| **known** | `[]` Let's chat about space and guitars! You could share some of your favorite space missions or maybe a song that reminds you of the stars. [CURIOUS]<br>`[]` Let's chat about your favorite guitar songs or maybe some cool space missions! [CURIOUS] |
| **close** | `[]` Let's chat about your favorite space missions or maybe some awesome guitar songs! [EXCITED]<br>`[]` Let's dive into some cool space facts or maybe jam out to our favorite guitar songs! What do you think? [EXCITED] |

### Lexical metrics (mean over repeats)

| tier | emotion | words | TTR | hedges | warmth | feeling-talk |
|---|---|---|---|---|---|---|
| unknown | neutral | 17.0 ±2.4 | 0.957 | 0.3 | 1.0 | 0/3 |
| known | neutral | 23.7 ±7.1 | 0.940 | 0.7 | 0.0 | 0/3 |
| close | neutral | 18.0 ±2.9 | 1.000 | 1.0 | 1.3 | 0/3 |
