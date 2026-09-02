# PAD prompt grid

_2026-08-14T02:55:03+00:00_  
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
-• With this person, stay quiet and responsive — answer what they ask, ask a little back, and do not propose topics of your own.
+• Your manner right now is even, calm, even-handed. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
+• With this person, mostly follow their lead — answer directly, and suggest something only now and then.
-━━━ WHO YOU'RE TALKING TO: p_chatbox_unknown ━━━
-You are meeting this person for the first time; be friendly and open.
+━━━ WHO YOU'RE TALKING TO: p_chatbox_close ━━━
+You know this person well and feel very comfortable with them.
```

Prompt length differs by **29 characters** (of 2067).

### Replies

| tier | neutral |
|---|---|
| **unknown** | `[]` Hi there! How about we chat about space? I know a bit about it and it's something you seem really into. What’s your favorite planet or celestial body? [CURIOUS]<br>`[]` Space could be fun! We could discuss the solar system or maybe even dream about future space travel. [CURIOUS] |
| **known** | `[]` How about space exploration? We could discuss the latest missions or maybe even dream up our own imaginary spacecraft. [CURIOUS]<br>`[]` Space, of course! I could go on for hours about the cosmos. [CURIOUS] Have you ever thought about which planet you'd most want to visit and why? |
| **close** | `[CURIOUS]` How about talking about your favorite constellations in the sky? You know, there are so many fascinating ones to discuss!<br>`[]` Hey there! How about we chat about space exploration? I know you're into it, and there's so much to explore! [CURIOUS] |

### Lexical metrics (mean over repeats)

| tier | emotion | words | TTR | hedges | warmth | feeling-talk |
|---|---|---|---|---|---|---|
| unknown | neutral | 22.0 ±5.7 | 0.943 | 0.3 | 0.3 | 1/3 |
| known | neutral | 21.7 ±3.9 | 0.969 | 0.3 | 0.0 | 1/3 |
| close | neutral | 19.3 ±1.7 | 0.948 | 0.3 | 0.0 | 0/3 |

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
-• With this person, mostly follow their lead — answer directly, and suggest something only now and then.
+• Your manner right now is warm, lively, commanding. That colours your WORDING only — it never changes what you know, what you answer, or whether you bring up how they feel.
+• With this person, lead the exchange — say what you think plainly and suggest what to do next.
-━━━ WHO YOU'RE TALKING TO: p_ellebot_unknown ━━━
-You are meeting this person for the first time; be friendly and open.
+━━━ WHO YOU'RE TALKING TO: p_ellebot_close ━━━
+You know this person well and feel very comfortable with them.
+  – They clearly like guitar.
-  – They clearly like guitar.
```

Prompt length differs by **19 characters** (of 1917).

### Replies

| tier | neutral |
|---|---|
| **unknown** | `[CURIOUS]` How about we chat about your love for space and guitars? Maybe you could tell me about a favorite celestial body or musician!<br>`[EXCITED]` How about we talk about your love for guitars and space! Maybe you could tell me about a favorite guitarist or an astronaut who inspires you? |
| **known** | `[]` Let's chat about your favorite guitarist and maybe add a cosmic twist by discussing their influence on modern music! [CURIOUS]<br>`[]` Let's chat about your favorite guitar songs or bands! You know, I remember you mentioned how much you enjoy space-themed music. Maybe we can find some cosmic tunes together? [CURIOUS] |
| **close** | `[]` Let's dive into your favorite topic: guitars! Or maybe explore the wonders of space a bit more? What do you think? [CURIOUS]<br>`[]` Let's talk about your favorite guitars! You've got some amazing ones, right? |

### Lexical metrics (mean over repeats)

| tier | emotion | words | TTR | hedges | warmth | feeling-talk |
|---|---|---|---|---|---|---|
| unknown | neutral | 23.0 ±2.4 | 0.916 | 1.0 | 0.7 | 0/3 |
| known | neutral | 22.3 ±6.3 | 0.958 | 1.0 | 0.0 | 0/3 |
| close | neutral | 15.7 ±4.5 | 1.000 | 0.7 | 0.0 | 0/3 |
