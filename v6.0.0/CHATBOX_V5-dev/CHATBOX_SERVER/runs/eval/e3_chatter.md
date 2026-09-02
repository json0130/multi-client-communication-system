# E3 — directive chattering under permuted source-to-axis assignment

_generated 2026-09-01T00:15:09+00:00 at 2be86bb_

Reproduce: `python3 -m pytest padeval/tests/test_e3_chatter.py` and
`python3 -c "import sys;sys.path.insert(0,'.');from padeval.analysis.e3_permute import *;print(format_table(sweep()))"`

The commanded behavioural directive is a per-SESSION control. Under the deployed
assignment the face cannot write Dominance, so the setpoint is constant for the
whole session at **any** camera noise level — zero switches by construction, not
by tuning. Permute emotion and relationship and a 30 Hz signal drives a
per-session effector.

`sigma` is additive Gaussian frame noise on (valence, arousal). The pooled
within-class dispersion of the deployed regression head over 320 real faces
(AffectNet-HQ + RAF-DB) is **sd_v = 0.279, sd_a = 0.292**, which is an upper
bound on frame-to-frame noise since it also mixes identity, pose and lighting.
The sweep therefore spans the empirically plausible range.

Trace: 8 expression blocks x 4 s at 30 fps = 32 s, schedule
neutral/happy/neutral/sad/neutral/happy/neutral/surprise, corner V/A from
`affect.CATEGORY_VA`, smoothed by the deployed 5-sample `AffectStream`.

### frame noise sigma = 0.00

| assignment | robot | tier | switches/min | distinct rungs | median dwell (s) | min dwell (s) | clamp rate (D) |
|---|---|---|---|---|---|---|---|
| `identity` | CHATBOX | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | CHATBOX | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `perm_emotion_D` | CHATBOX | known | **11.2** | 3 | 4.03 | 3.933 | 0.00% |
| `perm_emotion_D` | CHATBOX | close | **11.2** | 3 | 4.03 | 3.933 | 0.00% |
| `perm_emotion_D` | ELLEBOT | known | **18.8** | 4 | 3.87 | 0.100 | 0.00% |
| `perm_emotion_D` | ELLEBOT | close | **18.8** | 4 | 3.87 | 0.100 | 0.00% |
| `perm_arousal_D` | CHATBOX | known | **5.6** | 3 | 8.00 | 3.867 | 0.00% |
| `perm_arousal_D` | CHATBOX | close | **11.2** | 3 | 3.87 | 0.067 | 0.00% |
| `perm_arousal_D` | ELLEBOT | known | **15.0** | 4 | 3.97 | 0.100 | 0.00% |
| `perm_arousal_D` | ELLEBOT | close | **15.0** | 4 | 3.97 | 0.100 | 0.00% |
| `collapse_D` | CHATBOX | known | **11.2** | 3 | 4.07 | 3.867 | 0.00% |
| `collapse_D` | CHATBOX | close | **20.6** | 4 | 3.88 | 0.133 | 0.00% |
| `collapse_D` | ELLEBOT | known | **13.1** | 3 | 4.00 | 3.900 | 0.00% |
| `collapse_D` | ELLEBOT | close | **3.8** | 2 | 12.00 | 4.133 | 24.17% |

### frame noise sigma = 0.05

| assignment | robot | tier | switches/min | distinct rungs | median dwell (s) | min dwell (s) | clamp rate (D) |
|---|---|---|---|---|---|---|---|
| `identity` | CHATBOX | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | CHATBOX | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `perm_emotion_D` | CHATBOX | known | **11.2** | 3 | 4.03 | 3.900 | 0.00% |
| `perm_emotion_D` | CHATBOX | close | **11.2** | 3 | 4.03 | 3.900 | 0.00% |
| `perm_emotion_D` | ELLEBOT | known | **30.0** | 4 | 0.63 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | close | **30.0** | 4 | 0.63 | 0.033 | 0.00% |
| `perm_arousal_D` | CHATBOX | known | **5.6** | 3 | 8.00 | 3.867 | 0.00% |
| `perm_arousal_D` | CHATBOX | close | **69.4** | 3 | 0.12 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | known | **28.1** | 4 | 1.52 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | close | **15.0** | 4 | 3.97 | 0.100 | 0.00% |
| `collapse_D` | CHATBOX | known | **30.0** | 3 | 0.80 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | close | **39.4** | 4 | 0.30 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | known | **13.1** | 3 | 3.98 | 3.900 | 0.00% |
| `collapse_D` | ELLEBOT | close | **3.8** | 2 | 12.00 | 4.133 | 24.27% |

### frame noise sigma = 0.10

| assignment | robot | tier | switches/min | distinct rungs | median dwell (s) | min dwell (s) | clamp rate (D) |
|---|---|---|---|---|---|---|---|
| `identity` | CHATBOX | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | CHATBOX | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `perm_emotion_D` | CHATBOX | known | **18.8** | 4 | 3.90 | 0.033 | 0.00% |
| `perm_emotion_D` | CHATBOX | close | **18.8** | 4 | 3.90 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | known | **101.2** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | close | **101.2** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | CHATBOX | known | **9.4** | 3 | 3.57 | 0.067 | 0.00% |
| `perm_arousal_D` | CHATBOX | close | **106.9** | 3 | 0.12 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | known | **69.4** | 4 | 0.20 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | close | **60.0** | 4 | 0.47 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | known | **71.2** | 3 | 0.10 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | close | **80.6** | 4 | 0.10 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | known | **16.9** | 3 | 3.95 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | close | **30.0** | 3 | 0.27 | 0.033 | 24.27% |

### frame noise sigma = 0.20

| assignment | robot | tier | switches/min | distinct rungs | median dwell (s) | min dwell (s) | clamp rate (D) |
|---|---|---|---|---|---|---|---|
| `identity` | CHATBOX | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | CHATBOX | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `perm_emotion_D` | CHATBOX | known | **60.0** | 4 | 0.17 | 0.033 | 0.00% |
| `perm_emotion_D` | CHATBOX | close | **60.0** | 4 | 0.17 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | known | **240.0** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | close | **240.0** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | CHATBOX | known | **52.5** | 3 | 0.20 | 0.033 | 0.00% |
| `perm_arousal_D` | CHATBOX | close | **170.6** | 4 | 0.13 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | known | **204.4** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | close | **178.1** | 4 | 0.17 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | known | **86.2** | 3 | 0.17 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | close | **118.1** | 4 | 0.17 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | known | **95.6** | 4 | 0.13 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | close | **116.2** | 3 | 0.13 | 0.033 | 24.58% |

### frame noise sigma = 0.28

| assignment | robot | tier | switches/min | distinct rungs | median dwell (s) | min dwell (s) | clamp rate (D) |
|---|---|---|---|---|---|---|---|
| `identity` | CHATBOX | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | CHATBOX | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | known | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `identity` | ELLEBOT | close | **0.0** | 1 | 32.00 | 32.000 | 0.00% |
| `perm_emotion_D` | CHATBOX | known | **78.8** | 4 | 0.17 | 0.033 | 0.00% |
| `perm_emotion_D` | CHATBOX | close | **78.8** | 4 | 0.17 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | known | **328.1** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_emotion_D` | ELLEBOT | close | **328.1** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | CHATBOX | known | **101.2** | 3 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | CHATBOX | close | **198.8** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | known | **260.6** | 4 | 0.10 | 0.033 | 0.00% |
| `perm_arousal_D` | ELLEBOT | close | **283.1** | 4 | 0.10 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | known | **93.8** | 3 | 0.17 | 0.033 | 0.00% |
| `collapse_D` | CHATBOX | close | **166.9** | 4 | 0.17 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | known | **155.6** | 4 | 0.10 | 0.033 | 0.00% |
| `collapse_D` | ELLEBOT | close | **172.5** | 3 | 0.13 | 0.033 | 23.44% |
