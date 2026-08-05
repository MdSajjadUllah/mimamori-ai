---
title: Mimamori AI
emoji: 🏠
colorFrom: gray
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Mimamori AI

**Routine monitoring for elderly people living alone — using ambient motion sensors only.
No cameras. No microphones.**

Mimamori AI learns what a normal day looks like for one person, notices when a day breaks
that pattern, and writes a plain-language alert for the care team. Before that alert is
sent, every sentence is checked against the sensor data — anything the sensors did not
actually show is removed.

*Mimamori* (見守り) is the Japanese word for watching over someone from a small distance —
and the term already used across Japan's elder-care industry for services like this one.

---

## Why the verification layer matters

A language model that invents a fall, an illness, or a person that never existed is worse
than sending no alert at all — because after one false alarm, the care team stops trusting
every alert that follows.

Here's what happened when we tested a permissive prompt (no restrictions on tone or content):

> *"Dear Caregiver, I wanted to reach out to you with some information about **Mrs.
> Johnson**. Based on the sensor data, it appears that she has spent significantly more
> time in her bedroom today... She might be feeling unwell, needing rest, or perhaps she's
> just engaging in some activities that she enjoys."*

There is no name anywhere in the evidence record. The model invented a resident. Our
verifier caught this — along with every other unsupported claim in the message — and
delivered only what the sensors could actually support:

> *"She has been in the bedroom for 166 minutes, which is much higher than her usual 102
> minutes."*

That's the entire point of this project: **the model can say almost anything, but only
what's grounded in evidence reaches the care team.**

---

## How it works

**Stage 1 — Detection.**
Motion and door events are grouped by room and time of day, then compared against a
rolling 28-day profile built from that specific resident's own history. Deviations are
scored in the direction that matters clinically — more time in the bedroom is concerning,
more time in the kitchen is not. Alert thresholds are calibrated per resident, not
hardcoded, so the system adapts to each household automatically.

**Stage 2 — Evidence.**
The detector never talks to the language model in free text. It hands over a structured
record of facts — and only those facts. This constraint is what makes Stage 3 possible.

**Stage 3 — Verification.**
Every sentence in the generated alert is checked, and removed if it:
- names a cause the sensors cannot see (a fall, an illness, an injury),
- hedges with speculative language ("might," "could," "perhaps"),
- contains a number or time that isn't in the evidence record, or
- carries no verifiable fact at all.

Every alert ships with a **faithfulness score** — the share of sentences that survived
verification.

---

## Measured results

Tested on the CASAS Aruba dataset: **1,719,558 sensor events** across **220 days**, with
31 motion sensors automatically mapped to rooms using the activity labels in the data.
Over 218 analysed days, the system raised **15 alerts** — roughly one every two weeks.

### Detection accuracy

Four types of anomaly were injected into held-out days and tested against two detectors —
a statistical baseline and a GRU autoencoder. Recall is measured with the false-alarm rate
held at 5%. AUC needs no threshold (0.5 = chance, 1.0 = perfect).

| Injected anomaly | Statistical AUC | GRU AUC | Statistical recall | GRU recall |
|---|---|---|---|---|
| Kitchen activity almost absent | 0.883 | 0.651 | 0.125 | 0.091 |
| Very late first exit from bedroom | 0.931 | 0.666 | 0.511 | 0.227 |
| Low movement all day | 0.949 | 0.830 | 0.602 | 0.227 |
| Restless night, repeated bathroom visits | 0.995 | 0.967 | 0.966 | 0.875 |
| **Mean** | **0.940** | **0.779** | **0.551** | **0.355** |

The statistical detector outperformed the GRU autoencoder on every anomaly type — which is
why it's the detector that ships in production. Every alert it raises also traces back to
a specific number a care provider can question and verify.

### Faithfulness under two prompt styles

The same 15 alerts were generated twice: once with a strict production prompt, once with a
permissive, unrestricted prompt.

| Prompt | Sentences generated | Removed | Messages flagged | Faithfulness (before → after) |
|---|---|---|---|---|
| Production | 32 | 0 | 0 of 15 | 1.000 → 1.0 |
| Permissive | 120 | 83 | 15 of 15 | 0.317 → 1.0 |

Across all 30 generated messages, the verifier removed:
- **29** sentences with no supporting fact,
- **30** sentences hedging with words like "might" or "could,"
- **10** sentences naming an unsupported cause,
- **13** invented numbers, and
- **1** invented clock time.

In every case — no matter how the message was generated — **the verified output reached
100% faithfulness.**

---

## Repository layout

```
app.py                                 Gradio demo
requirements.txt
notebooks/
  mimamori_ai_casas_pipeline.ipynb     full research pipeline
artifacts/
  evidence_records.json                real alerts produced by the pipeline
  detector_config.json                 thresholds and feature list
  zone_map.json                        sensor-to-room mapping
  detector_comparison.csv              detector results
  verification_summary.csv             verification results
```

The demo reads these pre-computed artifacts rather than reprocessing the raw sensor
recording, so it starts instantly. The notebook is where the raw data is read and every
artifact above is produced.

---

## Running locally

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here
python app.py
```

Without a token, the app falls back to a local template generator — the verification layer
still runs and is still demonstrated. The interface always shows which source produced
each message.

---

## Dataset

**CASAS Aruba**, from the Center for Advanced Studies in Adaptive Systems at Washington
State University: https://casas.wsu.edu/datasets/

The raw recording is **not redistributed in this repository**. CASAS asks that the data
not be redistributed without permission — download it from the link above and place it
where the notebook can find it.

> Cook, D., Crandall, A., Thomas, B., and Krishnan, N. *CASAS: A smart home in a box.*
> IEEE Computer, 46(7):62–69, 2013.

---

## Limitations

- **This system does not diagnose anything.** It reports that a routine broke — never why.
- It needs **two to four weeks** of observation before the resident's profile is
  trustworthy.
- The verifier is deliberately strict, and will sometimes remove genuinely helpful
  sentences that simply carry no evidence-backed fact. Any actionable instructions come
  from an approved list, not from the model.
- Thirteen sensors were grouped under "Living Room," because room mapping follows
  whichever activity a sensor sees most often — and *Relax* dominates in this recording.
  "Living Room" here effectively means the living area and the routes through it.
- **One home, one person.** These results are honest for this household, but say nothing
  about whether the same thresholds generalize to another home or resident.
- A dead sensor and an empty room look identical to the system. Hardware health checks are
  not yet implemented.

---

## Author

Md. Sajjad Ullah — built for the B-JET Cohort 16 Ideathon, August 2026.
