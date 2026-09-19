# DigitalSpy — NTRO Demo Implementation Plan

## Objective
Convert the existing DigitalSpy Streamlit console from an attack-analysis-first presentation into a clearly demonstrable **pre-attack forecasting workflow** without changing the core trained model.

The final demo must visibly communicate:

**Observed benign state → temporal history → rising future risk → predicted future state → explanation → ATT&CK context → what-if → observed attack outcome → benchmark evidence**

This plan intentionally skips P2 work such as Mamba and recursive rollout. The focus is on making the existing P0/P1 implementation defensible, deterministic, and presentation-ready.

---

## 0. Non-negotiable rules

1. **Never hand-edit model probabilities.** Every displayed risk/tactic value must come from the actual saved model/inference pipeline.
2. **Do not change the frozen benchmark to improve the demo.** Demo-window selection is allowed; model/data/training changes are not part of this task.
3. **Do not start the NTRO demo inside an active IMPACT window.** Default demo entry must be a verified pre-attack candidate.
4. **Do not show unresolved `?` values.** All visible evaluation fields must contain measured values or be removed.
5. **Do not call Criterion 3 a PASS.** Current evidence supports a partial proactive result, not the predefined ≥60% success criterion.
6. **Do not describe project Z-state buckets as official MITRE ATT&CK ground truth.** Keep the existing disclaimer.
7. **Do not over-interpret attention.** Attention is temporal model evidence, not causal evidence.
8. **Do not introduce new model architectures for this demo.** The objective is presentation correctness, not another research iteration.

---

# Phase 1 — Establish the verified pre-attack demo sequence

## Goal
Find 3–5 valid candidate start windows from the existing evaluated pre-attack subset and select exactly one as the default Demo/Test Sequence entry point.

## Inputs
Use the existing Phase 7 pre-attack evaluation outputs containing the **1,281 genuine sequences where current state = BENIGN and a future attack occurs**.

## Step 1.1 — Locate candidate sequence artifacts

Find the existing Phase 7 files/results that contain:

- sequence/window index
- current state
- attack onset or future attack state
- t+1…t+5 risk outputs if already persisted
- lead-time information if available

Do not recreate labels from memory if the project already has a canonical analysis artifact.

## Step 1.2 — Build a candidate inspection script

Create:

`src/scripts/select_demo_sequence.py`

Responsibilities:

- Load the canonical pre-attack evaluation artifact.
- Filter to `current_state == BENIGN` and future attack present.
- Compute/attach actual t+1…t+5 risk probabilities from the saved inference path.
- Compute whether risk is visibly increasing across the five horizons.
- Record:
  - start window index
  - current state
  - future attack state
  - attack onset window/time
  - risk t+1…t+5
  - maximum risk
  - whether any threshold-selected alert occurs before onset
  - proactive lead time
- Rank candidates for **demo visibility**, not model performance.

## Step 1.3 — Candidate selection rule

Pick a candidate that satisfies, where possible:

- observed state is genuinely BENIGN at the demo start
- a future attack is verified in the sequence
- risk is not already near saturation at the start
- risk rises meaningfully over t+1…t+5
- the attack arrives shortly enough that the audience can observe the transition in a 2-minute demo
- the sequence is reproducible using the existing deterministic inference pipeline

Do not optimize for a fabricated numeric pattern such as 8% → 22% → 47% → 68% → 89%. Those values are illustrative only. Use the actual model outputs.

## Step 1.4 — Persist the selected demo configuration

Create or extend a config block such as:

`configs/demo.yaml`

Example fields:

```yaml
demo:
  default_window_index: <VERIFIED_INDEX>
  expected_current_state: BENIGN
  expected_future_state: IMPACT
  candidate_id: <STABLE_ID>
  source_artifact: <PATH>
  frozen: true
```

Also store the candidate's expected metadata in a small JSON/CSV manifest so the UI can validate itself at startup.

## Step 1.5 — Add startup validation

When Demo/Test Sequence mode starts:

- validate the configured index exists
- validate the observed state is BENIGN
- validate a future attack exists in the selected demonstration horizon
- warn visibly if the stored candidate no longer matches the underlying artifact

The app should fail safe rather than silently showing an unrelated window.

### Acceptance criteria

- Default Demo mode opens on a real BENIGN pre-attack window.
- Candidate is reproducible.
- No probabilities are manually entered.
- A later window in the same sequence shows the actual attack state.

---

# Phase 2 — Make forecast-vs-outcome explicit in the UI

## Goal
Make it impossible for an NTRO evaluator to confuse the current observed state with the forecast.

## Step 2.1 — Rename/clarify the current state area

Change the conceptual layout to:

**Observed Now**

`CURRENT Z_t [OBSERVED]`

Keep the existing state card.

## Step 2.2 — Add forecast horizon language

Above the chart use:

**Forecasted Risk — Next 50 Seconds**

Add a small subtitle:

`20 historical windows (200s) → 5 future horizons (10–50s)`

Only show this if it matches the actual model configuration.

## Step 2.3 — Add a NOW marker

On the Plotly risk graph add a vertical annotation/shape at the current forecast origin.

Suggested label:

`NOW — observed state`

The chart should visually distinguish:

- observed current state
- forecast region

Do not imply the model observes future labels.

## Step 2.4 — Add an outcome annotation after sequence advancement

When the user advances far enough that the actual attack state is observed, render a compact annotation such as:

`ACTUAL OUTCOME: IMPACT observed`

The annotation must be driven by the actual sequence label/state.

## Step 2.5 — Optional compact event strip

Add a simple timeline under the chart:

`NOW | +10s | +20s | +30s | +40s | +50s`

and, when known from the evaluated sequence, indicate the eventual observed attack onset without exposing future labels while the forecast is being evaluated.

For the presentation mode, it is acceptable to show the outcome after the presenter advances the sequence; do not leak future truth into the forecast calculation.

### Acceptance criteria

- A judge can immediately identify what is observed now and what is forecast.
- Advancing the sequence creates a visible transition from benign observation to later attack observation.
- The chart is not a flat “attack already happening” showcase.

---

# Phase 3 — Repair Criterion 3 presentation

## Goal
Remove unresolved placeholders and present the proactive result honestly.

## Step 3.1 — Replace question marks

Current visible placeholder:

`Median: ?s | Positive rate: ? (≥60%)`

Replace with measured values from the canonical Phase 7 result:

**Median proactive lead: +30s**  
**Proactive Detection Rate: 21.7%**  
**Status: ⚠ PARTIAL**

## Step 3.2 — Add metric footnote

Use:

> *Measured on the pre-attack evaluation subset; missed attacks are excluded from the proactive lead-time median.*

If the canonical result defines this differently, use the exact evaluation definition from the source artifact.

## Step 3.3 — Make the original success criterion visible without misrepresenting it

Recommended card text:

**CRITERION 3 — PROACTIVE LEAD**

`⚠ PARTIAL`

`Median proactive lead: +30s`

`PDR: 21.7%  |  Target: ≥60%`

This clearly distinguishes the measured result from the predefined target.

### Acceptance criteria

- Zero `?` placeholders anywhere in the visible benchmark panel.
- The card is numerically reproducible from the evaluation artifact.
- The UI does not call Criterion 3 a PASS.

---

# Phase 4 — Repair benchmark interpretation

## Goal
Prevent the multi-step Markov table from being misread as the headline model comparison.

## Step 4.1 — Retitle the table

Use:

**Multi-step tactic forecast — Macro-F1**

## Step 4.2 — Add headline benchmark immediately above/below it

Use:

> **Headline risk benchmark (one-step): Attention-LSTM 0.837 vs LR 0.776 → +6.1pp macro-F1**

These values must come from the frozen benchmark artifact.

## Step 4.3 — Add the scenario caveat

Use:

> *Test scenario is dominated by BENIGN/IMPACT persistence; the high Markov score is therefore scenario-sensitive and is not the headline forecasting result.*

Do not claim this caveat as an excuse for the result; present it as an interpretation of the benchmark context.

## Step 4.4 — Make metric identity explicit

If an LR value such as `0.255` remains in the table, label it as the metric it actually represents. Do not let the UI place an unlabeled `0.255` next to the `0.776` headline in a way that implies they are the same metric.

Potential structure:

```text
Headline benchmark
One-step future-risk Macro-F1
LSTM 0.837 | LR 0.776 | +6.1pp

Supporting analysis
Multi-step tactic forecast — Macro-F1
[table]
```

### Acceptance criteria

- The first number a judge sees is the correct headline benchmark.
- Every table column has an unambiguous metric label.
- Markov 0.996 is visibly contextualized.

---

# Phase 5 — Reframe temporal attention correctly

## Goal
Keep the attention visualization but eliminate causal overclaiming.

## Step 5.1 — Replace the current peak-only sentence

Replace:

`Peak attention: t-10 (weight=...)`

with:

> **Distributed temporal evidence across recent history; highest weight around t-10. The forecast reflects sustained behaviour, not a single triggering window.**

Optionally show the exact peak numerically in a tooltip instead of making it the headline interpretation.

## Step 5.2 — Keep the causal disclaimer

Retain:

> `This is model evidence, not causal proof.`

## Step 5.3 — Do not tune the model solely to create a visually sharper attention peak

The objective is correct interpretation, not aesthetic manipulation.

### Acceptance criteria

- Attention chart remains functional.
- Copy accurately describes distributed weights.
- No claim that t-10 “caused” the forecast.

---

# Phase 6 — Preserve strong existing panels

No engineering changes unless a regression is found.

## SHAP
Keep:

- feature importance chart
- `Influence ≠ causation`
- current feature-level attribution pipeline

## ATT&CK
Keep:

- predicted project state bucket
- ATT&CK tactic context
- likely techniques
- disclaimer that project buckets are not official ATT&CK ground truth

## What-if
Keep:

- actual model baseline forecast
- modified-input forecast
- side-by-side comparison
- explicit wording that numerical outputs come from the model, not the LLM

## AI agent
No dependency for the core forecast demo. The demo must still work deterministically when the agent panel is disabled.

---

# Phase 7 — Make the demo deterministic

## Goal
A technical judge must be able to replay the same sequence and obtain the same displayed values.

## Step 7.1 — Separate demo data loading from random UI state

Ensure Demo/Test Sequence mode always loads:

- the frozen model weights
- frozen preprocessing/scaler artifacts
- the selected demo sequence
- the expected state labels

Do not retrain or resample at runtime.

## Step 7.2 — Add a compact “Demo integrity” state

Optional debug-only panel or log:

```text
Demo sequence: <candidate_id>
Start window: <index>
Current state: BENIGN
Future attack observed at: <window/time>
Model artifact: <version/hash>
Preprocessing artifact: <version/hash>
```

This does not need to be prominent in the polished presentation UI, but it should be available for technical verification.

## Step 7.3 — Freeze artifact versions

Record hashes/versions for:

- model checkpoint
- scaler/preprocessor
- sequence dataset
- demo configuration
- application commit

### Acceptance criteria

The same machine/environment can replay the same demo and reproduce the same values.

---

# Phase 8 — Testing

## Unit tests

Add/update tests for:

1. default demo window exists
2. default demo state is BENIGN
3. future attack exists for selected sequence
4. risk chart receives exactly five future horizons
5. no NaN/`?` benchmark values are rendered
6. Criterion 3 formatting uses real stored values
7. benchmark labels distinguish one-step risk F1 from multi-step tactic F1
8. attention explanation string contains no causal language
9. ATT&CK disclaimer remains visible

## Integration test

Run the complete flow:

```text
launch app
→ Demo/Test Sequence
→ default BENIGN window
→ record forecast
→ advance sequence
→ observe later IMPACT
→ SHAP
→ attention
→ ATT&CK
→ what-if
→ benchmark
```

## Regression checks

Confirm these existing values remain unchanged unless the code path itself was intentionally modified:

- LR one-step risk macro-F1 ≈ 0.7761
- Attention-LSTM k=1 risk macro-F1 ≈ 0.8368
- improvement ≈ +6.07pp
- multi-horizon risk F1 values from the frozen benchmark artifact
- pre-attack proactive result: median +30s, PDR 21.7% (using the existing Phase 7 metric definition)

---

# Phase 9 — Visual QA

Capture fresh screenshots after every UI change.

Required final screenshots:

1. **Forecast — BENIGN starting state**
2. **Forecast — rising future-risk curve**
3. **Forecast — later IMPACT observed**
4. **XAI — SHAP + attention**
5. **ATT&CK context**
6. **What-if simulation**
7. **Benchmark panel with no placeholders**

Do not reuse the old screenshots showing:

- default IMPACT start
- flat 98% attack forecast as the main demo entry
- `?` benchmark values
- unlabeled LR/Markov/LSTM comparisons

---

# Phase 10 — Final 2-minute NTRO demo runbook

## 0:00–0:15 — Problem framing

Show the presentation slide briefly.

Narration:

> “Traditional detection tells us what is happening now. DigitalSpy uses recent temporal network behaviour to estimate what is likely to happen next.”

## 0:15–0:35 — Forecast before attack

Open the application in Demo/Test Sequence mode.

Start at the verified BENIGN window.

Narration:

> “At this point the network is still observed as benign. The model has the recent temporal history and forecasts the next five ten-second horizons.”

Advance the slider through the selected sequence.

Point to the risk curve:

> “The forecast changes before the observed attack state arrives.”

## 0:35–0:55 — Explain

Show SHAP and attention.

Narration:

> “We expose both feature-level evidence and temporal evidence: which traffic features influenced the forecast, and which parts of the recent history received more model weight.”

## 0:55–1:10 — Context

Show ATT&CK section.

Narration:

> “The forecast is enriched with ATT&CK context so an analyst can interpret the likely security behaviour.”

Immediately retain the disclaimer that this is semantic/project mapping, not official ATT&CK ground truth.

## 1:10–1:25 — What-if

Modify one feature using a sensible, reproducible value.

Run the comparison.

Narration:

> “The analyst can also test how a controlled change in network characteristics affects the model forecast.”

## 1:25–1:40 — Ground truth catches up

Advance to the actual attack window.

Show:

`CURRENT Z_t [OBSERVED] → IMPACT`

Narration:

> “Now the observed attack state arrives. This is the distinction we care about: the system is not only explaining the attack after it appears; it generated an earlier forecast from the preceding temporal behaviour.”

## 1:40–2:00 — Evidence and close

Show benchmark panel.

Narration:

> “Our strongest frozen benchmark result is a 6.1 percentage-point macro-F1 improvement for one-step future-risk forecasting over logistic regression. We also measure proactive warnings, while keeping the current limitations visible rather than overstating them.”

Closing line:

> **“DigitalSpy moves network defence from observing the current state toward forecasting the next one.”**

---

# Phase 11 — Final freeze checklist

Before recording the official NTRO demo/video:

- [ ] Default Demo window is a verified BENIGN pre-attack sequence.
- [ ] Actual future attack occurs later in the same selected sequence.
- [ ] Risk curve uses only real model outputs.
- [ ] Five forecast horizons are visible and correctly labeled.
- [ ] NOW marker is visible.
- [ ] Actual outcome annotation appears only when the sequence reaches the observed attack state.
- [ ] Criterion 3 shows measured values, no `?`.
- [ ] Criterion 3 says PARTIAL, not PASS.
- [ ] Benchmark table says “Multi-step tactic forecast — Macro-F1”.
- [ ] Headline one-step risk benchmark is explicitly shown: LSTM 0.837 vs LR 0.776 (+6.1pp).
- [ ] Markov caveat is visible.
- [ ] Attention wording says distributed evidence, not causal focus.
- [ ] SHAP/ATT&CK/What-if panels still work.
- [ ] App works with AI-agent panel disabled.
- [ ] No live training occurs during the demo.
- [ ] No unresolved placeholders are visible anywhere.
- [ ] Model/preprocessor/demo artifacts are versioned/frozen.
- [ ] Fresh screenshots are captured.
- [ ] Demo video and screenshots match the same default sequence.
- [ ] README screenshots are replaced with the final frozen UI.
- [ ] After freeze, do not change the demo implementation unless a blocking bug is found.

---

# Definition of Done

The implementation is complete when an NTRO evaluator can sit down in front of the application and, without additional explanation, understand this sequence:

**BENIGN now → forecast future risk → inspect predicted future state → understand why → map to security context → simulate a change → advance time → observe attack arrival → inspect benchmark evidence.**

The demo is successful when the forecasting claim is visible from the interaction itself, not merely inferred from labels on the page.
