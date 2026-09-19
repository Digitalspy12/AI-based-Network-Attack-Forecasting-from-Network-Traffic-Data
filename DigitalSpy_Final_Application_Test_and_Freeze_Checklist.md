# DigitalSpy — Final Application Test & Freeze Checklist

## Purpose

This document defines the **final application testing, UI design corrections, scientific validation, and freeze criteria** for the DigitalSpy NTRO demo.

The goal is not to add new AI capability. The goal is to prove that the application is:

- scientifically coherent,
- visually understandable,
- reproducible,
- consistent across Demo and Upload modes,
- honest about current limitations,
- and ready for the NTRO presentation.

The central story that the application must make visible is:

```text
BENIGN observed now
        ↓
200 seconds of temporal history
        ↓
DigitalSpy forecasts next 10–50 seconds
        ↓
risk/state forecast is shown
        ↓
model evidence is explained
        ↓
actual attack appears later
        ↓
forecast can be compared with observed outcome
```

---

# 1. TEST ORDER

Run tests in this order:

```text
TEST 1  → Upload CSV ingestion
TEST 2  → Forecast output correctness
TEST 3  → Risk/state interpretation coherence
TEST 4  → Demo-mode sequence verification
TEST 5  → Forecast-before-attack verification
TEST 6  → Cross-mode consistency
TEST 7  → XAI verification
TEST 8  → ATT&CK context verification
TEST 9  → What-if verification
TEST 10 → Benchmark/metrics verification
TEST 11 → Clean offline regression
TEST 12 → Final visual QA
TEST 13 → Freeze
```



---

# 2. TEST 1 — UPLOAD CSV INGESTION

## Input

Use the verified CIC-IDS2017 CSV already used during testing.

Example:

```text
Tuesday..._ISCX.csv
```

## Verify

```text
☐ Upload mode accepts the CSV
☐ File is parsed without error
☐ Expected windows are generated
☐ Current state is calculated
☐ 20-window temporal history is available
☐ Model inference starts successfully
```

## Record

```text
filename
number of rows
number of generated windows
first timestamp
last timestamp
current window
current observed state
```

## Pass condition

The uploaded CSV produces a valid sequence and reaches the Forecast Dashboard without exceptions.

---

# 3. TEST 2 — FORECAST OUTPUT CORRECTNESS

For every tested sequence, verify:

```text
☐ t+1 exists
☐ t+2 exists
☐ t+3 exists
☐ t+4 exists
☐ t+5 exists

☐ all risk values are finite
☐ all risk values are between 0 and 1
☐ five future horizons correspond to 10/20/30/40/50 seconds
☐ history length is exactly 20 windows
```

The model input must be:

```text
t-19 ... t0
```

The future labels must never be part of the model input.

---

# 4. TEST 3 — RISK / STATE INTERPRETATION COHERENCE

This is a critical test.

The application has two outputs:

```text
Risk head:
P(attack | H_t)

State head:
P(Z_t | H_t)
```

They may disagree because they are separate heads.

The UI must not present their outputs as contradictory operational decisions.

## Example

An acceptable state is:

```text
Risk probability: 81.9%
Risk severity: HIGH

Operational state:
BELOW THRESHOLD

Raw state-head signal:
RECON (58%)
```

This is coherent because:

```text
HIGH risk severity
≠
operational alert activation
```

## Verify

```text
☐ risk severity is shown independently
☐ operational threshold is visible/understandable
☐ operational state uses the frozen threshold
☐ raw state-head output is clearly labeled analytical
☐ ATT&CK activation follows the same interpretation layer
```

## Forbidden

Do not display:

```text
81.9% attack risk
C2 84%
ATT&CK active
```

as though all three are the same operational decision.

---

# 5. TEST 4 — DEMO MODE SEQUENCE

Switch to:

```text
Demo — Test Sequence
```

Verify:

```text
☐ default window loads
☐ current state is BENIGN
☐ demo.yaml loads correctly
☐ scenario_id matches candidate artifact
☐ attack onset is known
☐ selected sequence is reproducible
```

Record:

```text
demo window index
scenario_id
start timestamp
attack onset window
attack onset timestamp
threshold
```

---

# 6. TEST 5 — FORECAST-BEFORE-ATTACK PROOF

This is the **most important test in the entire document**.

The application is only ready for NTRO when this temporal ordering is true:

```text
T0
Observed state = BENIGN
        ↓
Model receives previous 200 seconds
        ↓
Model generates forecast
        ↓
Forecast signal appears
        ↓
T+N
Observed attack begins
```

## Verify

At the selected demo starting window:

```text
☐ current state = BENIGN
☐ forecast is generated before onset
☐ attack onset occurs within +10 ... +50 seconds
☐ threshold crossing occurs before attack onset
☐ proactive lead is positive
☐ ground truth later becomes attack
```

## Record exact evidence

```text
start_window
start_timestamp
attack_onset_window
attack_onset_timestamp

risk_t1
risk_t2
risk_t3
risk_t4
risk_t5

threshold
first_threshold_crossing
proactive_lead_seconds

predicted_state_t1...t5
actual_state_t1...t5
```

## Scientific acceptance condition

The sequence must demonstrate:

> **forecast first → attack later**

If this cannot be proven for the selected sequence:

```text
☐ do not freeze the demo sequence
☐ rerun candidate selection
☐ do not edit probabilities
☐ do not move labels manually
```

---

# 7. TEST 6 — CROSS-MODE CONSISTENCY

Run the same logical sequence through:

```text
A. Upload Mode
B. Demo Mode
```

The interpretation layer must be identical.

## Compare

| Output | Upload | Demo | Must match? |
|---|---|---|---|
| Risk probability | — | — | YES |
| Forecast threshold | — | — | YES |
| Operational state | — | — | YES |
| Raw state-head distribution | — | — | YES |
| ATT&CK activation | — | — | YES |
| Horizon definitions | — | — | YES |

Only the input source should differ.

## Pass condition

No mode-specific interpretation logic changes the meaning of the same model output.

---

# 8. TEST 7 — FORECAST DASHBOARD DESIGN

## Required structure

The top dashboard should visually communicate:

```text
CURRENT OBSERVED STATE
        ↓
FORECASTED ATTACK RISK
        ↓
FORECASTED NETWORK STATE
```

## Current Network State

Use:

```text
CURRENT Z_t [OBSERVED]
```

This must represent the state currently observed in the selected window.

Do not label it as a prediction.

## Risk panel

Use:

```text
FORECASTED ATTACK RISK — NEXT 50 SECONDS
```

Subtitle:

```text
20 historical windows (200s) → 5 future horizons (10–50s)
```

The risk graph must show:

```text
+10s
+20s
+30s
+40s
+50s
```

## NOW marker

The `NOW` marker must clearly indicate the boundary between:

```text
observed history
```

and:

```text
future forecast
```

Do not imply a t0 forecast value if one is not actually plotted.

## Pass condition

A judge can immediately answer:

> “What is happening now?”
>
> “What does the model predict next?”

without presenter clarification.

---

# 9. TEST 8 — FORECASTED NETWORK STATE DESIGN

Use:

```text
FORECASTED NETWORK STATE
```

Do not use the previous:

```text
Predicted Attack Tactics
```

because the cards contain operational risk + state interpretation.

## Recommended card hierarchy

Each horizon should show:

```text
T+1 (+10s)

HIGH RISK
81.9%

Operational state:
BELOW THRESHOLD

Raw state-head signal:
RECON (58%)
```

The numerical risk should be visually primary.

The raw state-head signal should be secondary.

## Verify

```text
☐ Risk and state are visually separated
☐ Operational state is threshold-gated
☐ Raw state signal is clearly marked analytical
☐ Horizon labels are clear
☐ No contradictory operational claim appears
```

---

# 10. TEST 9 — HEATMAP DESIGN

The heatmap currently represents the multi-class state-head distribution.

Do not present it as though it were the operational decision.

## Rename

Use:

```text
RAW STATE-HEAD PROBABILITY DISTRIBUTION
```

Subtitle:

```text
Analytical distribution across project-defined network-state buckets.
```

Add:

```text
Operational state is gated by overall attack risk threshold.
```

## Rows

```text
BENIGN
RECON
INITIAL_ACCESS
LATERAL_MOVEMENT
C2
IMPACT
```

## Columns

```text
t+1
t+2
t+3
t+4
t+5
```

## Important

The six states are project-defined engineering buckets.

Do not call them official ATT&CK stages.

## Pass condition

A judge can understand that:

```text
Heatmap = raw analytical state distribution
```

while:

```text
Operational alert = threshold-gated decision
```

---

# 11. TEST 10 — XAI DESIGN

## SHAP

Keep:

```text
Feature Importance (SHAP)
```

Use:

```text
Most influential features for the t+1 risk forecast.
```

Keep:

```text
Influence ≠ causation.
```

Verify:

```text
☐ actual feature values are used
☐ bars render
☐ feature names are correct
☐ no future feature leakage
```

## Temporal attention

Use dynamic wording based on actual weights.

Recommended:

```text
Attention is distributed across recent history, with the highest weight at <t-X>.
The forecast reflects temporal evidence rather than a single causal trigger.
Attention is model evidence, not causal proof.
```

Verify:

```text
☐ peak label matches chart
☐ text changes when sequence changes
☐ no hard-coded t-10/t-1/etc.
☐ no causal claim
```

If attention is broadly uniform, say so.

Do not force a strong temporal-focus narrative.

---

# 12. TEST 11 — ATT&CK SECURITY CONTEXT

The ATT&CK panel is a semantic enrichment layer.

Use:

```text
ATT&CK Security Context
```

Keep:

```text
Semantic context only — project labels are not official ATT&CK ground truth.
```

## Operational state

Use:

```text
Operational Forecast State
```

Possible values:

```text
BELOW THRESHOLD
or
ACTIVATED
```

## Raw state

If useful, show:

```text
Raw State-Head Signal
RECON (58%)
```

## ATT&CK

When operational risk is below threshold:

```text
ATT&CK Tactic:
Not Activated
```

When threshold is crossed, use the appropriate project state mapping and corresponding ATT&CK context.

## Forbidden

Do not imply:

```text
official ATT&CK ground truth
attacker intent
causal attribution
```

---

# 13. TEST 12 — ATT&CK ICON / STATUS DESIGN

Avoid using:

```text
❓ BELOW THRESHOLD
```

because it looks like system uncertainty or an error.

Use:

```text
— BELOW THRESHOLD
```

or another neutral indicator.

Meaning:

> The system deliberately did not activate the operational ATT&CK alert.

This is a policy/decision state, not missing knowledge.

---

# 14. TEST 13 — WHAT-IF SIMULATION

Keep:

```text
What-If Scenario Simulation
```

Verify:

```text
☐ original value is shown
☐ modified value is shown
☐ baseline forecast comes from actual model
☐ modified forecast comes from actual model
☐ outputs are compared
☐ LLM does not invent numerical probabilities
```

Recommended narrative:

> “The analyst can modify a network feature and observe how the actual forecasting model responds.”

Do not present the what-if result as causal proof.

---

# 15. TEST 14 — BENCHMARK PANEL

## Criterion 1

Must show:

```text
PASS

LSTM k=1: 0.837
LR: 0.776
Δ: +6.1pp
```

## Criterion 2

Must show:

```text
FAIL / RESEARCH RESULT

LSTM: 0.277
Markov: 0.992
```

Keep the scenario caveat.

## Criterion 3

Must show:

```text
PARTIAL

Median proactive lead: +30s*
PDR: 21.7%
Target: ≥60%
```

Inline footnote:

```text
*Among successful proactive detections.
```

## Never show

```text
Median: ?s
Positive rate: ?
```

## Headline

Keep:

```text
Headline risk benchmark (one-step):
Attention-LSTM 0.837 vs LR 0.776 → +6.1pp macro-F1
```

## Caveat

Keep the explanation that the evaluated test scenario is dominated by BENIGN/IMPACT persistence and that the Markov score is scenario-sensitive.

---

# 16. TEST 15 — METRIC ARCHITECTURE

The UI should read from:

```text
configs/evaluation_metrics.yaml
```

Avoid manually duplicating numbers in `app.py`.

Recommended flow:

```text
Evaluation artifacts
        ↓
verified metrics
        ↓
evaluation_metrics.yaml
        ↓
Streamlit UI
```

Verify:

```text
☐ Criterion 1 matches evaluation artifact
☐ Criterion 2 matches evaluation artifact
☐ Criterion 3 matches evaluation artifact
☐ no duplicated conflicting values
```

---

# 17. TEST 16 — DEMO CONFIGURATION INTEGRITY

`configs/demo.yaml` should identify the real sequence:

```yaml
demo:
  default_window_index: <selected_index>
  scenario_id: "<candidate_id>"
  start_state: "BENIGN"
  attack_onset_window: <onset_index>
  attack_onset_state: "<ground_truth_state>"
  threshold: <frozen_validation_threshold>
```

Do not store:

```yaml
expected_model_prediction: ...
```

The model must generate the prediction at runtime.

---

# 18. TEST 17 — AUTOMATED VALIDATION

Run:

```bash
python src/scripts/validate_demo.py
```

It must validate:

```text
☐ demo.yaml loads
☐ evaluation_metrics.yaml loads
☐ scenario_id exists
☐ scenario_id matches candidate artifact
☐ window index matches candidate
☐ starting state = BENIGN
☐ attack onset exists
☐ attack onset within +50s
☐ all five probabilities exist
☐ probabilities ∈ [0,1]
☐ threshold exists
☐ threshold crossing occurs before onset
☐ positive lead time exists
☐ history length = 20
☐ no future leakage
☐ no '?' placeholders
```

Critical failure must produce a non-zero exit code.

---

# 19. TEST 18 — HEAD CONSISTENCY DIAGNOSTIC

Run:

```text
src/scripts/diagnose_head_consistency.py
```

Compare:

```text
P(attack)
```

against:

```text
1 - P(BENIGN)
```

Compute:

```text
Pearson correlation
Spearman correlation
mean absolute disagreement
median absolute disagreement
threshold agreement
```

Save:

```text
artifacts/head_consistency_report.json
artifacts/head_consistency_summary.md
```

## Purpose

This is a scientific diagnostic.

It does not need to be fixed by retraining.

If disagreement exists, document it as a limitation and retain the threshold-gated operational interpretation.

---

# 20. TEST 19 — CLEAN OFFLINE TEST

Use a clean checkout/environment.

Disable:

```text
internet
cloud APIs
external model services
```

Run:

```bash
python src/scripts/validate_demo.py
streamlit run src/digitalspy/ui/app.py
```

Perform the same replay twice.

## Pass condition

```text
Run 1 → successful
Run 2 → successful
```

Expected:

```text
same demo sequence
same model artifact
same forecast
same displayed metrics
same interpretation
```

No errors or external network dependency.

---

# 21. TEST 20 — FINAL VISUAL QA

Check screenshots for:

### Top dashboard

```text
☐ BENIGN current state visible
☐ forecast panel immediately below
☐ 200s history → 10–50s forecast explanation
☐ NOW boundary clear
```

### Forecast cards

```text
☐ risk is primary
☐ operational state is secondary
☐ analytical state is tertiary
```

### Heatmap

```text
☐ raw state-head label visible
☐ operational threshold explanation visible
```

### XAI

```text
☐ SHAP readable
☐ attention wording dynamic
☐ no causal claims
```

### ATT&CK

```text
☐ semantic disclaimer visible
☐ operational activation consistent with risk
☐ neutral icon for BELOW THRESHOLD
```

### Benchmark

```text
☐ no question marks
☐ PASS / FAIL / PARTIAL statuses correct
☐ headline one-step result visible
```

---

# 22. FINAL DEMO REHEARSAL

## Scene 1 — Start

Show:

```text
CURRENT Z_t [OBSERVED]
BENIGN
```

Say:

> “The network is currently observed as benign. DigitalSpy has the previous 200 seconds of temporal network behaviour.”

## Scene 2 — Forecast

Show the risk curve.

Say:

> “The temporal model now forecasts attack risk over the next 10 to 50 seconds.”

## Scene 3 — Explain

Show SHAP and attention.

Say:

> “We provide feature-level influence and temporal model evidence so the analyst can inspect the forecast.”

## Scene 4 — Context

Show ATT&CK context.

Say:

> “The model output is translated into security context for analyst interpretation.”

## Scene 5 — What-if

Run one scenario.

Say:

> “The analyst can test a hypothetical network change and observe the model's resulting forecast.”

## Scene 6 — Outcome

Advance until the actual attack appears.

Say:

> “The important temporal evidence is that the forecast was generated while the network was still observed as benign, and the attack state appears later.”

## Scene 7 — Evidence

Show benchmark panel.

Say:

> “Our strongest validated result is a 6.1 percentage-point improvement in one-step future-risk macro-F1 over logistic regression. Proactive lead time is demonstrated on a subset of successful pre-attack detections and remains a limitation.”

---

# 23. FINAL FREEZE CRITERIA

All must be true.

## Scientific

```text
☐ Real evaluated sequence
☐ BENIGN starting state
☐ Attack within +50s
☐ Forecast occurs before onset
☐ Positive proactive lead
☐ No future leakage
```

## Coherence

```text
☐ Risk and state heads are clearly separated
☐ Operational state uses frozen threshold
☐ ATT&CK uses same interpretation layer
☐ Raw heatmap is clearly labeled analytical
```

## UI

```text
☐ Forecast story understandable without explanation
☐ Heatmap understandable
☐ SHAP visible
☐ Attention explanation correct
☐ ATT&CK context consistent
☐ What-if works
☐ Benchmark readable
```

## Metrics

```text
☐ Criterion 1 = PASS
☐ Criterion 2 = FAIL / RESEARCH RESULT
☐ Criterion 3 = PARTIAL
☐ +30s is scoped to successful proactive detections
☐ no '?' placeholders
```

## Reproducibility

```text
☐ demo.yaml matches candidate artifact
☐ Upload and Demo modes use the same interpretation layer
☐ validation script passes
☐ clean offline run passes twice
```

---

# 24. FINAL GO / NO-GO DECISION

## GO

Only when:

```text
BENIGN
   ↓
real forecast
   ↓
positive lead
   ↓
actual attack
```

is reproducibly demonstrated.

## NO-GO

Stop and fix if any of these occur:

```text
☐ attack starts before forecast
☐ no positive lead
☐ selected sequence not reproducible
☐ UI displays contradictory operational states
☐ benchmark contains placeholders
☐ attention text does not match chart
☐ Upload and Demo modes disagree
☐ future information leaks into inference
☐ clean offline replay fails
```

---

# 25. AFTER FREEZE

Once all GO conditions pass:

**STOP APP DEVELOPMENT.**

Do not add:

```text
P2
Mamba
new model architectures
recursive rollout
new dashboard features
demo-only retraining
manual probability adjustments
```

Remaining work:

```text
PPT
README
2-minute video
GitHub/package cleanup
architecture document
technical presentation
rehearsal
```

---

# Definition of Done

DigitalSpy is ready for the NTRO demo when the evaluator can see:

```text
NOW
Observed = BENIGN
        ↓
20-window temporal history
        ↓
FORECAST
10–50s future risk
        ↓
EXPLAIN
SHAP + temporal evidence
        ↓
CONTEXT
ATT&CK-oriented interpretation
        ↓
OUTCOME
Attack observed later
        ↓
EVIDENCE
Benchmark results
```

The final proof is not a polished screenshot.

The final proof is:

> **The application makes a genuine forecast first, and the evaluated attack appears later in the same reproducible sequence.**
