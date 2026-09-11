# DigitalSpy — Next Implementation Plan
## NTRO / SIH 2026: AI-Based Network Attack Forecasting from Network Traffic Data

**Document purpose:** This file is the operational development plan for taking DigitalSpy from the current research prototype to a stronger, demonstrable, NTRO-aligned solution.

**Current development principle:** Preserve all completed experiments as immutable baselines. New work must be added as controlled experiments with explicit inputs, outputs, acceptance gates, and rollback points.

---

# 0. Executive Status

DigitalSpy has completed the initial research and implementation cycle from data auditing through temporal modelling and proactive-forecasting investigation.

The most important validated result is:

$$
LR_{k=1}=0.7761
$$

$$
Attention\text{-}LSTM_{k=1}=0.8368
$$

$$
\Delta F1=+6.07pp
$$

This passes the predeclared Criterion 1 requirement of at least +5 percentage points.

However, the system is **not yet fully NTRO-ready**. The remaining work is concentrated in five areas:

1. real packet-level feature extraction and flow+packet fusion,
2. demonstrable explainability,
3. defensible unseen-attack/generalisation evaluation,
4. stronger proactive forecasting methodology,
5. production-quality offline demo and submission packaging.

The immediate priority is **not** to keep adding model complexity. The immediate priority is to close explicit challenge requirements and establish stronger evidence around the forecasting claim.

---

# 1. Current System — What Exists Now

## 1.1 Current research pipeline

```text
Network Traffic / CIC-IDS2017 CSV
        ↓
CSV ingestion and normalization
        ↓
10-second temporal state windows
        ↓
24-dimensional S_t
        ↓
20-step history H_t
        ↓
┌────────────────────────────────────────────┐
│ Baselines                                  │
│                                            │
│ Logistic Regression → static/one-step     │
│ Markov Model         → state persistence  │
│ Attention-LSTM       → temporal forecast  │
└────────────────────────────────────────────┘
        ↓
K = 5 future horizons
        ↓
Risk forecast + tactic forecast
        ↓
Calibration + lead-time evaluation
        ↓
Explainability / future SOC interface
```

## 1.2 State representation

Current Week-1 state representation uses 24 features across:

- volume/timing,
- TCP flags,
- inter-arrival time,
- packet size,
- source/destination diversity,
- directionality,
- packet-derived fields in the schema.

The temporal configuration is:

- 10-second windows,
- non-overlapping window aggregation,
- source-IP-as-host identity assumption,
- 20-step history,
- 200 seconds of observed history,
- five future horizons,
- 10–50 seconds forecast range.

The final processed pipeline produced 107,301 training windows, 22,948 validation windows and 11,287 test windows, with 20-step sequences and 24 features. The resulting sequence counts were 107,277 train, 22,924 validation and 11,263 test. 

## 1.3 Target representation

The project uses two forecast targets.

### Future risk

Binary:

$$
Y_{t+k}^{risk}\in\{0,1\}
$$

meaning whether an attack is active for the relevant host at future horizon $t+k$.

### Future tactic

Six project-level classes:

```text
BENIGN
RECON
INITIAL_ACCESS
LATERAL_MOVEMENT
C2
IMPACT
```

The tactic taxonomy is a **project-specific engineering mapping** from CIC-IDS2017 labels. It should not be described as native ATT&CK ground truth.

## 1.4 Current model

The principal neural model is:

```text
Input: (batch, 20, 24)
        ↓
1-layer LSTM
hidden size = 64
        ↓
Bahdanau/additive temporal attention
        ↓
Risk head: 5 sigmoid outputs
Tactic head: 5 × 6-class softmax
```

The model uses direct multi-horizon prediction rather than recursive prediction for the primary benchmark.

## 1.5 Current baselines

### Logistic Regression

The fair one-step forecasting baseline is:

$$
LR(S_t)\rightarrow Y_{t+1}
$$

Experiment B result:

- Macro-F1: **0.7761**
- Precision: **0.8687**
- Recall: **0.7696**
- FPR: **0.4589**

### Markov

The discrete temporal baseline estimates:

$$
P(Z_{t+1}|Z_t)
$$

and multi-step state probabilities using:

$$
P^k
$$

The current test scenario is strongly concentrated in BENIGN + IMPACT, so the very high Markov tactic F1 is treated as a **scenario-specific persistence result**, not evidence of general superiority over the LSTM.

---

# 2. Completed Experiment Record

## 2.1 Experiment A — Original chronological scenario split

Original structure:

```text
Monday–Wednesday → Train
Thursday          → Validation
Friday            → Test
```

This preserved whole scenario blocks but produced severe class-support mismatch.

Observed state distributions:

| State | Train | Validation | Test |
|---|---:|---:|---:|
| BENIGN | 73,707 | 21,720 | 4,863 |
| RECON | 8,298 | 0 | 0 |
| INITIAL_ACCESS | 5,292 | 1,193 | 0 |
| LATERAL_MOVEMENT | 0 | 35 | 0 |
| C2 | 898 | 0 | 0 |
| IMPACT | 19,106 | 0 | 6,424 |

Conclusion: retain this experiment as a stress-test/generalisation record; do not use it as the primary six-class benchmark.

## 2.2 Experiment B — corrected class-covered temporal benchmark

The split was redesigned to make the supervised risk comparison meaningful while continuing to use whole scenario/file blocks and avoiding temporal contamination.

One class remains an intrinsic dataset limitation:

```text
LATERAL_MOVEMENT = only 36 Infiltration samples in CIC-IDS2017
```

It cannot be distributed meaningfully across train, validation and test without either duplicating samples or breaking the scenario structure.

## 2.3 Experiment B — Criterion 1

Results:

```text
LR one-step risk       = 0.7761 Macro-F1
Attention-LSTM k=1    = 0.8368 Macro-F1
Improvement            = +6.07pp
Required improvement   = +5pp
```

**Criterion 1 = PASS.**

This is the project's primary validated quantitative result.

## 2.4 Multi-horizon risk forecasting

Current LSTM risk Macro-F1:

| Horizon | Risk Macro-F1 |
|---|---:|
| k=1 | 0.8368 |
| k=2 | 0.8357 |
| k=3 | 0.8358 |
| k=4 | 0.8417 |
| k=5 | 0.8347 |

Average risk F1:

$$
F1_K\approx0.837
$$

## 2.5 Multi-horizon tactic forecasting

Current LSTM tactic Macro-F1:

| Horizon | Tactic Macro-F1 |
|---|---:|
| k=1 | 0.2812 |
| k=2 | 0.2748 |
| k=3 | 0.2788 |
| k=4 | 0.2793 |
| k=5 | 0.2726 |

Average:

$$
F1_K=0.2773
$$

Interpretation: binary future-risk forecasting is considerably easier than reliable future attack-state classification.

## 2.6 Criterion 2 — Markov vs LSTM

Current comparison:

```text
Markov F1_K = 0.9922
LSTM F1_K   = 0.2773
```

This fails the numerical criterion, but the test contains only BENIGN and IMPACT. The Markov result is dominated by persistent IMPACT behaviour.

**Status:** scenario-limited; do not use as a headline claim.

## 2.7 Calibration

Current k=1 calibration:

```text
Brier score = 0.1147
ECE         = 0.1178
```

Probabilities should continue to be described as probabilistic estimates, not guarantees.

## 2.8 Criterion 3 — initial lead-time benchmark

At a threshold of 0.90:

```text
Attack sequences = 6,512
Median lead time = 0.0 s
Positive lead rate = 0.2%
```

**Criterion 3 = NOT MET.**

The result indicates detector-like behaviour at the strict operating point.

## 2.9 Phase 7.1 — threshold sensitivity

With proper Val-A / Val-B separation:

| Threshold | Proactive Detection Rate | Median Proactive Lead |
|---|---:|---:|
| 0.30 | 20.0% | +25s |
| 0.50 | 18.3% | +30s |
| 0.90 | 6.7% | +25s |

The important result is that the model contains some earlier signal which is suppressed by a very conservative threshold.

**Do not declare 0.30 optimal solely because it has higher proactive detection.** The operating point must be selected using PDR together with precision, recall, FPR and lead time.

## 2.10 Phase 7.3 — pre-attack signal analysis

Pre-attack versus pure-benign analysis identified measurable differences.

Strongest reported features:

| Rank | Feature | Cohen's d | Interpretation |
|---|---|---:|---|
| 1 | std_iat | 0.75 | higher before attacks |
| 2 | max_iat | 0.60 | higher before attacks |
| 3 | mean_iat | 0.42 | higher before attacks |
| 4 | psh_ratio | -0.42 | lower before attacks |
| 5 | packets_per_sec | -0.41 | lower before attacks |
| 6 | distinct_dst_ips | -0.38 | lower before attacks |

This establishes that measurable distributional differences exist in the feature space.

## 2.11 Critical distinction — signal exists vs model learns it

The trained LSTM assigned:

```text
Pre-attack mean risk  = 0.0761
Pure-benign mean risk = 0.2255
```

Therefore the current model does not consistently rank pre-attack histories as more risky.

The correct conclusion is:

> Statistical precursor signal exists, but the current forecasting model does not yet reliably exploit that signal.

## 2.12 Phase 7 controlled progression experiment

Eligible sequences:

```text
1,281 BENIGN-now → ATTACK-in-future sequences
```

Original strict-threshold experiment:

- proactive detections = 13,
- positive proactive rate ≈ 1%,
- positive-only median lead ≈ +20s.

Lead-time analysis should distinguish:

```text
PROACTIVE → forecast before onset
AT-ONSET  → forecast coincides with onset
REACTIVE  → forecast after onset
MISSED    → no qualifying forecast
```

Missed attacks should be recorded as **missed / undefined lead time**, not as artificial negative lead times.

## 2.13 Phase 7.4 — temporal feature augmentation

The 24-feature representation was extended to 120 features using:

$$
x_t,\quad \Delta x_t,\quad \Delta^2x_t,
$$

rolling standard deviation and rolling maximum.

Results:

```text
PDR                = 21.7%
Median proactive lead = +30s
Best epoch          = 1
```

This was essentially unchanged from the simpler delta-feature experiment.

Conclusion:

> Increasing handcrafted trajectory features alone does not materially solve proactive forecasting.

---

# 3. What Is Scientifically Proven Now

DigitalSpy currently supports the following claims:

## Proven / strongly supported

### A. Temporal history adds value for one-step future-risk prediction

The Attention-LSTM outperformed the fair LR one-step baseline by +6.07 percentage points.

### B. Temporal persistence is informative

The Markov model demonstrates strong persistence for some attack states.

### C. Pre-attack feature differences exist

The feature analysis identified measurable differences, especially in IAT statistics.

### D. Threshold choice strongly affects early-warning behaviour

Lower thresholds expose more proactive predictions.

### E. More handcrafted trajectory features alone are insufficient

The 120-feature experiment produced no material improvement in proactive detection.

## Not yet proven

### A. Robust proactive 20–50 second attack forecasting

Not yet demonstrated.

### B. Generalization to truly unseen attack families

Not yet demonstrated.

### C. Robust six-class future ATT&CK-stage/tactic prediction

Not yet demonstrated.

### D. Mandatory flow + real packet-level fusion

Not yet demonstrated in the current audited implementation.

### E. Production-quality offline end-to-end demo evidence

Needs explicit verification and screenshots/video evidence.

---

# 4. NTRO Requirement Gap Closure Plan

This section defines the next implementation cycle.

---

# PHASE 8A — Real Packet-Level Telemetry and Flow + Packet Fusion

## Priority: P0 / Mandatory

This is the first development task because the NTRO requirement explicitly calls for both flow-level and packet-level features.

### Objective

Build a real PCAP ingestion path that extracts packet-level evidence rather than imputing packet features from flow-only data.

### Required packet feature families

At minimum implement and test:

```text
TTL statistics
TCP window statistics
IP fragmentation indicators
Payload-size distribution
Retransmission indicators
Port-scan signatures
```

### Proposed pipeline

```text
PCAP
 ↓
Scapy / packet parser
 ↓
packet-level records
 ↓
5–10s aggregation
 ↓
packet feature state P_t
 ↓
merge with flow state F_t
 ↓
combined network state S_t
 ↓
forecast model
```

### Fusion design

Do not replace the existing 24 features.

Create:

$$
S_t=[S_t^{flow};S_t^{packet}]
$$

Then run an ablation:

```text
Experiment 8A-1: Flow-only
Experiment 8A-2: Flow + Packet
```

### Required acceptance tests

```text
[ ] PCAP parser reads real packet records
[ ] TTL variance comes from packets
[ ] TCP window statistics come from packets
[ ] fragmentation is packet-derived
[ ] payload-size distribution is packet-derived
[ ] retransmission logic has unit tests
[ ] scan signature logic has unit tests
[ ] packet timestamps align to the same 10s window definition
[ ] flow and packet records join without future leakage
[ ] missing PCAP coverage is explicitly marked
[ ] packet coverage statistics are reported
```

### Deliverables

```text
src/digitalspy/pcap/
src/digitalspy/features/packet_features.py
configs/packet_features.yaml
scripts/audit_pcap.py
scripts/build_packet_state.py
reports/packet_feature_coverage.json
```

### Gate

**Do not claim NTRO packet-level compliance until real PCAP-derived features are demonstrated in the pipeline and visible in the evaluation report/demo.**

---

# PHASE 8B — Explainable Forecasting Evidence

## Priority: P0

The system already contains attention and SHAP-related components, but the next step is to make explainability visible and reproducible.

### Required output

Every qualifying forecast should be capable of producing:

```text
Forecast probability
Forecast horizon
Predicted tactic / behavioural state
Top contributing features
Historical timesteps receiving high attention
Evidence summary
```

### Example UI object

```text
FORECAST ALERT

Risk in +10s       0.82
Risk in +20s       0.91
Risk in +30s       0.94

Predicted tactic   INITIAL_ACCESS

Top evidence
1. std_iat
2. max_iat
3. syn_ratio

Temporal attention
T-190   ██
T-180   ███
...
T-20    ████████
T       ██████████
```

### Important caveat

Attention must be described as **temporal evidence**, not as causal proof.

SHAP/feature attribution should be attached to numerical model output, not generated by the LLM agent.

### Required examples

Create at least three reproducible explanation cases:

```text
1. true positive / proactive
2. false positive
3. missed pre-attack sequence
```

This makes the limitations as visible as the successes.

### Deliverables

```text
reports/xai_examples/
reports/attention_heatmaps/
reports/shap_examples/
src/digitalspy/explainability/
```

### Gate

A judge should be able to look at one alert and answer:

> “Why did the system make this prediction?”

without reading the source code.

---

# PHASE 8C — Offline End-to-End Demonstration

## Priority: P0

The current project already contains the planned Streamlit architecture, but it must be explicitly verified as a reproducible deliverable.

### Required user flow

```text
Upload CSV or PCAP
       ↓
Validate input
       ↓
Extract flow + packet state
       ↓
Build temporal history
       ↓
Run forecast
       ↓
Probability timeline
       ↓
Predicted tactic/context
       ↓
Top evidence
       ↓
Lead-time view
```

### Interface requirements

The dashboard should expose:

1. current network state,
2. 0–50s future risk curve,
3. predicted behavioural state/tactic where sufficiently confident,
4. confidence/probability,
5. top features,
6. temporal attention,
7. alert timeline,
8. explanation panel,
9. what-if analysis without allowing the LLM to manufacture numbers.

### Offline requirement

No cloud inference should be required for the main demo.

The agent layer may use local Ollama.

### Agent safety boundary

The local agent may:

```text
read forecasts
read history
lookup technique context
summarize alerts
compare what-if runs
```

It may not:

```text
modify probabilities
invent model scores
execute destructive actions
override numerical model outputs
```

### Gate

Run the complete demo on a clean environment using only the documented dependencies and local models.

---

# PHASE 8D — Generalization / Unseen Attack Experiment

## Priority: P0

The challenge explicitly values generalization beyond memorized attack signatures.

The current Experiment A exposed scenario/class mismatch but does not by itself demonstrate successful unseen-attack generalization.

### Option A — held-out attack family

Use whole scenario/file blocks.

Example design:

```text
TRAIN
other attack families + BENIGN

VALIDATION
seen attack families

TEST
entire held-out attack family
```

No random rows from the held-out attack family may appear in training.

### Option B — cross-dataset

Use CIC-IDS2017 for training and an independent benchmark such as UNSW-NB15 for external evaluation, subject to a documented feature-mapping layer.

### Recommended first implementation

Start with **held-out attack family**, because it is less disruptive to the current system.

### Required metrics

```text
macro-F1
per-class recall
FPR
precision
risk calibration
lead time where meaningful
```

### Gate

The experiment is successful scientifically even if performance drops, provided the drop is measured honestly and the failure mode is explained.

The requirement is to demonstrate that the system was tested against unseen patterns, not to manufacture a high score.

---

# PHASE 8E — Dedicated Pre-Attack Forecasting Objective

## Priority: P1

The current LSTM is trained primarily for general future-risk classification. The remaining problem is specifically:

```text
BENIGN NOW
       ↓
ATTACK LATER
```

versus:

```text
BENIGN NOW
       ↓
BENIGN LATER
```

### New target

Define:

$$
Y_t^{pre}=1
$$

when current state is benign and an attack occurs within the future prediction horizon.

Define:

$$
Y_t^{pre}=0
$$

for benign histories remaining benign throughout the horizon.

### Why this experiment exists

It tests whether explicitly optimizing the model for **early-warning discrimination** is more effective than general future-risk multitask training.

### Keep architecture initially unchanged

First compare:

```text
Current Attention-LSTM
        vs
Pre-attack-objective Attention-LSTM
```

without changing the architecture.

### Metrics

```text
PDR
FPR
Precision
Recall
AUROC / PR-AUC where appropriate
median successful proactive lead time
miss rate
```

### Gate

Do not declare success from F1 alone. A model that has high F1 but no positive lead time is still not solving the proactive forecasting problem.

---

# PHASE 8F — Controlled LSTM Regularization Study

## Priority: P1

The LSTM repeatedly reaches its best validation checkpoint at epoch 1 while training loss continues falling. This indicates a generalization/optimization problem that should be tested before declaring an architectural ceiling.

### Single controlled run

Keep the feature set and split fixed and test:

```text
learning rate = 1e-4
weight decay   = 1e-4
dropout        = 0.3
```

Do not launch a large sweep initially.

### Compare against the frozen 7.4 baseline

```text
PDR
median proactive lead
FPR
validation loss
k=1 risk F1
```

### Gate

If regularization still produces immediate validation deterioration and no meaningful proactive gain, confidence increases that the limitation is more likely related to the objective/data/model inductive bias rather than simply optimization.

---

# PHASE 8G — Optional Recursive State Rollout

## Priority: P2

The primary LSTM implementation is direct multi-horizon prediction:

$$
P(Y_{t+k}|H_t)
$$

This is legitimate and avoids error compounding, but NTRO's world-model framing emphasizes future-state evolution.

### Add an experimental rollout mode

```text
S_t
 ↓
predict S_(t+1)
 ↓
feed predicted state
 ↓
predict S_(t+2)
 ↓
...
```

Start with a 2–3-step demonstration rather than replacing the primary benchmark.

### Purpose

Demonstrate explicitly that DigitalSpy can reason about state evolution, while retaining direct multi-horizon prediction as the more stable benchmark.

---

# PHASE 8H — Mamba / State-Space World Model

## Priority: P2 / Research Upgrade

Only begin this phase after the mandatory and evaluation gaps above are closed.

### Motivation

The Phase 7.4 result shows that simply increasing handcrafted temporal features from 24 to 120 does not materially improve proactive forecasting.

Therefore a stronger temporal inductive bias may be justified.

### Proposed architecture

```text
Flow + Packet state
       ↓
Feature projection
       ↓
Mamba / SSM blocks
       ↓
Temporal representation
       ↓
Risk head K=5
Tactic head K=5
Uncertainty head
       ↓
Explainability
```

### Important scientific rule

Do not claim:

> “Mamba will solve forecasting.”

The correct hypothesis is:

> “A state-space sequence model may capture temporal dependencies that the current LSTM configuration does not exploit reliably.”

Test it.

### Required ablation

```text
24-feature LSTM
120-feature LSTM
Flow+Packet LSTM
Mamba / SSM
Flow+Packet Mamba / SSM
```

Only comparable splits and evaluation code may be used.

---

# 5. Data Integrity Rules — Frozen

These rules apply to every future experiment.

## 5.1 No random row-level train/test split

Scenario-aware or temporal separation must be preserved for the primary research benchmark.

## 5.2 No test tuning

The test set must never determine:

- feature selection,
- threshold selection,
- hyperparameter choice,
- model architecture choice.

## 5.3 Validation must be separated when required

For proactive threshold work:

```text
Val-A → threshold selection
Val-B → proactive evaluation
Test  → final untouched benchmark
```

## 5.4 No artificial class manufacturing for the main benchmark

Do not duplicate the 36 Infiltration samples to pretend that LATERAL_MOVEMENT has normal support.

## 5.5 Source-IP limitation remains documented

Source IP is treated as host identity for the benchmark, but real deployments may have NAT/DHCP/shared-address complications.

## 5.6 Missing packet coverage must be explicit

Do not silently convert “packet unavailable” into “packet-derived.”

---

# 6. Lead-Time Measurement — New Frozen Definition

For every eligible attack scenario:

$$
LeadTime=t_{attack\ onset}-t_{first\ qualifying\ forecast}
$$

Classify outcomes as:

```text
lead_time > 0 → PROACTIVE
lead_time = 0 → AT-ONSET
lead_time < 0 → REACTIVE
no forecast    → MISSED
```

For MISSED sequences:

$$
LeadTime=Undefined
$$

Do not assign an artificial negative value.

Report separately:

### Proactive Detection Rate

$$
PDR=\frac{N_{proactive}}{N_{eligible}}
$$

### Successful proactive lead time

Median and distribution of positive lead times only.

### Reactive detection delay

Median and distribution of post-onset detection delay.

### Miss rate

$$
MissRate=\frac{N_{missed}}{N_{eligible}}
$$

This produces a cleaner operational picture than a single lead-time number.

---

# 7. Threshold Selection Protocol

The operating threshold must not be selected simply by maximizing F1.

Generate a threshold curve:

```text
0.10
0.20
0.30
0.40
0.50
0.60
0.70
0.80
0.90
```

For each threshold measure:

```text
PDR
Precision
Recall
FPR
Miss rate
Median proactive lead
Alert volume
```

Then select one operating point on **Val-A** according to a clearly documented defender utility criterion.

Freeze it before Val-B/test evaluation.

---

# 8. Evaluation Matrix for the Next Release

Every model release should be evaluated in the same table structure.

| Model | Features | Task | k=1 F1 | F1_K | Precision | Recall | FPR | PDR | Median +Lead |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR | Flow | One-step risk | 0.7761 baseline | — | — | — | — | — | — |
| Markov | State | Tactic | scenario-limited | 0.9922 | — | — | — | — | — |
| LSTM | 24 Flow | Risk | 0.8368 | ~0.837 | — | — | — | — | — |
| LSTM | 24+trend | Risk | TBD | TBD | TBD | TBD | TBD | 21.7% | +30s |
| LSTM | Flow+Packet | Risk | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Pre-attack LSTM | Flow+Packet | Early warning | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mamba | Flow+Packet | Early warning | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

TBD values must remain genuinely unknown until measured.

---

# 9. Product / Demo Behaviour

DigitalSpy's final user-facing experience should answer four questions immediately:

### 1. What is happening now?

Current network state and risk.

### 2. What is likely to happen next?

Probability curve over +10s to +50s.

### 3. Why does the model think that?

Top features and temporal evidence.

### 4. What should the defender investigate?

Likely behavioural stage/tactic, affected host, confidence and evidence.

The interface must clearly distinguish:

```text
OBSERVED
FORECAST
INFERRED CONTEXT
```

This is essential to avoid presenting predictions as facts.

---

# 10. ATT&CK Integration Rules

ATT&CK context is a semantic layer around the forecast, not the numerical forecasting engine.

Preferred UI wording:

```text
Predicted behavioural tactic
Likely ATT&CK technique
Evidence supporting forecast
```

Avoid wording that implies the six project-level buckets are native ATT&CK ground-truth stages.

Where tactic confidence is low, display:

```text
Tactic: uncertain
Risk forecast: available
Top evidence: available
```

Do not force a tactic label purely because the interface requires one.

---

# 11. Agent Layer — Frozen Safety Contract

The local Ollama-based agent remains an orchestration and explanation component.

### Allowed tools

```text
get_recent_flows()
get_host_history()
lookup_attack_technique()
explain_forecast()
run_what_if()
generate_incident_summary()
```

### Numerical truth rule

The LLM must never generate or modify:

- risk probabilities,
- F1 values,
- lead times,
- thresholds,
- model confidence values.

Numerical values must come directly from the model/evaluation engine.

### What-if

The agent may request:

```text
baseline forecast H_t
modified scenario H'_t
comparison
```

but both forecast curves must come from the deterministic forecasting engine.

---

# 12. Repository Development Plan

Recommended next repository structure:

```text
DigitalSpy/
├── README.md
├── IMPLEMENTATION.md
├── NEXT-IMPLEMENTATION.md
├── pyproject.toml
│
├── configs/
│   ├── system.yaml
│   ├── features.yaml
│   ├── packet_features.yaml
│   ├── labels.yaml
│   ├── splits.yaml
│   ├── markov.yaml
│   ├── lstm.yaml
│   └── mamba.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── experiment_a_original_split/
│   ├── experiment_b_primary/
│   ├── phase7/
│   ├── packet_features/
│   ├── xai_examples/
│   ├── generalization/
│   └── final/
│
├── scripts/
│   ├── audit_dataset.py
│   ├── audit_pcap.py
│   ├── build_states.py
│   ├── build_packet_state.py
│   ├── train_logistic.py
│   ├── train_markov.py
│   ├── train_lstm.py
│   ├── phase7_threshold.py
│   ├── phase7_pre_attack.py
│   ├── phase7_4_augmented_lstm.py
│   ├── train_pre_attack.py
│   ├── train_mamba.py
│   ├── evaluate_all.py
│   ├── evaluate_lead_time.py
│   ├── evaluate_generalization.py
│   └── generate_xai_examples.py
│
├── src/digitalspy/
│   ├── data/
│   ├── features/
│   │   ├── flow_features.py
│   │   ├── packet_features.py
│   │   └── fusion.py
│   ├── states/
│   ├── baselines/
│   ├── markov/
│   ├── models/
│   │   ├── attention_lstm.py
│   │   └── mamba.py
│   ├── forecasting/
│   ├── explainability/
│   ├── attack_context/
│   ├── agent/
│   └── ui/
│
└── tests/
```

---

# 13. Testing Requirements for the Next Release

Add automated tests for:

## Packet extraction

```text
[ ] TTL aggregation
[ ] TCP window aggregation
[ ] fragmentation detection
[ ] payload statistics
[ ] retransmission detection
[ ] port-scan signature
```

## Temporal correctness

```text
[ ] timestamps sorted
[ ] 10-second window alignment
[ ] no window overlap
[ ] no future features in current state
[ ] 20-step history exactly maintained
[ ] sequence continuity enforcement
```

## Forecast correctness

```text
[ ] k=1 target shift
[ ] k=5 target shift
[ ] forecast shape
[ ] probability range [0,1]
[ ] threshold application
[ ] lead-time classification
```

## Explainability

```text
[ ] attribution shape
[ ] feature names preserved
[ ] attention weights sum to 1 per sequence
[ ] no fabricated explanation values
```

## Reproducibility

```text
[ ] fixed random seeds where applicable
[ ] config snapshot saved
[ ] model checkpoint saved
[ ] environment/dependency lock
[ ] report generated automatically
```

Target: **20+ meaningful tests per major pipeline release**, not superficial assertion counts.

---

# 14. Experimental Governance

Every new experiment must create a record containing:

```text
Experiment ID
Purpose
Hypothesis
Input dataset version
Feature version
Split version
Model version
Hyperparameters
Threshold protocol
Metrics
Observations
Limitations
Decision
Next action
```

Do not overwrite old JSON reports.

Use:

```text
reports/experiment_8a_...
reports/experiment_8b_...
```

or versioned report files.

This ensures that failed experiments remain reproducible evidence rather than disappearing history.

---

# 15. Recommended Execution Order

The correct sequence is:

```text
CURRENT BASELINE — FROZEN
        ↓
8A  Real PCAP + packet features
        ↓
8A ablation: Flow-only vs Flow+Packet
        ↓
8B  Explainability examples
        ↓
8C  Verify offline Streamlit demo
        ↓
8D  Held-out attack generalization
        ↓
8E  Dedicated pre-attack objective
        ↓
8F  Controlled LSTM regularization
        ↓
Re-evaluate proactive forecasting
        ↓
8G  Optional recursive state rollout
        ↓
8H  Mamba / stronger world model
        ↓
Final integrated evaluation
        ↓
Submission packaging
```

Do not skip directly from the current LSTM to Mamba simply because Mamba is more advanced.

---

# 16. NTRO Submission-Readiness Checklist

## Technical requirement

```text
[ ] Flow features demonstrated
[ ] Packet features demonstrated from real PCAP
[ ] Flow+packet fusion demonstrated
[ ] Structured network state demonstrated
[ ] Temporal dynamics demonstrated
[ ] K-step forecast demonstrated
[ ] Proactive lead-time measurement demonstrated
[ ] ATT&CK semantic context demonstrated
[ ] Explainability shown with real examples
[ ] Offline operation verified
[ ] Numerical outputs traceable to the model
```

## Generalization

```text
[ ] Original scenario-shift experiment retained
[ ] Held-out attack experiment completed
[ ] Class-support limitations documented
[ ] Cross-dataset evaluation considered
```

## Engineering

```text
[ ] GitHub repository clean
[ ] README complete
[ ] setup instructions tested
[ ] configuration documented
[ ] weights/checkpoints packaged
[ ] architecture document <= required limit
[ ] demo video <= required limit
[ ] technical deck <= required limit
```

## Demonstration

```text
[ ] User can load data
[ ] Model generates risk forecast
[ ] Forecast timeline visible
[ ] Top features visible
[ ] ATT&CK context visible
[ ] Lead time visible
[ ] Explainability visible
[ ] Offline mode works
```

---

# 17. What Should Not Be Claimed Yet

Until the corresponding experiments are completed, the project must not claim:

```text
❌ Reliable 20–50s proactive attack forecasting
❌ Robust unseen-attack generalization
❌ Full packet-level NTRO compliance
❌ Superior six-class tactic forecasting over Markov
❌ Causal explanations from attention
❌ Production-ready CII deployment
❌ Mamba superiority
```

Use evidence-backed wording instead:

```text
✅ Improved one-step future-risk prediction
✅ Measurable pre-attack feature differences
✅ Some proactive forecasts at lower thresholds
✅ Modular path toward flow+packet forecasting
✅ Explainability mechanism under development
✅ Research prototype for proactive cyber defence
```

---

# 18. Definition of Done for the Next Major Release

The next major DigitalSpy release is complete only when all of the following are true:

### Data

Real PCAP packet features are extracted and fused with flow features.

### Model

The model receives a documented combined state representation.

### Evaluation

At least one unseen-attack experiment is complete.

### Explainability

At least three reproducible alerts have feature-level and temporal explanations.

### Proactive forecasting

Lead-time metrics distinguish proactive, onset, reactive and missed cases.

### Demo

The offline Streamlit path works end-to-end from input data to forecast and explanation.

### Reproducibility

A fresh environment can reproduce the benchmark using documented commands/configuration.

### Scientific integrity

All previous baseline results remain archived and no test-set tuning has been introduced.

---

# 19. Final Strategic Direction

DigitalSpy should now move from:

```text
“Can a temporal model outperform a static classifier?”
```

to:

```text
“Can a multimodal temporal network state provide
credible early warning of future malicious behaviour?”
```

The completed experiments already answer the first question positively for one-step future risk:

$$
0.8368 > 0.7761
$$

with a:

$$
+6.07pp
$$

improvement.

The remaining challenge is the second question.

The development strategy is therefore:

```text
FLOW DATA
   +
REAL PACKET DATA
   ↓
RICH NETWORK STATE
   ↓
TEMPORAL DYNAMICS
   ↓
PRE-ATTACK OBJECTIVE
   ↓
K-STEP RISK FORECAST
   ↓
UNCERTAINTY + CALIBRATION
   ↓
ATT&CK SEMANTIC CONTEXT
   ↓
EXPLAINABLE ALERT
   ↓
DEFENDER DECISION
```

The purpose of the next implementation cycle is not to hide the weaknesses already found. It is to systematically remove the **mandatory requirement gaps**, test the **unresolved scientific hypotheses**, and turn DigitalSpy into a defensible, demonstrable NTRO/SIH prototype.

---

# 20. Immediate Next Command Sequence

When development resumes, use this order:

```bash
# 1. Freeze and record current baseline
python3 scripts/evaluate_all.py
python3 scripts/evaluate_lead_time.py

# 2. Audit available PCAP data
python3 scripts/audit_pcap.py

# 3. Build packet-derived state
python3 scripts/build_packet_state.py

# 4. Build fused flow + packet state
python3 scripts/build_states.py

# 5. Verify state integrity
python3 scripts/diagnostics.py

# 6. Run flow-only vs flow+packet baseline
python3 scripts/train_logistic.py
python3 scripts/train_lstm.py
python3 scripts/evaluate_all.py

# 7. Generate explainability examples
python3 scripts/generate_xai_examples.py

# 8. Verify offline interface
streamlit run src/digitalspy/ui/app.py

# 9. Run unseen-attack evaluation
python3 scripts/evaluate_generalization.py

# 10. Then investigate pre-attack objective / regularization
python3 scripts/train_pre_attack.py
```

The exact commands should only be executed after checking the current repository filenames and configuration files. Do not assume that old scripts still have identical paths or arguments after the PCAP/fusion refactor.

---

# 21. Final Development Principle

DigitalSpy should be developed according to four rules:

**Measure before modifying.**

**Preserve every experiment.**

**Never trade scientific validity for a better-looking score.**

**Build the NTRO-required capability before adding optional complexity.**

The immediate target is therefore not “the biggest model.”

The immediate target is:

$$
\boxed{\text{Flow + Packet + Temporal + Explainable + Reproducible + Offline}}
$$

Once that foundation is complete, Mamba/TGN/world-model upgrades become meaningful engineering decisions rather than speculative complexity.
