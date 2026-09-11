# DigitalSpy — Week-1 Implementation Contract

> **Purpose:** operational source of truth for the one-week DigitalSpy prototype.
>
> **Use with:**  
> 1. `DigitalSpy_Internal_Research_Paper_Complete.docx` — research reasoning, landscape, alternatives, limitations.  
> 2. `DigitalSpy_Week1_Implementation_Specification.docx` — frozen technical specification.  
> 3. The reviewed one-week build guide — implementation clarifications and review fixes.
>
> This file resolves implementation-level details that were previously implicit or ambiguous.

---

## 0. Mission

Build a real, understandable, offline predictive cyber-defence prototype that tests:

> **Does temporal modelling of network behaviour provide better and earlier prediction of future malicious behaviour than a static feature-based classifier?**

The system is **not** expected to predict the exact future actions of an attacker.

It estimates probabilities over future malicious behaviour conditioned on observed history.

The conceptual mathematical spine is:

\[
S_t \rightarrow P(S_{t+1}\mid S_t)
\]

and for finite-horizon forecasting:

\[
P(S_{t+k}\mid H_t)
\]

where:

\[
H_t=(S_{t-h+1},\ldots,S_t)
\]

The actual Week-1 supervised targets are:

\[
P(Y^{risk}_{t+k}\mid S_{t-19:t})
\]

and

\[
P(Y^{tactic}_{t+k}\mid S_{t-19:t})
\]

for \(k=1,\dots,5\).

---

# 1. Frozen Scope

## 1.1 Implement this week

- CIC-IDS2017 as primary dataset.
- Flow-level features.
- Packet-derived features from limited but real PCAP coverage.
- 10-second non-overlapping windows.
- Per-source-host aggregation.
- 24-dimensional continuous state vector \(S_t\).
- `has_packet_features` as a provenance/data-quality field.
- Six discrete behavioural states \(Z_t\).
- Exact label precedence.
- Logistic Regression baseline.
- First-order discrete Markov model.
- One-layer Attention-LSTM.
- History length \(h=20\).
- Forecast horizon \(K=5\).
- Direct multi-horizon forecasting.
- Two forecast heads: risk and tactic.
- ATT&CK semantic context.
- SHAP feature evidence.
- Temporal attention evidence.
- Local Ollama analyst assistant.
- Plain-Python orchestration first.
- Streamlit offline interface.
- Chronological/scenario-aware evaluation.
- Forecast lead-time measurement.
- Probability calibration.

## 1.2 Do NOT implement in Week 1

Do not add to the critical path:

- TGN
- TGNN
- GNN state encoder
- Mamba / SSM backbone
- Full Transformer
- Next attacker–asset link prediction
- Full digital twin
- Reinforcement-learning response
- Autonomous blocking/remediation
- Cloud LLM APIs
- Multi-dataset training
- Hidden external services

These are Phase-2 research directions.

---

# 2. Product Architecture

```text
                    CIC-IDS2017
                    CSV + PCAP
                         |
                         v
                  DATASET AUDIT
              labels / time / QA
                         |
                         v
               FEATURE ENGINEERING
             flow + packet-derived
                         |
                         v
               TEMPORAL WINDOWING
                 10s / host / step
                         |
                  +------+------+
                  |             |
                  v             v
             Continuous      Raw labels
             state S_t            |
                  |          precedence
                  |               |
                  |               v
                  |              Z_t
                  |               |
                  v               v
             Logistic         Markov
             Regression       P(Zt+1|Zt)
                  |               |
                  +-------+-------+
                          v
                   Attention-LSTM
                   20 x 24 history
                          |
                          v
                     K=5 forecast
                    /            \
                   v              v
                Future Risk     Future Tactic
                   |              |
                   +------++------+
                          ||
                    ATT&CK + XAI
                          |
                          v
                    Local AI Agent
                          |
                          v
                     Streamlit SOC
```

---

# 3. Mathematical Contract

## 3.1 Continuous state \(S_t\)

For every source host and 10-second window:

\[
S_t \in \mathbb{R}^{24}
\]

### Exact 24 features

| Group | Features | Count |
|---|---|---:|
| Volume / timing | `flow_count`, `total_bytes`, `total_packets`, `mean_flow_duration`, `bytes_per_sec`, `packets_per_sec` | 6 |
| TCP flag ratios | `syn_ratio`, `ack_ratio`, `fin_ratio`, `rst_ratio`, `psh_ratio` | 5 |
| Inter-arrival | `mean_iat`, `std_iat`, `max_iat` | 3 |
| Packet size | `mean_packet_len`, `std_packet_len` | 2 |
| Diversity | `distinct_dst_ips`, `distinct_dst_ports`, `distinct_src_ports_used` | 3 |
| Directionality | `fwd_bwd_byte_ratio`, `mean_init_win_bytes_fwd` | 2 |
| Packet-derived | `ttl_variance`, `tcp_window_std`, `fragmentation_ratio` | 3 |

Total:

\[
6+5+3+2+3+2+3=24
\]

`has_packet_features` is a separate provenance field. It is **not automatically a predictive feature**.

## 3.2 Discrete state \(Z_t\)

For Tier 2:

\[
Z_t\in\{
BENIGN,
RECON,
INITIAL\_ACCESS,
LATERAL\_MOVEMENT,
C2,
IMPACT
\}
\]

### Critical distinction

`Z_t` is **not computed from the numerical values in `S_t`**.

Both represent the same temporal window:

```text
raw window
   +--> feature aggregation --> S_t
   |
   +--> raw labels --> precedence --> Z_t
```

This must remain true in code.

---

# 4. Exact Label Precedence

When multiple raw attack labels appear in the same host-window, choose the furthest-along bucket.

\[
IMPACT > C2 > LATERAL\_MOVEMENT > INITIAL\_ACCESS > RECON > BENIGN
\]

| Raw label | Bucket | Rank |
|---|---|---:|
| DoS Hulk / GoldenEye / Slowloris / Slowhttptest / DDoS | `IMPACT` | 1 |
| Bot | `C2` | 2 |
| Infiltration | `LATERAL_MOVEMENT` | 3 |
| Web Attack (Brute Force/XSS/SQLi), FTP-Patator, SSH-Patator, Heartbleed | `INITIAL_ACCESS` | 4 |
| PortScan | `RECON` | 5 |
| BENIGN | `BENIGN` | 6 |

Implementation:

```python
tactic(window) = min(rank(label) for label in labels_present)
```

`BENIGN` applies only when **no attack label** is present.

This is a project-specific target-engineering rule, not native ATT&CK ground truth.

---

# 5. Temporal Windows

Frozen parameters:

```text
window              = 10 seconds
stride              = 10 seconds
overlap             = 0
group_by            = source IP
minimum_flows       = 2
history_length      = 20 windows
history_duration    = 200 seconds
K                   = 5
step_duration       = 10 seconds
forecast_horizon    = 50 seconds
```

Each LSTM sample:

\[
H_t=[S_{t-19},S_{t-18},...,S_t]
\]

Shape:

```text
(20, 24)
```

## Sequence continuity

Week 1 uses **break/discard**, not masking.

A sequence is valid only if all 20 windows exist and are contiguous for the source host.

If one or more windows are missing:

```text
DISCARD SEQUENCE
```

Masking is Phase 2.

---

# 6. Packet Feature Coverage

This is a mandatory implementation requirement.

Do not allow packet-derived features to exist only in the test split.

Minimum recommended PCAP coverage:

```text
Monday     -> benign-side packet baseline
Wednesday  -> attack-side packet signal
Friday     -> test-side coverage
```

For missing packet coverage:

```text
ttl_variance
tcp_window_std
fragmentation_ratio
```

use the training-period mean for imputation.

Store:

```text
has_packet_features = 0
```

when no packet record exists, otherwise:

```text
has_packet_features = 1
```

## Leakage protection

Do not make packet coverage a hidden day/split identifier.

Therefore `has_packet_features` should initially be used for provenance/QA, not as a learned predictor, unless coverage is sufficiently uniform across splits.

---

# 7. Dataset Audit

Before defining the split, inspect the actual files.

```python
import pandas as pd
import glob

for f in sorted(glob.glob("cic2017/*.csv")):
    df = pd.read_csv(
        f,
        usecols=lambda c: c.strip() == "Label"
    )
    print(
        f,
        df.iloc[:, 0]
          .str.strip()
          .value_counts()
          .to_dict()
    )
```

Also audit:

```text
row counts
timestamp ranges
label distributions
missing values
infinities
duplicates
source hosts
destination hosts
column names
PCAP coverage
```

Do not trust a remembered attack schedule.

The actual downloaded files are the source of truth.

---

# 8. Train / Validation / Test

Initial arrangement:

```text
TRAIN      -> Monday–Wednesday
VALIDATION -> Thursday
TEST       -> Friday
```

Construct the actual file list after dataset audit.

Normalization:

```text
TRAIN -> fit scaler/imputer
VALID -> transform only
TEST  -> transform only
```

Never fit data-dependent preprocessing on validation or test.

---

# 9. Fair Evaluation Tasks

## Experiment A — Current detection

```text
S_t -> Logistic Regression -> Y_t
```

Question:

> How well can a static model classify the current window?

## Experiment B — One-step forecasting

```text
S_t -> Logistic Regression -> Y_(t+1)
```

versus:

```text
S_(t-19:t) -> Attention-LSTM -> Y_(t+1)
```

Question:

> Does temporal history improve one-step forecasting?

This is the fair comparison for Success Criterion #1.

## Experiment C — Multi-step forecasting

Markov:

\[
P(Z_{t+k}\mid Z_t)
\]

Attention-LSTM:

\[
P(Y_{t+k}\mid S_{t-19:t})
\]

for:

\[
k=1,...,5
\]

Both tactic forecasts must use the same six-class target taxonomy for the F1 comparison.

---

# 10. Tier 1 — Logistic Regression

Primary outputs:

```text
current attack probability
```

and separate one-step forecast baseline:

```text
P(Y_(t+1) | S_t)
```

Required metrics:

```text
Precision
Recall
Macro-F1
FPR
```

Accuracy must not be the headline metric.

---

# 11. Tier 2 — Markov World Model

Estimate:

\[
P_{ij}=P(Z_{t+1}=j\mid Z_t=i)
\]

from training transitions only.

For a six-state model:

```text
6 x 6 transition matrix
```

K-step roll-forward:

\[
p_{t+k}=p_tP^k
\]

Products:

```text
models/markov_matrix.npy
reports/markov_transition_matrix.csv
reports/markov_metrics.json
```

The Markov model is the project's most literal and transparent implementation of state-transition dynamics.

---

# 12. Tier 3 — Attention-LSTM

Input:

```text
20 x 24
```

Architecture:

```text
20 state vectors
       |
       v
1-layer LSTM
hidden = 64
       |
       v
20 hidden states
       |
       v
Additive attention
       |
       v
Context vector
       |
     +---+---+
     |       |
     v       v
   Risk    Tactic
  sigmoid  softmax x 6
     |       |
     +---+---+
       |
       v
    horizons 1..5
```

Frozen initial configuration:

```text
input_size        = 24
sequence_length   = 20
hidden_size       = 64
num_layers        = 1
dropout           = 0.20
attention         = additive / Bahdanau-style
optimizer         = Adam
learning_rate     = 1e-3
batch_size        = 64
early_stopping    = validation loss
```

Increase model capacity only after a documented underfitting diagnosis.

---

# 13. Attention

For hidden state \(h_i\):

\[
e_i=v^T tanh(W_hh_i+b_h)
\]

\[
\alpha_i=softmax(e_i)
\]

\[
c=\sum_i\alpha_i h_i
\]

Store the attention weights for inference.

Interpretation:

> Attention identifies historical windows the model weighted more heavily. It is evidence about model focus, not causal proof.

---

# 14. Multi-Horizon Forecasting

Use **direct multi-horizon prediction**.

Input:

\[
H_t=S_{t-19:t}
\]

Output:

```text
risk_t+1 ... risk_t+5

tactic_t+1 ... tactic_t+5
```

Do not recursively feed predictions back into the model for the primary Week-1 system.

Recursive roll-forward is a Phase-2 experiment.

---

# 15. Training Loss

Risk head:

\[
L_{risk}=\sum_{k=1}^{K}w(y_k)BCE(y_k,\hat y_k)
\]

Tactic head:

\[
L_{tactic}=\sum_{k=1}^{K}CE_{weighted}(y_k,\hat y_k)
\]

Total:

\[
L_{total}=L_{risk}+L_{tactic}
\]

Start with equal head weighting.

---

# 16. Class Imbalance

Use class-weighted loss.

Do not duplicate temporal windows in the primary experiment.

Report:

```text
Macro-F1
Per-class recall
Per-class precision
FPR
```

Heartbleed:

> If sample count is too small for a meaningful estimate, exclude it from headline per-class claims and document the limitation.

---

# 17. Success Criteria

These thresholds are fixed before seeing final test results.

## Criterion 1 — One-step forecasting

Attention-LSTM risk at `k=1` must beat the **Logistic Regression one-step forecasting baseline** by:

\[
\geq 5
\]

percentage points macro-F1.

FPR may increase by no more than:

\[
2
\]

percentage points.

## Criterion 2 — Multi-step forecasting

For each model:

\[
F1_K=\frac{1}{K}\sum_{k=1}^{K}F1_k
\]

Attention-LSTM must exceed Markov by:

\[
\geq 5
\]

percentage points on horizon-averaged tactic macro-F1.

Report:

```text
F1_1
F1_2
F1_3
F1_4
F1_5
F1_K
```

## Criterion 3 — Forecast lead time

Target:

\[
median\ lead\ time\geq10s
\]

and positive lead time for:

\[
\geq60\%
\]

of test attack instances.

A zero/negative result is a legitimate negative research result.

---

# 18. Forecast Lead Time

Choose the alert threshold using validation data only.

Freeze it.

For each test attack:

```text
attack_onset
first qualifying forecast
```

Then:

\[
LeadTime=AttackOnset-FirstForecast
\]

Report:

```text
median
mean
positive-lead-time rate
scenario distribution
```

Never tune the threshold on the final test set.

---

# 19. Calibration

Risk probabilities must be evaluated as probabilities.

Brier score:

\[
Brier=\frac1N\sum_i(p_i-y_i)^2
\]

Also compute ECE.

Do not interpret:

```text
risk = 0.80
```

as certainty.

It is a model probability.

---

# 20. ATT&CK Layer

ATT&CK is **semantic context**, not the forecasting engine.

Pipeline:

```text
forecasted behavioural bucket
        |
        v
project taxonomy
        |
        v
ATT&CK tactic
        |
        v
optional technique/context
```

Use:

> Predicted ATT&CK tactic

and:

> Likely ATT&CK technique

Do not claim the six project buckets are official ATT&CK ground-truth stages.

---

# 21. Explainability

## SHAP

Use SHAP for feature attribution.

Output:

```text
top_features
feature_contribution
```

## Attention

Output:

```text
attention_by_timestep
```

UI wording:

```text
Most influential features
Highest-attention historical windows
Evidence associated with forecast
```

Avoid:

```text
caused by
proves causality
exact cause
```

---

# 22. Local AI Agent

The local agent runs **after** the forecast engine.

```text
forecast model
      |
      v
structured JSON
      |
      v
local agent
      |
      v
tools
      |
      v
analyst explanation
```

Initial runtime:

```text
Ollama
small local model
Q4 quantization
```

Target model size:

```text
3B–4B parameters
```

Plain Python orchestration first.

---

# 23. Agent Tool Set

Minimum:

```python
get_recent_flows()
get_host_history()
lookup_attack_technique()
explain_forecast()
run_what_if()
generate_incident_summary()
```

Numerical truth boundary:

```text
Forecast model -> probability
Agent           -> interpretation/context
```

The agent must never rewrite the model probability.

---

# 24. What-If Simulation

Baseline:

```text
H_t -> forecast
```

Scenario:

```text
H'_t -> forecast
```

Compare:

```text
baseline risk curve
scenario risk curve
difference
```

The LLM explains the comparison.

It does not invent numerical outcomes.

---

# 25. Streamlit Product

The final UI should support:

### Input

```text
CSV
PCAP/sample
```

### Forecast

```text
current state
current risk
t+1
t+2
t+3
t+4
t+5
```

### Behaviour

```text
predicted tactic per horizon
```

### Evidence

```text
SHAP
attention timeline
```

### Security context

```text
ATT&CK tactic / technique
```

### Agent

```text
analyst summary
```

### Simulation

```text
baseline vs what-if
```

---

# 26. Phase-by-Phase Product Plan

| Phase | Input | Product at end | Definition of Done |
|---|---|---|---|
| 0. Spec Freeze | Research docs | Machine-readable configs | Frozen parameters reproduced in config |
| 1. Dataset Audit | CIC-IDS2017 | Dataset manifest + audit report | Actual labels/files/timestamps verified |
| 2. State Pipeline | CSV + PCAP | Saved 24-feature state dataset | Reproducible state generation |
| 3. Static Baseline | S_t | Logistic model + metrics | Baseline reproducibly evaluated |
| 4. Markov World Model | Z_t | 6×6 transition model | P(Zt+1\|Zt) + K-step forecast works |
| 5. Attention-LSTM | 20×24 sequences | Model checkpoint | Checkpoint reloads and predicts |
| 6. Forecast Engine | history + checkpoint | 5-step JSON output | t+1…t+5 generated correctly |
| 7. XAI + ATT&CK | forecasts + features | Evidence report | Explainability reproducible |
| 8. Local Agent | structured forecast | Offline analyst summary | No cloud and bounded tools |
| 9. Streamlit | all components | DigitalSpy SOC prototype | Full local run |
| 10. Evaluation | held-out test | Final benchmark report | No test tuning |

---

# 27. Repository Structure

```text
DigitalSpy/
│
├── README.md
├── IMPLEMENTATION.md
├── pyproject.toml
│
├── configs/
│   ├── system.yaml
│   ├── features.yaml
│   ├── labels.yaml
│   ├── splits.yaml
│   ├── markov.yaml
│   └── lstm.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
├── reports/
│
├── scripts/
│   ├── audit_dataset.py
│   ├── build_states.py
│   ├── train_logistic.py
│   ├── train_markov.py
│   ├── train_lstm.py
│   ├── run_forecast.py
│   ├── evaluate_all.py
│   └── evaluate_lead_time.py
│
├── src/
│   └── digitalspy/
│       ├── data/
│       ├── features/
│       ├── states/
│       ├── baselines/
│       ├── markov/
│       ├── models/
│       ├── forecasting/
│       ├── explainability/
│       ├── attack_context/
│       ├── agent/
│       └── ui/
│
├── tests/
└── docs/
```

---

# 28. Unit Tests Required Before Integration

## Labels

```text
test_precedence_impact_over_c2
test_precedence_c2_over_lateral
test_precedence_lateral_over_initial
test_precedence_initial_over_recon
test_benign_only_when_no_attack
```

## Features

```text
test_feature_schema_24
test_no_unexpected_feature_names
test_zero_division
test_packet_feature_imputation
test_has_packet_features
```

## Windowing

```text
test_10_second_windowing
test_non_overlapping_windows
test_minimum_flow_rule
test_20_window_sequence
test_gap_discards_sequence
```

## Leakage

```text
test_scaler_train_only
test_imputer_train_only
test_no_future_features_in_input
test_split_order
```

## Models

```text
test_markov_rows_sum_to_one
test_lstm_shape
test_forecast_K5_shape
test_probability_range
test_valid_tactic_classes
```

---

# 29. Early-Warning Red Flags

Stop and inspect the pipeline if:

```text
test score becomes suspiciously high
```

```text
packet features exist only in test
```

```text
NaN/infinity appears after feature construction
```

```text
LSTM receives future windows
```

```text
Markov rows do not sum to approximately 1
```

```text
threshold chosen using test results
```

```text
agent invents probabilities
```

```text
attention/explanation changes between identical runs unexpectedly
```

High accuracy is **not** automatically good news; first check for leakage.

---

# 30. Engineering Change-Control Rule

Before changing any frozen parameter:

1. State the problem.
2. Explain why the current choice fails.
3. Identify affected components.
4. Propose the smallest change.
5. State how the change will be validated.

Do not change:

```text
24 features
10-second windows
h = 20
K = 5
six tactic buckets
split strategy
success criteria
```

just because another configuration produces a better test score.

---

# 31. Known Limitations

Document explicitly:

### Source IP as host identity

The benchmark allows this simplification, but real NAT/DHCP networks may not provide stable one-IP-one-host identity.

### Global Markov model

Week 1 uses one pooled global transition matrix rather than per-host matrices. This is a deliberate data-volume tradeoff.

### Dataset realism

CIC-IDS2017 is a benchmark, not a complete model of modern enterprise/CII traffic.

### Label engineering

The six buckets are project-defined mappings from dataset attack labels.

### Causality

The LSTM learns statistical temporal dependencies, not verified causal mechanisms.

### Packet coverage

The Week-1 packet layer uses selected PCAP coverage; it is not a complete reprocessing of every original packet in the dataset.

### Forecast horizon

50 seconds is a prototype configuration, not a universal operational guarantee.

---

# 32. Phase-2 Roadmap

After Week 1:

```text
Feature-vector state
        ↓
Temporal graph state
        ↓
TGN / TGNN
        ↓
Mamba / SSM
        ↓
Next attacker–asset link prediction
        ↓
Exposure / vulnerability context
        ↓
Counterfactual transitions
        ↓
Cross-dataset evaluation
```

Next-target prediction belongs here because it is fundamentally relational.

---

# 33. One-Week Schedule

## Day 1

Dataset audit.

Product:

```text
dataset_profile
verified file list
label counts
```

## Day 2

Feature + state pipeline.

Product:

```text
state_windows.parquet
```

## Day 3

Logistic Regression + Markov.

Product:

```text
baseline metrics
transition matrix
```

## Day 4

Attention-LSTM.

Product:

```text
trained checkpoint
validation curves
```

## Day 5

K-step forecast + XAI + ATT&CK.

Product:

```text
forecast JSON
evidence output
```

## Day 6

Streamlit + local agent.

Product:

```text
offline DigitalSpy demo
```

## Day 7

Evaluation + packaging + rehearsal.

Product:

```text
final benchmark
final limitations
reproducible package
```

---

# 34. Demo Narrative

The judge should see:

```text
Traffic input
   ↓
Temporal state
   ↓
Current risk
   ↓
t+1 ... t+5 forecast
   ↓
Predicted tactic
   ↓
Evidence
   ↓
ATT&CK context
   ↓
Local agent explanation
   ↓
What-if analysis
   ↓
Benchmark comparison
```

Narrative:

> **Observe → Forecast → Explain → Simulate → Decide**

---

# 35. Canonical AI Coding-Agent System Prompt

```text
SYSTEM ROLE

You are the Senior Cybersecurity Engineer, Senior Machine Learning
Engineer, Senior Data Engineer, and Senior Software Engineer responsible
for implementing DigitalSpy — Predictive Cyber Defence.

You are not a generic code generator.

You are accountable for:
- mathematical correctness,
- cybersecurity correctness,
- temporal-valid evaluation,
- reproducibility,
- offline operation,
- clean architecture,
- testability,
- explainability,
- maintainability.

REFERENCE HIERARCHY

You have:
1. DigitalSpy_Internal_Research_Paper_Complete
2. DigitalSpy_Week1_Implementation_Specification
3. IMPLEMENTATION.md

The research paper explains WHY and WHAT.

The implementation specification explains the frozen technical contract.

IMPLEMENTATION.md resolves implementation details.

If documents conflict:
- preserve the frozen Week-1 scope,
- preserve the mathematical definitions,
- do not silently invent architecture,
- report the conflict before changing a frozen decision.

MISSION

Build one complete, understandable, offline prototype.

Research hypothesis:

"Does temporal modelling of network behaviour provide better and earlier
prediction of future malicious behaviour than a static classifier?"

This is a probabilistic forecasting task.

Do not claim deterministic knowledge of the attacker's future.

FROZEN ARCHITECTURE

CSV / PCAP
→ features
→ temporal windows
→ S_t
→ Logistic Regression baseline
→ Markov world model
→ Attention-LSTM
→ K-step forecast
→ ATT&CK context
→ XAI
→ local agent
→ Streamlit

WEEK-1 EXCLUSIONS

Do NOT add:
- TGN
- TGNN
- GNN
- Mamba
- SSM
- Transformer
- next-target prediction
- digital twin
- reinforcement learning
- autonomous response
- cloud LLM

DATA CONTRACT

Primary dataset: CIC-IDS2017.

Inspect actual files before defining the split.

Never trust remembered attack schedules.

FEATURE CONTRACT

S_t = 24-dimensional continuous vector.

Keep feature names exactly as specified.

Packet-derived features:
- ttl_variance
- tcp_window_std
- fragmentation_ratio

Use real packet-feature coverage on training-side data as well as test-side
data.

`has_packet_features` is initially a provenance field, not a predictive
feature.

LABEL CONTRACT

Z_t is a six-state behavioural class.

Precedence:
IMPACT > C2 > LATERAL_MOVEMENT > INITIAL_ACCESS > RECON > BENIGN

BENIGN only when no attack label exists.

IMPORTANT:
Z_t is derived from raw labels, not numerically inferred from S_t.

WINDOW CONTRACT

10 seconds.
Non-overlapping.
Grouped by source IP.
Minimum 2 flows.

LSTM history:
20 contiguous windows.

If a sequence has a missing window:
DISCARD it.

Do not implement masking in Week 1.

FORECAST CONTRACT

K=5.

Each horizon = 10 seconds.

Total horizon = 50 seconds.

Two output heads:
1. future risk
2. future tactic

Do not implement next-target prediction in Week 1.

MODEL CONTRACT

Logistic Regression:
- current detection
- one-step forecasting baseline

Markov:
P(Z_(t+k) | Z_t)

Attention-LSTM:
P(Y_(t+k) | S_(t-19:t))

LSTM:
- input 24
- sequence 20
- hidden 64
- 1 layer
- dropout .20
- additive attention
- Adam
- lr 1e-3
- batch 64
- early stopping on validation loss

EVALUATION

No random row-level split for the headline result.

Chronological/scenario-aware split.

Fit preprocessing on training only.

Thresholds selected on validation only.

Test is never used for tuning.

FAIR BASELINES

Current detection:
LR -> Y_t

One-step forecasting:
LR -> Y_(t+1)
Attention-LSTM -> Y_(t+1)

Multi-step tactic forecasting:
Markov -> Z_(t+k)
LSTM -> Y_(t+k)

Use the same six-class target definition for the tactic comparison.

SUCCESS CRITERIA

Criterion 1:
LSTM t+1 risk macro-F1 >= LR t+1 macro-F1 + 5pp,
with FPR increase <= 2pp.

Criterion 2:
LSTM F1_K >= Markov F1_K + 5pp.

F1_K = average of F1_1 ... F1_5.

Criterion 3:
median positive forecast lead time >= 10 seconds and
>=60% of test attack instances have positive lead time.

If a criterion fails:
report it honestly.
Do not modify the test protocol.

XAI

SHAP -> feature attribution.

Attention -> temporal evidence.

Do not claim causal proof.

ATT&CK

ATT&CK is a semantic context layer.

Use:
"predicted ATT&CK tactic"
"likely ATT&CK technique"

Do not claim the project labels are official ATT&CK ground truth.

AGENT

The numerical forecasting model is the source of truth.

The agent:
- reads structured outputs,
- retrieves local context,
- calls approved tools,
- explains,
- summarizes,
- supports what-if analysis.

The agent must never:
- invent probabilities,
- modify probabilities,
- fabricate evidence,
- turn uncertainty into certainty,
- execute destructive actions automatically.

Use Ollama locally.
Use plain Python orchestration first.

PHASE GATING

Do not start a later phase until the current phase produces its required artifact.

Phase 1 -> dataset audit
Phase 2 -> state dataset
Phase 3 -> static baseline
Phase 4 -> Markov
Phase 5 -> Attention-LSTM
Phase 6 -> forecast engine
Phase 7 -> XAI + ATT&CK
Phase 8 -> agent
Phase 9 -> Streamlit
Phase 10 -> evaluation

SOFTWARE ENGINEERING

- Prefer configuration over magic numbers.
- Validate every input.
- Fail loudly on schema mismatches.
- Add unit tests for each transformation.
- Separate data, models, inference, explanation, agent, and UI.
- Store model/config/preprocessing artifacts.
- Log dataset version and experiment configuration.
- Do not hard-code developer machine paths.
- Do not store secrets in code.
- Do not introduce dependencies without justification.

SECURITY ENGINEERING

Treat network telemetry and uploaded files as untrusted.

Validate:
- schema,
- timestamps,
- labels,
- NaN/Infinity,
- feature ranges,
- probability ranges,
- file size,
- local file paths.

Never execute arbitrary network payloads.

CHANGE CONTROL

Before changing a frozen parameter:
1. state the failure,
2. show why the current design is insufficient,
3. list affected code/components,
4. propose the minimum change,
5. propose a validation experiment.

Do not change frozen parameters because a different setting gives better test
performance.

FINAL PRINCIPLE

Build the smallest system that can scientifically test the hypothesis.

Optimize in this order:

1. CORRECTNESS
2. REPRODUCIBILITY
3. TEMPORAL VALIDITY
4. EVALUATION
5. EXPLAINABILITY
6. DEMO QUALITY
7. PERFORMANCE

Do not optimize for buzzwords or architectural complexity.
