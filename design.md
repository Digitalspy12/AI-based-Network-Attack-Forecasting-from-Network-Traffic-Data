# DigitalSpy Dashboard — `design.md`

## 1. Purpose

Design a professional, offline-first cybersecurity dashboard for **DigitalSpy — Predictive Cyber Defence**.

Core story:

> **Observe → Forecast → Explain → Simulate → Decide**

This is a scientific demonstration interface for temporal network-attack forecasting, not a generic SOC dashboard.

The UI must clearly separate:
- **Observed now**
- **ML forecast**
- **Analytical/XAI evidence**
- **ATT&CK semantic context**
- **Actually observed later**
- **What-if simulation**
- **Benchmark/evaluation results**

Never visually imply that a forecast is a confirmed attack.

---

## 2. Product Identity

**Product:** DigitalSpy  
**Subtitle:** Predictive Cyber Defence  
**Description:** Temporal AI forecasting of network attacks  
**Dataset:** CIC-IDS2017  
**Model:** Attention-LSTM  
**Input:** 20 historical windows × 24 features  
**Forecast:** 5 horizons × 10 seconds = 50 seconds  
**Mode:** Offline / no cloud APIs  
**Optional agent:** Local Ollama agent for explanation/orchestration only

---

## 3. Visual Direction

Use a **dark enterprise cybersecurity / intelligence interface**.

### Style
- Deep navy/blue-black background.
- Dark blue panels.
- Cyan/blue analytical accents.
- Red/orange only for elevated risk or attack states.
- Green for benign/success/verified.
- Thin borders and subtle glow.
- High information density without clutter.
- Professional enough for a government/cybersecurity presentation.

Avoid:
- Excessive neon.
- Gaming/HUD aesthetics.
- Giant decorative graphics.
- Fake terminal text.
- Excessive rounded cards.
- Decorative content that does not support the scientific story.

Use a modern sans-serif font with clear hierarchy:
1. Product title
2. Section title
3. Large metric
4. Supporting label
5. Scientific disclaimer

---

# 4. Application Shell

Desktop-first.

## Header

Left:
- Shield icon
- `DigitalSpy`
- `Predictive Cyber Defence`

Center:
- `Temporal AI forecasting of network attacks`
- `Observe → Forecast → Explain → Simulate → Decide`

Right:
- `Offline Mode`
- Dataset/version
- Application version
- Settings/help

Keep the header visually stable while navigating.

---

# 5. Left Sidebar

Width: approximately 210–240 px.

## Configuration

### Input Mode
- `Demo — Test Sequence`
- `Upload File (CSV / PCAP)`

Demo mode can show:
- Dataset
- Scenario/day
- Window selector

Upload mode:
- CSV/PCAP uploader
- Validation status
- Feature/packet coverage

## Navigation
1. Overview
2. Forecast Dashboard
3. ATT&CK Analysis
4. Explainability (XAI)
5. What-If Simulation
6. Benchmark Comparison
7. Data Information

## About

Show:
- DigitalSpy is a Week-1 prototype.
- Dataset: CIC-IDS2017.
- Model: Attention-LSTM (20×24 → K=5).
- Agent: Ollama, optional.
- Offline — no cloud APIs.

Do not describe the prototype as an autonomous defence system.

---

# 6. Main Overview Dashboard

The Overview is the primary demo screen.

Recommended layout:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Header                                                               │
├──────────────┬───────────────────────────────────────────────────────┤
│ Sidebar      │ Network Traffic Input / Timeline                    │
│              ├───────────────────────────────────────────────────────┤
│              │ Current State | T+1 Risk | Max Risk | Forecast State │
│              ├──────────────────────────┬────────────────────────────┤
│              │ Forecast Risk Chart      │ Security Context ATT&CK    │
│              ├──────────────────────────┼────────────────────────────┤
│              │ Raw State Forecast       │                            │
│              ├──────────────────────────┴────────────────────────────┤
│              │ Forecast → Observed Outcome Timeline                  │
│              ├──────────────────────┬────────────────────────────────┤
│              │ SHAP                 │ Temporal Attention             │
│              ├──────────────────────┼────────────────────────────────┤
│              │ Packet Telemetry     │ What-If Simulation             │
│              ├──────────────────────┴────────────────────────────────┤
│              │ Benchmark Comparison                                  │
└──────────────┴───────────────────────────────────────────────────────┘
```

---

# 7. Network Traffic Input

Top of main content.

Display:
- Dataset name
- Scenario/day
- Selected window
- Windows available
- Time per window
- Timeline/slider

Example:

`CIC-IDS2017 · Friday-WorkingHours`

`Window 4381 / 11287`

`10 seconds / window`

The selected window represents **NOW**.

Clearly distinguish:
- Historical input
- Current observed state
- Future forecast
- Later observed outcome

---

# 8. Current Network State Card

Title:

**Current Network State (Observed)**

Examples:

`✓ BENIGN`

or

`⚡ IMPACT`

Include a short explanation.

This is the observed state for the selected window.

Never label a forecast as observed.

---

# 9. T+1 Risk Card

Title:

**T+1 Risk Probability**

Display:
- Large percentage
- Risk level
- Threshold comparison

Example:

`90.7%`

`HIGH`

`Exceeds 90% operational threshold`

The numerical probability must come directly from the ML model.

The LLM must never generate or alter it.

---

# 10. Max Risk Card

Title:

**Max Risk (Next 50s)**

Show:
- Maximum predicted risk
- Horizon where it occurs

Example:

`93.4%`

`Peak risk at t+2 (+20s)`

This is a forecast summary, not a confirmed future event.

---

# 11. Operational Forecast State

Title:

**Operational Forecast State**

Example:

`⚡ IMPACT`

Subtitle:

`Forecasted attack state (t+1)`

Distinction:
- Current Network State = observed now.
- Operational Forecast State = predicted future state.

Never call the forecast the current attack state.

---

# 12. Main Forecast Chart

Title:

**Forecasted Risk — Next 50 Seconds**

X-axis:
- t+1 (+10s)
- t+2 (+20s)
- t+3 (+30s)
- t+4 (+40s)
- t+5 (+50s)

Y-axis:
`P(attack)`

Show:
1. Predicted risk
2. Operational threshold
3. Observed/future outcome marker when available
4. NOW marker
5. Actual outcome marker after replay reaches attack onset

Use clear legend labels.

Example annotations:

`NOW — BENIGN`

`Actual outcome — IMPACT`

The chart must make the temporal relationship obvious.

---

# 13. Operational Threshold

Use the configured evaluation threshold.

Example UI label:

`90% operational threshold`

Do not present this as a universal cybersecurity standard.

It means:

> The model crossed the configured alert threshold.

It does not mean:

> An attack is guaranteed.

---

# 14. Raw State Forecast Heatmap

Title:

**Raw State Forecast — Probability by Horizon**

Rows:
- BENIGN
- RECON
- INITIAL_ACCESS
- LATERAL_MOVEMENT
- C2
- IMPACT

Columns:
- t+1
- t+2
- t+3
- t+4
- t+5

Show probability values.

Subtitle:

> Project-defined network-state buckets. Analytical distribution, not official ATT&CK ground truth.

Do not confuse these project buckets with official MITRE ATT&CK tactics.

---

# 15. Security Context / ATT&CK Panel

Title:

**Security Context (ATT&CK)**

Show:
- Forecast status
- Risk probability
- Predicted project state
- Confidence, if legitimately computed
- ATT&CK tactic
- Likely technique

Example:

`EARLY WARNING`  
`90.7%`  
`RECON`  
`Reconnaissance`  
`T1046 — Network Service Scanning`

Disclaimer:

> ATT&CK mapping provides semantic security context; it is not ground-truth ATT&CK classification.

Important:
- Tactics describe adversary objectives/behavior.
- Techniques describe how activity is performed.
- Do not display tactics as guaranteed sequential model states.

---

# 16. Forecast → Observed Outcome Timeline

This is one of the most important demo components.

Title:

**Forecast → Observed Outcome Timeline**

Example:

```text
Window 4381                    Window 4405
(BENIGN)                       (IMPACT)
    ●────────────────────────────●
          Lead Time: +40s
```

Left:
`AI Early Warning`

`90.7% risk predicted`

`t+1 (+10s)`

Right:
`Actual Attack Observed`

`State changed to IMPACT`

`+40.0 seconds`

Only show this panel when the future observed outcome is actually available.

Never manufacture an outcome from the forecast.

---

# 17. Explainability — SHAP

Title:

**Feature Importance (SHAP)**

Subtitle:

> Most influential features for the selected forecast. Influence ≠ causation.

Use a horizontal bar chart.

Possible features:
- mean_iat
- std_iat
- max_iat
- psh_ratio
- packets_per_sec
- bytes_per_sec
- flow_count
- syn_ratio

Correct terminology:

`Model feature attribution`

Do not label SHAP as root cause or causal proof.

---

# 18. Temporal Attention

Title:

**Temporal Attention Weights**

X-axis:
Historical windows

Example:
`t-19 ... t`

Y-axis:
`Attention Weight`

Subtitle:

> Attention indicates which historical windows influenced the forecast; it is not causal evidence.

Do not claim that a high-attention window is necessarily the attack trigger.

---

# 19. Packet-Level Telemetry

Title:

**Packet-Level Telemetry (from PCAP)**

This panel demonstrates packet-level compliance.

Show:
- TTL variance
- TCP window standard deviation
- Fragmentation rate
- Payload-size statistics
- Retransmission indicator, where implemented
- Port-scan signature, where implemented
- PCAP coverage
- Parser status

Example:

```text
TTL Variance             14.2
TCP Window Std           1820
Fragmentation Rate       0.7%
Payload Size Std         341 bytes
Coverage                 100%
Source                   PCAP
Status                   ✓ Parsed
```

All values must be derived from the actual PCAP parser.

Do not use placeholder values in the final scientific demo.

---

# 20. Flow + Packet Fusion

Communicate both feature families.

### Flow-level
- IPs
- Ports
- Protocol
- Bytes
- Packets
- Duration
- Timing
- TCP flags
- Directionality

### Packet-level
- TTL
- TCP window
- Fragmentation
- Payload size
- Retransmission
- Packet signatures

Visual:

```text
Flow Features ─────┐
                   ├──> Temporal State S_t ──> Attention-LSTM
Packet Features ───┘
```

The dashboard should make the architecture understandable without exposing implementation complexity.

---

# 21. What-If Scenario Simulation

Title:

**What-If Scenario Simulation**

Purpose:

Compare:

`Baseline H_t → Forecast`

versus

`Modified H'_t → Forecast`

Example:

Feature:
`flow_count`

Original:
`20`

Modified:
`40`

Button:

`▶ Run What-If`

Show two forecast curves.

Rules:
- Both curves come from the actual model.
- The LLM may explain the difference.
- The LLM must not generate the numerical probabilities.
- Do not describe this as a real attack simulation.

Use label:

`Model-based scenario simulation`

---

# 22. Benchmark Comparison

Title:

**Benchmark Comparison**

Show scientific evaluation.

### Criterion 1 — One-Step Risk
Example:
`PASS`
`LSTM F1: 0.837`
`LR F1: 0.776`
`Δ = +6.07pp`

### Criterion 2 — Multi-Step Tactic F1K
Example:
`FAIL / RESEARCH RESULT`
`LSTM: 0.277`
`Markov: 0.992`

Explain that this result is scenario-limited and must not be hidden.

### Criterion 3 — Proactive Lead
Example:
`PARTIAL`
`Median proactive lead: +30s`
`PDR: 21.7%`

Never display PASS unless the frozen success criterion is actually satisfied.

---

# 23. Scientific Honesty Rules

Always distinguish:

### Observed
What exists in the dataset at the selected time.

### Predicted
What the ML model estimates for future horizons.

### Analytical
SHAP, attention, distributions, metrics.

### Semantic
ATT&CK mapping.

### Simulated
What-if model output.

### Evaluated
Benchmark results.

Never merge these into one generic “AI decision.”

---

# 24. Demo Replay Design

The primary demonstration should be deterministic.

### Step 1
Select a known benign window.

Display:

`NOW — BENIGN`

### Step 2
Generate t+1 through t+5 forecasts.

Display:
- Risk curve
- State probabilities
- ATT&CK context
- XAI

### Step 3
Show threshold crossing.

Example:

`90.7% HIGH`

`EARLY WARNING`

### Step 4
Advance the timeline.

### Step 5
At actual attack onset:

`ACTUAL OUTCOME OBSERVED`

### Step 6
Show:

`Lead Time: +40 seconds`

This provides the clearest visual evidence for the forecasting story.

---

# 25. Data Information Page

Show:
- Dataset name
- File names
- Number of windows
- Attack windows
- Benign windows
- Feature count
- Window size
- History length
- Forecast horizon
- Train/validation/test methodology
- Packet coverage
- Missing-data handling

Limitations:
- CIC-IDS2017 is a benchmark dataset, not modern CII telemetry.
- Source IP is used as host identity and may not represent a real asset under NAT/DHCP.
- Some attack classes are highly sparse.
- Six state buckets are project-defined engineering labels.

---

# 26. ATT&CK Analysis Page

Provide:
- Forecasted project state
- Mapped ATT&CK tactic
- Mapped technique
- Probability
- Evidence
- Confidence, if valid

Allow inspection across horizons.

Always show:

`Mapping is semantic context, not model ground truth.`

---

# 27. Explainability Page

Show:
1. SHAP feature attribution
2. Temporal attention
3. Input feature snapshot
4. Forecast probability
5. Forecast horizon
6. Actual outcome, if known

Prefer three documented cases:
- Proactive true positive
- False positive
- Missed attack

This demonstrates evaluation rather than only successful examples.

---

# 28. Responsive Behaviour

Primary targets:
- 1440×900
- 1920×1080

Minimum usable:
- 1280 px wide

For smaller screens:
- Stack cards.
- Resize charts.
- Keep critical metrics visible.
- Never squeeze charts until labels become unreadable.

---

# 29. Accessibility

Use:
- Sufficient contrast.
- Icons plus text, not color alone.
- Readable chart labels.
- Clear observed/predicted distinction.
- Tooltips for technical terms.
- Keyboard-accessible controls.

Never rely only on color to communicate state.

---

# 30. Performance

The dashboard should **not train the model during normal demo interaction**.

Recommended:

```text
Precomputed model artifacts
        ↓
Precomputed/loaded feature windows
        ↓
Fast inference
        ↓
Dashboard rendering
```

Use caching.

PCAP parsing should happen during preprocessing or an explicitly controlled upload pipeline, not repeatedly on every Streamlit rerun.

---

# 31. AI Agent Rules

The optional local AI agent is an explanation/orchestration layer.

Allowed:
- Summarize forecast.
- Explain SHAP.
- Explain attention.
- Retrieve host history.
- Look up ATT&CK context.
- Explain what-if changes.
- Generate incident summary.

Not allowed:
- Invent probability values.
- Modify model outputs.
- Claim an attack is confirmed when only predicted.
- Execute destructive actions.
- Make autonomous containment decisions.

The numerical ML model remains the source of truth.

---

# 32. Empty / Loading / Error States

### Loading
`Loading forecast...`

### No sequence
`Select a host/window with at least 20 historical windows.`

### Missing packet data
`Packet telemetry unavailable for this window.`

### Partial packet coverage
`Packet features available for 78% of relevant windows.`

### Parser failure
`PCAP parsing failed — flow-only forecast remains available.`

### No future outcome
`Observed outcome not yet available for this window.`

Never replace missing evidence with fabricated data.

---

# 33. Color Semantics

### Cyan / Blue
Information, forecast, analytical content.

### Green
BENIGN / verified / successful parsing.

### Orange
Warning / medium risk / partial research result.

### Red
High risk / observed attack / failed criterion.

### Purple
ATT&CK semantic context.

Keep semantics consistent across the application.

---

# 34. Final Design Principle

The dashboard should answer five questions immediately:

1. **What is happening now?**
2. **What does the model predict next?**
3. **Why did the model make that prediction?**
4. **What happens if conditions change?**
5. **Did the forecast match the later observed outcome?**

Visual hierarchy:

```text
CURRENT STATE
      ↓
FUTURE RISK
      ↓
FORECASTED STATE
      ↓
WHY
      ↓
OBSERVED OUTCOME
      ↓
SIMULATION
      ↓
BENCHMARK / EVIDENCE
```

The final interface should look like a **scientific predictive-cybersecurity instrument**, not a generic dashboard and not a fictional autonomous SOC.

---

# 35. Definition of Done

- [ ] Current observed state is clearly separated from prediction.
- [ ] 5-horizon risk forecast is visible.
- [ ] Operational threshold is clearly labelled.
- [ ] Raw state probabilities are visible.
- [ ] ATT&CK mapping is explicitly semantic context.
- [ ] Forecast → observed outcome is demonstrated with real future labels.
- [ ] SHAP evidence is visible.
- [ ] Temporal attention is visible.
- [ ] Real PCAP-derived packet telemetry is visible.
- [ ] Flow + packet fusion is traceable.
- [ ] What-if uses the actual model.
- [ ] Benchmark results show PASS/FAIL/PARTIAL honestly.
- [ ] Demo runs offline without training.
- [ ] No LLM-generated numerical claims are presented as model output.
