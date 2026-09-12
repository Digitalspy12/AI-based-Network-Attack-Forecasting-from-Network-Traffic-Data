# DigitalSpy — Predictive Cyber Defence

> **Research Hypothesis:** Does temporal modelling of network behaviour provide better and earlier prediction of future malicious behaviour than a static feature-based classifier?

Week-1 prototype using CIC-IDS2017. **Fully offline. No cloud APIs.**

## Architecture

```
CSV (CIC-IDS2017)
    ↓
Feature Engineering (24 features, 10s windows, per-source-host)
    ↓
S_t ∈ ℝ²⁴  +  Z_t ∈ {BENIGN, RECON, INITIAL_ACCESS, LATERAL_MOVEMENT, C2, IMPACT}
    ↓                    ↓
Logistic Regression   Markov World Model P(Z_{t+k}|Z_t)
 (baseline)               ↓
    ↓              Attention-LSTM
    ↓              20×24 → K=5 forecast
    ↓                    ↓
         ATT&CK Context + SHAP + Attention Evidence
                         ↓
              Local Ollama Agent (qwen2.5:7b)
                         ↓
               Streamlit SOC Dashboard
```

## Setting Up on a New Device

### 1. Prerequisites
- Python 3.10 or higher
- Git
- At least 8GB of RAM (16GB recommended for data processing)
- (Optional but recommended) NVIDIA GPU for faster LSTM training

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd "AI-based-Network-Attack-Forecasting-from-Network-Traffic-Data/DigitalSpy"

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install core dependencies and development tools
pip install -e ".[dev]"

# Install SHAP XAI dependencies (TensorFlow required for DeepExplainer compat)
pip install tensorflow-cpu shap matplotlib seaborn
```

### 3. Prepare the Dataset
DigitalSpy expects the CIC-IDS2017 dataset (CSVs) to be located in the parent directory under `MachineLearningCVE/`.
If you only want to test the Streamlit Demo, you just need a single valid CSV or PCAP to upload.

To run the full pipeline, ensure the dataset is structured as follows:
```bash
ls ../MachineLearningCVE/*.csv
# Expected: 8 CSV files (Mon–Fri) from CIC-IDS2017
```

### 3. Run the pipeline (phase by phase)

```bash
# Phase 1: Audit
python scripts/audit_dataset.py

# Phase 2: Build state windows
python scripts/build_states.py

# Phase 3: Logistic Regression baselines
python scripts/train_logistic.py

# Phase 4: Markov World Model
python scripts/train_markov.py

# Phase 5: Attention-LSTM (GPU recommended)
python scripts/train_lstm.py

# Phase 10: Final evaluation
python scripts/evaluate_all.py
python scripts/evaluate_lead_time.py
```

### 5. Launch the Streamlit SOC Dashboard (Demo)

Once models are trained (or if you already have the pre-trained checkpoints in `models/`), you can launch the interactive dashboard:

```bash
streamlit run src/digitalspy/ui/app.py
```

**Using the Demo UI:**
1. **Demo — Test Sequence Mode:** Automatically loads a 20-window sequence from the processed test dataset. Use the slider to pick a different sequence segment.
2. **Upload File Mode:** Upload your own CIC-IDS2017 CSV or PCAP file. 
   - CSVs will be processed dynamically to extract the 24 flow features.
   - PCAPs will be streamed (up to 100k packets limit) to extract packet-level features and fused dynamically.
3. Toggle XAI Evidence to see SHAP feature importance and temporal attention heatmaps.

### 5. Run unit tests

```bash
pytest tests/ -v
```

## Frozen Parameters (Do NOT change without §30 Change Control)

| Parameter | Value |
|---|---|
| Feature dimension | 24 |
| Window size | 10 seconds |
| History length h | 20 windows |
| Forecast horizon K | 5 steps (50 seconds) |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| Dropout | 0.20 |
| Attention | Additive (Bahdanau) |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 64 |
| Tactic states | 6 (BENIGN → IMPACT) |

## Success Criteria (Pre-fixed)

| Criterion | Target |
|---|---|
| 1 — One-step risk | LSTM ≥ LR + 5pp macro-F1, FPR increase ≤ 2pp |
| 2 — Multi-step tactic F1_K | LSTM ≥ Markov + 5pp |
| 3 — Forecast lead time | median ≥ 10s, ≥60% positive |

## Repository Structure

```
configs/       ← Frozen YAML configuration (system, features, labels, splits, markov, lstm)
data/          ← raw/ interim/ processed/
models/        ← Saved checkpoints and scalers
reports/       ← Audit, metrics, lead-time JSON reports
scripts/       ← Runnable pipeline phases (audit → build → train → evaluate)
src/digitalspy/
  data/        ← CIC-IDS2017 loader
  features/    ← 24-feature engineering
  states/      ← Windowing + label precedence
  baselines/   ← Logistic Regression
  markov/      ← 6×6 transition model
  models/      ← Attention-LSTM + trainer
  forecasting/ ← Forecast engine (direct multi-horizon)
  explainability/ ← SHAP + attention
  attack_context/ ← ATT&CK mapping
  agent/       ← Ollama analyst agent
  ui/          ← Streamlit SOC dashboard
tests/         ← Unit tests (§28)
```

## Limitations (§31)

- Source IP used as host identity (NAT/DHCP limitation)
- Global pooled Markov matrix (not per-host)
- CIC-IDS2017 is a benchmark dataset, not real enterprise traffic
- Six tactic buckets are project-defined, not official ATT&CK ground truth
- No PCAP files available; packet-derived features imputed from training means
- 50-second forecast horizon is a prototype configuration

## Phase-2 Roadmap

TGN/TGNN → Mamba/SSM → Next attacker–asset link prediction → Counterfactual transitions → Cross-dataset evaluation
