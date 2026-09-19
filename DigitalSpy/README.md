# DigitalSpy — Predictive Cyber Defence

> **Research Hypothesis:** Does temporal modelling of network behaviour provide better and earlier prediction of future malicious behaviour than a static feature-based classifier?

DigitalSpy is a Week-1 prototype for temporal network attack forecasting using the CIC-IDS2017 dataset. **Fully offline. No cloud APIs.**

---

## 🏗️ Architecture

```text
CSV (CIC-IDS2017) / PCAP Telemetry
    ↓
Feature Engineering (24 flow features + 12 packet features, 10s windows, per-source-host)
    ↓
S_t ∈ ℝ²⁴  +  Z_t ∈ {BENIGN, RECON, INITIAL_ACCESS, LATERAL_MOVEMENT, C2, IMPACT}
    ↓                    ↓
Logistic Regression   Markov World Model P(Z_{t+k}|Z_t)
 (baseline)               ↓
    ↓              Attention-LSTM
    ↓              20×24 → K=5 forecast (50s horizon)
    ↓                    ↓
         ATT&CK Context + SHAP + Attention Evidence
                         ↓
              Local Ollama Agent (qwen2.5:7b)
                         ↓
               Streamlit SOC Dashboard
```

---

## 💻 Setting Up on a Completely New Laptop

Follow these step-by-step instructions to set up and run DigitalSpy on a fresh machine (Linux, macOS, or Windows WSL2).

### 1. Prerequisites
- **Python**: 3.10, 3.11, or 3.12 (`python3 --version`)
- **Git**: Installed and configured (`git --version`)
- **System Memory**: At least 8 GB of RAM (16 GB recommended for full raw dataset processing)
- **Optional**: NVIDIA GPU for faster Attention-LSTM training (CPU is sufficient for demo replay)

---

### 2. Clone Repository & Setup Virtual Environment

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd "AI-bassed Network attack Forecast"

# 2. Enter the DigitalSpy application directory
cd DigitalSpy

# 3. Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows WSL2 / Git Bash: source venv/bin/activate
                          # On Windows Command Prompt: venv\Scripts\activate.bat
                          # On Windows PowerShell: venv\Scripts\Activate.ps1

# 4. Upgrade pip and install core dependencies in editable mode
pip install --upgrade pip
pip install -e ".[dev]"

# 5. Install PCAP parsing dependencies (for live PCAP upload features)
pip install scapy dpkt
```

---

### 3. Fast-Track Demo Launch (No Dataset Download Required)

DigitalSpy includes lightweight pre-trained model checkpoints (`lstm_checkpoint.pt`, `scaler.pkl`, `imputer.pkl`, `markov_matrix.npy` ~600 KB total) in the repository.

You can immediately launch the interactive Streamlit SOC Dashboard on a new laptop:

```bash
# From inside the DigitalSpy/ directory with virtual environment activated:
streamlit run src/digitalspy/ui/app.py
```

* Open your browser at `http://localhost:8501`.
* **Demo — Test Sequence Mode:** Replays temporal sequences from pre-processed test data. Use the timeline slider to navigate sequence windows.
* **Upload File Mode:** Upload your own custom CIC-IDS2017 CSV or `.pcap` file for real-time feature extraction and risk forecasting.

---

### 4. Full Pipeline Execution (From Raw Dataset)

If you wish to re-process the raw dataset and re-train all models from scratch:

#### Step 4a: Place Raw Data
Download the CIC-IDS2017 CSV files (~2.3 GB) and place them in the `MachineLearningCVE/` folder at the repository root:
```text
AI-bassed Network attack Forecast/
├── MachineLearningCVE/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── DigitalSpy/
```

#### Step 4b: Execute Pipeline Phases
From inside the `DigitalSpy/` directory:

```bash
# Phase 1: Audit raw dataset schemas and verify labels
python scripts/audit_dataset.py

# Phase 2: Build 10-second host windows and feature vectors
python scripts/build_states.py

# Phase 3: Train Logistic Regression baselines
python scripts/train_logistic.py

# Phase 4: Train Markov World Model transition matrix
python scripts/train_markov.py

# Phase 5: Train Attention-LSTM neural network
python scripts/train_lstm.py

# Phase 6: Run comprehensive benchmark evaluation & lead-time analysis
python scripts/evaluate_all.py
python scripts/evaluate_lead_time.py
```

---

### 5. Running Tests & DOM Verification

```bash
# Run unit test suite:
pytest tests/ -v

# Run Playwright UI DOM verification suite (requires Streamlit running on port 8501):
python src/scripts/full_ui_dom_verification.py
```

---

## 📁 Repository Structure & Git Rules (`.gitignore`)

### What is Tracked in Git
- 📂 `src/digitalspy/` — Source code (models, feature engineering, windowing, UI, explainability)
- 📂 `configs/` — Frozen YAML configurations (`system.yaml`, `lstm.yaml`, `features.yaml`, `labels.yaml`, `splits.yaml`)
- 📂 `scripts/` & `src/scripts/` — Pipeline execution and DOM verification scripts
- 📂 `tests/` — PyTest suite
- 📄 `models/` — Lightweight pre-trained checkpoints (`lstm_checkpoint.pt`, `scaler.pkl`, `imputer.pkl`, `markov_matrix.npy` ~600 KB total)
- 📄 `pyproject.toml`, `README.md`, `design.md` — Project configuration and documentation

### What is Ignored by `.gitignore` (Cannot / Should Not Be Committed)
- 🚫 `MachineLearningCVE/` & `*.csv`, `*.zip` — Raw 2.3 GB dataset files (too large for Git)
- 🚫 `Tuesday-WorkingHours.pcap` & `*.pcap` — Raw 11 GB packet captures
- 🚫 `data/` & `*.parquet` — Intermediate generated window datasets (~30 MB)
- 🚫 `.venv/`, `venv/`, `env/`, `.env` — Local Python virtual environments & secrets
- 🚫 `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.DS_Store` — Bytecode & local test caches

---

## ⚙️ Frozen Parameters (§30 Change Control)

| Parameter | Value |
|---|---|
| Feature dimension | 24 (flow) / 36 (fused packet) |
| Window size | 10 seconds |
| History length $h$ | 20 windows (200s) |
| Forecast horizon $K$ | 5 steps (50 seconds) |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| Dropout | 0.20 |
| Attention mechanism | Additive (Bahdanau) |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 64 |
| Tactic states | 6 (BENIGN → IMPACT) |

---

## 🎯 Evaluation Success Criteria

| Criterion | Target | Status |
|---|---|---|
| **Criterion 1 — One-Step Risk** | LSTM $\ge$ LR + 5pp macro-F1 | **PASS** (+6.07pp, LSTM 0.837 vs LR 0.776) |
| **Criterion 2 — Multi-Step Tactic F1_K** | LSTM $\ge$ Markov + 5pp | **FAIL / RESEARCH RESULT** (Scenario persistence effect) |
| **Criterion 3 — Forecast Lead Time** | Median $\ge$ 10s, PDR $\ge$ 60% | **PARTIAL** (+30s median lead, +40s verified on sequence 4381) |

---

## 📜 Limitations (§31)

- Source IP used as host identity (NAT/DHCP limitation).
- Global pooled Markov matrix (not per-host).
- CIC-IDS2017 is a benchmark dataset, not live enterprise telemetry.
- Six tactic buckets are project-defined labels, not official MITRE ATT&CK ground truth.
- 50-second forecast horizon is a prototype configuration.

---

## 📄 License & Attribution

Developed for predictive cyber defence research. Includes components adapted for offline research with CIC-IDS2017.
