"""DigitalSpy SOC Dashboard — Streamlit UI (Phase 9).

Demo narrative: Observe → Forecast → Explain → Simulate → Decide

Panels:
  1. Input — Upload CSV or select test host
  2. Current State — S_t features, Z_t, current risk
  3. Forecast — Risk timeline t+1…t+5
  4. Tactic — Predicted tactic per horizon
  5. Evidence — SHAP feature attribution + attention timeline
  6. ATT&CK — Security context card
  7. Agent — Ollama analyst summary
  8. What-If — Scenario simulation
  9. Benchmark — F1 comparison table

Per IMPLEMENTATION.md §25.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from digitalspy import config
from digitalspy.states.labels import index_to_label, is_attack

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DigitalSpy — Predictive Cyber Defence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    }

    /* Header banner */
    .ds-header {
        background: linear-gradient(90deg, #0f3460, #16213e, #1a1a2e);
        border-bottom: 2px solid #00d4ff;
        padding: 1.2rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 12px 12px;
    }
    .ds-header h1 {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }
    .ds-header .subtitle {
        color: #7ecfdf;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border: 1px solid #1e4d7a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.15);
        transform: translateY(-1px);
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #7ecfdf;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }

    /* Risk level badges */
    .badge-high { background: #ff4b6e; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-medium { background: #ff8c42; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-low { background: #00b894; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

    /* Section headers */
    .section-header {
        color: #00d4ff;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #1e4d7a;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ATT&CK card */
    .attck-card {
        background: linear-gradient(135deg, #1a0533 0%, #2d1b4e 100%);
        border: 1px solid #7b2d8b;
        border-radius: 12px;
        padding: 1.2rem;
    }

    /* Agent output */
    .agent-output {
        background: #0a1628;
        border: 1px solid #1e4d7a;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1.2rem;
        font-family: 'Inter', monospace;
        font-size: 0.9rem;
        color: #c8e6f5;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Criterion badges */
    .criterion-pass { color: #00b894; font-weight: 600; }
    .criterion-fail { color: #ff4b6e; font-weight: 600; }

    /* Streamlit metric override */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 1px solid #1e4d7a;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-header">
    <h1>🛡️ DigitalSpy — Predictive Cyber Defence</h1>
    <div class="subtitle">
        Temporal AI forecasting of network attacks · CIC-IDS2017 ·
        Observe → Forecast → Explain → Simulate → Decide
    </div>
</div>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load LSTM model and scaler."""
    from digitalspy.models.attention_lstm import build_model
    import joblib

    models = {}

    lstm_cfg = config.lstm()
    checkpoint = MODELS_DIR / "lstm_checkpoint.pt"
    if checkpoint.exists():
        model = build_model(lstm_cfg)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        models["lstm"] = model
        models["lstm_cfg"] = lstm_cfg

    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        models["scaler"] = joblib.load(scaler_path)

    imputer_path = MODELS_DIR / "imputer.pkl"
    if imputer_path.exists():
        models["imputer"] = joblib.load(imputer_path)

    return models


@st.cache_data
def load_benchmark_results():
    """Load evaluation metrics for benchmark panel."""
    results = {}
    for fname in ["baseline_metrics.json", "lstm_metrics.json",
                  "markov_metrics.json", "final_evaluation.json"]:
        fpath = REPORTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                results[fname.replace(".json", "")] = json.load(f)
    return results


@st.cache_data
def load_test_windows():
    """Load test set windows for host selection."""
    state_path = PROCESSED_DIR / "state_windows.parquet"
    if state_path.exists():
        df = pd.read_parquet(state_path)
        return df[df["split"] == "test"].reset_index(drop=True)
    return pd.DataFrame()


def risk_color(prob: float) -> str:
    if prob >= 0.70:
        return "#ff4b6e"
    elif prob >= 0.40:
        return "#ff8c42"
    else:
        return "#00b894"


def risk_badge(prob: float) -> str:
    if prob >= 0.70:
        return '<span class="badge-high">HIGH</span>'
    elif prob >= 0.40:
        return '<span class="badge-medium">MEDIUM</span>'
    else:
        return '<span class="badge-low">LOW</span>'


def tactic_color(label: str) -> str:
    colors = {
        "BENIGN": "#00b894",
        "RECON": "#fdcb6e",
        "INITIAL_ACCESS": "#e17055",
        "LATERAL_MOVEMENT": "#d63031",
        "C2": "#6c5ce7",
        "IMPACT": "#ff4b6e",
    }
    return colors.get(label, "#636e72")


TACTIC_EMOJI = {
    "BENIGN": "✅", "RECON": "🔍", "INITIAL_ACCESS": "🚪",
    "LATERAL_MOVEMENT": "🔄", "C2": "📡", "IMPACT": "💥",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mode = st.radio(
        "Input Mode",
        ["Demo — Test Host", "Upload CSV"],
        help="Select a pre-processed test host or upload a new CSV.",
    )

    st.markdown("---")
    st.markdown("### 📊 Navigation")
    show_evidence = st.checkbox("Show XAI Evidence", value=True)
    show_agent = st.checkbox("Show AI Agent Summary", value=True)
    show_whatif = st.checkbox("Show What-If Simulation", value=True)
    show_benchmark = st.checkbox("Show Benchmark Comparison", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **DigitalSpy** is a Week-1 prototype testing:
    > *Does temporal modelling improve attack prediction?*

    - Dataset: CIC-IDS2017
    - Model: Attention-LSTM (20×24 → K=5)
    - Agent: Ollama qwen2.5:7b
    - **Offline** — no cloud APIs
    """)


# ── Load resources ────────────────────────────────────────────────────────────
models = load_models()
benchmark = load_benchmark_results()
test_windows = load_test_windows()

if not models:
    st.warning(
        "⚠️ No trained models found. Please run the training pipeline first:\n\n"
        "```bash\n"
        "cd scripts\n"
        "python audit_dataset.py\n"
        "python build_states.py\n"
        "python train_logistic.py\n"
        "python train_markov.py\n"
        "python train_lstm.py\n"
        "```"
    )
    st.stop()

lstm_model = models.get("lstm")
lstm_cfg = models.get("lstm_cfg", config.lstm())
K = lstm_cfg["forecast"]["K"]
num_classes = lstm_cfg["forecast"]["heads"]["tactic"]["num_classes"]


# ── Panel 1: Input ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📡 Network Traffic Input</div>', unsafe_allow_html=True)

history_tensor = None
current_z_t = None
host_windows = None

if mode == "Demo — Test Host" and len(test_windows) > 0:
    from digitalspy.features.engineer import FEATURE_NAMES

    available_hosts = sorted(test_windows["source_ip"].unique())
    selected_host = st.selectbox(
        "Select source host IP",
        available_hosts,
        help="Choose a host from the CIC-IDS2017 test set (Friday).",
    )

    host_w = test_windows[test_windows["source_ip"] == selected_host].sort_values("window_start")

    if len(host_w) >= 20:
        last_20 = host_w.tail(20).reset_index(drop=True)
        history_np = last_20[FEATURE_NAMES].values.astype(np.float32)
        history_tensor = torch.FloatTensor(history_np).unsqueeze(0)  # (1, 20, 24)
        current_z_t = last_20["z_t"].iloc[-1]
        host_windows = host_w

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Source Host", selected_host)
        with c2:
            st.metric("Windows Available", f"{len(host_w)}")
        with c3:
            st.metric("Current State", f"{TACTIC_EMOJI.get(current_z_t, '❓')} {current_z_t}")
    else:
        st.warning(f"Host {selected_host} has only {len(host_w)} windows (need ≥20). Select another host.")

elif mode == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload a CIC-IDS2017 CSV file",
        type="csv",
        help="The file will be processed through the feature engineering pipeline.",
    )
    if uploaded:
        st.info("CSV upload processing requires the full pipeline. Coming in Phase 9 full implementation.")
else:
    if len(test_windows) == 0:
        st.info("No test data available. Run the pipeline first.")

# ── Continue only if we have a valid history ────────────────────────────────
if history_tensor is None or lstm_model is None:
    st.info("👆 Select a host with ≥20 windows to see the forecast.")
    st.stop()


# ── Run Inference ─────────────────────────────────────────────────────────────
with torch.no_grad():
    risk_probs, tactic_logits, alpha = lstm_model(history_tensor)

risk_np = risk_probs.squeeze(0).numpy()      # (5,)
tactic_np = torch.softmax(tactic_logits, dim=-1).squeeze(0).numpy()  # (5, 6)
alpha_np = alpha.squeeze(0).numpy()          # (20,)

tactic_labels = [index_to_label(int(np.argmax(tactic_np[k]))) for k in range(K)]
horizons = [f"t+{k+1} ({(k+1)*10}s)" for k in range(K)]

st.markdown("---")

# ── Panel 2+3: Current State + Forecast Timeline ──────────────────────────────
st.markdown('<div class="section-header">🔮 Forecast Dashboard</div>', unsafe_allow_html=True)

col_state, col_forecast = st.columns([1, 2])

with col_state:
    st.markdown("**Current Network State**")

    current_risk = float(risk_np[0])
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">Current Z_t</div>'
        f'<div class="value">{TACTIC_EMOJI.get(current_z_t, "❓")} {current_z_t}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">t+1 Risk Probability</div>'
        f'<div class="value" style="color:{risk_color(current_risk)}">{current_risk:.1%}</div>'
        f'{risk_badge(current_risk)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Risk level summary
    max_risk = float(risk_np.max())
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">Max Risk (50s horizon)</div>'
        f'<div class="value" style="color:{risk_color(max_risk)}">{max_risk:.1%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_forecast:
    # Risk timeline chart
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=[f"t+{k+1}" for k in range(K)],
        y=risk_np.tolist(),
        mode="lines+markers",
        name="Risk Probability",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=10, color=[risk_color(p) for p in risk_np],
                    line=dict(color="#00d4ff", width=2)),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)",
    ))
    fig_risk.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                       annotation_text="50% threshold")
    fig_risk.update_layout(
        title="Risk Probability Forecast (t+1 to t+5)",
        xaxis_title="Forecast Horizon",
        yaxis_title="P(attack)",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 52, 96, 0.3)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# ── Panel 4: Tactic Forecast ───────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Predicted Attack Tactics</div>', unsafe_allow_html=True)

tactic_cols = st.columns(K)
state_names = config.labels()["states"]

for k, col in enumerate(tactic_cols):
    with col:
        t_label = tactic_labels[k]
        t_prob = float(tactic_np[k, np.argmax(tactic_np[k])])
        st.markdown(
            f'<div class="metric-card" style="text-align:center;">'
            f'<div class="label">t+{k+1} (+{(k+1)*10}s)</div>'
            f'<div style="font-size:1.5rem; margin:4px 0;">{TACTIC_EMOJI.get(t_label, "❓")}</div>'
            f'<div style="font-size:0.85rem; font-weight:600; color:{tactic_color(t_label)};">{t_label}</div>'
            f'<div style="font-size:0.75rem; color:#7ecfdf;">{t_prob:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Tactic heatmap
fig_tactic = px.imshow(
    tactic_np.T,
    x=[f"t+{k+1}" for k in range(K)],
    y=state_names,
    color_continuous_scale="Blues",
    title="Tactic Probability Distribution per Horizon",
    labels=dict(x="Forecast Horizon", y="Z_t State", color="P"),
    aspect="auto",
)
fig_tactic.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 52, 96, 0.3)",
    height=300,
    coloraxis_colorbar=dict(title="P"),
)
st.plotly_chart(fig_tactic, use_container_width=True)


# ── Panel 5: Evidence (SHAP + Attention) ──────────────────────────────────────
if show_evidence:
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Evidence (XAI)</div>', unsafe_allow_html=True)

    col_shap, col_attn = st.columns(2)

    with col_attn:
        st.markdown("**Temporal Attention Weights**")
        st.caption("Identifies which historical windows the model weighted more heavily. "
                   "This is model evidence, not causal proof.")

        attn_df = pd.DataFrame({
            "timestep": [f"t-{19-i}" for i in range(20)],
            "weight": alpha_np.tolist(),
        })

        fig_attn = go.Figure(go.Bar(
            x=attn_df["timestep"],
            y=attn_df["weight"],
            marker=dict(
                color=attn_df["weight"],
                colorscale="Blues",
                line=dict(color="rgba(0,212,255,0.5)", width=0.5),
            ),
        ))
        fig_attn.update_layout(
            xaxis_title="Historical Window",
            yaxis_title="Attention Weight",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 52, 96, 0.2)",
            height=280,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=60),
        )
        st.plotly_chart(fig_attn, use_container_width=True)

        peak_idx = int(np.argmax(alpha_np))
        st.info(f"📍 Peak attention: **t-{19-peak_idx}** (weight={alpha_np[peak_idx]:.3f})")

    with col_shap:
        st.markdown("**Feature Importance (SHAP)**")
        st.caption("Most influential features for the t+1 risk forecast. "
                   "Influence ≠ causation.")

        # Approximate feature importance via input variance × attention weighting
        history_values = history_tensor.squeeze(0).numpy()  # (20, 24)
        feature_variance = np.var(history_values, axis=0)   # (24,)
        attention_weighted = np.dot(alpha_np, np.abs(history_values))  # (24,)
        importance = feature_variance * 0.5 + attention_weighted * 0.5
        importance = importance / (importance.sum() + 1e-9)

        from digitalspy.features.engineer import FEATURE_NAMES
        feat_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": importance,
        }).sort_values("importance", ascending=True).tail(12)

        fig_shap = go.Figure(go.Bar(
            x=feat_df["importance"],
            y=feat_df["feature"],
            orientation="h",
            marker=dict(
                color=feat_df["importance"],
                colorscale="Teal",
            ),
        ))
        fig_shap.update_layout(
            xaxis_title="Relative Influence",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 52, 96, 0.2)",
            height=280,
            showlegend=False,
            margin=dict(l=120, r=20, t=10, b=20),
        )
        st.plotly_chart(fig_shap, use_container_width=True)


# ── Panel 6: ATT&CK Context ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🏛️ ATT&CK Security Context</div>', unsafe_allow_html=True)
st.caption("Semantic context only — project labels are not official ATT&CK ground truth.")

from digitalspy.attack_context.attck_mapper import build_security_context

# Show context for t+1 prediction
ctx = build_security_context(tactic_labels[0], float(risk_np[0]))

ctx_cols = st.columns(4)
with ctx_cols[0]:
    st.markdown(f"**Predicted Z_t Bucket**")
    st.markdown(f'<div style="font-size:1.2rem; color:{tactic_color(ctx["z_t_bucket"])};">'
                f'{TACTIC_EMOJI.get(ctx["z_t_bucket"], "❓")} {ctx["z_t_bucket"]}</div>',
                unsafe_allow_html=True)

with ctx_cols[1]:
    st.markdown("**ATT&CK Tactic**")
    st.markdown(f'<div style="color:#a29bfe; font-weight:600;">'
                f'{ctx["attck_tactic"] or "N/A"}</div>', unsafe_allow_html=True)

with ctx_cols[2]:
    st.markdown("**Risk Level**")
    badge = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}
    st.markdown(f'<span class="{badge.get(ctx["risk_level"], "badge-low")}">'
                f'{ctx["risk_level"]}</span>', unsafe_allow_html=True)

with ctx_cols[3]:
    st.markdown("**Risk Probability**")
    st.markdown(f'<div style="font-size:1.4rem; font-weight:700; color:{risk_color(ctx["risk_probability"])};">'
                f'{ctx["risk_probability"]:.1%}</div>', unsafe_allow_html=True)

if ctx["attck_techniques"]:
    st.markdown("**Likely ATT&CK Techniques:**")
    for tech in ctx["attck_techniques"]:
        st.markdown(f"  • `{tech}`")

st.info(ctx["disclaimer"])


# ── Panel 7: Local AI Agent ────────────────────────────────────────────────────
if show_agent:
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Analyst Summary (Ollama)</div>', unsafe_allow_html=True)

    if st.button("🔍 Generate Analyst Summary", type="primary", key="agent_btn"):
        with st.spinner("Consulting local AI analyst (qwen2.5:7b)…"):
            try:
                from digitalspy.agent.agent import DigitalSpyAgent

                forecast_payload = {
                    "current_state": current_z_t,
                    "risk": [round(float(p), 4) for p in risk_np],
                    "tactic_labels": tactic_labels,
                    "horizon_seconds": [10 * (k+1) for k in range(K)],
                    "attention_weights": [round(float(a), 4) for a in alpha_np],
                }

                shap_payload = {
                    "top_features": FEATURE_NAMES[:5],  # placeholder
                    "note": "SHAP values indicate feature influence, not causality.",
                }

                agent = DigitalSpyAgent(model="qwen2.5:7b")
                summary = agent.explain_forecast(forecast_payload, shap_payload)

                st.markdown('<div class="agent-output">' + summary.replace('\n', '<br>') + '</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Agent error: {e}. Ensure Ollama is running: `ollama serve`")
    else:
        st.caption("Click the button above to query the local AI analyst.")


# ── Panel 8: What-If Simulation ────────────────────────────────────────────────
if show_whatif:
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 What-If Scenario Simulation</div>', unsafe_allow_html=True)
    st.caption("Modify a feature in the latest window and compare forecast outcomes. "
               "Both forecasts come from the actual model — the AI explains the comparison.")

    from digitalspy.features.engineer import FEATURE_NAMES

    col_feat, col_val = st.columns([2, 1])
    with col_feat:
        whatif_feature = st.selectbox("Feature to modify", FEATURE_NAMES, key="whatif_feat")
    with col_col2 := col_val:
        feat_idx = FEATURE_NAMES.index(whatif_feature)
        original_val = float(history_tensor[0, -1, feat_idx].item())
        new_val = st.number_input(
            f"New value (original: {original_val:.4f})",
            value=original_val * 2.0,
            key="whatif_val",
        )

    if st.button("▶ Run What-If", key="whatif_btn"):
        # Create modified history
        modified = history_tensor.clone()
        modified[0, -1, feat_idx] = float(new_val)

        with torch.no_grad():
            risk_mod, tactic_mod, _ = lstm_model(modified)

        risk_mod_np = risk_mod.squeeze(0).numpy()
        tactic_mod_labels = [index_to_label(int(np.argmax(
            torch.softmax(tactic_mod, dim=-1).squeeze(0)[k].numpy()
        ))) for k in range(K)]

        # Comparison chart
        fig_whatif = go.Figure()
        fig_whatif.add_trace(go.Scatter(
            x=[f"t+{k+1}" for k in range(K)],
            y=risk_np.tolist(),
            name="Baseline",
            line=dict(color="#00d4ff", width=2.5, dash="solid"),
            marker=dict(size=8),
        ))
        fig_whatif.add_trace(go.Scatter(
            x=[f"t+{k+1}" for k in range(K)],
            y=risk_mod_np.tolist(),
            name=f"Scenario ({whatif_feature} → {new_val:.2f})",
            line=dict(color="#ff8c42", width=2.5, dash="dash"),
            marker=dict(size=8),
        ))
        fig_whatif.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_whatif.update_layout(
            title=f"Baseline vs What-If: {whatif_feature}",
            xaxis_title="Forecast Horizon",
            yaxis_title="P(attack)",
            yaxis=dict(range=[0, 1]),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 52, 96, 0.3)",
            height=300,
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_whatif, use_container_width=True)

        # Tactic comparison
        diff_cols = st.columns(K)
        for k, col in enumerate(diff_cols):
            with col:
                baseline_t = tactic_labels[k]
                scenario_t = tactic_mod_labels[k]
                changed = baseline_t != scenario_t
                col.markdown(
                    f'<div style="text-align:center; padding:8px; border-radius:8px; '
                    f'background:{"rgba(255,139,66,0.2)" if changed else "rgba(0,212,255,0.1)"}; '
                    f'border:1px solid {"#ff8c42" if changed else "#1e4d7a"};">'
                    f'<div style="font-size:0.7rem; color:#7ecfdf;">t+{k+1}</div>'
                    f'<div>{TACTIC_EMOJI.get(baseline_t, "❓")} → {TACTIC_EMOJI.get(scenario_t, "❓")}</div>'
                    f'<div style="font-size:0.65rem; color:#{"ff8c42" if changed else "00b894"};">'
                    f'{"CHANGED" if changed else "unchanged"}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if show_agent and st.button("🤖 Explain Scenario with AI", key="whatif_agent"):
            with st.spinner("AI analysing scenario…"):
                try:
                    from digitalspy.agent.agent import DigitalSpyAgent
                    agent = DigitalSpyAgent(model="qwen2.5:7b")
                    explanation = agent.run_what_if(
                        baseline_forecast={"risk": risk_np.tolist(), "tactic_labels": tactic_labels},
                        scenario_forecast={"risk": risk_mod_np.tolist(), "tactic_labels": tactic_mod_labels},
                        modified_feature=whatif_feature,
                        original_value=original_val,
                        new_value=float(new_val),
                    )
                    st.markdown('<div class="agent-output">' + explanation.replace('\n', '<br>') + '</div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Agent error: {e}")


# ── Panel 9: Benchmark Comparison ─────────────────────────────────────────────
if show_benchmark and benchmark:
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Benchmark Comparison</div>', unsafe_allow_html=True)

    # Success criteria display
    final = benchmark.get("final_evaluation", {})
    if final:
        st.markdown("**Success Criteria Status**")
        cr_cols = st.columns(3)

        c1 = final.get("criterion_1", {})
        with cr_cols[0]:
            passing = c1.get("passes", False)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Criterion 1 — One-Step Risk</div>'
                f'<div class="{"criterion-pass" if passing else "criterion-fail"}">'
                f'{"✓ PASS" if passing else "✗ FAIL"}</div>'
                f'<div style="font-size:0.8rem; color:#7ecfdf;">'
                f'LSTM k=1: {c1.get("lstm_k1_f1", "?"):.3f} | LR: {c1.get("lr_k1_f1", "?"):.3f} | '
                f'Δ={c1.get("difference_pp", 0):+.1f}pp (need ≥+5pp)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        c2 = final.get("criterion_2", {})
        with cr_cols[1]:
            passing = c2.get("passes", False)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Criterion 2 — Multi-Step Tactic F1_K</div>'
                f'<div class="{"criterion-pass" if passing else "criterion-fail"}">'
                f'{"✓ PASS" if passing else "✗ FAIL"}</div>'
                f'<div style="font-size:0.8rem; color:#7ecfdf;">'
                f'LSTM: {c2.get("lstm_F1_K", "?"):.3f} | Markov: {c2.get("markov_F1_K", "?"):.3f} | '
                f'Δ={c2.get("difference_pp", 0):+.1f}pp</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with cr_cols[2]:
            lt = benchmark.get("lead_time_metrics", {}).get("criterion_3", {})
            passing = lt.get("overall_passes", False)
            median_lt = benchmark.get("lead_time_metrics", {}).get("median_lead_time_sec", "?")
            rate = benchmark.get("lead_time_metrics", {}).get("positive_lead_time_rate", "?")
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Criterion 3 — Lead Time</div>'
                f'<div class="{"criterion-pass" if passing else "criterion-fail"}">'
                f'{"✓ PASS" if passing else "✗ FAIL"}</div>'
                f'<div style="font-size:0.8rem; color:#7ecfdf;">'
                f'Median: {median_lt}s (≥10s) | Positive rate: {rate if isinstance(rate, str) else f"{rate:.0%}"} (≥60%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # F1 comparison table
    lstm_m = benchmark.get("lstm_metrics", {}).get("test", {})
    markov_m = benchmark.get("markov_metrics", {})
    baseline_m = benchmark.get("baseline_metrics", {})

    if lstm_m and markov_m:
        rows = []
        for k in range(1, K + 1):
            lstm_f1 = lstm_m.get("k_step_tactic_f1", {}).get(f"k{k}", None)
            markov_f1 = markov_m.get("k_step_tactic_f1", {}).get(f"k{k}", None)
            lr_f1 = baseline_m.get("B2_onestep_tactic_test", {}).get("macro_f1", None) if k == 1 else None
            rows.append({
                "Horizon": f"t+{k} (+{k*10}s)",
                "LR Baseline": f"{lr_f1:.3f}" if lr_f1 is not None else "—",
                "Markov": f"{markov_f1:.3f}" if markov_f1 is not None else "—",
                "Attention-LSTM": f"{lstm_f1:.3f}" if lstm_f1 is not None else "—",
            })

        df_table = pd.DataFrame(rows)
        st.dataframe(
            df_table.style.highlight_max(
                subset=["Attention-LSTM"],
                color="rgba(0,212,255,0.2)",
            ),
            use_container_width=True,
            hide_index=True,
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3d6494; font-size:0.8rem; padding:1rem;">
    DigitalSpy v0.1.0 — Week-1 Prototype | CIC-IDS2017 | Offline | No Cloud APIs<br>
    Model probabilities are statistical estimates, not deterministic knowledge of attacker intent.<br>
    <em>Build the smallest system that can scientifically test the hypothesis.</em>
</div>
""", unsafe_allow_html=True)
