import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

# Ensure we can import from digitalspy
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model
from digitalspy.features.engineer import FEATURE_NAMES

def index_to_label(idx):
    states = config.labels()["states"]
    return states[idx] if idx < len(states) else "UNKNOWN"

def main():
    # Load configuration
    lstm_cfg = config.lstm()
    sys_cfg = config.system()
    K = lstm_cfg["forecast"]["K"]
    model_input_size = lstm_cfg["architecture"]["input_size"]
    
    # Frozen validation threshold
    FROZEN_THRESHOLD = 0.90 

    # Load test windows
    processed_dir = config.resolve_path("processed_data")
    state_path = processed_dir / "state_windows.parquet"
    if not state_path.exists():
        print(f"Error: {state_path} does not exist.")
        return

    df = pd.read_parquet(state_path)
    test_windows = df[df["split"] == "test"].reset_index(drop=True)
    if test_windows.empty:
        print("Error: No test windows found.")
        return

    # Load model
    models_dir = config.resolve_path("models")
    checkpoint = models_dir / "lstm_checkpoint.pt"
    if not checkpoint.exists():
        print(f"Error: {checkpoint} does not exist.")
        return

    model = build_model(lstm_cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    print(f"Loaded {len(test_windows)} test windows.")
    max_idx = len(test_windows) - 20 - K

    candidates = []

    for start_idx in range(0, max_idx, 1):
        # The history is from start_idx to start_idx + 20
        last_20 = test_windows.iloc[start_idx : start_idx + 20]
        current_z_t = last_20["z_t"].iloc[-1]
        
        if current_z_t != "BENIGN":
            continue

        # Look ahead up to 50s (K horizons) to see if an attack happens
        future_windows = test_windows.iloc[start_idx + 20 : start_idx + 20 + K]
        future_states = future_windows["z_t"].values
        
        attack_indices = [i for i, s in enumerate(future_states) if s != "BENIGN"]
        if not attack_indices:
            continue
            
        onset_offset = attack_indices[0] 
        attack_onset_window = start_idx + 20 + onset_offset
        attack_onset_state = future_states[onset_offset]

        # Prepare input tensor
        history_np = last_20[FEATURE_NAMES].values.astype(np.float32)
        if history_np.shape[-1] != model_input_size:
             history_np = np.pad(history_np, ((0,0), (0, max(0, model_input_size - history_np.shape[-1]))))[:, :model_input_size]
        history_tensor = torch.FloatTensor(history_np).unsqueeze(0)

        with torch.no_grad():
            risk_probs, tactic_logits, _ = model(history_tensor)
        
        risk_np = risk_probs.squeeze(0).numpy()
        tactic_np = torch.softmax(tactic_logits, dim=-1).squeeze(0).numpy()
        tactic_labels = [index_to_label(int(np.argmax(tactic_np[k]))) for k in range(K)]
        
        max_risk = float(risk_np.max())
        
        # Check if the risk crosses the threshold before onset
        # The horizons are +10s, +20s, ... up to onset_offset*10s + 10s
        first_threshold_idx = -1
        for i in range(K):
            if risk_np[i] >= FROZEN_THRESHOLD:
                first_threshold_idx = i
                break
        
        if first_threshold_idx != -1 and first_threshold_idx <= onset_offset:
            # Positive proactive lead!
            lead_time = (onset_offset - first_threshold_idx) * 10
        elif first_threshold_idx != -1 and first_threshold_idx > onset_offset:
            # Crossed threshold but too late (reactive)
            lead_time = -1
        else:
            # Never crossed threshold (missed, but maybe rising?)
            lead_time = -1
            
        # We also want to accept candidates where the risk increases significantly towards onset 
        # even if it doesn't cross 0.9, just to show the forecast mechanic if 0.9 is never crossed proactively.
        # But per Phase 1 docs, rank primarily by positive lead. Let's record all that have positive lead OR 
        # measurable rising risk toward the onset.
        
        if lead_time >= 0 or (max_risk > 0.4 and risk_np[0] < 0.6):
            candidates.append({
                "candidate_id": f"seq_{start_idx}",
                "start_idx": start_idx,
                "current_z_t": current_z_t,
                "attack_onset_window": int(attack_onset_window),
                "attack_onset_time": (onset_offset + 1) * 10,
                "attack_onset_state": attack_onset_state,
                "risk_t1": float(risk_np[0]),
                "risk_t2": float(risk_np[1]),
                "risk_t3": float(risk_np[2]),
                "risk_t4": float(risk_np[3]),
                "risk_t5": float(risk_np[4]),
                "max_risk": max_risk,
                "risk_delta": max_risk - float(risk_np[0]),
                "forecast_threshold": FROZEN_THRESHOLD,
                "first_threshold_crossing": (first_threshold_idx + 1) * 10 if first_threshold_idx != -1 else -1,
                "proactive_lead_seconds": lead_time,
                "predicted_state_t1": tactic_labels[0],
                "predicted_state_t2": tactic_labels[1],
                "predicted_state_t3": tactic_labels[2],
                "predicted_state_t4": tactic_labels[3],
                "predicted_state_t5": tactic_labels[4],
                "actual_state_t1": future_states[0],
                "actual_state_t2": future_states[1],
                "actual_state_t3": future_states[2],
                "actual_state_t4": future_states[3],
                "actual_state_t5": future_states[4],
            })

    print(f"Found {len(candidates)} candidates.")
    
    if not candidates:
        print("No candidates found with rising risk.")
        return

    df_cands = pd.DataFrame(candidates)
    # Ranking logic:
    # 1. Positive proactive lead (sort descending)
    # 2. Threshold crossing before attack onset (implicitly handled by lead)
    # 3. Risk increase toward onset (risk_delta)
    # 4. Prediction/outcome temporal alignment (if they predict attack right around onset)
    
    # We will prioritize those with actual proactive_lead_seconds >= 0 first
    df_cands = df_cands.sort_values(
        by=["proactive_lead_seconds", "risk_delta", "max_risk"], 
        ascending=[False, False, False]
    )
    
    # Save artifacts
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    df_cands.to_csv(artifacts_dir / "demo_candidates.csv", index=False)
    
    selected = df_cands.iloc[0].to_dict()
    print(f"Selected candidate {selected['candidate_id']}:")
    print(f"  Proactive lead: {selected['proactive_lead_seconds']}s")
    print(f"  Max risk: {selected['max_risk']:.2f}")

    # Write to config/demo.yaml
    demo_yaml_path = project_root / "configs" / "demo.yaml"
    demo_config = {
        "demo": {
            "default_window_index": int(selected["start_idx"]),
            "scenario_id": selected["candidate_id"],
            "start_state": selected["current_z_t"],
            "attack_onset_window": selected["attack_onset_window"],
            "attack_onset_state": selected["attack_onset_state"],
            "threshold": FROZEN_THRESHOLD,
            "frozen": True
        }
    }
    
    with open(demo_yaml_path, "w") as f:
        yaml.dump(demo_config, f, default_flow_style=False)
        
    print(f"Saved to {demo_yaml_path}")

if __name__ == "__main__":
    main()
