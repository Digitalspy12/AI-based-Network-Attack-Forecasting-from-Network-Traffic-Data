import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

# Ensure we can import from digitalspy
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model
from digitalspy.features.engineer import FEATURE_NAMES

def main():
    print("--- DigitalSpy Head Consistency Diagnostic ---")
    
    # Load configuration
    lstm_cfg = config.lstm()
    model_input_size = lstm_cfg["architecture"]["input_size"]
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
    
    # For a diagnostic, we will evaluate 1000 sequences randomly or sequentially.
    # To be thorough but fast, we'll evaluate 1000 random sequences of length 20.
    np.random.seed(42)
    max_idx = len(test_windows) - 20
    sample_indices = np.random.choice(max_idx, size=min(1000, max_idx), replace=False)
    
    risk_probs_list = []
    one_minus_p_benign_list = []
    
    benign_idx = config.labels()["states"].index("BENIGN")
    
    for start_idx in sample_indices:
        last_20 = test_windows.iloc[start_idx : start_idx + 20]
        history_np = last_20[FEATURE_NAMES].values.astype(np.float32)
        if history_np.shape[-1] != model_input_size:
             history_np = np.pad(history_np, ((0,0), (0, max(0, model_input_size - history_np.shape[-1]))))[:, :model_input_size]
        history_tensor = torch.FloatTensor(history_np).unsqueeze(0)

        with torch.no_grad():
            risk_probs, tactic_logits, _ = model(history_tensor)
            
        risk_np = risk_probs.squeeze(0).numpy()[0] # Check t+1 horizon
        tactic_np = torch.softmax(tactic_logits, dim=-1).squeeze(0).numpy()[0]
        
        risk_probs_list.append(float(risk_np))
        one_minus_p_benign_list.append(float(1.0 - tactic_np[benign_idx]))

    risk_arr = np.array(risk_probs_list)
    ompb_arr = np.array(one_minus_p_benign_list)
    
    pearson_corr, _ = pearsonr(risk_arr, ompb_arr)
    spearman_corr, _ = spearmanr(risk_arr, ompb_arr)
    
    abs_diff = np.abs(risk_arr - ompb_arr)
    mean_abs_diff = np.mean(abs_diff)
    median_abs_diff = np.median(abs_diff)
    
    # Agreement rates based on threshold
    # Both above or both below
    both_above = np.sum((risk_arr >= FROZEN_THRESHOLD) & (ompb_arr >= FROZEN_THRESHOLD))
    both_below = np.sum((risk_arr < FROZEN_THRESHOLD) & (ompb_arr < FROZEN_THRESHOLD))
    
    above_agreement = both_above / np.sum(risk_arr >= FROZEN_THRESHOLD) if np.sum(risk_arr >= FROZEN_THRESHOLD) > 0 else 1.0
    below_agreement = both_below / np.sum(risk_arr < FROZEN_THRESHOLD) if np.sum(risk_arr < FROZEN_THRESHOLD) > 0 else 1.0
    
    report = {
        "n_samples": len(sample_indices),
        "pearson_correlation": float(pearson_corr),
        "spearman_correlation": float(spearman_corr),
        "mean_absolute_disagreement": float(mean_abs_diff),
        "median_absolute_disagreement": float(median_abs_diff),
        "agreement_rate_above_threshold": float(above_agreement),
        "agreement_rate_below_threshold": float(below_agreement),
        "finding": "The binary risk and state heads are not fully probability-consistent." if mean_abs_diff > 0.05 else "Strong consistency."
    }
    
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Write JSON
    with open(artifacts_dir / "head_consistency_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Write Markdown
    md_content = f"""# Head Consistency Diagnostic

## Summary
The current prototype uses separate binary-risk and multi-class state heads, which can produce probability disagreement in some windows. The demo presentation therefore gates state display by the frozen overall-risk threshold. A future unified risk definition or joint calibration would remove this inconsistency at the modeling layer.

## Metrics (N={report['n_samples']} samples)
- **Pearson Correlation**: {report['pearson_correlation']:.4f}
- **Spearman Correlation**: {report['spearman_correlation']:.4f}
- **Mean Absolute Disagreement**: {report['mean_absolute_disagreement']:.4f}
- **Median Absolute Disagreement**: {report['median_absolute_disagreement']:.4f}
- **Agreement Above Threshold (>0.90)**: {report['agreement_rate_above_threshold']:.1%}
- **Agreement Below Threshold (<0.90)**: {report['agreement_rate_below_threshold']:.1%}
"""
    with open(artifacts_dir / "head_consistency_summary.md", "w") as f:
        f.write(md_content)
        
    print(f"Generated head consistency diagnostic in artifacts/")

if __name__ == "__main__":
    main()
