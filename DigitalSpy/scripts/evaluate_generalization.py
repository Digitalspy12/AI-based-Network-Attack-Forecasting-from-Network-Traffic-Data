#!/usr/bin/env python3
"""Phase 8D — Evaluate Generalization on Held-Out Attack Family.

Evaluates an already trained LSTM model on the held-out Thursday WebAttacks file.
By default, the model is trained on a split that includes BENIGN + DDoS + Bot + PortScan + FTP/SSH-Patator + Infiltration.
WebAttacks (Brute Force, XSS, Sql Injection) are held out in the 'test' split.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model
from scripts.train_lstm import load_sequences, evaluate_lstm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=None)
    args = parser.parse_args()

    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")
    model_path = args.model_path or models_dir / "lstm_checkpoint.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run train_lstm.py first.")
        sys.exit(1)

    lstm_cfg = config.lstm()
    K = lstm_cfg["forecast"]["K"]
    num_classes = lstm_cfg["forecast"]["heads"]["tactic"]["num_classes"]

    # Load test sequences (which is the held-out WebAttacks now)
    logger.info("Loading held-out test sequences (WebAttacks)...")
    X_te, yr_te, yt_te = load_sequences("test", processed_dir)
    
    if len(X_te) == 0:
        logger.error("No test sequences found.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(X_te):,} sequences.")

    # Determine input size dynamically based on data
    input_size = X_te.shape[-1]
    lstm_cfg["architecture"]["input_size"] = input_size
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(lstm_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info("Evaluating generalization on held-out WebAttacks...")
    metrics = evaluate_lstm(model, X_te, yr_te, yt_te, K, num_classes, device, "held_out_webattacks")

    # Save report
    reports_dir = _ROOT / "reports" / "generalization"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "experiment_8d_held_out.json"
    
    report_data = {
        "experiment": "8D_Held_Out_WebAttacks",
        "description": "Evaluate frozen LSTM on unseen WebAttacks family.",
        "input_features": "flow_packet" if input_size == 36 else "flow",
        "input_dim": input_size,
        "test_sequences": len(X_te),
        "metrics": metrics
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 8D: GENERALIZATION EXPERIMENT (WEB ATTACKS)")
    print("=" * 60)
    print(f"Features      : {report_data['input_features']} ({input_size} dims)")
    print(f"Sequences     : {len(X_te):,}")
    print(f"\nHeld-out k=1 Risk macro-F1 : {metrics['k1_risk_f1']:.4f}")
    print(f"Held-out F1_K Tactic       : {metrics['F1_K_tactic']:.4f}")
    print(f"\nPer-horizon Risk F1   : {metrics['k_step_risk_f1']}")
    print(f"Per-horizon Tactic F1 : {metrics['k_step_tactic_f1']}")
    print(f"\n✓ Report saved → {report_path}")

if __name__ == "__main__":
    main()
