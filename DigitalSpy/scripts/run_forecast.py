#!/usr/bin/env python3
"""Phase 6 — Run Forecast on a Specific Host.

Runs the ForecastEngine on a given source IP from the test set.
Outputs structured forecast JSON to stdout and optionally to a file.

Usage:
    python run_forecast.py --host 192.168.10.50
    python run_forecast.py --host 192.168.10.50 --output forecasts/host_forecast.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.features.engineer import FEATURE_NAMES
from digitalspy.forecasting.engine import ForecastEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run DigitalSpy forecast for a host.")
    parser.add_argument("--host", required=True, help="Source IP address.")
    parser.add_argument("--output", default=None, help="Optional JSON output file path.")
    args = parser.parse_args()

    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")

    state_path = processed_dir / "state_windows.parquet"
    checkpoint_path = models_dir / "lstm_checkpoint.pt"

    if not state_path.exists():
        logger.error("state_windows.parquet not found. Run build_states.py first.")
        sys.exit(1)

    if not checkpoint_path.exists():
        logger.error("lstm_checkpoint.pt not found. Run train_lstm.py first.")
        sys.exit(1)

    # Load host windows
    all_windows = pd.read_parquet(state_path)
    host_windows = all_windows[
        (all_windows["source_ip"] == args.host) &
        (all_windows["split"] == "test")
    ].sort_values("window_start").reset_index(drop=True)

    if len(host_windows) < 20:
        logger.error(
            f"Host {args.host} has only {len(host_windows)} windows in test set. "
            f"Need at least 20."
        )
        sys.exit(1)

    # Take last 20 windows as history
    last_20 = host_windows.tail(20).reset_index(drop=True)
    history = last_20[FEATURE_NAMES].values.astype("float32")
    current_z_t = last_20["z_t"].iloc[-1]

    # Run inference
    engine = ForecastEngine(checkpoint_path)
    result = engine.predict(history, current_z_t=current_z_t)

    # Add metadata
    result["host_ip"] = args.host
    result["history_window_start"] = str(last_20["window_start"].iloc[0])
    result["history_window_end"] = str(last_20["window_start"].iloc[-1])

    output_json = json.dumps(result, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output_json)
        logger.info(f"Forecast saved to {out_path}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
