#!/usr/bin/env python3
"""Phase 1 — Dataset Audit Script.

Inspects all CIC-IDS2017 CSV files and produces:
  reports/audit_report.json

Checks:
  - Row counts
  - Timestamp ranges
  - Label distributions (with whitespace stripped)
  - Missing values (NaN count per column)
  - Infinity values
  - Duplicate rows
  - Source host count
  - Column names (whitespace warning)

Per IMPLEMENTATION.md §7:
  "Do not trust a remembered attack schedule.
   The actual downloaded files are the source of truth."
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def audit_file(csv_path: Path) -> dict:
    """Audit a single CIC-IDS2017 CSV file."""
    logger.info(f"Auditing: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return {"file": csv_path.name, "error": str(e)}

    # Strip column names
    original_cols = list(df.columns)
    df.columns = df.columns.str.strip()
    stripped_cols = list(df.columns)
    whitespace_cols = [o for o, s in zip(original_cols, stripped_cols) if o != s]

    # Strip label
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip()

    # Timestamp
    ts_info = {}
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        ts_info = {
            "min": str(ts.min()),
            "max": str(ts.max()),
            "null_count": int(ts.isna().sum()),
        }

    # Label distribution
    label_dist = {}
    if "Label" in df.columns:
        label_dist = df["Label"].value_counts().to_dict()

    # NaN / Inf counts
    nan_total = int(df.isnull().sum().sum())
    inf_total = int((df == np.inf).sum().sum() + (df == -np.inf).sum().sum())

    # Top NaN columns
    nan_by_col = df.isnull().sum()
    nan_by_col = nan_by_col[nan_by_col > 0].sort_values(ascending=False)

    # Duplicates
    dup_count = int(df.duplicated().sum())

    # Source IPs
    src_ip_count = int(df["Source IP"].nunique()) if "Source IP" in df.columns else None

    return {
        "file": csv_path.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "timestamp": ts_info,
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "nan_total": nan_total,
        "inf_total": inf_total,
        "nan_by_column": {str(k): int(v) for k, v in nan_by_col.head(10).items()},
        "duplicate_rows": dup_count,
        "source_ip_count": src_ip_count,
        "columns_with_whitespace": whitespace_cols,
        "file_size_mb": round(csv_path.stat().st_size / (1024 ** 2), 2),
    }


def main():
    raw_dir = config.resolve_path("raw_data")
    reports_dir = config.resolve_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset directory: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {raw_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} CSV files.")

    results = []
    for f in csv_files:
        result = audit_file(f)
        results.append(result)
        # Print summary
        if "error" not in result:
            logger.info(
                f"  {result['file']}: {result['rows']:,} rows | "
                f"NaN={result['nan_total']:,} | Inf={result['inf_total']:,} | "
                f"Labels={list(result['label_distribution'].keys())}"
            )

    report = {
        "audit_version": "1.0",
        "dataset": "CIC-IDS2017",
        "total_files": len(csv_files),
        "files": results,
        "split_plan": {
            "train": ["Monday-WorkingHours.pcap_ISCX.csv",
                      "Tuesday-WorkingHours.pcap_ISCX.csv",
                      "Wednesday-workingHours.pcap_ISCX.csv"],
            "validation": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                           "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
            "test": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                     "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                     "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
        },
        "notes": [
            "No PCAP files available. Packet-derived features will be imputed with training-period means.",
            "has_packet_features will be 0 for all rows.",
            "Column names had leading/trailing whitespace (stripped).",
        ],
    }

    output_path = reports_dir / "audit_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n✓ Audit report saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("DATASET AUDIT SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  ERROR {r['file']}: {r['error']}")
        else:
            labels = list(r['label_distribution'].keys())
            print(f"  {r['file']}")
            print(f"    Rows: {r['rows']:>10,} | NaN: {r['nan_total']:>6,} | Inf: {r['inf_total']:>4,}")
            print(f"    Labels: {labels}")
            if r.get("timestamp"):
                print(f"    Time: {r['timestamp']['min']} → {r['timestamp']['max']}")
            print()


if __name__ == "__main__":
    main()
