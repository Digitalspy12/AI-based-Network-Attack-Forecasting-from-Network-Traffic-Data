#!/usr/bin/env python3
"""Phase 8A — Audit PCAP coverage.

Scans the available PCAP file(s) and produces a coverage report:
  - Total packets read
  - Time range (first ts, last ts, duration)
  - Packet count per protocol
  - Unique source IPs
  - Windows available (floor-aligned to 10s)
  - Overlap with flow-state windows (if state_windows.parquet exists)

Output: reports/packet_feature_coverage.json

Usage:
    python scripts/audit_pcap.py
    python scripts/audit_pcap.py --pcap /path/to/other.pcap
    python scripts/audit_pcap.py --max-packets 500000   # quick audit
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from digitalspy.pcap.reader import stream_pcap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_PCAP = (
    Path(__file__).resolve().parents[2]
    / "Tuesday-WorkingHours.pcap"
)
_WINDOW_SECONDS = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PCAP coverage for Phase 8A.")
    parser.add_argument(
        "--pcap",
        type=Path,
        default=_DEFAULT_PCAP,
        help="Path to PCAP file (default: Tuesday-WorkingHours.pcap in project root).",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Stop after this many packets (None = read all). Use for quick audit.",
    )
    args = parser.parse_args()

    pcap_path = args.pcap
    if not pcap_path.exists():
        logger.error("PCAP file not found: %s", pcap_path)
        logger.error("Provide --pcap /path/to/file.pcap")
        sys.exit(1)

    logger.info("Auditing PCAP: %s (size: %.1f GB)", pcap_path, pcap_path.stat().st_size / 1e9)
    logger.info("Window size: %.0f seconds", _WINDOW_SECONDS)
    if args.max_packets:
        logger.info("Quick audit mode: reading up to %d packets.", args.max_packets)

    # ── Streaming scan ────────────────────────────────────────────────────────
    total_packets = 0
    protocol_counts: Counter[int] = Counter()
    src_ips: set[str] = set()
    window_set: set[float] = set()
    ts_min = float("inf")
    ts_max = float("-inf")
    ttl_samples: list[int] = []
    payload_samples: list[int] = []

    for pkt in stream_pcap(pcap_path, max_packets=args.max_packets):
        total_packets += 1
        protocol_counts[pkt.protocol] += 1
        src_ips.add(pkt.src_ip)
        w = math.floor(pkt.timestamp / _WINDOW_SECONDS) * _WINDOW_SECONDS
        window_set.add(w)

        if pkt.timestamp < ts_min:
            ts_min = pkt.timestamp
        if pkt.timestamp > ts_max:
            ts_max = pkt.timestamp

        # Sample TTL and payload for statistics (every 1000th packet)
        if total_packets % 1000 == 0:
            ttl_samples.append(pkt.ttl)
            payload_samples.append(pkt.payload_len)

        if total_packets % 500_000 == 0:
            logger.info(
                "  … %d packets read | windows so far: %d | unique IPs: %d",
                total_packets,
                len(window_set),
                len(src_ips),
            )

    duration_s = ts_max - ts_min if ts_min < float("inf") else 0.0

    # ── Flow overlap check ────────────────────────────────────────────────────
    processed_dir = _ROOT / "data" / "processed"
    state_path = processed_dir / "state_windows.parquet"
    flow_overlap: dict = {"status": "state_windows.parquet not found — run build_states.py first"}

    if state_path.exists():
        try:
            import pandas as pd
            sw = pd.read_parquet(state_path, columns=["source_ip", "window_start"])
            flow_windows = set(sw["window_start"].unique().tolist())
            overlap = window_set & flow_windows
            flow_overlap = {
                "flow_windows": len(flow_windows),
                "pcap_windows": len(window_set),
                "overlapping_windows": len(overlap),
                "coverage_pct": round(100.0 * len(overlap) / max(len(flow_windows), 1), 2),
            }
            logger.info(
                "Flow window overlap: %d / %d (%.1f%%)",
                len(overlap), len(flow_windows), flow_overlap["coverage_pct"],
            )
        except Exception as exc:
            flow_overlap = {"error": str(exc)}

    # ── Summary ────────────────────────────────────────────────────────────────
    proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
    protocol_summary = {
        proto_map.get(k, f"proto_{k}"): v
        for k, v in protocol_counts.most_common(10)
    }

    ttl_arr = np.array(ttl_samples, dtype=np.float32)
    pay_arr = np.array(payload_samples, dtype=np.float32)

    report = {
        "pcap_path": str(pcap_path),
        "pcap_size_gb": round(pcap_path.stat().st_size / 1e9, 2),
        "max_packets_limit": args.max_packets,
        "total_packets_read": total_packets,
        "time_range": {
            "first_ts": round(ts_min, 3) if ts_min < float("inf") else None,
            "last_ts": round(ts_max, 3) if ts_max > float("-inf") else None,
            "duration_seconds": round(duration_s, 1),
            "duration_hours": round(duration_s / 3600, 2),
        },
        "protocol_distribution": protocol_summary,
        "unique_source_ips": len(src_ips),
        "windows_10s": len(window_set),
        "sampled_statistics": {
            "ttl": {
                "mean": round(float(ttl_arr.mean()), 2) if len(ttl_arr) > 0 else None,
                "std": round(float(ttl_arr.std()), 2) if len(ttl_arr) > 0 else None,
                "min": int(ttl_arr.min()) if len(ttl_arr) > 0 else None,
                "max": int(ttl_arr.max()) if len(ttl_arr) > 0 else None,
            },
            "payload_bytes": {
                "mean": round(float(pay_arr.mean()), 2) if len(pay_arr) > 0 else None,
                "max": int(pay_arr.max()) if len(pay_arr) > 0 else None,
            },
        },
        "flow_window_overlap": flow_overlap,
        "ntro_note": (
            "This report documents PCAP coverage. "
            "Phase 8A NTRO compliance requires that packet features "
            "come exclusively from real PCAP data."
        ),
    }

    # ── Save report ────────────────────────────────────────────────────────────
    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "packet_feature_coverage.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PCAP AUDIT REPORT")
    print("=" * 60)
    print(f"  File           : {pcap_path.name}")
    print(f"  Size           : {report['pcap_size_gb']:.1f} GB")
    print(f"  Packets read   : {total_packets:,}")
    print(f"  Duration       : {report['time_range']['duration_hours']:.2f} hours")
    print(f"  Unique src IPs : {len(src_ips):,}")
    print(f"  10s windows    : {len(window_set):,}")
    print(f"  Protocols      : {protocol_summary}")
    print(f"\n  Flow overlap   : {flow_overlap}")
    print(f"\n✓ Report saved → {out_path}")


if __name__ == "__main__":
    main()
