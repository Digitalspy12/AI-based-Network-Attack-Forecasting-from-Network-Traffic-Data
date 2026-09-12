#!/usr/bin/env python3
"""Phase 8A — Build packet-level state from PCAP.

Streams the PCAP file, aggregates packets into 10-second host-IP windows,
extracts the 12 packet features per window, and saves a parquet table:

    data/interim/packet_state.parquet

Columns:
    source_ip       : str
    window_start    : float  (Unix epoch, floor-aligned to 10s)
    pk_ttl_mean     : float
    pk_ttl_std      : float
    pk_ttl_min      : float
    pk_ttl_max      : float
    pk_tcp_win_mean : float
    pk_tcp_win_std  : float
    pk_frag_ratio   : float
    pk_payload_mean : float
    pk_payload_std  : float
    pk_payload_max  : float
    pk_retx_ratio   : float
    pk_scan_score   : float
    packet_count    : int    (number of packets in this window for this host)

Usage:
    python scripts/build_packet_state.py
    python scripts/build_packet_state.py --pcap /path/to/custom.pcap
    python scripts/build_packet_state.py --max-packets 2000000  # subset run

After this script, run build_states.py with --mode flow_packet to produce
the fused state.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from digitalspy.pcap.reader import stream_pcap, PacketRecord
from digitalspy.features.packet_features import (
    extract_packet_features,
    PACKET_FEATURE_NAMES,
)

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
_LOG_INTERVAL = 1_000_000  # log every 1M packets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build packet-level state windows from PCAP (Phase 8A)."
    )
    parser.add_argument(
        "--pcap",
        type=Path,
        default=_DEFAULT_PCAP,
        help="Path to PCAP file.",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Read at most N packets (for testing / partial runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "interim" / "packet_state.parquet",
        help="Output parquet path.",
    )
    args = parser.parse_args()

    pcap_path = args.pcap
    if not pcap_path.exists():
        logger.error("PCAP file not found: %s", pcap_path)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building packet state from: %s (%.1f GB)",
        pcap_path,
        pcap_path.stat().st_size / 1e9,
    )
    if args.max_packets:
        logger.info("Packet cap: %d", args.max_packets)

    # ── Accumulate per-(host, window) buckets ──────────────────────────────
    # bucket key: (src_ip, window_start_int) to avoid float hashing issues
    # window_start_int = int(floor(ts / 10.0) * 10.0 * 1000)  → ms precision
    buckets: dict[tuple[str, int], list[PacketRecord]] = defaultdict(list)

    total_pkts = 0
    for pkt in stream_pcap(pcap_path, max_packets=args.max_packets):
        total_pkts += 1
        w_int = int(math.floor(pkt.timestamp / _WINDOW_SECONDS) * _WINDOW_SECONDS * 1000)
        buckets[(pkt.src_ip, w_int)].append(pkt)

        if total_pkts % _LOG_INTERVAL == 0:
            logger.info(
                "  … %d packets processed | %d (host, window) buckets",
                total_pkts,
                len(buckets),
            )

    logger.info(
        "Streaming complete: %d packets → %d (host, window) buckets.",
        total_pkts,
        len(buckets),
    )

    # ── Extract features per bucket ────────────────────────────────────────
    logger.info("Extracting packet features per window bucket…")
    rows = []
    for (src_ip, w_int), pkt_list in buckets.items():
        window_start = float(w_int) / 1000.0
        feats = extract_packet_features(pkt_list, window_src_ip=src_ip)
        row = {
            "source_ip": src_ip,
            "window_start": window_start,
            "packet_count": len(pkt_list),
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        logger.error("No packets yielded any windows. Check the PCAP file.")
        sys.exit(1)

    pkt_df = pd.DataFrame(rows)
    pkt_df = pkt_df.sort_values(["source_ip", "window_start"]).reset_index(drop=True)

    # ── Type enforcement ───────────────────────────────────────────────────
    pkt_df["window_start"] = pkt_df["window_start"].astype("float64")
    pkt_df["source_ip"] = pkt_df["source_ip"].astype(str)
    pkt_df["packet_count"] = pkt_df["packet_count"].astype("int32")
    for feat in PACKET_FEATURE_NAMES:
        pkt_df[feat] = pkt_df[feat].astype("float32")

    # ── Save ───────────────────────────────────────────────────────────────
    pkt_df.to_parquet(args.output, index=False)
    logger.info(
        "✓ Saved packet_state.parquet: %d rows → %s",
        len(pkt_df),
        args.output,
    )

    # ── Coverage report update ─────────────────────────────────────────────
    n_hosts = int(pkt_df["source_ip"].nunique())
    n_windows = len(pkt_df)
    ts_min = float(pkt_df["window_start"].min())
    ts_max = float(pkt_df["window_start"].max())
    duration_h = (ts_max - ts_min) / 3600.0

    coverage_update = {
        "packet_state_build": {
            "total_packets_processed": total_pkts,
            "unique_hosts": n_hosts,
            "host_window_rows": n_windows,
            "time_range_hours": round(duration_h, 2),
            "output_path": str(args.output),
            "feature_names": PACKET_FEATURE_NAMES,
        }
    }

    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cov_path = reports_dir / "packet_feature_coverage.json"
    if cov_path.exists():
        with open(cov_path) as f:
            existing = json.load(f)
        existing.update(coverage_update)
        with open(cov_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(cov_path, "w") as f:
            json.dump(coverage_update, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PACKET STATE BUILD SUMMARY")
    print("=" * 60)
    print(f"  Packets processed  : {total_pkts:,}")
    print(f"  Unique hosts       : {n_hosts:,}")
    print(f"  (host, window) rows: {n_windows:,}")
    print(f"  Time span          : {duration_h:.2f} hours")
    print(f"  Packet features    : {len(PACKET_FEATURE_NAMES)}")
    print(f"\n✓ Output → {args.output}")
    print(f"✓ Coverage report updated → {cov_path}")
    print("\nNext: python scripts/build_states.py --mode flow_packet")


if __name__ == "__main__":
    main()
