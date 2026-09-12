"""Phase 8A — Packet-level feature extraction.

Takes a list of PacketRecords from one (host_ip, window_start) bucket
and computes the packet feature state P_t.

Packet feature families (Phase 8A requirement):
  - TTL statistics          : mean, std, min, max
  - TCP window statistics   : mean, std
  - IP fragmentation        : fragmentation rate
  - Payload size distribution: mean, std, max
  - Retransmission indicator: SYN-then-SYN-ACK retx ratio
  - Port-scan signature     : single-src → many-dst-port ratio

All features are derived exclusively from real packet-level fields.
None are imputed from flow-level CSV data.

Output feature names (PACKET_FEATURE_NAMES) — 12 features:
  ttl_mean, ttl_std, ttl_min, ttl_max,
  tcp_win_mean, tcp_win_std,
  frag_ratio,
  payload_mean, payload_std, payload_max,
  retx_ratio,
  scan_score
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from digitalspy.pcap.reader import PacketRecord

logger = logging.getLogger(__name__)

# TCP flag bit masks
_TCP_SYN = 0x02
_TCP_ACK = 0x10
_TCP_SYN_ACK = _TCP_SYN | _TCP_ACK  # 0x12

# Ordered feature name contract (must stay stable — see packet_features.yaml)
PACKET_FEATURE_NAMES: list[str] = [
    "pk_ttl_mean",
    "pk_ttl_std",
    "pk_ttl_min",
    "pk_ttl_max",
    "pk_tcp_win_mean",
    "pk_tcp_win_std",
    "pk_frag_ratio",
    "pk_payload_mean",
    "pk_payload_std",
    "pk_payload_max",
    "pk_retx_ratio",
    "pk_scan_score",
]

_N_PACKET_FEATURES = len(PACKET_FEATURE_NAMES)
assert _N_PACKET_FEATURES == 12


def extract_packet_features(
    packets: Sequence[PacketRecord],
    window_src_ip: str | None = None,
) -> dict[str, float]:
    """Compute P_t from a list of PacketRecords for one host-window.

    Args:
        packets: All PacketRecords in the 10-second window for a given host.
        window_src_ip: The host IP being tracked (source perspective).
                       Used to scope port-scan detection.

    Returns:
        dict mapping each of PACKET_FEATURE_NAMES to a float.
        Returns all-zero dict if packets is empty.
    """
    zero = {f: 0.0 for f in PACKET_FEATURE_NAMES}
    if not packets:
        return zero

    ttl_vals = np.array([p.ttl for p in packets], dtype=np.float32)
    tcp_pkts = [p for p in packets if p.protocol == 6]  # TCP
    payload_vals = np.array([p.payload_len for p in packets], dtype=np.float32)

    # ── TTL statistics ──────────────────────────────────────────────────────
    feats: dict[str, float] = {}
    feats["pk_ttl_mean"] = float(ttl_vals.mean())
    feats["pk_ttl_std"] = float(ttl_vals.std()) if len(ttl_vals) > 1 else 0.0
    feats["pk_ttl_min"] = float(ttl_vals.min())
    feats["pk_ttl_max"] = float(ttl_vals.max())

    # ── TCP window statistics ───────────────────────────────────────────────
    if tcp_pkts:
        win_vals = np.array([p.tcp_window for p in tcp_pkts], dtype=np.float32)
        feats["pk_tcp_win_mean"] = float(win_vals.mean())
        feats["pk_tcp_win_std"] = float(win_vals.std()) if len(win_vals) > 1 else 0.0
    else:
        feats["pk_tcp_win_mean"] = 0.0
        feats["pk_tcp_win_std"] = 0.0

    # ── IP fragmentation rate ───────────────────────────────────────────────
    n = len(packets)
    n_frag = sum(1 for p in packets if p.ip_frag)
    feats["pk_frag_ratio"] = float(n_frag) / float(n)

    # ── Payload size distribution ───────────────────────────────────────────
    feats["pk_payload_mean"] = float(payload_vals.mean())
    feats["pk_payload_std"] = float(payload_vals.std()) if len(payload_vals) > 1 else 0.0
    feats["pk_payload_max"] = float(payload_vals.max())

    # ── Retransmission indicator ────────────────────────────────────────────
    # Heuristic: count SYN packets and SYN-ACK packets within the window.
    # A high SYN-ACK ratio relative to SYN suggests repeated handshake attempts.
    n_syn = sum(
        1 for p in tcp_pkts
        if (p.tcp_flags & _TCP_SYN) and not (p.tcp_flags & _TCP_ACK)
    )
    n_syn_ack = sum(
        1 for p in tcp_pkts
        if (p.tcp_flags & _TCP_SYN_ACK) == _TCP_SYN_ACK
    )
    total_tcp = len(tcp_pkts)
    # retx_ratio: proportion of TCP packets that are SYN-ACK (re-responses)
    feats["pk_retx_ratio"] = float(n_syn_ack) / float(total_tcp) if total_tcp > 0 else 0.0

    # ── Port-scan signature ─────────────────────────────────────────────────
    # Score: number of distinct destination ports from this source / total packets.
    # High value means one source is touching many ports (scan behaviour).
    if window_src_ip is not None:
        src_packets = [p for p in packets if p.src_ip == window_src_ip]
    else:
        src_packets = list(packets)

    if src_packets:
        distinct_dst_ports = len(set(p.dst_port for p in src_packets))
        feats["pk_scan_score"] = float(distinct_dst_ports) / float(len(src_packets))
    else:
        feats["pk_scan_score"] = 0.0

    # ── Sanitise ────────────────────────────────────────────────────────────
    for k in feats:
        if not np.isfinite(feats[k]):
            feats[k] = 0.0

    return feats


def empty_packet_features() -> dict[str, float]:
    """Return a zero-filled packet feature dict (coverage = False case)."""
    return {f: 0.0 for f in PACKET_FEATURE_NAMES}


def validate_packet_features(feats: dict[str, float]) -> None:
    """Raise ValueError if the packet feature dict is not the expected 12 keys."""
    actual = set(feats.keys())
    expected = set(PACKET_FEATURE_NAMES)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Packet feature dict missing: {missing}")
    if extra:
        raise ValueError(f"Packet feature dict has extra keys: {extra}")
    if len(feats) != _N_PACKET_FEATURES:
        raise ValueError(f"Expected {_N_PACKET_FEATURES} packet features, got {len(feats)}")
