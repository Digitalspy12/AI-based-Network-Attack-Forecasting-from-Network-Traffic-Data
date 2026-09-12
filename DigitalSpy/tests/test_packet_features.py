"""Phase 8A automated tests — Packet features and Fusion.

Tests:
  PacketFeature extraction:
    - TTL aggregation correctness
    - TCP window aggregation correctness
    - Fragmentation detection
    - Payload statistics
    - Retransmission detection
    - Port-scan signature

  Temporal correctness:
    - Empty packet list returns zeros
    - Feature name contract (exactly 12 names)
    - No NaN/Inf in output

  Fusion:
    - Left join preserves all flow rows
    - Missing packet coverage filled with 0.0
    - has_packet_features flag correct
    - No future leakage (timestamps not modified)
    - Combined feature count is 36

Run with:
    cd DigitalSpy && python -m pytest tests/test_packet_features.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Imports under test ────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digitalspy.pcap.reader import PacketRecord
from digitalspy.features.packet_features import (
    extract_packet_features,
    empty_packet_features,
    validate_packet_features,
    PACKET_FEATURE_NAMES,
    _TCP_SYN,
    _TCP_ACK,
    _TCP_SYN_ACK,
)
from digitalspy.features.fusion import (
    fuse_flow_packet,
    COMBINED_FEATURE_NAMES,
    get_feature_names,
)
from digitalspy.features.engineer import FEATURE_NAMES as FLOW_FEATURE_NAMES


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pkt(
    ts=1000.0,
    src_ip="192.168.1.1",
    dst_ip="10.0.0.1",
    src_port=12345,
    dst_port=80,
    protocol=6,
    ttl=64,
    tcp_window=65535,
    tcp_flags=0x10,   # ACK
    ip_frag=False,
    payload_len=100,
    pkt_len=154,
) -> PacketRecord:
    return PacketRecord(
        timestamp=ts,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        ttl=ttl,
        tcp_window=tcp_window,
        tcp_flags=tcp_flags,
        ip_frag=ip_frag,
        payload_len=payload_len,
        pkt_len=pkt_len,
    )


def make_flow_df(n_rows=5, source_ip="192.168.1.1", window_starts=None) -> pd.DataFrame:
    """Create a minimal flow state DataFrame for fusion tests."""
    if window_starts is None:
        window_starts = [float(i * 10) for i in range(n_rows)]
    rows = []
    for i, ws in enumerate(window_starts):
        row = {"source_ip": source_ip, "window_start": float(ws), "z_t": "BENIGN", "split": "test"}
        for feat in FLOW_FEATURE_NAMES:
            row[feat] = float(i + 1)  # non-zero placeholder
        rows.append(row)
    return pd.DataFrame(rows)


def make_packet_df(source_ip="192.168.1.1", window_starts=None) -> pd.DataFrame:
    """Create a minimal packet state DataFrame for fusion tests."""
    if window_starts is None:
        window_starts = [0.0, 10.0]
    rows = []
    for ws in window_starts:
        row = {"source_ip": source_ip, "window_start": float(ws)}
        for feat in PACKET_FEATURE_NAMES:
            row[feat] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Packet feature extraction tests ──────────────────────────────────────────

class TestPacketFeatureNames:
    def test_exactly_12_features(self):
        assert len(PACKET_FEATURE_NAMES) == 12

    def test_all_pk_prefixed(self):
        for name in PACKET_FEATURE_NAMES:
            assert name.startswith("pk_"), f"Feature {name!r} missing 'pk_' prefix"

    def test_no_duplicates(self):
        assert len(PACKET_FEATURE_NAMES) == len(set(PACKET_FEATURE_NAMES))


class TestEmptyPacketList:
    def test_returns_zero_dict(self):
        result = extract_packet_features([])
        assert all(v == 0.0 for v in result.values())

    def test_returns_correct_keys(self):
        result = extract_packet_features([])
        assert set(result.keys()) == set(PACKET_FEATURE_NAMES)


class TestTTLAggregation:
    def test_mean_correct(self):
        pkts = [make_pkt(ttl=64), make_pkt(ttl=128), make_pkt(ttl=32)]
        result = extract_packet_features(pkts)
        assert result["pk_ttl_mean"] == pytest.approx(74.666, abs=0.01)

    def test_std_correct(self):
        pkts = [make_pkt(ttl=100), make_pkt(ttl=100)]
        result = extract_packet_features(pkts)
        assert result["pk_ttl_std"] == pytest.approx(0.0, abs=1e-6)

    def test_min_max(self):
        pkts = [make_pkt(ttl=10), make_pkt(ttl=200), make_pkt(ttl=64)]
        result = extract_packet_features(pkts)
        assert result["pk_ttl_min"] == pytest.approx(10.0)
        assert result["pk_ttl_max"] == pytest.approx(200.0)


class TestTCPWindowAggregation:
    def test_mean_over_tcp_only(self):
        tcp_pkt = make_pkt(protocol=6, tcp_window=8192)
        udp_pkt = make_pkt(protocol=17, tcp_window=0)  # not TCP
        result = extract_packet_features([tcp_pkt, udp_pkt])
        assert result["pk_tcp_win_mean"] == pytest.approx(8192.0)

    def test_zero_if_no_tcp(self):
        pkts = [make_pkt(protocol=17, tcp_window=0)]
        result = extract_packet_features(pkts)
        assert result["pk_tcp_win_mean"] == 0.0
        assert result["pk_tcp_win_std"] == 0.0

    def test_std_nonzero_for_different_windows(self):
        pkts = [make_pkt(protocol=6, tcp_window=1024), make_pkt(protocol=6, tcp_window=4096)]
        result = extract_packet_features(pkts)
        assert result["pk_tcp_win_std"] > 0.0


class TestFragmentationDetection:
    def test_no_frag(self):
        pkts = [make_pkt(ip_frag=False), make_pkt(ip_frag=False)]
        result = extract_packet_features(pkts)
        assert result["pk_frag_ratio"] == pytest.approx(0.0)

    def test_all_frag(self):
        pkts = [make_pkt(ip_frag=True), make_pkt(ip_frag=True)]
        result = extract_packet_features(pkts)
        assert result["pk_frag_ratio"] == pytest.approx(1.0)

    def test_half_frag(self):
        pkts = [make_pkt(ip_frag=True), make_pkt(ip_frag=False)]
        result = extract_packet_features(pkts)
        assert result["pk_frag_ratio"] == pytest.approx(0.5)


class TestPayloadStatistics:
    def test_mean_payload(self):
        pkts = [make_pkt(payload_len=100), make_pkt(payload_len=200)]
        result = extract_packet_features(pkts)
        assert result["pk_payload_mean"] == pytest.approx(150.0)

    def test_max_payload(self):
        pkts = [make_pkt(payload_len=50), make_pkt(payload_len=1400), make_pkt(payload_len=200)]
        result = extract_packet_features(pkts)
        assert result["pk_payload_max"] == pytest.approx(1400.0)

    def test_std_payload_uniform(self):
        pkts = [make_pkt(payload_len=500), make_pkt(payload_len=500)]
        result = extract_packet_features(pkts)
        assert result["pk_payload_std"] == pytest.approx(0.0)


class TestRetransmissionIndicator:
    def test_syn_ack_ratio(self):
        """2 SYN-ACK packets out of 4 TCP packets → retx_ratio = 0.5"""
        pkts = [
            make_pkt(protocol=6, tcp_flags=_TCP_SYN),           # SYN
            make_pkt(protocol=6, tcp_flags=_TCP_SYN_ACK),       # SYN-ACK
            make_pkt(protocol=6, tcp_flags=_TCP_SYN_ACK),       # SYN-ACK
            make_pkt(protocol=6, tcp_flags=_TCP_ACK),           # ACK
        ]
        result = extract_packet_features(pkts)
        assert result["pk_retx_ratio"] == pytest.approx(0.5)

    def test_zero_retx_pure_data(self):
        """Only ACK data packets → no SYN-ACK → retx_ratio = 0"""
        pkts = [make_pkt(protocol=6, tcp_flags=_TCP_ACK) for _ in range(5)]
        result = extract_packet_features(pkts)
        assert result["pk_retx_ratio"] == pytest.approx(0.0)


class TestPortScanSignature:
    def test_scan_score_high_for_port_sweep(self):
        """One source IP touching 10 different dst ports → scan_score close to 1.0"""
        pkts = [make_pkt(src_ip="10.0.0.1", dst_port=p) for p in range(10, 20)]
        result = extract_packet_features(pkts, window_src_ip="10.0.0.1")
        assert result["pk_scan_score"] == pytest.approx(1.0)

    def test_scan_score_low_single_port(self):
        """All packets to same port → scan_score = 1/N"""
        pkts = [make_pkt(src_ip="10.0.0.1", dst_port=80) for _ in range(10)]
        result = extract_packet_features(pkts, window_src_ip="10.0.0.1")
        assert result["pk_scan_score"] == pytest.approx(1.0 / 10.0)


class TestNoNaNInf:
    def test_no_nan_or_inf(self):
        pkts = [make_pkt(ttl=255, payload_len=0, tcp_window=0)] * 3
        result = extract_packet_features(pkts)
        for k, v in result.items():
            assert np.isfinite(v), f"Feature {k} is {v}"


class TestValidatePacketFeatures:
    def test_valid_dict_passes(self):
        feats = empty_packet_features()
        validate_packet_features(feats)  # should not raise

    def test_missing_key_raises(self):
        feats = empty_packet_features()
        del feats["pk_ttl_mean"]
        with pytest.raises(ValueError, match="missing"):
            validate_packet_features(feats)

    def test_extra_key_raises(self):
        feats = empty_packet_features()
        feats["extra_key"] = 1.0
        with pytest.raises(ValueError, match="extra"):
            validate_packet_features(feats)


# ── Fusion tests ──────────────────────────────────────────────────────────────

class TestCombinedFeatureNames:
    def test_total_36(self):
        assert len(COMBINED_FEATURE_NAMES) == 36

    def test_flow_names_first(self):
        assert COMBINED_FEATURE_NAMES[:24] == list(FLOW_FEATURE_NAMES)

    def test_packet_names_last(self):
        assert COMBINED_FEATURE_NAMES[24:] == PACKET_FEATURE_NAMES

    def test_get_feature_names_flow(self):
        assert get_feature_names("flow") == list(FLOW_FEATURE_NAMES)

    def test_get_feature_names_packet(self):
        assert get_feature_names("packet") == PACKET_FEATURE_NAMES

    def test_get_feature_names_fused(self):
        assert get_feature_names("fused") == list(COMBINED_FEATURE_NAMES)


class TestFuseFlowPacket:
    def test_all_flow_rows_preserved(self):
        flow_df = make_flow_df(n_rows=5, window_starts=[0.0, 10.0, 20.0, 30.0, 40.0])
        pkt_df = make_packet_df(window_starts=[0.0, 10.0])  # only 2 of 5 windows covered
        fused = fuse_flow_packet(flow_df, pkt_df)
        assert len(fused) == 5  # no rows dropped

    def test_covered_windows_have_packet_features(self):
        flow_df = make_flow_df(n_rows=2, window_starts=[0.0, 10.0])
        pkt_df = make_packet_df(window_starts=[0.0])  # only first window covered
        fused = fuse_flow_packet(flow_df, pkt_df)
        covered = fused[fused["window_start"] == 0.0]
        uncovered = fused[fused["window_start"] == 10.0]
        assert bool(covered["has_packet_features"].iloc[0]) is True
        assert bool(uncovered["has_packet_features"].iloc[0]) is False

    def test_missing_coverage_filled_with_zero(self):
        flow_df = make_flow_df(n_rows=1, window_starts=[99.0])  # no matching packet window
        pkt_df = make_packet_df(window_starts=[0.0])
        fused = fuse_flow_packet(flow_df, pkt_df)
        for feat in PACKET_FEATURE_NAMES:
            assert fused[feat].iloc[0] == 0.0

    def test_flow_features_unchanged_after_fusion(self):
        flow_df = make_flow_df(n_rows=3, window_starts=[0.0, 10.0, 20.0])
        pkt_df = make_packet_df(window_starts=[0.0, 10.0, 20.0])
        fused = fuse_flow_packet(flow_df, pkt_df)
        for feat in FLOW_FEATURE_NAMES:
            assert (fused[feat].values == flow_df[feat].values).all(), (
                f"Flow feature {feat} was modified by fusion"
            )

    def test_window_start_timestamps_unchanged(self):
        """Fusion must not modify window_start values (no future leakage)."""
        ws = [0.0, 10.0, 20.0]
        flow_df = make_flow_df(n_rows=3, window_starts=ws)
        pkt_df = make_packet_df(window_starts=ws)
        fused = fuse_flow_packet(flow_df, pkt_df)
        assert sorted(fused["window_start"].tolist()) == ws

    def test_no_nan_in_fused_packet_columns(self):
        flow_df = make_flow_df(n_rows=3, window_starts=[0.0, 10.0, 20.0])
        pkt_df = make_packet_df(window_starts=[0.0])  # partial coverage
        fused = fuse_flow_packet(flow_df, pkt_df)
        for feat in PACKET_FEATURE_NAMES:
            assert not fused[feat].isna().any(), f"NaN in fused column {feat}"

    def test_missing_columns_raise(self):
        bad_flow = pd.DataFrame({"wrong_col": [1, 2]})
        pkt_df = make_packet_df()
        with pytest.raises(ValueError, match="flow_df missing"):
            fuse_flow_packet(bad_flow, pkt_df)
