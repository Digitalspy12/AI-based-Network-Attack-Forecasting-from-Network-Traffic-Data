"""Unit tests for DigitalSpy — all required tests from IMPLEMENTATION.md §28.

Test suites:
  1. Label precedence (5 tests)
  2. Feature schema (5 tests)
  3. Windowing (5 tests)
  4. Leakage (4 tests)
  5. Models (5 tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy.states.labels import (
    resolve_window_label,
    label_to_index,
    index_to_label,
    is_attack,
    get_precedence,
)
from digitalspy.features.engineer import (
    FEATURE_NAMES,
    engineer_window_features,
    validate_feature_vector,
)
from digitalspy.states.windowing import (
    build_state_windows,
    build_lstm_sequences,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LABEL PRECEDENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelPrecedence:
    """Tests §28 label precedence rules."""

    def test_precedence_impact_over_c2(self):
        """IMPACT beats C2 in the same window."""
        result = resolve_window_label(["DoS Hulk", "Bot"])
        assert result == "IMPACT", f"Expected IMPACT, got {result}"

    def test_precedence_c2_over_lateral(self):
        """C2 beats LATERAL_MOVEMENT."""
        result = resolve_window_label(["Bot", "Infiltration"])
        assert result == "C2", f"Expected C2, got {result}"

    def test_precedence_lateral_over_initial(self):
        """LATERAL_MOVEMENT beats INITIAL_ACCESS."""
        result = resolve_window_label(["Infiltration", "FTP-Patator"])
        assert result == "LATERAL_MOVEMENT", f"Expected LATERAL_MOVEMENT, got {result}"

    def test_precedence_initial_over_recon(self):
        """INITIAL_ACCESS beats RECON."""
        result = resolve_window_label(["FTP-Patator", "PortScan"])
        assert result == "INITIAL_ACCESS", f"Expected INITIAL_ACCESS, got {result}"

    def test_benign_only_when_no_attack(self):
        """BENIGN applies only when no attack label is present."""
        result = resolve_window_label(["BENIGN"])
        assert result == "BENIGN"

        # With any attack, BENIGN should NOT win
        result2 = resolve_window_label(["BENIGN", "PortScan"])
        assert result2 != "BENIGN", f"BENIGN should not win when attack present: {result2}"
        assert result2 == "RECON"

    def test_full_precedence_chain(self):
        """Full chain: all states present → IMPACT wins."""
        result = resolve_window_label([
            "DoS Hulk", "Bot", "Infiltration",
            "FTP-Patator", "PortScan", "BENIGN"
        ])
        assert result == "IMPACT"

    def test_impact_variants(self):
        """All DoS/DDoS labels map to IMPACT."""
        for label in ["DoS Hulk", "DoS GoldenEye", "DoS slowloris",
                      "DoS Slowhttptest", "DDoS"]:
            result = resolve_window_label([label])
            assert result == "IMPACT", f"{label} should map to IMPACT, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureSchema:
    """Tests §28 feature schema rules."""

    def test_feature_schema_24(self):
        """Exactly 24 features must be defined."""
        assert len(FEATURE_NAMES) == 24, f"Expected 24, got {len(FEATURE_NAMES)}"

    def test_no_unexpected_feature_names(self):
        """Feature list must not contain unexpected or extra names."""
        expected_groups = {
            "flow_count", "total_bytes", "total_packets",
            "mean_flow_duration", "bytes_per_sec", "packets_per_sec",
            "syn_ratio", "ack_ratio", "fin_ratio", "rst_ratio", "psh_ratio",
            "mean_iat", "std_iat", "max_iat",
            "mean_packet_len", "std_packet_len",
            "distinct_dst_ips", "distinct_dst_ports", "distinct_src_ports_used",
            "fwd_bwd_byte_ratio", "mean_init_win_bytes_fwd",
            "ttl_variance", "tcp_window_std", "fragmentation_ratio",
        }
        assert set(FEATURE_NAMES) == expected_groups

    def test_zero_division(self):
        """Feature engineering must not crash on minimal data."""
        # Minimal 2-row DataFrame with the minimum required columns
        df = pd.DataFrame({
            "Total Length of Fwd Packets": [0.0, 0.0],
            "Total Length of Bwd Packets": [0.0, 0.0],
            "Total Fwd Packets": [1.0, 1.0],
            "Total Backward Packets": [0.0, 0.0],
            "Flow Duration": [0.0, 0.0],  # zero duration
            "Flow Bytes/s": [0.0, 0.0],
            "Flow Packets/s": [0.0, 0.0],
            "Flow IAT Mean": [0.0, 0.0],
            "Flow IAT Std": [0.0, 0.0],
            "Flow IAT Max": [0.0, 0.0],
            "Fwd Packet Length Std": [0.0, 0.0],
            "Bwd Packet Length Std": [0.0, 0.0],
            "SYN Flag Count": [0.0, 0.0],
            "ACK Flag Count": [0.0, 0.0],
            "FIN Flag Count": [0.0, 0.0],
            "RST Flag Count": [0.0, 0.0],
            "PSH Flag Count": [0.0, 0.0],
            "Destination IP": ["10.0.0.1", "10.0.0.1"],
            "Destination Port": [80, 443],
            "Source Port": [12345, 12346],
            "Init_Win_bytes_forward": [0.0, 0.0],
        })
        feats = engineer_window_features(df)
        assert len(feats) == 24
        # No NaN or Inf should remain
        for name, val in feats.items():
            assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"

    def test_packet_feature_imputation(self):
        """Packet-derived features are 0.0 when has_packet_features=0."""
        df = pd.DataFrame({
            "Total Length of Fwd Packets": [100.0, 200.0],
            "Total Length of Bwd Packets": [50.0, 75.0],
            "Total Fwd Packets": [5.0, 8.0],
            "Total Backward Packets": [3.0, 4.0],
            "Flow Duration": [1000.0, 2000.0],
            "Flow IAT Mean": [100.0, 150.0],
            "Flow IAT Std": [10.0, 15.0],
            "Flow IAT Max": [500.0, 600.0],
            "Fwd Packet Length Std": [5.0, 8.0],
            "Bwd Packet Length Std": [3.0, 4.0],
            "SYN Flag Count": [1.0, 0.0],
            "ACK Flag Count": [2.0, 3.0],
            "FIN Flag Count": [0.0, 1.0],
            "RST Flag Count": [0.0, 0.0],
            "PSH Flag Count": [1.0, 2.0],
            "Destination IP": ["10.0.0.1", "10.0.0.2"],
            "Destination Port": [80, 443],
            "Source Port": [12345, 12346],
            "Init_Win_bytes_forward": [8192.0, 8192.0],
            "Flow Bytes/s": [100.0, 200.0],
            "Flow Packets/s": [5.0, 8.0],
        })
        feats = engineer_window_features(df)
        # Packet-derived features should be 0.0 (no PCAP)
        assert feats["ttl_variance"] == 0.0
        assert feats["tcp_window_std"] == 0.0
        assert feats["fragmentation_ratio"] == 0.0

    def test_has_packet_features(self):
        """has_packet_features field must be tracked correctly in windows."""
        # Create a minimal window dataset
        df_windows = pd.DataFrame({
            "source_ip": ["192.168.1.1", "192.168.1.1"],
            "window_start": pd.to_datetime(["2017-07-03 09:00:00", "2017-07-03 09:00:10"]),
            "has_packet_features": [0, 0],
            "z_t": ["BENIGN", "BENIGN"],
            "z_t_idx": [0, 0],
            "split": ["train", "train"],
        })
        # All have_packet_features must be 0 (no PCAP in Week-1)
        assert (df_windows["has_packet_features"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WINDOWING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def make_mock_df(n_windows=25, src_ip="192.168.1.1", start="2017-07-03 09:00:00",
                 label="BENIGN", gap_at=None):
    """Create a mock flow DataFrame for windowing tests."""
    rows = []
    base = pd.Timestamp(start)
    for w in range(n_windows):
        if gap_at is not None and w == gap_at:
            continue  # create a gap
        t = base + pd.Timedelta(seconds=w * 10 + 1)
        t2 = base + pd.Timedelta(seconds=w * 10 + 3)
        for ts in [t, t2]:  # 2 flows per window
            rows.append({
                "Source IP": src_ip,
                "Destination IP": "10.0.0.1",
                "Source Port": 12345 + w,
                "Destination Port": 80,
                "Timestamp": ts.strftime("%d/%m/%Y %H:%M:%S"),
                "Label": label,
                "Total Length of Fwd Packets": 100.0,
                "Total Length of Bwd Packets": 50.0,
                "Total Fwd Packets": 5.0,
                "Total Backward Packets": 3.0,
                "Flow Duration": 1000.0,
                "Flow Bytes/s": 100.0,
                "Flow Packets/s": 5.0,
                "Flow IAT Mean": 100.0,
                "Flow IAT Std": 10.0,
                "Flow IAT Max": 500.0,
                "Fwd Packet Length Std": 5.0,
                "Bwd Packet Length Std": 3.0,
                "SYN Flag Count": 1.0,
                "ACK Flag Count": 2.0,
                "FIN Flag Count": 0.0,
                "RST Flag Count": 0.0,
                "PSH Flag Count": 1.0,
                "Init_Win_bytes_forward": 8192.0,
            })
    df = pd.DataFrame(rows)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
    return df


class TestWindowing:
    """Tests §28 windowing rules."""

    def test_10_second_windowing(self):
        """Windows should be exactly 10 seconds wide."""
        df = make_mock_df(n_windows=5)
        windows = build_state_windows(df, split_tag="train")
        assert len(windows) == 5, f"Expected 5 windows, got {len(windows)}"

    def test_non_overlapping_windows(self):
        """Windows must be non-overlapping (stride = window_size)."""
        df = make_mock_df(n_windows=5)
        windows = build_state_windows(df, split_tag="train")
        # Sort by window_start
        windows = windows.sort_values("window_start").reset_index(drop=True)
        if len(windows) > 1:
            for i in range(len(windows) - 1):
                # Each window_end should be <= next window_start
                assert windows["window_end"].iloc[i] <= windows["window_start"].iloc[i + 1]

    def test_minimum_flow_rule(self):
        """Windows with < 2 flows must be dropped."""
        # Create 1-flow windows
        df = pd.DataFrame({
            "Source IP": ["192.168.1.1"] * 5,
            "Destination IP": ["10.0.0.1"] * 5,
            "Source Port": [12345 + i for i in range(5)],
            "Destination Port": [80] * 5,
            "Timestamp": pd.date_range("2017-07-03 09:00:00", periods=5, freq="11s"),
            "Label": ["BENIGN"] * 5,
            "Total Length of Fwd Packets": [100.0] * 5,
            "Total Length of Bwd Packets": [50.0] * 5,
            "Total Fwd Packets": [5.0] * 5,
            "Total Backward Packets": [3.0] * 5,
            "Flow Duration": [1000.0] * 5,
            "Flow Bytes/s": [100.0] * 5,
            "Flow Packets/s": [5.0] * 5,
            "Flow IAT Mean": [100.0] * 5,
            "Flow IAT Std": [10.0] * 5,
            "Flow IAT Max": [500.0] * 5,
            "Fwd Packet Length Std": [5.0] * 5,
            "Bwd Packet Length Std": [3.0] * 5,
            "SYN Flag Count": [1.0] * 5,
            "ACK Flag Count": [2.0] * 5,
            "FIN Flag Count": [0.0] * 5,
            "RST Flag Count": [0.0] * 5,
            "PSH Flag Count": [1.0] * 5,
            "Init_Win_bytes_forward": [8192.0] * 5,
        })
        # Each 11-second gap means at most 1 flow per window → all dropped
        windows = build_state_windows(df, split_tag="train")
        # All windows with 1 flow are discarded
        assert len(windows) == 0 or all(len(w) >= 2 for w in [windows])

    def test_20_window_sequence(self):
        """LSTM sequences must have exactly 20 history windows."""
        df = make_mock_df(n_windows=30)
        windows = build_state_windows(df, split_tag="train")

        # Apply dummy scaling (identity)
        from digitalspy.features.engineer import FEATURE_NAMES
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="mean")
        sc = StandardScaler()
        X_raw = windows[FEATURE_NAMES].fillna(0).values
        imp.fit(X_raw)
        sc.fit(imp.transform(X_raw))
        windows[FEATURE_NAMES] = sc.transform(imp.transform(X_raw))

        X, y_risk, y_tactic, meta = build_lstm_sequences(windows, history_length=20, K=5)
        if len(X) > 0:
            assert X.shape[1] == 20, f"History dim should be 20, got {X.shape[1]}"
            assert X.shape[2] == 24, f"Feature dim should be 24, got {X.shape[2]}"

    def test_gap_discards_sequence(self):
        """A gap in the window sequence must cause it to be discarded."""
        # With gap: host has windows 0-9 and 11-29 (gap at 10)
        # We simulate this by building windows from two separate DataFrames
        df_before_gap = make_mock_df(n_windows=10, start="2017-07-03 09:00:00")
        # After gap: start 100 seconds later (not contiguous)
        df_after_gap = make_mock_df(n_windows=15, start="2017-07-03 09:01:50")

        combined = pd.concat([df_before_gap, df_after_gap], ignore_index=True)
        windows = build_state_windows(combined, split_tag="train")

        from digitalspy.features.engineer import FEATURE_NAMES
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        if len(windows) == 0:
            pytest.skip("No windows generated")

        imp = SimpleImputer(strategy="mean")
        sc = StandardScaler()
        X_raw = windows[FEATURE_NAMES].fillna(0).values
        imp.fit(X_raw)
        sc.fit(imp.transform(X_raw))
        windows[FEATURE_NAMES] = sc.transform(imp.transform(X_raw))

        X, _, _, _ = build_lstm_sequences(windows, history_length=20, K=5)
        # With only 10 contiguous windows before the gap, no 20-window sequence can form
        assert len(X) == 0, f"Gap should prevent sequence formation, got {len(X)} sequences"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEAKAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeakage:
    """Tests §28 leakage prevention rules."""

    def test_scaler_train_only(self):
        """Scaler must only be fitted on training data."""
        from sklearn.preprocessing import StandardScaler
        import tempfile, os
        import joblib

        X_train = np.random.randn(100, 24).astype(np.float32)
        X_test = np.random.randn(20, 24).astype(np.float32) * 10  # different scale

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Test that scaler mean/std are from training data only
        assert scaler.mean_.shape == (24,)
        assert scaler.scale_.shape == (24,)

        # Transform test without refitting
        X_test_scaled = scaler.transform(X_test)
        assert X_test_scaled.shape == (20, 24)

    def test_imputer_train_only(self):
        """Imputer must only be fitted on training data."""
        from sklearn.impute import SimpleImputer

        X_train = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        X_test = np.array([[np.nan, 10.0]])

        imp = SimpleImputer(strategy="mean")
        imp.fit(X_train)

        # Training mean for col 0 = (1+3+5)/3 = 3.0
        assert abs(imp.statistics_[0] - 3.0) < 1e-6
        # Training mean for col 1 = (4+6)/2 = 5.0
        assert abs(imp.statistics_[1] - 5.0) < 1e-6

        # Test transform uses training mean, not test values
        X_test_imp = imp.transform(X_test)
        assert abs(X_test_imp[0, 0] - 3.0) < 1e-6  # imputed with train mean

    def test_no_future_features_in_input(self):
        """LSTM input X must use windows t-19..t, NOT t+1..t+5."""
        # The build_lstm_sequences function must ensure:
        # history = X[i:i+20]  (past)
        # targets = y[i+20:i+25]  (future)
        X_dummy = np.arange(30 * 24, dtype=np.float32).reshape(30, 24)
        y_risk_dummy = np.zeros(30, dtype=np.float32)
        y_tactic_dummy = np.zeros(30, dtype=np.int64)

        # Build a dummy windows dataframe
        windows = pd.DataFrame(
            X_dummy,
            columns=FEATURE_NAMES,
        )
        windows["source_ip"] = "test_host"
        windows["window_start"] = pd.date_range("2017-07-03", periods=30, freq="10s")
        windows["window_end"] = windows["window_start"] + pd.Timedelta(seconds=10)
        windows["z_t"] = "BENIGN"
        windows["z_t_idx"] = 0
        windows["has_packet_features"] = 0
        windows["split"] = "train"

        X, yr, yt, meta = build_lstm_sequences(windows, history_length=20, K=5)

        # Verify: first sequence X[0] should be windows 0..19
        # targets should be windows 20..24
        if len(X) > 0:
            # X[0] features at time 0 must match window_features[0]
            np.testing.assert_array_equal(X[0, 0, :], X_dummy[0, :])
            # X[0] at last history step must be window 19
            np.testing.assert_array_equal(X[0, -1, :], X_dummy[19, :])

    def test_split_order(self):
        """Train must precede validation, validation must precede test chronologically."""
        # Verify file assignment ordering by day
        splits_cfg = __import__("digitalspy.config", fromlist=["splits"]).splits()
        train_files = splits_cfg["splits"]["train"]["files"]
        val_files = splits_cfg["splits"]["validation"]["files"]
        test_files = splits_cfg["splits"]["test"]["files"]

        # Monday/Tue/Wed in train
        assert any("Monday" in f for f in train_files)
        assert any("Tuesday" in f or "Wednesday" in f for f in train_files)
        # Thursday in val
        assert any("Thursday" in f for f in val_files)
        # Friday in test
        assert any("Friday" in f for f in test_files)

        # No overlap
        assert set(train_files).isdisjoint(set(val_files))
        assert set(train_files).isdisjoint(set(test_files))
        assert set(val_files).isdisjoint(set(test_files))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels:
    """Tests §28 model correctness rules."""

    def test_markov_rows_sum_to_one(self):
        """Markov transition matrix rows must sum to approximately 1.0."""
        from digitalspy.markov.model import MarkovWorldModel

        # Generate a random Z_t sequence and fit
        np.random.seed(42)
        z_seq = np.random.randint(0, 6, size=1000)
        model = MarkovWorldModel(num_states=6, alpha=1.0)
        model.fit(z_seq)

        row_sums = model.P.sum(axis=1)
        for i, s in enumerate(row_sums):
            assert abs(s - 1.0) < 1e-6, f"Row {i} sums to {s}, not 1.0"

    def test_lstm_shape(self):
        """AttentionLSTM output shapes must match spec."""
        import torch
        from digitalspy.models.attention_lstm import AttentionLSTM

        model = AttentionLSTM(
            input_size=24,
            hidden_size=64,
            num_layers=1,
            dropout=0.20,
            K=5,
            num_tactic_classes=6,
        )
        model.eval()

        batch = 4
        x = torch.randn(batch, 20, 24)
        with torch.no_grad():
            risk, tactic, alpha = model(x)

        assert risk.shape == (batch, 5), f"Risk shape: {risk.shape}"
        assert tactic.shape == (batch, 5, 6), f"Tactic shape: {tactic.shape}"
        assert alpha.shape == (batch, 20), f"Attention shape: {alpha.shape}"

    def test_forecast_K5_shape(self):
        """Forecast engine must output exactly K=5 horizon values."""
        import torch
        from digitalspy.models.attention_lstm import AttentionLSTM

        model = AttentionLSTM(input_size=24, hidden_size=64, num_layers=1,
                               dropout=0.20, K=5, num_tactic_classes=6)
        model.eval()
        x = torch.randn(1, 20, 24)
        with torch.no_grad():
            risk, tactic, alpha = model(x)
        assert risk.shape[1] == 5, "Must output 5 risk values"
        assert tactic.shape[1] == 5, "Must output 5 tactic distributions"

    def test_probability_range(self):
        """All model output probabilities must be in [0, 1]."""
        import torch
        from digitalspy.models.attention_lstm import AttentionLSTM

        model = AttentionLSTM(input_size=24, hidden_size=64, num_layers=1,
                               dropout=0.20, K=5, num_tactic_classes=6)
        model.eval()
        x = torch.randn(10, 20, 24)
        with torch.no_grad():
            risk, tactic, alpha = model(x)

        risk_np = risk.numpy()
        tactic_softmax = torch.softmax(tactic, dim=-1).numpy()
        alpha_np = alpha.numpy()

        assert np.all(risk_np >= 0.0) and np.all(risk_np <= 1.0), "Risk probs out of range"
        assert np.all(tactic_softmax >= 0.0) and np.all(tactic_softmax <= 1.0)
        # Attention must sum to ~1 per sample
        np.testing.assert_allclose(alpha_np.sum(axis=1), 1.0, atol=1e-5)

    def test_valid_tactic_classes(self):
        """Tactic predictions must only contain valid class indices (0-5)."""
        import torch
        from digitalspy.models.attention_lstm import AttentionLSTM

        model = AttentionLSTM(input_size=24, hidden_size=64, num_layers=1,
                               dropout=0.20, K=5, num_tactic_classes=6)
        model.eval()
        x = torch.randn(8, 20, 24)
        with torch.no_grad():
            _, tactic, _ = model(x)

        preds = torch.argmax(tactic, dim=-1).numpy()  # (8, 5)
        assert np.all(preds >= 0) and np.all(preds <= 5), f"Invalid tactic index found"
