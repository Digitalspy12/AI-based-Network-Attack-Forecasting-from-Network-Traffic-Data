# Final Demo Verification

This artifact serves as the final evidence that the presentation sequence is real, scientifically coherent, and reproducible.

## Scenario Details
- **Scenario ID**: seq_4381
- **Start Window Index**: 4381
- **Start Timestamp**: 88000
- **Start State**: BENIGN

## Attack Ground Truth
- **Attack Onset Window**: 4405
- **Attack Onset Timestamp**: 88100
- **Attack Onset State**: IMPACT

## Model Forecast Evidence
- **Threshold**: 0.9
- **First Threshold Crossing**: +10.0s
- **Proactive Lead Time**: 40.0s
- **Risk Forecast**:
  - t+1 (+10s): 0.9071
  - t+2 (+20s): 0.9337
  - t+3 (+30s): 0.8856
  - t+4 (+40s): 0.8825
  - t+5 (+50s): 0.8976

## Verification Conclusion
**VALID**: True
The forecast was generated based on the BENIGN history, and the threshold was crossed proactively before the attack onset occurred.
