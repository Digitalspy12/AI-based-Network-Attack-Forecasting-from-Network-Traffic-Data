# Head Consistency Diagnostic

## Summary
The current prototype uses separate binary-risk and multi-class state heads, which can produce probability disagreement in some windows. The demo presentation therefore gates state display by the frozen overall-risk threshold. A future unified risk definition or joint calibration would remove this inconsistency at the modeling layer.

## Metrics (N=1000 samples)
- **Pearson Correlation**: 0.8288
- **Spearman Correlation**: 0.9608
- **Mean Absolute Disagreement**: 0.1383
- **Median Absolute Disagreement**: 0.0167
- **Agreement Above Threshold (>0.90)**: 100.0%
- **Agreement Below Threshold (<0.90)**: 83.2%
