# Benchmark reports

Reports must state exact rows, bytes, runtime, cluster configuration, failures, and cost. Compare
only decisions that matter: JSON normalization, parent mapping, incremental sequence/features,
hard negatives, embedding refresh, and ANN synchronization.

`portfolio_evidence.json` is the checked, machine-readable source for the README charts. Its
Databricks section records the final successful replay and direct post-run SQL assertions; its
model sections are copied from the deterministic real-data experiment artifacts. Regenerate all
three public visuals with `make assets`.
