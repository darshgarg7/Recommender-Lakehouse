# Benchmark reports

Reports must state exact rows, bytes, runtime, cluster configuration, failures, and cost. Compare
only decisions that matter: JSON normalization, parent mapping, incremental sequence/features,
hard negatives, embedding refresh, and ANN synchronization.

`portfolio_evidence.json` is the checked, machine-readable source for the README charts. Its
Databricks section records the final successful replay and direct post-run SQL assertions; its
model sections are copied from deterministic local artifacts and live certified Spark ALS, causal
SASRec, MLflow-lineage, and managed AI Search benchmarks. A model can be registered while remaining
ineligible for a serving alias; evidence records both states explicitly. Regenerate the public
visuals with `make assets`.
