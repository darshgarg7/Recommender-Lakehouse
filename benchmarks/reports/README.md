# Benchmark and evidence report

This directory is the detailed evidence layer behind the repository landing page.
`portfolio_evidence.json` is the machine-readable source for every published number and generated
chart; `assets/screenshots/` contains sanitized captures from the live Databricks workspace.

The evidence is intentionally separated into four claims:

1. replay-safe lakehouse execution across two Amazon Reviews 2023 categories;
2. multi-million-row batch scale on Appliances;
3. temporal model evaluation with paired uncertainty and fail-closed release decisions;
4. managed vector retrieval quality and load under a bounded Free Edition experiment.

These are offline portfolio results, not production traffic, causal marketplace lift, or a paid
capacity benchmark.

## Evidence policy

Every published managed-platform run must include a Databricks job ID and run ID, terminal state,
input identity, output counts, contract version, failed-check count, and implementation identity
where model logic is involved. Screenshots make the private workspace state visually inspectable;
the checked JSON makes exact values diffable and testable.

The repository follows three release rules:

- validation selects parameters; the final temporal test is touched once;
- matched-user bootstrap intervals, not point estimates alone, govern release;
- registration records lineage, while an alias records deployment eligibility.

Consequently, a registered model may correctly have no serving alias.

## Cross-domain lakehouse portability

The same Asset Bundle and package run against two isolated targets.

| Target | Category | Landed JSONL | Silver interactions | Gold labels | Interpretation |
|---|---|---:|---:|---:|---|
| `dev` | Magazine Subscriptions | 74,888 | 70,922 | 53,897 | second-domain portability |
| `scale` | Appliances | 2,222,932 | 2,105,948 | 1,921,223 | scale and replay benchmark |
| **Aggregate** | **2 categories** | **2,297,820** | **2,176,870** | **1,975,120** | not full-corpus scale |

This demonstrates schema portability across materially different catalog shapes. It does not
establish multi-region, multi-tenant, streaming, or all-category production scale.

The current-code receipts are Magazine run `1080552669029878` (168.212 seconds) and Appliances run
`926465572087562` (179.316 seconds). Both passed `lakehouse-certification/v2` with 13/13 checks,
including duplicate-content checks that are independent of ID uniqueness.

The Magazine input is 37,393,554 bytes: 71,497 landed review lines and 3,391 metadata lines.
Content identity reduces that to 70,922 unique Bronze review payloads and 3,391 metadata payloads.
The certified state contains zero duplicate Bronze IDs, duplicate Bronze content, duplicate Silver
interaction IDs, quarantined rows, flagged interactions, item-feature leakage rows, or
sequence-leakage rows.

## Appliances batch scale

The `scale` target fixes both landed SHA-256 digests in `databricks.yml`, uses an isolated Unity
Catalog schema prefix, and recomputes each digest in bounded memory before committing a manifest.

| Boundary | Observed result |
|---|---:|
| Landed source | 2,222,932 lines; 1,214,749,887 bytes |
| Review object | 2,128,605 lines; 929,451,412 bytes |
| Metadata object | 94,327 lines; 285,298,475 bytes |
| Unique Bronze state | 2,105,949 reviews; 94,327 products |
| Canonical Silver state | 2,105,948 interactions; 94,327 parent products |
| Strict point-in-time Gold | 1,921,223 aligned labels, sequences, and snapshots |
| Principal Delta footprint | 1,066,761,382 bytes; 106 files; 7 tables |

Exact-content addressing collapses byte-identical immutable-source rows without assigning identity
from file order. Gold uses partitioned strict-prior windows rather than label-to-history
self-joins, excluding timestamp ties and avoiding quadratic join growth.

The original cold run completed in 695.653 seconds. The integrity gate consumed 539 seconds on
Free Edition; the post-integrity critical path transformed and certified 2.22M landed lines in
152.653 seconds, or 14,561.99 lines/second. The counted replay completed in 172.780 seconds with
the same source fingerprint and business row counts.

### Delta footprint

| Table | Rows | Files | Bytes |
|---|---:|---:|---:|
| `bronze_reviews` | 2,105,949 | 32 | 348,457,994 |
| `bronze_product_metadata` | 94,327 | 32 | 66,164,245 |
| `silver_products` | 94,327 | 17 | 39,821,773 |
| `silver_interactions` | 2,105,948 | 11 | 290,390,029 |
| `gold_training_labels` | 1,921,223 | 2 | 111,749,431 |
| `gold_user_sequences_asof` | 1,921,223 | 6 | 106,850,590 |
| `gold_item_statistics_asof` | 1,921,223 | 6 | 103,327,320 |

## Distributed ALS benchmark

The independent Spark job reads the versioned Appliances Silver table and applies a global
80/10/10 temporal split. Reciprocal-rank-fusion weight is selected on validation, the model is
refit through the validation boundary, and the future test is used only for release qualification.
Targets are warm, previously unseen positive products; training history is excluded from every
recommendation list.

The reported fit used rank 64, 12 iterations, 345,729 training events, 145,551 user factors,
24,443 item factors, 4,467 validation users, and every 4,316 eligible test user. Exact distributed
percentiles define the temporal cutoffs, so table file layout cannot move the split.

| Test policy | Candidate R@100 | Recall@10 | NDCG@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|
| Popularity, serving | 0.09430 | 0.01691 | 0.00890 | 0.00057 |
| Spark implicit ALS | 0.06117 | 0.01321 | 0.00644 | **0.03257** |
| Temporal hybrid RRF | **0.10009** | **0.01923** | **0.00961** | 0.00217 |

The 10,000-sample paired bootstrap compares the selected hybrid with popularity over identical
users:

| Delta | Estimate | Paired 95% interval | Two-sided p |
|---|---:|---:|---:|
| Candidate Recall@100 | +0.00579 | [+0.00301, +0.00857] | 0.0002 |
| Recall@10 | +0.00232 | [-0.00070, +0.00533] | 0.1454 |
| NDCG@10 | +0.00070 | [-0.00099, +0.00262] | 0.4242 |

Retrieval breadth improves significantly, but top-ten ranking does not. The release rule requires
the lower bound of paired NDCG@10 improvement to exceed zero, so popularity remains champion.

## Causal SASRec and MLflow lineage

The sequence candidate is a two-layer causal PyTorch TransformerEncoder with four heads,
64-dimensional states, 20-event histories, sampled pairwise loss, and validation-only epoch
selection. It trains on 47,853 next-novel-product examples from 30,000 users and a 40,622-item
vocabulary, then evaluates 4,000 future users.

| Test policy | Candidate R@100 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.09325 | 0.01425 | 0.00606 |
| SASRec | 0.00625 | 0.00000 | 0.00000 |

The paired SASRec-minus-popularity NDCG@10 delta is -0.00606 with a 95% interval of
[-0.00786, -0.00447]. The model is rejected. MLflow still records the Delta source version,
parameters, metrics, TorchScript artifact, run, and Unity Catalog registration; the serving alias
remains unset. That distinction is the purpose of the lineage path.

`NVIDIA_API_KEY` is not used. Nemotron is an LLM, not a sequential recommender, and no external LLM
is required by ingestion, ALS, SASRec, evaluation, retrieval, or deployment lineage.

## Managed AI Search

All 24,443 ALS item factors are transformed from 64-dimensional maximum-inner-product vectors to
equivalent 65-dimensional L2 vectors and synchronized into a Databricks AI Search Delta Sync
index. The benchmark waits for no pending update and a processed Delta commit at least as new as
the source before measuring quality.

| Measure | Observed | Contract |
|---|---:|---:|
| Indexed rows | 24,443 / 24,443 | complete |
| Exact-oracle Recall@10 | 0.904 | at least 0.85 |
| Latency p50 | 213.1 ms | reported |
| Latency p95 | 399.3 ms | at most 1,000 ms |
| Latency p99 | 439.1 ms | reported |
| Completed throughput | 67.0 requests/s | 500/500, concurrency 16 |

Recall is measured against a complete exact ALS inner-product ranking for 200 users. Latency
includes managed retrieval and exact rescoring of the 500-candidate pool. It is not browser
latency, autoscaling evidence, or a paid-workspace service-level objective.

## Local reference path

The dependency-light local path is a deterministic contract oracle for content cold start,
capability routing, ranking, reranking, fail-closed promotion, batch serving, and tamper-evident
receipts. It is intentionally smaller than the distributed benchmark so CI can execute it on
Python 3.11–3.13 without Spark or Torch.

The local real-data scope uses 400 repeat users, 1,683 interactions, 263 future positives, and the
same temporal rules. Popularity wins aggregate NDCG@10; content provides the strongest
zero-history recall. The serving output consumes the actual promotion decision rather than a
hard-coded learned-model label.

## Verification surfaces

- `portfolio_evidence.json`: exact published metrics, run identities, hashes, and decisions.
- `assets/screenshots/databricks-job-dag.png`: real six-task Databricks job graph.
- `assets/screenshots/mlflow-model-lineage.png`: real registered-model lineage view.
- `assets/screenshots/databricks-ai-search-benchmark.png`: real managed retrieval benchmark output.
- `diagram.md`: architecture and failure semantics without performance claims.
- `make verify`: formatting, Ruff, MyPy, branch coverage, tests, replay, receipt, asset drift, wheel.

The 85% branch-coverage floor applies to the deterministic core. Runtime-only Spark, Torch,
Databricks, MLflow, AI Search, and serving adapters are explicitly omitted from that local number
and are instead bound to live managed-platform contracts. This avoids inflating coverage with
mocks that cannot establish platform behavior.

Regenerate derived SVGs with `make assets`. Contract tests fail when the checked benchmark
contract version or implementation tree hash no longer matches the live evidence.
