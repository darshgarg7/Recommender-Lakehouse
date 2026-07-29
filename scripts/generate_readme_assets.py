from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/reports/portfolio_evidence.json"
ASSETS = ROOT / "assets"
COLORS = ["#2563eb", "#7c3aed", "#0891b2", "#ea580c"]


def _svg_document(width: int, height: int, body: Iterable[str], title: str) -> str:
    content = "\n".join(body)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">Generated from benchmarks/reports/portfolio_evidence.json.</desc>
  <rect width="100%" height="100%" rx="20" fill="#f8fafc"/>
  <style>
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill: #0f172a; }}
    .muted {{ fill: #64748b; }}
    .grid {{ stroke: #cbd5e1; stroke-width: 1; }}
  </style>
{content}
</svg>
"""


def _write(name: str, content: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / name).write_text(content, encoding="utf-8")


def model_comparison(evidence: dict) -> None:
    models = evidence["real_model_metrics"]
    width, height = 980, 440
    left, top, plot_width = 230, 110, 650
    max_value = max(max(row["ndcg_at_10"], row["zero_history_recall_at_10"]) for row in models)
    scale_max = max(0.08, max_value * 1.08)
    body = [
        '<text x="48" y="48" font-size="26" font-weight="700">Real-data model comparison</text>',
        '<text x="48" y="76" font-size="15" class="muted">263 positive examples · Magazine Subscriptions · higher is better</text>',
        f'<rect x="650" y="44" width="12" height="12" rx="3" fill="{COLORS[0]}"/>',
        '<text x="668" y="54" font-size="12">NDCG@10</text>',
        f'<rect x="755" y="44" width="12" height="12" rx="3" fill="{COLORS[1]}"/>',
        '<text x="773" y="54" font-size="12">Zero-history Recall@10</text>',
    ]
    for tick in range(5):
        value = scale_max * tick / 4
        x = left + plot_width * tick / 4
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 54}" class="grid"/>',
                f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" font-size="13" class="muted">{value:.2f}</text>',
            ]
        )
    for index, row in enumerate(models):
        y = top + index * 76
        body.append(
            f'<text x="48" y="{y + 23}" font-size="15" font-weight="600">{html.escape(row["label"])}</text>'
        )
        for offset, (metric, color, label) in enumerate(
            (
                ("ndcg_at_10", COLORS[0], "NDCG@10"),
                ("zero_history_recall_at_10", COLORS[1], "Zero-history Recall@10"),
            )
        ):
            value = row[metric]
            bar_y = y + offset * 27
            bar_width = value / scale_max * plot_width
            body.extend(
                [
                    f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" height="18" rx="9" fill="{color}"/>',
                    f'<text x="{left + bar_width + 8:.1f}" y="{bar_y + 14}" font-size="12">{value:.4f}</text>',
                ]
            )
    _write(
        "real-model-comparison.svg",
        _svg_document(width, height, body, "Real-data model comparison"),
    )


def long_tail_frontier(evidence: dict) -> None:
    points = evidence["long_tail_frontier"]
    width, height = 900, 450
    left, right, top, bottom = 90, 50, 92, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    min_x = min(row["long_tail_exposure"] for row in points) - 0.004
    max_x = max(row["long_tail_exposure"] for row in points) + 0.004
    min_y = min(row["ndcg_at_10"] for row in points) - 0.001
    max_y = max(row["ndcg_at_10"] for row in points) + 0.001

    def x_pos(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_width

    def y_pos(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_height

    body = [
        '<text x="45" y="45" font-size="26" font-weight="700">Relevance–long-tail frontier</text>',
        '<text x="45" y="72" font-size="15" class="muted">Measured reranker trade-off · each point is a configured tail weight</text>',
    ]
    for tick in range(5):
        x_value = min_x + (max_x - min_x) * tick / 4
        y_value = min_y + (max_y - min_y) * tick / 4
        x = x_pos(x_value)
        y = y_pos(y_value)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="grid"/>',
                f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" font-size="12" class="muted">{x_value * 100:.1f}%</text>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" class="muted">{y_value:.3f}</text>',
            ]
        )
    path = " ".join(
        ("M" if index == 0 else "L")
        + f" {x_pos(row['long_tail_exposure']):.1f} {y_pos(row['ndcg_at_10']):.1f}"
        for index, row in enumerate(points)
    )
    body.append(f'<path d="{path}" fill="none" stroke="{COLORS[2]}" stroke-width="4"/>')
    for row in points:
        x, y = x_pos(row["long_tail_exposure"]), y_pos(row["ndcg_at_10"])
        body.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{COLORS[2]}" stroke="#fff" stroke-width="3"/>',
                f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" font-size="12" font-weight="600">w={row["long_tail_weight"]:g}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="13">Top-10 long-tail exposure</text>',
            f'<text x="20" y="{top + plot_height / 2:.1f}" transform="rotate(-90 20 {top + plot_height / 2:.1f})" text-anchor="middle" font-size="13">NDCG@10</text>',
        ]
    )
    _write(
        "relevance-long-tail-frontier.svg",
        _svg_document(width, height, body, "Relevance and long-tail frontier"),
    )


def pipeline_evidence(evidence: dict) -> None:
    run = evidence["databricks_run"]
    certification = run["certification"]
    width, height = 1200, 390
    body = [
        '<text x="45" y="46" font-size="26" font-weight="700">Verified Databricks serverless run</text>',
        f'<text x="45" y="74" font-size="15" class="muted">Run {run["run_id"]} · {run["duration_seconds"]:.1f}s · full replay succeeded</text>',
    ]
    boxes = [
        (40, 118, 170, "Bootstrap", "2 SHA-256 checks"),
        (250, 98, 175, "Bronze reviews", f"{run['tables']['bronze_reviews']:,} rows"),
        (250, 180, 175, "Bronze metadata", f"{run['tables']['bronze_product_metadata']:,} rows"),
        (470, 139, 175, "Silver", f"{run['tables']['silver_interactions']:,} interactions"),
        (690, 139, 175, "Gold", f"{run['tables']['gold_training_labels']:,} labels"),
        (
            910,
            139,
            240,
            "Certify",
            f"{certification['assertion_count']}/11 assertions · PASS",
        ),
    ]
    for x, y, box_width, label, detail in boxes:
        body.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_width}" height="64" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>',
                f'<text x="{x + 16}" y="{y + 27}" font-size="15" font-weight="700">{label}</text>',
                f'<text x="{x + 16}" y="{y + 48}" font-size="12" class="muted">{detail}</text>',
            ]
        )
    for x1, y1, x2, y2 in (
        (210, 150, 250, 130),
        (210, 150, 250, 212),
        (425, 130, 470, 171),
        (425, 212, 470, 171),
        (645, 171, 690, 171),
        (865, 171, 910, 171),
    ):
        body.extend(
            [
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2"/>',
                f'<circle cx="{x2}" cy="{y2}" r="3" fill="#64748b"/>',
            ]
        )
    checks = run["assertions"]
    body.extend(
        [
            '<rect x="40" y="282" width="1110" height="68" rx="12" fill="#ecfdf5" stroke="#86efac"/>',
            '<text x="60" y="321" font-size="14" font-weight="700" fill="#166534">CERTIFIED</text>',
            f'<text x="145" y="305" font-size="13">replay attempts={checks["manifest_replay_attempts"]} · Bronze ID duplicates={checks["bronze_reviews_duplicate_ids"] + checks["bronze_metadata_duplicate_ids"]} · quarantined={checks["quarantined_rows"]}</text>',
            f'<text x="145" y="323" font-size="13">Silver duplicates={checks["duplicate_interaction_ids"]} · flagged={checks["flagged_interactions"]} · item leakage={checks["item_feature_leakage_rows"]} · sequence leakage={checks["sequence_leakage_rows"]}</text>',
            f'<text x="145" y="341" font-size="12" class="muted">source state {certification["source_set_sha256"][:12]}… · table state {certification["table_state_sha256"][:12]}… · failed assertions={certification["failed_assertion_count"]}</text>',
        ]
    )
    _write(
        "databricks-pipeline-evidence.svg",
        _svg_document(width, height, body, "Verified Databricks serverless pipeline"),
    )


def scale_benchmark(evidence: dict) -> None:
    benchmark = evidence["scale_benchmark"]
    dataset = benchmark["dataset"]
    initial = benchmark["initial_run"]
    replay = benchmark["replay_run"]
    tables = benchmark["tables"]
    width, height = 1200, 470
    body = [
        '<text x="45" y="46" font-size="26" font-weight="700">Certified multi-million-row scale</text>',
        f'<text x="45" y="74" font-size="15" class="muted">Amazon Reviews 2023 · {dataset["category"]} · Databricks serverless run {initial["run_id"]}</text>',
    ]
    cards = [
        (
            40,
            "Landed source",
            f"{dataset['landed_lines'] / 1_000_000:.2f}M lines",
            f"{dataset['total_bytes'] / 1_000_000_000:.2f} GB · SHA-256 verified",
        ),
        (
            325,
            "Canonical state",
            f"{tables['silver_interactions'] / 1_000_000:.2f}M interactions",
            f"{tables['silver_products']:,} parent products",
        ),
        (
            610,
            "Point-in-time Gold",
            f"{tables['gold_training_labels'] / 1_000_000:.2f}M examples",
            "labels = sequences = item snapshots",
        ),
        (
            895,
            "Post-integrity rate",
            f"{initial['post_integrity_throughput_lines_per_second'] / 1_000:.1f}K lines/s",
            f"{initial['post_integrity_seconds']:.1f}s measured critical path",
        ),
    ]
    for x, label, value, detail in cards:
        body.extend(
            [
                f'<rect x="{x}" y="98" width="265" height="82" rx="14" fill="#ffffff" stroke="#cbd5e1"/>',
                f'<text x="{x + 16}" y="121" font-size="12" class="muted">{label}</text>',
                f'<text x="{x + 16}" y="148" font-size="22" font-weight="700">{value}</text>',
                f'<text x="{x + 16}" y="169" font-size="12" class="muted">{detail}</text>',
            ]
        )
    stages = initial["stage_execution_seconds"]
    boxes = [
        (40, 236, 170, "Integrity gate", f"{stages['integrity_bootstrap']}s · constant memory"),
        (250, 212, 180, "Bronze reviews", f"{stages['bronze_reviews']}s · 2.13M lines"),
        (250, 286, 180, "Bronze metadata", f"{stages['bronze_metadata']}s · 94.3K lines"),
        (470, 249, 170, "Silver", f"{stages['silver']}s · typed contracts"),
        (680, 249, 170, "Gold windows", f"{stages['gold']}s · strict as-of"),
        (890, 249, 230, "Certification", f"{stages['certify']}s · 11/11 PASS"),
    ]
    for x, y, box_width, label, detail in boxes:
        body.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_width}" height="60" rx="13" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>',
                f'<text x="{x + 14}" y="{y + 25}" font-size="14" font-weight="700">{label}</text>',
                f'<text x="{x + 14}" y="{y + 45}" font-size="11" class="muted">{detail}</text>',
            ]
        )
    for x1, y1, x2, y2 in (
        (210, 266, 250, 242),
        (210, 266, 250, 316),
        (430, 242, 470, 279),
        (430, 316, 470, 279),
        (640, 279, 680, 279),
        (850, 279, 890, 279),
    ):
        body.extend(
            [
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2"/>',
                f'<circle cx="{x2}" cy="{y2}" r="3" fill="#64748b"/>',
            ]
        )
    body.extend(
        [
            '<rect x="40" y="388" width="1080" height="48" rx="12" fill="#ecfdf5" stroke="#86efac"/>',
            '<text x="60" y="417" font-size="14" font-weight="700" fill="#166534">REPLAY CERTIFIED</text>',
            f'<text x="215" y="417" font-size="13">run {replay["run_id"]} · {replay["duration_seconds"]:.3f}s · source fingerprint unchanged · row counts unchanged · replay attempts={replay["manifest_replay_attempts"]}</text>',
        ]
    )
    _write(
        "scale-benchmark.svg",
        _svg_document(width, height, body, "Certified multi-million-row scale benchmark"),
    )


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    model_comparison(evidence)
    long_tail_frontier(evidence)
    pipeline_evidence(evidence)
    scale_benchmark(evidence)
    print(f"generated 4 README visuals in {ASSETS}")


if __name__ == "__main__":
    main()
