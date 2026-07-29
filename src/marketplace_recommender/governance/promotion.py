from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionPolicy:
    """Fail-closed policy for turning an experiment into a serving decision."""

    version: str = "promotion-policy/v1"
    max_relative_relevance_regression: float = 0.02
    max_relative_retrieval_regression: float = 0.02
    max_relative_cohort_regression: float = 0.02
    required_cohorts: tuple[str, ...] = ("zero-history", "sparse")
    cohort_metric: str = "recall_at_10"

    def __post_init__(self) -> None:
        for name in (
            "max_relative_relevance_regression",
            "max_relative_retrieval_regression",
            "max_relative_cohort_regression",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not self.required_cohorts:
            raise ValueError("promotion policy requires at least one protected cohort")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "PromotionPolicy":
        values = dict(raw or {})
        if "required_cohorts" in values:
            values["required_cohorts"] = tuple(values["required_cohorts"])
        return cls(**values)

    def contract(self) -> dict[str, Any]:
        body = asdict(self)
        body["required_cohorts"] = list(self.required_cohorts)
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return {**body, "policy_id": hashlib.sha256(canonical).hexdigest()}


def _metric_guardrail(
    *,
    name: str,
    metric: str,
    baseline: float | None,
    candidate: float | None,
    max_relative_regression: float,
    sample_count: int | None = None,
) -> dict[str, Any]:
    threshold = baseline * (1.0 - max_relative_regression) if baseline is not None else None
    passed = candidate is not None and threshold is not None and candidate >= threshold
    return {
        "name": name,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "threshold": threshold,
        "max_relative_regression": max_relative_regression,
        "sample_count": sample_count,
        "passed": passed,
        "failure_reason": None if passed else "missing_metric_or_below_threshold",
    }


def _nested_float(report: dict[str, Any], *path: str) -> float | None:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def promotion_decision(
    reports: dict[str, dict[str, Any]],
    *,
    candidate: str = "full_reranked",
    policy: PromotionPolicy | None = None,
) -> dict[str, Any]:
    """Select a champion only when every declared guardrail is supported and passes."""
    chosen_policy = policy or PromotionPolicy()
    if candidate not in reports:
        raise ValueError(f"missing candidate report: {candidate}")
    baselines = [name for name in reports if name != candidate]
    if not baselines:
        raise ValueError("promotion requires at least one baseline")
    best_baseline = max(
        baselines,
        key=lambda name: _nested_float(reports[name], "ranking", "ndcg_at_10") or 0.0,
    )
    baseline_report = reports[best_baseline]
    candidate_report = reports[candidate]
    guardrails = [
        _metric_guardrail(
            name="aggregate_relevance_noninferiority",
            metric="ndcg_at_10",
            baseline=_nested_float(baseline_report, "ranking", "ndcg_at_10"),
            candidate=_nested_float(candidate_report, "ranking", "ndcg_at_10"),
            max_relative_regression=chosen_policy.max_relative_relevance_regression,
            sample_count=int(candidate_report.get("example_count", 0)),
        ),
        _metric_guardrail(
            name="retrieval_coverage_noninferiority",
            metric="candidate_coverage",
            baseline=_nested_float(baseline_report, "retrieval", "candidate_coverage"),
            candidate=_nested_float(candidate_report, "retrieval", "candidate_coverage"),
            max_relative_regression=chosen_policy.max_relative_retrieval_regression,
            sample_count=int(candidate_report.get("example_count", 0)),
        ),
    ]
    for cohort in chosen_policy.required_cohorts:
        candidate_cohort = candidate_report.get("cohorts", {}).get(cohort, {})
        guardrails.append(
            _metric_guardrail(
                name=f"cohort_{cohort}_noninferiority",
                metric=chosen_policy.cohort_metric,
                baseline=_nested_float(
                    baseline_report, "cohorts", cohort, chosen_policy.cohort_metric
                ),
                candidate=_nested_float(
                    candidate_report, "cohorts", cohort, chosen_policy.cohort_metric
                ),
                max_relative_regression=chosen_policy.max_relative_cohort_regression,
                sample_count=int(candidate_cohort.get("example_count", 0)),
            )
        )
    promoted = all(check["passed"] for check in guardrails)
    by_name = {check["name"]: check for check in guardrails}
    relevance = by_name["aggregate_relevance_noninferiority"]
    retrieval = by_name["retrieval_coverage_noninferiority"]
    cold = by_name.get("cohort_zero-history_noninferiority")
    return {
        "candidate": candidate,
        "best_baseline": best_baseline,
        "serving_champion": candidate if promoted else best_baseline,
        "promoted": promoted,
        "policy": chosen_policy.contract(),
        "guardrails": guardrails,
        "relevance_guardrail_passed": relevance["passed"],
        "retrieval_guardrail_passed": retrieval["passed"],
        "cold_start_guardrail_passed": bool(cold and cold["passed"]),
        "cohort_guardrails_passed": all(
            check["passed"] for check in guardrails if check["name"].startswith("cohort_")
        ),
        "max_relative_relevance_regression": (chosen_policy.max_relative_relevance_regression),
        "baseline_ndcg_at_10": relevance["baseline"],
        "candidate_ndcg_at_10": relevance["candidate"],
        "baseline_zero_history_recall_at_10": cold["baseline"] if cold else None,
        "candidate_zero_history_recall_at_10": cold["candidate"] if cold else None,
    }
