from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidenceCapabilities:
    """The signals an item can legitimately use at a decision timestamp."""

    content_available: bool
    behavioral_events: int
    observed_retrieval_channels: tuple[str, ...]
    representation_strategy: str
    evidence_capabilities: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["observed_retrieval_channels"] = list(self.observed_retrieval_channels)
        record["evidence_capabilities"] = list(self.evidence_capabilities)
        return record


def resolve_capabilities(
    *,
    content_available: bool,
    behavioral_events: int,
    observed_retrieval_channels: tuple[str, ...] = (),
) -> EvidenceCapabilities:
    """Resolve a progressive representation path without inventing unavailable evidence."""
    if behavioral_events < 0:
        raise ValueError("behavioral_events cannot be negative")
    capabilities = []
    if content_available:
        capabilities.append("content")
    if behavioral_events > 0:
        capabilities.extend(("collaborative", "popularity"))
    if "bought_together" in observed_retrieval_channels:
        capabilities.append("catalog_graph")
    if behavioral_events == 0 and content_available:
        strategy = "content_cold_start"
    elif 0 < behavioral_events <= 10 and content_available:
        strategy = "distilled_sparse_hybrid"
    elif behavioral_events > 10 and content_available:
        strategy = "warm_hybrid"
    elif behavioral_events > 0:
        strategy = "collaborative_only"
    elif "bought_together" in observed_retrieval_channels:
        strategy = "graph_seeded_fallback"
    else:
        strategy = "catalog_fallback"
    return EvidenceCapabilities(
        content_available=content_available,
        behavioral_events=behavioral_events,
        observed_retrieval_channels=tuple(sorted(set(observed_retrieval_channels))),
        representation_strategy=strategy,
        evidence_capabilities=tuple(capabilities),
    )
