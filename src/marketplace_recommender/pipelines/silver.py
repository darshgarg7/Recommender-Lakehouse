from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from marketplace_recommender.schemas import (
    DATASET_END_MS,
    DATASET_START_MS,
    as_bool,
    normalize_timestamp,
    stable_interaction_id,
    validate_rating,
)
from marketplace_recommender.storage import read_jsonl, write_jsonl_atomic


KNOWN_PRODUCT_FIELDS = {
    "parent_asin",
    "asin",
    "title",
    "store",
    "brand",
    "main_category",
    "categories",
    "description",
    "features",
    "details",
    "price",
    "images",
    "bought_together",
}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _structured_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw_value": parsed}
        except json.JSONDecodeError:
            return {"raw_value": value}
    return {}


def _image_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key in ("hi_res", "large", "thumb"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                urls.extend(str(item) for item in candidate if item)
            elif candidate:
                urls.append(str(candidate))
    elif isinstance(value, list):
        for image in value:
            if isinstance(image, dict):
                candidate = image.get("hi_res") or image.get("large") or image.get("thumb")
                if candidate:
                    urls.append(str(candidate))
            elif image:
                urls.append(str(image))
    return list(dict.fromkeys(urls))


def build_silver(bronze_dir: str | Path, silver_dir: str | Path) -> dict[str, int]:
    source = Path(bronze_dir)
    target = Path(silver_dir)
    target.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    products_by_parent: dict[str, dict[str, Any]] = {}
    asin_to_parent: dict[str, str] = {}
    variants: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    product_images: list[dict[str, Any]] = []

    metadata_path = source / "bronze_product_metadata.jsonl"
    for row in read_jsonl(metadata_path):
        raw = row["raw_payload"]
        try:
            parent = str(raw.get("parent_asin") or raw.get("asin") or "").strip()
            asin = str(raw.get("asin") or parent).strip()
            if not parent or not asin:
                raise ValueError("missing asin and parent_asin")
            previous = asin_to_parent.get(asin)
            if previous is not None and previous != parent:
                raise ValueError(
                    f"non-deterministic parent mapping for {asin}: {previous}, {parent}"
                )
            asin_to_parent[asin] = parent
            category_path = _strings(raw.get("categories") or raw.get("main_category"))
            processed_at = row["ingested_at"]
            image_urls = _image_urls(raw.get("images"))
            product = {
                "parent_asin": parent,
                "domain": row["source_domain"],
                "title": str(raw.get("title") or "").strip(),
                "brand_or_store": str(raw.get("store") or raw.get("brand") or "").strip(),
                "category_path": category_path,
                "description": _strings(raw.get("description")),
                "feature_bullets": _strings(raw.get("features")),
                "structured_attributes": _structured_attributes(raw.get("details")),
                "crawl_price": raw.get("price"),
                "image_references": image_urls,
                "source_file": row["source_file"],
                "processed_at": processed_at,
                "quality_status": "valid",
                "rescued_fields": {
                    key: value for key, value in raw.items() if key not in KNOWN_PRODUCT_FIELDS
                },
            }
            # Parent rows can appear once per variant. Pick the stable lexicographic source record.
            if (
                parent not in products_by_parent
                or asin < products_by_parent[parent]["representative_asin"]
            ):
                product["representative_asin"] = asin
                products_by_parent[parent] = product
            variants.append({"asin": asin, "parent_asin": parent, "domain": row["source_domain"]})
            categories.extend(
                {"parent_asin": parent, "category_level": level, "category": category}
                for level, category in enumerate(category_path)
            )
            details = _structured_attributes(raw.get("details"))
            attributes.extend(
                {"parent_asin": parent, "attribute_key": str(key), "attribute_value": value}
                for key, value in sorted(details.items())
            )
            for neighbor in _strings(raw.get("bought_together")):
                graph_edges.append({"parent_asin": parent, "related_parent_asin": neighbor})
            product_images.extend({"parent_asin": parent, "image_url": url} for url in image_urls)
        except (TypeError, ValueError) as exc:
            failures.append(_failure("product", row, exc))

    interaction_map: dict[str, dict[str, Any]] = {}
    review_text: list[dict[str, Any]] = []
    review_images: list[dict[str, Any]] = []
    reviews_path = source / "bronze_reviews.jsonl"
    for row in read_jsonl(reviews_path):
        raw = row["raw_payload"]
        try:
            user_id = str(raw.get("user_id") or "").strip()
            asin = str(raw.get("asin") or "").strip()
            if not user_id or not asin:
                raise ValueError("missing user_id or asin")
            timestamp = normalize_timestamp(raw.get("timestamp"))
            if not DATASET_START_MS <= timestamp < DATASET_END_MS:
                raise ValueError(f"timestamp outside published dataset range: {timestamp}")
            rating = validate_rating(raw.get("rating"))
            parent = str(raw.get("parent_asin") or asin_to_parent.get(asin) or asin).strip()
            interaction_id = stable_interaction_id(user_id, asin, timestamp, rating)
            interaction = {
                "interaction_id": interaction_id,
                "user_id": user_id,
                "asin": asin,
                "parent_asin": parent,
                "domain": row["source_domain"],
                "rating": rating,
                "verified_purchase": as_bool(raw.get("verified_purchase", False)),
                "review_timestamp": timestamp,
                "review_title": str(raw.get("title") or ""),
                "review_text": str(raw.get("text") or ""),
                "helpful_votes": int(raw.get("helpful_vote") or 0),
                "source_file": row["source_file"],
                "processed_at": row["ingested_at"],
                "quality_status": "valid" if asin in asin_to_parent else "missing_product_metadata",
            }
            # Stable ID deduplicates exact review replays. Lowest provenance tuple wins.
            old = interaction_map.get(interaction_id)
            if old is None or interaction["source_file"] < old["source_file"]:
                interaction_map[interaction_id] = interaction
            review_text.append(
                {
                    "interaction_id": interaction_id,
                    "review_title": interaction["review_title"],
                    "review_text": interaction["review_text"],
                }
            )
            for image in raw.get("images") or []:
                review_images.append({"interaction_id": interaction_id, "image_reference": image})
        except (TypeError, ValueError) as exc:
            failures.append(_failure("interaction", row, exc))

    interactions = sorted(
        interaction_map.values(),
        key=lambda item: (item["review_timestamp"], item["interaction_id"]),
    )
    first_seen: dict[str, int] = {}
    for interaction in interactions:
        parent = interaction["parent_asin"]
        first_seen[parent] = min(
            first_seen.get(parent, interaction["review_timestamp"]), interaction["review_timestamp"]
        )
    products = []
    for parent, product in sorted(products_by_parent.items()):
        product["first_observed_interaction_at"] = first_seen.get(parent)
        product.pop("representative_asin", None)
        products.append(product)

    tables = {
        "silver_interactions": interactions,
        "silver_products": products,
        "silver_product_variants": _unique(variants),
        "silver_categories": _unique(categories),
        "silver_product_attributes": _unique(attributes),
        "silver_bought_together_edges": _unique(graph_edges),
        "silver_review_text": _unique(review_text),
        "silver_review_images": _unique(review_images),
        "silver_product_images": _unique(product_images),
        "silver_quality_failures": failures,
    }
    for name, records in tables.items():
        write_jsonl_atomic(target / f"{name}.jsonl", records)
    return {name: len(records) for name, records in tables.items()}


def _failure(kind: str, row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "record_kind": kind,
        "source_file": row["source_file"],
        "source_row_number": row["source_row_number"],
        "reason": str(exc),
    }


def _unique(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_value = {repr(sorted(row.items())): row for row in records}
    return [by_value[key] for key in sorted(by_value)]
