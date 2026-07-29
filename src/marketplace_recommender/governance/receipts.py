from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from marketplace_recommender.storage import file_checksum

RECEIPT_SCHEMA = "marketplace-run-receipt/v1"


class ReceiptVerificationError(ValueError):
    """Raised when a run receipt or one of its bound artifacts has been altered."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_relative(root: Path, artifact: Path) -> str:
    resolved_root = root.resolve()
    resolved_artifact = artifact.resolve()
    if not resolved_artifact.is_relative_to(resolved_root):
        raise ValueError(f"artifact must be inside the run root: {artifact}")
    return resolved_artifact.relative_to(resolved_root).as_posix()


def write_run_receipt(
    destination: str | Path,
    *,
    run_root: str | Path,
    identity: dict[str, Any],
    source_contract: dict[str, str],
    temporal_contract: dict[str, Any],
    decision_contract: dict[str, Any],
    verified_claims: dict[str, Any],
    artifacts: dict[str, str | Path],
) -> dict[str, Any]:
    """Bind inputs, time boundaries, decisions, and outputs into one content-addressed receipt."""
    root = Path(run_root).resolve()
    bound_artifacts: dict[str, dict[str, Any]] = {}
    for logical_name, raw_path in sorted(artifacts.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        bound_artifacts[logical_name] = {
            "path": _safe_relative(root, path),
            "sha256": file_checksum(path),
            "bytes": path.stat().st_size,
        }
    payload = {
        "schema": RECEIPT_SCHEMA,
        "identity": identity,
        "source_contract": dict(sorted(source_contract.items())),
        "temporal_contract": temporal_contract,
        "decision_contract": decision_contract,
        "verified_claims": verified_claims,
        "artifacts": bound_artifacts,
    }
    envelope = {
        "algorithm": "sha256-canonical-json",
        "payload_sha256": _digest(payload),
        "payload": payload,
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return envelope


def verify_run_receipt(receipt_path: str | Path, run_root: str | Path) -> dict[str, Any]:
    """Independently verify the receipt envelope and every content-bound artifact."""
    path = Path(receipt_path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        raise ReceiptVerificationError("unsupported or missing receipt schema")
    expected_payload_digest = envelope.get("payload_sha256")
    actual_payload_digest = _digest(payload)
    if expected_payload_digest != actual_payload_digest:
        raise ReceiptVerificationError("receipt payload digest mismatch")
    root = Path(run_root).resolve()
    verified: list[str] = []
    for logical_name, descriptor in sorted(payload.get("artifacts", {}).items()):
        artifact = (root / descriptor["path"]).resolve()
        if not artifact.is_relative_to(root):
            raise ReceiptVerificationError(f"artifact escapes run root: {logical_name}")
        if not artifact.is_file():
            raise ReceiptVerificationError(f"artifact is missing: {logical_name}")
        if artifact.stat().st_size != descriptor["bytes"]:
            raise ReceiptVerificationError(f"artifact byte count mismatch: {logical_name}")
        if file_checksum(artifact) != descriptor["sha256"]:
            raise ReceiptVerificationError(f"artifact digest mismatch: {logical_name}")
        verified.append(logical_name)
    if not verified:
        raise ReceiptVerificationError("receipt binds no artifacts")
    return {
        "valid": True,
        "schema": RECEIPT_SCHEMA,
        "payload_sha256": actual_payload_digest,
        "verified_artifacts": verified,
    }
