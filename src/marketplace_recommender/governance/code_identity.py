from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def source_tree_sha256(paths: Iterable[str | Path]) -> str:
    """Hash a set of source files independently of its checkout or wheel location."""
    normalized = sorted((Path(path) for path in paths), key=lambda path: path.name)
    if not normalized:
        raise ValueError("at least one implementation source is required")
    digest = hashlib.sha256()
    for path in normalized:
        content = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()
