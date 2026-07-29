from __future__ import annotations

from typing import Any


def create_app(system: Any, timestamp: int, rerank_config: dict[str, Any]):
    """Create the optional online demo; FastAPI is imported only when this is used."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install the 'serve' extra to run the online endpoint") from exc

    class Request(BaseModel):
        user_id: str
        domain: str | None = None
        limit: int = Field(default=20, ge=1, le=100)

    app = FastAPI(title="Marketplace cold-start recommender", version="0.1.0")

    @app.post("/recommendations")
    def recommendations(request: Request) -> dict[str, Any]:
        if not request.user_id.startswith("demo_"):
            raise HTTPException(400, "the public demo accepts synthetic demo_ personas only")
        _, rows = system.recommend(
            request.user_id,
            timestamp,
            candidate_limit=max(100, request.limit * 5),
            limit=request.limit,
            rerank_config=rerank_config,
            domain=request.domain,
        )
        return {"recommendations": rows}

    return app
