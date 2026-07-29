from __future__ import annotations

from marketplace_recommender.retrieval.sasrec import SequentialCooccurrenceTeacher


class LocalImplicitBaseline(SequentialCooccurrenceTeacher):
    """Dependency-free smoke baseline; Databricks jobs select Spark MLlib ALS."""
