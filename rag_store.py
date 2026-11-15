"""
Schema retrieval used by the Bayesian

This is the same idea as the original ChromaDB retrieval (top-k table DDLs
by cosine similarity), but implemented in-process so that the *numeric*
similarity values are available to the risk model, and so that a database
that has never been indexed still works (plug-and-play, as in the web demo).

It is a thin layer: ``retrieve()`` returns both the prompt
text and the selected table indices.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from config import SCHEMA_TOP_K
from embeddings import BaseEmbedder, get_embedder
from feature_extractor import DbSchema, load_db_schema


class SchemaIndex:
    def __init__(self, embedder: BaseEmbedder | None = None, top_k: int = SCHEMA_TOP_K):
        self.embedder = embedder or get_embedder("auto")
        self.top_k = top_k
        self._cache: Dict[str, np.ndarray] = {}

    def _ddl_vectors(self, schema: DbSchema) -> np.ndarray:
        if schema.db_id not in self._cache:
            self._cache[schema.db_id] = self.embedder.encode(schema.ddl)
        return self._cache[schema.db_id]

    def retrieve(self, query: str, db_id: str, db_path: str,
                 top_k: int | None = None) -> Tuple[str, List[int]]:
        schema = load_db_schema(db_path, db_id)
        if not schema.ddl:
            return "", []
        k = min(top_k or self.top_k, len(schema.ddl))
        q_vec = self.embedder.encode([query])[0]
        sims = self._ddl_vectors(schema) @ q_vec
        indices = list(np.argsort(-sims)[:k])
        return schema.to_prompt(indices), indices

    def retrieve_text(self, query: str, db_id: str, db_path: str,
                      top_k: int | None = None) -> str:
        return self.retrieve(query, db_id, db_path, top_k)[0]


if __name__ == "__main__":
    from config import DATABASE_DIR
    idx = SchemaIndex()
    text, ids = idx.retrieve("How many singers do we have?", "concert_singer",
                             os.path.join(DATABASE_DIR, "concert_singer",
                                          "concert_singer.sqlite"))
    print("retrieved tables:", ids)
    print(text[:400])
