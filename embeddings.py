"""
Pluggable text embedders used by the feature extractor

The original project relies on ChromaDB's default embedding function,
``all-MiniLM-L6-v2``.  For the risk model we need the *numeric* similarity
values (not just the retrieved documents), therefore we wrap the same model
behind a tiny ``encode`` interface.

Two implementations are provided:

``ChromaEmbedder``
    Uses the exact model the existing pipeline already uses
    (``ONNXMiniLM_L6_V2`` == all-MiniLM-L6-v2, 384-d).  Requires the
    ``chromadb`` package; the ONNX weights (~80 MB) are downloaded once.

``TfidfEmbedder``
    Pure scikit-learn fallback (no external download).  Similarities are
    lexical rather than semantic, so metrics obtained with it are slightly
    pessimistic.  It exists so the whole pipeline stays runnable offline.

Both return L2-normalised float32 matrices, hence cosine similarity is a
plain dot product.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import numpy as np


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class BaseEmbedder:
    """Minimal interface: ``encode(list[str]) -> (n, d) float32 matrix``."""

    name = "base"

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def cosine(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        """Pairwise-free cosine utility: returns the full ``len(a) x len(b)`` matrix."""
        va, vb = self.encode(list(a)), self.encode(list(b))
        return va @ vb.T


class ChromaEmbedder(BaseEmbedder):
    """all-MiniLM-L6-v2 through ChromaDB's ONNX runtime (no PyTorch needed)."""

    name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions  # imported lazily

        self._ef = embedding_functions.ONNXMiniLM_L6_V2()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        # Chroma's EF already normalises; normalise again for safety.
        vecs = np.asarray(self._ef(list(texts)), dtype=np.float32)
        return _l2_normalise(vecs)


class TfidfEmbedder(BaseEmbedder):
    """Offline fallback: TF-IDF vectoriser fitted on the supplied corpus."""

    name = "tfidf"

    def __init__(self, corpus: Iterable[str] | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
        )
        corpus = list(corpus or [])
        if corpus:
            self._vec.fit(corpus)
            self._fitted = True
        else:
            self._fitted = False

    def fit(self, corpus: Iterable[str]) -> "TfidfEmbedder":
        self._vec.fit(list(corpus))
        self._fitted = True
        return self

    def _sparse_encode(self, texts: Sequence[str]):
        # Unseen vocabulary is ignored, which is the expected behaviour.
        return self._vec.transform(list(texts))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if not self._fitted:
            self.fit(texts)
        mat = self._sparse_encode(texts).astype(np.float32).toarray()
        return _l2_normalise(mat)


def get_embedder(kind: str = "auto", corpus: Iterable[str] | None = None) -> BaseEmbedder:
    """Factory.

    ``kind='auto'`` prefers the semantic model used by the original pipeline
    and silently degrades to TF-IDF when chromadb / the ONNX weights are not
    available.
    """
    kind = (kind or "auto").lower()
    if kind in ("auto", "chroma", "minilm"):
        try:
            emb = ChromaEmbedder()
            emb.encode(["warm up"])  # forces the one-off weight download
            return emb
        except Exception as exc:  # pragma: no cover - environment dependent
            if kind != "auto":
                raise
            print(f"[embeddings] MiniLM unavailable ({type(exc).__name__}); "
                  f"falling back to TF-IDF.")
    return TfidfEmbedder(corpus)


if __name__ == "__main__":
    e = get_embedder(os.environ.get("EMBEDDER", "auto"))
    v = e.cosine(["How many singers are there?", "count of singers"],
                 ["CREATE TABLE singer ( singer_id int, name text )"])
    print("embedder:", e.name, "shape:", v.shape, "sim:", float(v[0, 0]))
