"""Text embeddings: turning text into vectors whose closeness reflects closeness in meaning."""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

EMBEDDING_BATCH_SIZE = 32
# Role prefixes the E5 models were trained with; part of the index fingerprint (changing them invalidates it)
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class IEmbedder(ABC):
    @abstractmethod
    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embeds corpus chunks; returns an array of shape (len(texts), dimension)."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a user query; returns a vector of shape (dimension,)."""


class E5Embedder(IEmbedder):
    """Sentence-transformers wrapper for the multilingual E5 family.

    E5 models are trained with role prefixes: "query: " for questions and "passage: " for
    documents; omitting them noticeably hurts retrieval quality. Vectors are L2-normalized, so
    the dot product of two vectors equals their cosine similarity.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, kept local

        self._model = SentenceTransformer(model_name)

    @property
    def max_tokens(self) -> int:
        """Longer inputs are silently truncated by the model."""
        return int(self._model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(f"{PASSAGE_PREFIX}{text}")["input_ids"])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode([f"{PASSAGE_PREFIX}{text}" for text in texts], batch_size=EMBEDDING_BATCH_SIZE,
                                  normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(f"{QUERY_PREFIX}{text}", normalize_embeddings=True, show_progress_bar=False)
