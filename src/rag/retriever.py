"""Retrieval: query -> top-k chunks, with optional second-stage re-ranking."""

from abc import ABC, abstractmethod
from typing import List, Sequence

from src.config.lab4_config import RetrievalConfig
from src.rag.embeddings import IEmbedder
from src.rag.vector_store import IVectorStore, RetrievedChunk


class IReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        """Returns the candidates re-scored and sorted by relevance, best first."""


class CosineRanker(IReranker):
    """Ranks by cosine similarity between the query vector and the chunk vector (the first-stage score)."""

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        return sorted(candidates, key=lambda item: item.score, reverse=True)


class CrossEncoderReranker(IReranker):
    """Re-scores every (query, chunk) pair with a cross-encoder.

    The embedding model (bi-encoder) compresses the query and the chunk into vectors separately,
    so it never "reads" them together. A cross-encoder reads both texts jointly and predicts a
    relevance score in [0, 1]: more accurate but much slower, so it is applied only to the few
    candidates returned by the vector search.
    """

    def __init__(self, model_name: str, max_tokens: int) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, kept local

        # Pairs longer than max_tokens are truncated (the end of a long chunk is not read)
        self._model = CrossEncoder(model_name, max_length=max_tokens)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, item.chunk.text_for_embedding) for item in candidates],
                                     show_progress_bar=False)
        rescored = [RetrievedChunk(chunk=item.chunk, score=float(score)) for item, score in zip(candidates, scores)]
        return sorted(rescored, key=lambda item: item.score, reverse=True)


class Retriever:
    """Two-stage retrieval: vector search for candidates, then ranking, then cut to top_k."""

    def __init__(self, embedder: IEmbedder, store: IVectorStore, ranker: IReranker, config: RetrievalConfig) -> None:
        self._embedder = embedder
        self._store = store
        self._ranker = ranker
        self._config = config

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        # With plain cosine ranking the first stage already returns the final order,
        # so there is no need to fetch more candidates than top_k.
        fetch_k = max(self._config.fetch_k, self._config.top_k) if self._config.use_reranker else self._config.top_k
        candidates = self._store.search(self._embedder.embed_query(query), top_k=fetch_k)
        return self._ranker.rerank(query, candidates)[:self._config.top_k]
