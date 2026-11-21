"""Assembly of the RAG system: builds indexes and wires components together (Factory pattern).

Heavy objects (embedding model, re-ranker, LLMs, vector collections) are created lazily and
re-used, because the evaluation runs many system setups that share them.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.config.lab4_config import ChunkingConfig, EvaluatedSetup, Lab4Config, RetrievalConfig
from src.rag.corpus import Document
from src.rag.embeddings import PASSAGE_PREFIX, E5Embedder
from src.rag.llm import MlxLanguageModel
from src.rag.memory import ConversationMemory
from src.rag.pipeline import RagPipeline
from src.rag.preprocessing import Chunk, TextCleaner, WordChunker, measure_overlap
from src.rag.retriever import CosineRanker, CrossEncoderReranker, IReranker, Retriever
from src.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStats:
    """Facts about one built index (for the report tables)."""
    chunking: ChunkingConfig
    chunk_count: int
    mean_words: float
    max_words: int
    # Actual overlap of neighbouring chunks (whole sentences are preferred, so it can differ from the target)
    mean_overlap_words: float
    share_without_overlap: float
    mean_tokens: float
    # Share of chunks longer than the embedding model input limit (their tail is not embedded)
    truncated_share: float
    embedding_dimension: int
    build_seconds: float
    reused: bool


class RagSystemFactory:
    """Creates and caches the components of the RAG system for given configurations."""

    def __init__(self, config: Lab4Config, documents: Sequence[Document]) -> None:
        self._config = config
        self._documents = documents
        self._cleaner = TextCleaner(config.corpus.stop_sections)
        self._embedder: E5Embedder | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._llms: Dict[str, MlxLanguageModel] = {}
        self._stores: Dict[str, ChromaVectorStore] = {}
        self._chunks: Dict[str, List[Chunk]] = {}
        self.index_stats: Dict[str, IndexStats] = {}

    @property
    def embedder(self) -> E5Embedder:
        if self._embedder is None:
            logger.info("Loading embedding model %s", self._config.models.embedding_model)
            self._embedder = E5Embedder(self._config.models.embedding_model)
        return self._embedder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            logger.info("Loading re-ranker %s", self._config.models.reranker_model)
            self._reranker = CrossEncoderReranker(self._config.models.reranker_model,
                                                  self._config.models.reranker_max_tokens)
        return self._reranker

    def llm(self, model_name: str | None = None) -> MlxLanguageModel:
        name = model_name or self._config.models.llm_model
        if name not in self._llms:
            logger.info("Loading LLM %s", name)
            started = time.perf_counter()
            self._llms[name] = MlxLanguageModel(name, self._config.models.temperature)
            logger.info("LLM loaded in %.1f s", time.perf_counter() - started)
        return self._llms[name]

    def chunks(self, chunking: ChunkingConfig) -> List[Chunk]:
        if chunking.tag not in self._chunks:
            self._chunks[chunking.tag] = WordChunker(chunking, self._cleaner).split_corpus(self._documents)
        return self._chunks[chunking.tag]

    def store(self, chunking: ChunkingConfig) -> ChromaVectorStore:
        """Returns the vector collection for this chunking, embedding the corpus if it is not indexed yet."""
        if chunking.tag in self._stores:
            return self._stores[chunking.tag]
        chunks = self.chunks(chunking)
        store = ChromaVectorStore(Path(self._config.vector_db_dir), collection_name=chunking.tag)
        started = time.perf_counter()
        fingerprint = self._fingerprint(chunks)
        # Re-use the stored index only if it was built from exactly these chunk texts with this model
        reused = store.fingerprint == fingerprint and store.count() == len(chunks)
        if not reused:
            store.reset(fingerprint)
            embeddings = self.embedder.embed_passages([chunk.text_for_embedding for chunk in chunks])
            store.add(chunks, embeddings)
        build_seconds = time.perf_counter() - started
        words = np.array([chunk.word_count for chunk in chunks])
        tokens = np.array([self.embedder.count_tokens(chunk.text_for_embedding) for chunk in chunks])
        dimension = int(self.embedder.embed_query("test").shape[0])
        overlap = measure_overlap(chunks)
        self.index_stats[chunking.tag] = IndexStats(
            chunking=chunking, chunk_count=len(chunks), mean_words=float(words.mean()), max_words=int(words.max()),
            mean_overlap_words=overlap.mean_words, share_without_overlap=overlap.share_without_overlap,
            mean_tokens=float(tokens.mean()), truncated_share=float((tokens > self.embedder.max_tokens).mean()),
            embedding_dimension=dimension, build_seconds=build_seconds, reused=reused)
        logger.info("Index '%s': %d chunks, %.0f words on average, %.1f%% longer than %d tokens (%s, %.1f s)",
                    chunking.tag, len(chunks), words.mean(), 100 * (tokens > self.embedder.max_tokens).mean(),
                    self.embedder.max_tokens, "re-used from disk" if reused else "embedded and stored",
                    build_seconds)
        self._stores[chunking.tag] = store
        return store

    def chunk_tokens(self, chunking: ChunkingConfig) -> List[int]:
        """Length of every chunk in embedding-model tokens."""
        return [self.embedder.count_tokens(chunk.text_for_embedding) for chunk in self.chunks(chunking)]

    def _fingerprint(self, chunks: Sequence[Chunk]) -> str:
        """SHA-256 of the embedding model, the passage format and all chunk ids and texts."""
        digest = hashlib.sha256(self._config.models.embedding_model.encode("utf-8"))
        digest.update(PASSAGE_PREFIX.encode("utf-8"))
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(chunk.text_for_embedding.encode("utf-8"))
        return digest.hexdigest()

    def ranker(self, retrieval: RetrievalConfig) -> IReranker:
        return self.reranker if retrieval.use_reranker else CosineRanker()

    def retriever(self, chunking: ChunkingConfig, retrieval: RetrievalConfig) -> Retriever:
        return Retriever(self.embedder, self.store(chunking), self.ranker(retrieval), retrieval)

    def pipeline(self, setup: EvaluatedSetup, memory: ConversationMemory) -> RagPipeline:
        retriever = self.retriever(setup.chunking, setup.retrieval) if setup.use_retrieval else None
        models = self._config.models
        return RagPipeline(self.llm(setup.llm_model), retriever, memory, models.max_answer_tokens,
                           models.condense_max_tokens)
