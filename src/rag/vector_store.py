"""Vector database: persistent storage of chunk embeddings and nearest-neighbour search (ChromaDB)."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.rag.preprocessing import Chunk

logger = logging.getLogger(__name__)

ADD_BATCH_SIZE = 500
FINGERPRINT_KEY = "content_fingerprint"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by search together with its relevance score (higher = more relevant)."""
    chunk: Chunk
    score: float


class IVectorStore(ABC):
    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Stores chunks with their vectors."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        """Returns the top_k chunks most similar to the query vector, best first."""


class ChromaVectorStore(IVectorStore):
    """One ChromaDB collection on disk, configured for cosine distance.

    ChromaDB builds an HNSW index (approximate nearest-neighbour graph) and persists it together
    with texts and metadata, so the corpus is embedded only once.
    """

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        import chromadb  # heavy import, kept local
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
        self._name = collection_name
        self._collection = self._client.get_or_create_collection(name=self._name,
                                                                 metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self._collection.count()

    @property
    def fingerprint(self) -> str | None:
        """Hash of the content the collection was built from (None for an empty or legacy collection)."""
        return (self._collection.metadata or {}).get(FINGERPRINT_KEY)

    def reset(self, fingerprint: str) -> None:
        """Deletes all stored chunks and starts an empty collection tagged with a new content fingerprint."""
        self._client.delete_collection(self._name)
        # "hnsw:space": "cosine" makes the index compare vectors by cosine distance
        self._collection = self._client.create_collection(
            name=self._name, metadata={"hnsw:space": "cosine", FINGERPRINT_KEY: fingerprint})

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start:start + ADD_BATCH_SIZE]
            self._collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings[start:start + ADD_BATCH_SIZE].tolist(),
                documents=[chunk.text for chunk in batch],
                metadatas=[{"title": chunk.document_title, "section": chunk.section, "url": chunk.url}
                           for chunk in batch])

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        result = self._collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k,
                                        include=["documents", "metadatas", "distances"])
        retrieved = []
        # Chroma returns one list per query; we always send a single query, hence the [0]
        for chunk_id, text, metadata, distance in zip(result["ids"][0], result["documents"][0],
                                                      result["metadatas"][0], result["distances"][0]):
            chunk = Chunk(chunk_id=chunk_id, document_title=metadata["title"], section=metadata["section"],
                          text=text, url=metadata["url"])
            # Chroma reports cosine distance = 1 - cosine similarity
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return retrieved
