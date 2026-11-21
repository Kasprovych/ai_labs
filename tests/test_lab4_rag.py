"""Unit tests for the Lab 4 RAG building blocks (fake embedder / store / LLM: no models, no network)."""

from typing import List, Sequence

import numpy as np
import pytest

from src.config.lab4_config import ChunkingConfig, RetrievalConfig
from src.rag.corpus import Document
from src.rag.embeddings import IEmbedder
from src.rag.evaluation import (EvalQuestion, LlmJudge, RetrievalScorer, fact_coverage, is_partial_refusal,
                                is_refusal)
from src.rag.llm import ROLE_SYSTEM, ROLE_USER, ScriptedLanguageModel
from src.rag.memory import ConversationMemory, DialogueTurn
from src.rag.pipeline import RagPipeline
from src.rag.preprocessing import Chunk, TextCleaner, WordChunker, measure_overlap
from src.rag.retriever import CosineRanker, IReranker, Retriever
from src.rag.vector_store import IVectorStore, RetrievedChunk

ACCENT = chr(0x301)


def make_document(text: str) -> Document:
    return Document(title="Тест", url="https://example.org", revision_id=1, text=text)


def numbered_sentences(count: int) -> str:
    return " ".join(f"Речення номер {index} має рівно шість слів." for index in range(count))


class KeywordEmbedder(IEmbedder):
    """Embeds text as counts of three keywords, so similarity is predictable."""
    KEYWORDS = ("кіт", "пес", "риба")

    def _vector(self, text: str) -> np.ndarray:
        vector = np.array([text.lower().count(word) for word in self.KEYWORDS], dtype=float) + 1e-6
        return vector / np.linalg.norm(vector)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class InMemoryStore(IVectorStore):
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectors = np.empty((0, 3))

    def count(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        scores = self._vectors @ query_embedding
        order = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(self._chunks[index], float(scores[index])) for index in order]


class ReverseRanker(IReranker):
    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        return list(reversed(candidates))


def build_retriever(config: RetrievalConfig, ranker: IReranker | None = None) -> Retriever:
    embedder, store = KeywordEmbedder(), InMemoryStore()
    chunks = [Chunk(f"doc#{index}", "Тварини", "", text, "url") for index, text in
              enumerate(["кіт кіт кіт спить", "пес гавкає на кота, пес", "риба плаває, риба мовчить", "кіт і пес"])]
    store.add(chunks, embedder.embed_passages([chunk.text for chunk in chunks]))
    return Retriever(embedder, store, ranker or CosineRanker(), config)


# --- preprocessing ---------------------------------------------------------------------------

def test_cleaner_removes_stress_marks_and_service_sections():
    text = (f"Генети{ACCENT}чний алгоритм — це метод пошуку.\n\n== Історія ==\nПерші спроби були у 1954 році.\n"
            "== Примітки ==\nЦе службовий розділ, якого не має бути.\n== Застосування ==\nОптимізація розкладів.")
    sections = TextCleaner(["Примітки"]).split_sections(make_document(text))
    assert [section.heading for section in sections] == ["", "Історія", "Застосування"]
    assert "Генетичний" in sections[0].text
    assert all("службовий" not in section.text for section in sections)


def test_cleaner_drops_lines_without_words():
    sections = TextCleaner([]).split_sections(make_document("Звичайний текст речення.\n= 1 + 2 * 3 / 4 =\n12 34 56"))
    assert sections[0].text == "Звичайний текст речення."


def test_chunks_respect_size_and_consecutive_chunks_overlap():
    config = ChunkingConfig(chunk_size_words=30, overlap_words=12, min_chunk_words=5)
    chunks = WordChunker(config, TextCleaner([])).split_document(make_document(numbered_sentences(20)))
    assert len(chunks) > 2
    assert all(chunk.word_count <= 30 for chunk in chunks[:-1])
    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = current.text.split(".")[0]
        assert first_sentence in previous.text  # the next chunk starts with the tail of the previous one


def test_chunking_loses_no_sentences():
    config = ChunkingConfig(chunk_size_words=30, overlap_words=12, min_chunk_words=5)
    chunks = WordChunker(config, TextCleaner([])).split_document(make_document(numbered_sentences(23)))
    joined = " ".join(chunk.text for chunk in chunks)
    assert all(f"номер {index} " in joined for index in range(23))


def test_chunks_do_not_cross_top_level_sections():
    text = "Перша частина. " * 10 + "\n== Друга ==\n" + "Інший розділ тексту. " * 10
    config = ChunkingConfig(chunk_size_words=200, overlap_words=40, min_chunk_words=5)
    chunks = WordChunker(config, TextCleaner([])).split_document(make_document(text))
    assert [chunk.section for chunk in chunks] == ["", "Друга"]
    assert "Інший" not in chunks[0].text


def test_long_sentences_still_produce_word_level_overlap():
    long_sentence = " ".join(f"слово{index}" for index in range(25)) + "."
    config = ChunkingConfig(chunk_size_words=30, overlap_words=10, min_chunk_words=5)
    chunks = WordChunker(config, TextCleaner([])).split_document(make_document(" ".join([long_sentence] * 4)))
    stats = measure_overlap(chunks)
    assert len(chunks) > 2
    assert stats.share_without_overlap == 0.0 and stats.mean_words >= 5


def test_measure_overlap_counts_shared_words_between_neighbours():
    chunks = [Chunk("d#0", "Д", "", "а б в г", "u"), Chunk("d#1", "Д", "", "в г д е", "u"),
              Chunk("d#2", "Д", "", "ж з и", "u")]
    stats = measure_overlap(chunks)
    assert stats.mean_words == 1.0  # overlaps 2 and 0
    assert stats.share_without_overlap == 0.5


def test_chunker_rejects_overlap_not_smaller_than_chunk():
    with pytest.raises(ValueError):
        WordChunker(ChunkingConfig(chunk_size_words=10, overlap_words=10), TextCleaner([]))


# --- retrieval -------------------------------------------------------------------------------

def test_retriever_returns_top_k_sorted_by_cosine_similarity():
    results = build_retriever(RetrievalConfig(top_k=2)).retrieve("кіт")
    assert [item.chunk.chunk_id for item in results] == ["doc#0", "doc#3"]
    assert results[0].score >= results[1].score


def test_reranker_reorders_fetched_candidates_before_the_cut():
    config = RetrievalConfig(top_k=1, fetch_k=4, use_reranker=True)
    results = build_retriever(config, ReverseRanker()).retrieve("кіт")
    assert len(results) == 1
    assert results[0].chunk.chunk_id != "doc#0"  # the least similar of 4 candidates is promoted


# --- memory and pipeline ---------------------------------------------------------------------

def test_memory_persists_turns_and_exposes_a_sliding_window(tmp_path):
    path = tmp_path / "history.json"
    memory = ConversationMemory(max_turns=2, storage_path=path)
    for index in range(3):
        memory.add(DialogueTurn(f"q{index}", f"q{index}", f"a{index}", ("Стаття",)))
    restored = ConversationMemory(max_turns=2, storage_path=path)
    assert [turn.question for turn in restored.turns] == ["q0", "q1", "q2"]
    assert [turn.question for turn in restored.recent_turns] == ["q1", "q2"]
    assert restored.turns[0].sources == ("Стаття",)


def test_pipeline_puts_retrieved_context_into_the_prompt_and_stores_the_turn():
    llm = ScriptedLanguageModel(["Кіт спить [1]."])
    pipeline = RagPipeline(llm, build_retriever(RetrievalConfig(top_k=2)), ConversationMemory(max_turns=3), 100)
    result = pipeline.ask("Що робить кіт?")
    prompt = llm.calls[0]
    assert prompt[0].role == ROLE_SYSTEM and prompt[-1].role == ROLE_USER
    assert "[1] Стаття «Тварини»" in prompt[-1].content and "кіт кіт кіт спить" in prompt[-1].content
    assert result.answer == "Кіт спить [1]."
    assert pipeline.memory.turns[0].sources == ("Тварини",)


def test_follow_up_question_is_rewritten_and_the_rewritten_form_is_used_for_retrieval():
    llm = ScriptedLanguageModel(["Риба плаває.", "Що робить риба?", "Вона мовчить."])
    pipeline = RagPipeline(llm, build_retriever(RetrievalConfig(top_k=1)), ConversationMemory(max_turns=3), 100)
    pipeline.ask("Що робить риба?")
    result = pipeline.ask("А ще що?")
    assert result.standalone_question == "Що робить риба?"
    assert result.retrieved[0].chunk.chunk_id == "doc#2"
    final_prompt = llm.calls[-1]
    assert [message.content for message in final_prompt[1:3]] == ["Що робить риба?", "Риба плаває."]


def test_old_citations_are_removed_from_history_sent_to_the_llm():
    llm = ScriptedLanguageModel(["Кіт спить [1][2].", "Що робить кіт?", "Ще він їсть [1]."])
    pipeline = RagPipeline(llm, build_retriever(RetrievalConfig(top_k=1)), ConversationMemory(max_turns=3), 100)
    pipeline.ask("Що робить кіт?")
    pipeline.ask("А ще?")
    assert llm.calls[-1][2].content == "Кіт спить."
    assert pipeline.memory.turns[0].answer == "Кіт спить [1][2]."  # the stored history keeps the citations


def test_pipeline_without_retriever_sends_only_the_question():
    llm = ScriptedLanguageModel(["Відповідь."])
    result = RagPipeline(llm, None, ConversationMemory(max_turns=0), 100).ask("Питання?")
    assert result.retrieved == () and llm.calls[0][-1].content == "Питання?"


# --- evaluation ------------------------------------------------------------------------------

QUESTION = EvalQuestion(id="Q", question="?", reference="Голланд, 1975", expected_titles=("Стаття",),
                        evidence=("Голланд", "1975"), key_facts=("Голланд", "1975"))


def retrieved(*texts: str) -> List[RetrievedChunk]:
    return [RetrievedChunk(Chunk(f"c{index}", "Стаття", "", text, "url"), 1.0) for index, text in enumerate(texts)]


def test_retrieval_score_reports_rank_of_first_relevant_chunk_and_evidence_recall():
    score = RetrievalScorer().score(QUESTION, retrieved("нічого", "Джон Голланд", "теж нічого"))
    assert score.first_relevant_rank == 2
    assert score.reciprocal_rank == 0.5
    assert score.hit_at(3) and not score.hit_at(1)
    assert score.evidence_recall == 0.5


def test_retrieval_score_without_relevant_chunks_is_zero():
    score = RetrievalScorer().score(QUESTION, retrieved("нічого"))
    assert score.first_relevant_rank is None and score.reciprocal_rank == 0.0 and not score.hit_at(5)


def test_fact_coverage_counts_key_facts_in_the_answer():
    assert fact_coverage(QUESTION, "Це зробив Голланд у 1975 році.") == 1.0
    assert fact_coverage(QUESTION, "Це зробив голланд.") == 0.5
    assert fact_coverage(QUESTION, "Не знаю.") == 0.0


def test_unanswerable_question_is_covered_only_by_a_refusal_without_fabrication():
    question = EvalQuestion("Q", "?", "немає", (), (), (), answerable=False, fabrication_markers=("Гінтон",))
    assert fact_coverage(question, "У наданих документах немає відповіді на це запитання.") == 1.0
    assert fact_coverage(question, "У наданих документах немає відповіді. Інформація відсутня [1].") == 1.0
    assert fact_coverage(question, "У наданих документах немає відповіді, але премію отримав Гінтон.") == 0.0
    assert fact_coverage(question, "Премію отримав Джеффрі Гінтон.") == 0.0


def test_full_and_partial_refusals_are_distinguished():
    full = "У наданих документах немає відповіді на це запитання."
    partial = ("Генеративні змагальні мережі запровадив Ян Ґудфелоу у 2014 році [1]. Про автора нечіткої логіки "
               "у наданих документах немає відповіді.")
    assert is_refusal(full) and not is_partial_refusal(full)
    assert is_partial_refusal(partial) and not is_refusal(partial)
    assert not is_refusal("Модель не містить рекурентних шарів.")


def test_judge_parses_grades_from_noisy_output():
    llm = ScriptedLanguageModel(['Ось оцінка: ```json\n{"accuracy": 4, "completeness": 3, "coherence": 5}\n```'])
    score = LlmJudge(llm, max_tokens=60).grade(QUESTION, "відповідь")
    assert (score.accuracy, score.completeness, score.coherence) == (4, 3, 5)


def test_judge_returns_none_for_unparseable_output():
    score = LlmJudge(ScriptedLanguageModel(["не можу оцінити"]), max_tokens=60).grade(QUESTION, "відповідь")
    assert score.accuracy is None and score.coherence is None
