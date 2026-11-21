"""Configuration for Laboratory Work #4: Retrieval-Augmented Generation (RAG) system.

All tunable choices (corpus, chunking, models, retrieval and generation parameters, evaluated
configurations) live here, each with a short justification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from src.config.app_config import PathConfig


@dataclass(frozen=True)
class CorpusConfig:
    """Corpus: a selection of Ukrainian Wikipedia articles about artificial intelligence."""
    language: str = "uk"
    # Wikipedia asks automated clients to identify themselves
    user_agent: str = "ai-labs-student-project/1.0 (educational RAG laboratory work)"
    titles: Tuple[str, ...] = (
        "Штучний інтелект", "Машинне навчання", "Глибоке навчання", "Штучна нейронна мережа",
        "Кероване навчання", "Некероване навчання", "Навчання з підкріпленням",
        "Перцептрон", "Метод зворотного поширення помилки", "Градієнтний спуск", "Перенавчання",
        "Згорткова нейронна мережа", "Рекурентна нейронна мережа", "Довга короткочасна пам'ять",
        "Автокодувальник", "Генеративна змагальна мережа", "Увага (машинне навчання)",
        "Трансформер (архітектура глибокого навчання)", "BERT (мовна модель)", "Велика мовна модель",
        "ChatGPT", "GPT-4", "Інженерія запитів", "Галюцинація (штучний інтелект)",
        "Вкладання слів", "Векторна база даних", "Обробка природної мови", "Розпізнавання мовлення",
        "Комп'ютерний зір", "Метод опорних векторів", "Дерево ухвалення рішень", "Random forest",
        "Генетичний алгоритм", "Еволюційний алгоритм", "Нечітка логіка", "Експертна система",
        "Тест Тюрінга", "Алан Тюрінг", "Deep Blue", "AlphaGo",
    )
    # Trailing service sections carry no knowledge (link lists, references)
    stop_sections: Tuple[str, ...] = (
        "Див. також", "Примітки", "Виноски", "Джерела", "Література", "Посилання", "Додаткова література",
        "Зовнішні посилання", "Подальше читання", "Ресурси Інтернету", "Коментарі", "Бібліографія",
    )


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk size is measured in words; neighbouring chunks share `overlap_words` words."""
    # ~200 Ukrainian words is close to the 512-token input limit of the embedding model, so the
    # whole chunk is actually embedded; 3 such chunks still fit comfortably into the LLM prompt.
    chunk_size_words: int = 200
    # 20% overlap: a fact cut by a chunk border is likely to stay intact in one of the neighbours
    overlap_words: int = 40
    # Fragments shorter than this (stray headings, list leftovers) are dropped
    min_chunk_words: int = 20

    @property
    def tag(self) -> str:
        """Stable name of this chunking setup (used as the vector collection name)."""
        return f"chunks_{self.chunk_size_words}w_{self.overlap_words}o"


@dataclass(frozen=True)
class ModelConfig:
    """Models, all running locally (no API keys). Downloaded once from Hugging Face Hub."""
    # Multilingual (incl. Ukrainian) embedding model, 384 dimensions, ~470 MB, 512-token input
    embedding_model: str = "intfloat/multilingual-e5-small"
    # Multilingual cross-encoder used as the second-stage re-ranker, ~2.3 GB
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Main LLM: Google Gemma 3 4B Instruct, 4-bit QAT weights converted for Apple MLX (~3 GB).
    # Chosen after a pilot comparison: fluent Ukrainian and reliable use of the provided context.
    llm_model: str = "mlx-community/gemma-3-4b-it-qat-4bit"
    # Alternative LLM for comparison: Alibaba Qwen2.5 3B Instruct, 4-bit (~1.8 GB), Apache-2.0
    alternative_llm_model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    # Limit of the answer length in tokens
    max_answer_tokens: int = 350
    # The rewritten follow-up question is one sentence; a small limit stops a model that starts to ramble
    condense_max_tokens: int = 80
    # The judge outputs a short JSON with three grades
    judge_max_tokens: int = 60
    # Input limit of the cross-encoder (its tokenizer); longer (query + chunk) pairs are truncated
    reranker_max_tokens: int = 512
    # Temperature 0 = greedy decoding: the same prompt always gives the same answer (reproducible evaluation)
    temperature: float = 0.0


@dataclass(frozen=True)
class RetrievalConfig:
    """Parameters of one retrieval setup."""
    top_k: int = 3
    # Number of candidates fetched from the vector store before re-ranking
    fetch_k: int = 20
    use_reranker: bool = False

    @property
    def label(self) -> str:
        return f"k={self.top_k}, {'cross-encoder' if self.use_reranker else 'cosine'}"


@dataclass(frozen=True)
class MemoryConfig:
    """Dialogue history settings."""
    # Only the last N question-answer pairs are sent to the LLM: bounded prompt size
    max_turns_in_prompt: int = 3


@dataclass(frozen=True)
class EvaluatedSetup:
    """One system configuration compared in the quality evaluation."""
    name: str
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    # False = the LLM answers from its own memory, without retrieved context (baseline)
    use_retrieval: bool = True
    # None = the main LLM from ModelConfig; otherwise a Hugging Face model id
    llm_model: str | None = None
    # The reference setup of the study (its answers are listed in the summary)
    is_base: bool = False


def default_setups() -> Tuple[EvaluatedSetup, ...]:
    """One-factor-at-a-time study around the base setup (200 words, k=3, cosine ranking)."""
    base_chunks = ChunkingConfig()
    small_chunks = ChunkingConfig(chunk_size_words=100, overlap_words=20)
    large_chunks = ChunkingConfig(chunk_size_words=400, overlap_words=80)
    return (
        EvaluatedSetup("A: LLM without RAG", base_chunks, RetrievalConfig(), use_retrieval=False),
        EvaluatedSetup("B: 200w, k=3, cosine (base)", base_chunks, RetrievalConfig(top_k=3), is_base=True),
        EvaluatedSetup("C: 200w, k=1, cosine", base_chunks, RetrievalConfig(top_k=1)),
        EvaluatedSetup("D: 200w, k=5, cosine", base_chunks, RetrievalConfig(top_k=5)),
        EvaluatedSetup("E: 100w, k=3, cosine", small_chunks, RetrievalConfig(top_k=3)),
        EvaluatedSetup("F: 400w, k=3, cosine", large_chunks, RetrievalConfig(top_k=3)),
        EvaluatedSetup("G: 200w, k=3, re-ranker", base_chunks, RetrievalConfig(top_k=3, use_reranker=True)),
        EvaluatedSetup("H: 100w, k=5, re-ranker", small_chunks, RetrievalConfig(top_k=5, use_reranker=True)),
        EvaluatedSetup("I: as B, LLM Qwen2.5-3B", base_chunks, RetrievalConfig(top_k=3),
                       llm_model=ModelConfig().alternative_llm_model),
    )


@dataclass(frozen=True)
class DemoConfig:
    """Questions used to demonstrate the individual steps in the console output."""
    search_question: str = "Хто і коли запропонував тест Тюрінга?"
    ranking_question: str = "Які методики допомагають уникати перенавчання моделі?"
    dialogue: Tuple[str, ...] = (
        "Що таке генетичний алгоритм?",
        "Хто вважається його засновником?",
        "А які генетичні оператори в ньому застосовують?",
        "Які критерії зупинки він може мати?",
    )
    # How many chunks the search and ranking demos print
    shown_chunks: int = 5


@dataclass(frozen=True)
class Lab4Config:
    """Root configuration of Lab 4."""
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    setups: Tuple[EvaluatedSetup, ...] = field(default_factory=default_setups)
    demo: DemoConfig = field(default_factory=DemoConfig)
    paths: PathConfig = field(default_factory=lambda: PathConfig(artifacts_subdir="lab4"))

    @property
    def data_dir(self) -> Path:
        return self.paths.data_dir / "lab4"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def vector_db_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    # Chunk sizes and cut-offs of the retrieval-only study (no LLM involved, so the full grid is cheap)
    retrieval_grid_chunkings: Tuple[ChunkingConfig, ...] = (
        ChunkingConfig(chunk_size_words=100, overlap_words=20), ChunkingConfig(),
        ChunkingConfig(chunk_size_words=400, overlap_words=80))
    retrieval_grid_max_k: int = 5

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "eval_questions.json"
