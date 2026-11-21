"""Console/Markdown tables and the saved summary of the Lab 4 run (no computation of metrics here)."""

import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config.lab4_config import Lab4Config
from src.rag.corpus import Document
from src.rag.evaluation import AnswerRecord, EvalQuestion
from src.rag.factory import IndexStats
from src.rag.pipeline import RagAnswer
from src.rag.preprocessing import TextCleaner

WRAP_WIDTH = 110


def wrap(text: str, indent: str = "    ") -> str:
    """Wraps long text for the console, keeping paragraph breaks."""
    return "\n".join(textwrap.fill(line, WRAP_WIDTH, initial_indent=indent, subsequent_indent=indent)
                     for line in text.splitlines() if line.strip())


def format_answer(result: RagAnswer, show_chunks: bool = True) -> str:
    """Question, rewritten question, retrieved chunks with scores, answer and timings as console text."""
    lines = [f"\nQ: {result.question}"]
    if result.standalone_question != result.question:
        lines.append(f"   rewritten with dialogue history -> {result.standalone_question}")
    if show_chunks:
        lines.extend(f"   [{rank}] score={item.score:.3f}  {item.chunk.chunk_id}  ({item.chunk.section or 'вступ'})"
                     for rank, item in enumerate(result.retrieved, start=1))
    lines.append("A:\n" + wrap(result.answer))
    lines.append(f"   (retrieval {result.retrieval_seconds:.2f} s, generation {result.generation_seconds:.1f} s)")
    return "\n".join(lines)


@dataclass(frozen=True)
class CleaningStats:
    raw_words: int
    clean_words: int

    @property
    def removed_share(self) -> float:
        return (self.raw_words - self.clean_words) / self.raw_words


def cleaning_stats(documents: Sequence[Document], cleaner: TextCleaner) -> CleaningStats:
    """Word counts before and after noise removal."""
    clean_words = sum(len(section.text.split()) for document in documents
                      for section in cleaner.split_sections(document))
    return CleaningStats(raw_words=sum(document.word_count for document in documents), clean_words=clean_words)


def corpus_table(documents: Sequence[Document]) -> str:
    return pd.DataFrame([{"#": number, "Article": document.title, "Words": document.word_count,
                          "Revision": document.revision_id}
                         for number, document in enumerate(documents, start=1)]).to_markdown(index=False)


def index_table(stats: Sequence[IndexStats]) -> str:
    return pd.DataFrame([{
        "Chunk size / overlap, words": f"{s.chunking.chunk_size_words} / {s.chunking.overlap_words}",
        "Chunks": s.chunk_count, "Mean words": round(s.mean_words, 1), "Max words": s.max_words,
        "Actual overlap, words": round(s.mean_overlap_words, 1),
        "No overlap, %": round(100 * s.share_without_overlap, 1), "Mean tokens": round(s.mean_tokens, 1),
        "Truncated by embedder, %": round(100 * s.truncated_share, 1), "Vector dim.": s.embedding_dimension,
        "Build, s": round(s.build_seconds, 1), "Source": "disk" if s.reused else "embedded"}
        for s in stats]).to_markdown(index=False)


def answers_table(records: Sequence[AnswerRecord], questions: Sequence[EvalQuestion], setup_name: str) -> str:
    """All answers of one setup next to the reference answers."""
    references = {question.id: question.reference for question in questions}
    return pd.DataFrame([{"ID": record.question_id, "Question": record.question,
                          "Reference": references[record.question_id], "Answer": record.answer.replace("\n", " "),
                          "Facts": round(record.fact_coverage, 2),
                          "Acc/Compl/Coh": f"{record.accuracy}/{record.completeness}/{record.coherence}"}
                         for record in records if record.setup == setup_name]).to_markdown(index=False)


class SummaryWriter:
    """Saves the evaluation results: all answers as JSON and a Markdown summary."""

    def __init__(self, config: Lab4Config) -> None:
        self._config = config
        self._artifacts = config.paths.artifacts_dir

    def write(self, records: Sequence[AnswerRecord], sections: Sequence[tuple[str, str]],
              plot_paths: Sequence[tuple[str, Path]]) -> Path:
        with open(self._artifacts / "eval_answers.json", "w", encoding="utf-8") as file:
            json.dump([asdict(record) for record in records], file, ensure_ascii=False, indent=1)
        models = self._config.models
        summary_path = self._artifacts / "metrics_summary.md"
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write("# Laboratory Work #4: RAG System Evaluation Summary\n\n")
            file.write(f"Embeddings: {models.embedding_model}; re-ranker: {models.reranker_model}; "
                       f"LLM: {models.llm_model} (judge: the same model); temperature: {models.temperature}\n\n")
            for title, body in sections:
                file.write(f"## {title}\n\n{body}\n\n")
            file.write("## Visual Artifacts\n\n")
            for title, path in plot_paths:
                file.write(f"- [{title}](plots/{path.name})\n")
        return summary_path
