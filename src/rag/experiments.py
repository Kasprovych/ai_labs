"""Evaluation experiments: retrieval-only grid and end-to-end answer quality per system setup."""

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

import pandas as pd

from src.config.lab4_config import EvaluatedSetup, Lab4Config, RetrievalConfig
from src.rag.evaluation import (AnswerRecord, EvalQuestion, LlmJudge, RetrievalScorer, fact_coverage,
                                is_partial_refusal, is_refusal)
from src.rag.factory import RagSystemFactory
from src.rag.memory import ConversationMemory

logger = logging.getLogger(__name__)


class RetrievalGridExperiment:
    """Retrieval quality for every chunk size x ranking method, measured without the LLM.

    The top `max_k` chunks are retrieved once per question; Hit@k for smaller k is read from the
    rank of the first relevant chunk.
    """

    def __init__(self, config: Lab4Config, factory: RagSystemFactory) -> None:
        self._config = config
        self._factory = factory
        self._scorer = RetrievalScorer()

    def run(self, questions: Sequence[EvalQuestion]) -> pd.DataFrame:
        answerable = [question for question in questions if question.answerable]
        max_k = self._config.retrieval_grid_max_k
        rows: List[Dict[str, Any]] = []
        for chunking in self._config.retrieval_grid_chunkings:
            for use_reranker in (False, True):
                retrieval = RetrievalConfig(top_k=max_k, fetch_k=self._config.retrieval.fetch_k,
                                            use_reranker=use_reranker)
                retriever = self._factory.retriever(chunking, retrieval)
                scores = [self._scorer.score(question, retriever.retrieve(question.question))
                          for question in answerable]
                count = len(scores)
                rows.append({
                    "Chunk size, words": chunking.chunk_size_words,
                    "Ranking": "cross-encoder" if use_reranker else "cosine",
                    "Hit@1": sum(score.hit_at(1) for score in scores) / count,
                    "Hit@3": sum(score.hit_at(3) for score in scores) / count,
                    "Hit@5": sum(score.hit_at(5) for score in scores) / count,
                    "MRR": sum(score.reciprocal_rank for score in scores) / count,
                    f"Evidence recall@{max_k}": sum(score.evidence_recall for score in scores) / count,
                })
                logger.info("  %3d words, %-13s | Hit@1 %.2f | Hit@3 %.2f | Hit@5 %.2f | MRR %.3f",
                            chunking.chunk_size_words, rows[-1]["Ranking"], rows[-1]["Hit@1"], rows[-1]["Hit@3"],
                            rows[-1]["Hit@5"], rows[-1]["MRR"])
        return pd.DataFrame(rows)


class AnswerQualityExperiment:
    """Runs every setup on every control question and scores the answers."""

    def __init__(self, factory: RagSystemFactory, judge: LlmJudge) -> None:
        self._factory = factory
        self._judge = judge
        self._scorer = RetrievalScorer()

    def run(self, setups: Sequence[EvaluatedSetup], questions: Sequence[EvalQuestion]) -> List[AnswerRecord]:
        records: List[AnswerRecord] = []
        for setup in setups:
            logger.info("Setup %s", setup.name)
            for question in questions:
                # A fresh empty memory per question: control questions must not influence each other
                pipeline = self._factory.pipeline(setup, ConversationMemory(max_turns=0))
                result = pipeline.ask(question.question)
                retrieval = self._scorer.score(question, result.retrieved)
                grades = self._judge.grade(question, result.answer)
                record = AnswerRecord(
                    setup=setup.name, question_id=question.id, question=question.question, answer=result.answer,
                    retrieved_ids=tuple(item.chunk.chunk_id for item in result.retrieved),
                    retrieved_scores=tuple(round(item.score, 4) for item in result.retrieved),
                    first_relevant_rank=retrieval.first_relevant_rank, evidence_recall=retrieval.evidence_recall,
                    fact_coverage=fact_coverage(question, result.answer), accuracy=grades.accuracy,
                    completeness=grades.completeness, coherence=grades.coherence,
                    refused=is_refusal(result.answer), partially_refused=is_partial_refusal(result.answer),
                    answerable=question.answerable, retrieval_seconds=result.retrieval_seconds,
                    generation_seconds=result.generation_seconds, judge_output=grades.raw_output)
                records.append(record)
                logger.info("  %s | facts %.2f | judge acc/compl/coh %s/%s/%s | %.1f s | %s", question.id,
                            record.fact_coverage, record.accuracy, record.completeness, record.coherence,
                            record.retrieval_seconds + record.generation_seconds,
                            record.answer.replace("\n", " ")[:90])
        return records


def summarize_setups(records: Sequence[AnswerRecord]) -> pd.DataFrame:
    """One row per setup: mean scores over all control questions."""
    frame = pd.DataFrame([asdict(record) for record in records])
    rows = []
    # sort=False keeps the setups in the order they were run
    for setup, group in frame.groupby("setup", sort=False):
        answerable = group[group["answerable"]]
        unanswerable = group[~group["answerable"]]
        rows.append({
            "Setup": setup,
            "Accuracy (1-5)": group["accuracy"].mean(),
            "Completeness (1-5)": group["completeness"].mean(),
            "Coherence (1-5)": group["coherence"].mean(),
            "Key facts covered": answerable["fact_coverage"].mean(),
            "Evidence in context": answerable["evidence_recall"].mean(),
            "False refusals": int(answerable["refused"].sum()),
            "Partial refusals": int(answerable["partially_refused"].sum()),
            # Correct = honest refusal without fabricated facts (fact_coverage of an unanswerable question)
            "Correct on unanswerable": f"{int((unanswerable['fact_coverage'] == 1.0).sum())}/{len(unanswerable)}",
            "Retrieval, s": group["retrieval_seconds"].mean(),
            "Generation, s": group["generation_seconds"].mean(),
        })
    return pd.DataFrame(rows)
