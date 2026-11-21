"""Quality evaluation of the RAG system: control questions, retrieval metrics and answer scoring.

Two complementary kinds of measurement are used:
  * objective, rule-based: does the retrieved context contain the evidence? does the answer
    mention the key facts of the reference answer? (regular expressions written before the run);
  * LLM-as-judge: a language model grades accuracy, completeness and coherence on a 1-5 scale
    by comparing the answer with the reference answer.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.rag import prompts
from src.rag.llm import ROLE_SYSTEM, ROLE_USER, ChatMessage, ILanguageModel
from src.rag.vector_store import RetrievedChunk

REFUSAL_PATTERN = re.compile(r"немає відповіді|немає інформації|немає відомостей|не міститься інформац",
                             re.IGNORECASE)
# A refusal is "full" when the answer is essentially just the refusal phrase; a longer answer that
# also contains it answers part of the question and declines the rest ("partial" refusal)
FULL_REFUSAL_MAX_WORDS = 15
JUDGE_CRITERIA = ("accuracy", "completeness", "coherence")


@dataclass(frozen=True)
class EvalQuestion:
    """A control question with the reference answer and machine-checkable expectations."""
    id: str
    question: str
    reference: str
    # Articles that contain the answer
    expected_titles: Tuple[str, ...]
    # Regex per piece of evidence: a retrieved chunk "contains the evidence" if the regex matches it
    evidence: Tuple[str, ...]
    # Regex per key fact that a complete answer must mention
    key_facts: Tuple[str, ...]
    # False = the corpus has no answer; the correct behaviour is to say so
    answerable: bool = True
    # For unanswerable questions: regex of facts a fabricated answer would contain (e.g. a laureate's name)
    fabrication_markers: Tuple[str, ...] = ()


def load_questions(path: Path) -> List[EvalQuestion]:
    raw_questions = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(id=q["id"], question=q["question"], reference=q["reference"],
                         expected_titles=tuple(q["expected_titles"]), evidence=tuple(q["evidence"]),
                         key_facts=tuple(q["key_facts"]), answerable=q.get("answerable", True),
                         fabrication_markers=tuple(q.get("fabrication_markers", ())))
            for q in raw_questions]


@dataclass(frozen=True)
class RetrievalScore:
    """Retrieval quality for one question."""
    # Rank (1-based) of the first chunk that contains any evidence; None if there is none
    first_relevant_rank: Optional[int]
    # Share of evidence items found somewhere in the retrieved chunks (context recall)
    evidence_recall: float

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0

    def hit_at(self, k: int) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= k


class RetrievalScorer:
    """Checks retrieved chunks against the evidence patterns of a question."""

    def score(self, question: EvalQuestion, retrieved: Sequence[RetrievedChunk]) -> RetrievalScore:
        patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in question.evidence]
        texts = [item.chunk.text_for_embedding for item in retrieved]
        first_rank = next((rank for rank, text in enumerate(texts, start=1)
                           if any(pattern.search(text) for pattern in patterns)), None)
        found = sum(any(pattern.search(text) for text in texts) for pattern in patterns)
        return RetrievalScore(first_relevant_rank=first_rank,
                              evidence_recall=found / len(patterns) if patterns else 0.0)


def is_refusal(answer: str) -> bool:
    """True if the system declined to answer the whole question ("the documents contain no answer")."""
    return bool(REFUSAL_PATTERN.search(answer)) and len(answer.split()) <= FULL_REFUSAL_MAX_WORDS


def is_partial_refusal(answer: str) -> bool:
    """True if the answer gives some information but declines another part of the question."""
    return bool(REFUSAL_PATTERN.search(answer)) and len(answer.split()) > FULL_REFUSAL_MAX_WORDS


def fact_coverage(question: EvalQuestion, answer: str) -> float:
    """Share of the key facts mentioned in the answer (objective completeness, 0..1).

    For an unanswerable question the only "fact" is an honest refusal: the answer must say that the
    documents contain no answer and must not contain any fabricated fact (a fabrication marker).
    """
    if not question.answerable:
        refused = bool(REFUSAL_PATTERN.search(answer))
        fabricated = any(re.search(marker, answer, re.IGNORECASE) for marker in question.fabrication_markers)
        return 1.0 if refused and not fabricated else 0.0
    if not question.key_facts:
        return 0.0
    matched = sum(bool(re.search(pattern, answer, re.IGNORECASE)) for pattern in question.key_facts)
    return matched / len(question.key_facts)


@dataclass(frozen=True)
class JudgeScore:
    """Grades 1-5 given by the LLM judge; None if the judge's output could not be parsed."""
    accuracy: Optional[int]
    completeness: Optional[int]
    coherence: Optional[int]
    raw_output: str = field(default="", compare=False)


class LlmJudge:
    """LLM-as-judge: grades an answer against the reference answer on three criteria."""

    def __init__(self, llm: ILanguageModel, max_tokens: int) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def grade(self, question: EvalQuestion, answer: str) -> JudgeScore:
        raw = self._llm.generate(
            [ChatMessage(ROLE_SYSTEM, prompts.JUDGE_SYSTEM_PROMPT),
             ChatMessage(ROLE_USER, prompts.JUDGE_USER_TEMPLATE.format(
                 question=question.question, reference=question.reference, answer=answer))],
            self._max_tokens)
        grades = self.parse(raw)
        return JudgeScore(grades.get("accuracy"), grades.get("completeness"), grades.get("coherence"), raw)

    @staticmethod
    def parse(raw: str) -> Dict[str, int]:
        """Extracts `"criterion": digit` pairs; tolerant to extra text around the JSON."""
        grades: Dict[str, int] = {}
        for criterion in JUDGE_CRITERIA:
            match = re.search(rf'"?{criterion}"?\s*[:=]\s*([1-5])\b', raw, re.IGNORECASE)
            if match:
                grades[criterion] = int(match.group(1))
        return grades


@dataclass(frozen=True)
class AnswerRecord:
    """Full evaluation record: one question answered by one system setup."""
    setup: str
    question_id: str
    question: str
    answer: str
    retrieved_ids: Tuple[str, ...]
    retrieved_scores: Tuple[float, ...]
    first_relevant_rank: Optional[int]
    evidence_recall: float
    fact_coverage: float
    accuracy: Optional[int]
    completeness: Optional[int]
    coherence: Optional[int]
    refused: bool
    partially_refused: bool
    answerable: bool
    retrieval_seconds: float
    generation_seconds: float
    # Verbatim output of the LLM judge, kept so that every grade can be checked by hand
    judge_output: str
