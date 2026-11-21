"""The RAG pipeline: question -> (rewrite with history) -> retrieve -> build context -> generate."""

import re
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.rag import prompts
from src.rag.llm import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER, ChatMessage, ILanguageModel
from src.rag.memory import ConversationMemory, DialogueTurn
from src.rag.retriever import Retriever
from src.rag.vector_store import RetrievedChunk

# Citation markers like [1] or [2, 3]: they refer to the context of an old turn, not the new one
CITATION_PATTERN = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")


@dataclass(frozen=True)
class RagAnswer:
    """Everything produced for one question (kept for display, logging and evaluation)."""
    question: str
    standalone_question: str
    retrieved: Tuple[RetrievedChunk, ...]
    answer: str
    retrieval_seconds: float
    generation_seconds: float


class RagPipeline:
    """Answers questions with retrieved context and remembers the dialogue.

    `retriever=None` switches retrieval off: the LLM answers from its own parametric memory.
    This mode is the baseline that shows what retrieval actually adds.
    """

    def __init__(self, llm: ILanguageModel, retriever: Retriever | None, memory: ConversationMemory,
                 max_answer_tokens: int, condense_max_tokens: int = 80) -> None:
        self._llm = llm
        self._retriever = retriever
        self._memory = memory
        self._max_answer_tokens = max_answer_tokens
        self._condense_max_tokens = condense_max_tokens

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def ask(self, question: str) -> RagAnswer:
        history = self._memory.recent_turns
        standalone = self._condense(question, history) if history else question

        started = time.perf_counter()
        retrieved = tuple(self._retriever.retrieve(standalone)) if self._retriever else ()
        retrieval_seconds = time.perf_counter() - started

        started = time.perf_counter()
        answer = self._llm.generate(self._build_messages(question, retrieved, history), self._max_answer_tokens)
        generation_seconds = time.perf_counter() - started

        sources = tuple(dict.fromkeys(item.chunk.document_title for item in retrieved))  # unique, order kept
        self._memory.add(DialogueTurn(question=question, standalone_question=standalone, answer=answer,
                                      sources=sources))
        return RagAnswer(question, standalone, retrieved, answer, retrieval_seconds, generation_seconds)

    def _condense(self, question: str, history: Sequence[DialogueTurn]) -> str:
        """Rewrites a follow-up question ("А хто його створив?") into a self-contained search query."""
        history_text = "\n".join(f"Користувач: {turn.question}\nАсистент: {self.strip_citations(turn.answer)}"
                                 for turn in history)
        rewritten = self._llm.generate(
            [ChatMessage(ROLE_SYSTEM, prompts.CONDENSE_SYSTEM_PROMPT),
             ChatMessage(ROLE_USER, prompts.CONDENSE_USER_TEMPLATE.format(history=history_text, question=question))],
            self._condense_max_tokens)
        first_line = rewritten.strip().splitlines()[0].strip() if rewritten.strip() else ""
        return first_line or question

    def _build_messages(self, question: str, retrieved: Sequence[RetrievedChunk],
                        history: Sequence[DialogueTurn]) -> List[ChatMessage]:
        system_prompt = prompts.RAG_SYSTEM_PROMPT if self._retriever else prompts.NO_RAG_SYSTEM_PROMPT
        messages = [ChatMessage(ROLE_SYSTEM, system_prompt)]
        for turn in history:  # previous turns let the LLM resolve references in the new question
            messages.append(ChatMessage(ROLE_USER, turn.question))
            messages.append(ChatMessage(ROLE_ASSISTANT, self.strip_citations(turn.answer)))
        if self._retriever:
            user_text = prompts.RAG_USER_TEMPLATE.format(context=self.format_context(retrieved), question=question)
        else:
            user_text = question
        messages.append(ChatMessage(ROLE_USER, user_text))
        return messages

    @staticmethod
    def strip_citations(answer: str) -> str:
        """Removes [n] markers from an old answer so they are not confused with the new context numbering."""
        return CITATION_PATTERN.sub("", answer)

    @staticmethod
    def format_context(retrieved: Sequence[RetrievedChunk]) -> str:
        """Joins retrieved chunks into one numbered context block (numbers are used for citations)."""
        entries = []
        for number, item in enumerate(retrieved, start=1):
            section = f", розділ «{item.chunk.section}»" if item.chunk.section else ""
            entries.append(prompts.CONTEXT_ENTRY_TEMPLATE.format(
                number=number, title=item.chunk.document_title, section=section, text=item.chunk.text))
        return "\n\n".join(entries)
