"""Dialogue history: keeps question-answer turns in memory and on disk."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class DialogueTurn:
    """One exchange: what the user asked, what the system answered and which sources it used."""
    question: str
    # The question rewritten into a self-contained form (equals `question` for the first turn)
    standalone_question: str
    answer: str
    sources: Tuple[str, ...]


class ConversationMemory:
    """Stores the whole dialogue; exposes only the last `max_turns` turns for prompts.

    A sliding window bounds the prompt size: old turns stay on disk (full history is saved to a
    JSON file after every turn) but are not sent to the LLM again.
    """

    def __init__(self, max_turns: int, storage_path: Path | None = None) -> None:
        self._max_turns = max_turns
        self._storage_path = storage_path
        self._turns: List[DialogueTurn] = []
        if storage_path is not None and storage_path.exists():
            raw_turns = json.loads(storage_path.read_text(encoding="utf-8"))
            self._turns = [DialogueTurn(**{**turn, "sources": tuple(turn["sources"])}) for turn in raw_turns]

    @property
    def turns(self) -> Tuple[DialogueTurn, ...]:
        """Complete history."""
        return tuple(self._turns)

    @property
    def recent_turns(self) -> Tuple[DialogueTurn, ...]:
        """The window of the most recent turns that goes into prompts."""
        # Negative slice index: the last max_turns elements of the list
        return tuple(self._turns[-self._max_turns:]) if self._max_turns > 0 else ()

    def add(self, turn: DialogueTurn) -> None:
        self._turns.append(turn)
        self._save()

    def clear(self) -> None:
        self._turns = []
        self._save()

    def _save(self) -> None:
        if self._storage_path is not None:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text(
                json.dumps([asdict(turn) for turn in self._turns], ensure_ascii=False, indent=2), encoding="utf-8")
