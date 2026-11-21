"""Text preprocessing: noise removal and splitting documents into overlapping chunks."""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.config.lab4_config import ChunkingConfig
from src.rag.corpus import Document

HEADING_PATTERN = re.compile(r"^(={2,})\s*(.*?)\s*=+\s*$")
SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?…])\s+(?=[«\"(A-ZА-ЯІЇЄҐ0-9])")
LATEX_PATTERN = re.compile(r"\{\\displaystyle.*?\}\s*\}?")
# U+0301: the stress mark Wikipedia puts over vowels in article titles
COMBINING_ACUTE_ACCENT = chr(0x301)
# A longer "heading" is a markup glitch (heading glued to a paragraph) and is treated as text
MAX_HEADING_LENGTH = 100


@dataclass(frozen=True)
class Section:
    """A cleaned top-level section of a document."""
    heading: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A fragment of a document: the unit that is embedded, stored and retrieved."""
    chunk_id: str
    document_title: str
    section: str
    text: str
    url: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def text_for_embedding(self) -> str:
        """Title and section are prepended so that the vector also reflects what the chunk is about."""
        heading = f"{self.document_title}. {self.section}" if self.section else self.document_title
        return f"{heading}. {self.text}"


class TextCleaner:
    """Removes noise from Wikipedia plain text and splits it into top-level sections.

    Noise removed: stress marks over vowels ("Генети́чний" would not match the query word
    "Генетичний"), service sections (references, link lists), LaTeX leftovers of formulas,
    lines without words, repeated whitespace.
    """

    def __init__(self, stop_sections: Sequence[str]) -> None:
        self._stop_sections = {name.lower() for name in stop_sections}

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text.replace(COMBINING_ACUTE_ACCENT, ""))
        text = LATEX_PATTERN.sub(" ", text)
        text = text.replace("\xa0", " ")
        return re.sub(r"[ \t]+", " ", text).strip()

    def split_sections(self, document: Document) -> List[Section]:
        sections: List[Section] = []
        heading, lines, skipping = "", [], False

        def flush() -> None:
            body = " ".join(lines).strip()
            if body and not skipping:
                sections.append(Section(heading=heading, text=body))

        for raw_line in document.text.splitlines():
            match = HEADING_PATTERN.match(raw_line.strip())
            if match and len(match.group(2)) <= MAX_HEADING_LENGTH:
                level, title = len(match.group(1)), self.clean_text(match.group(2))
                if level == 2:  # "== Title ==" starts a new top-level section
                    flush()
                    heading, lines = title, []
                    skipping = title.lower() in self._stop_sections
                elif not skipping:  # deeper headings stay in the text as a short lead-in sentence
                    lines.append(f"{title}.")
                continue
            line = self.clean_text(raw_line)
            if self._is_informative(line):
                lines.append(line)
        flush()
        return sections

    @staticmethod
    def _is_informative(line: str) -> bool:
        """Drops empty lines and formula debris: a useful line is mostly letters."""
        if len(line) < 3:
            return False
        letters = sum(character.isalpha() for character in line)
        return letters / len(line) >= 0.5


class WordChunker:
    """Greedy sentence-aware chunking with overlap, measured in words.

    Sentences are appended to the current chunk until `chunk_size_words` would be exceeded; the
    next chunk starts with the trailing sentences of the previous one (up to `overlap_words`).
    If whole sentences give less than half of the overlap budget (long sentences), the overlap is
    taken word by word instead, so neighbouring chunks always share text. Cutting chunk ends at
    sentence borders keeps them readable for the LLM; chunks never cross a top-level section,
    so one chunk talks about one subtopic.
    """

    def __init__(self, config: ChunkingConfig, cleaner: TextCleaner) -> None:
        if config.overlap_words >= config.chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words")
        self._config = config
        self._cleaner = cleaner

    def split_corpus(self, documents: Sequence[Document]) -> List[Chunk]:
        # Nested comprehension = two nested for-loops that collect results into one flat list
        return [chunk for document in documents for chunk in self.split_document(document)]

    def split_document(self, document: Document) -> List[Chunk]:
        pieces: List[Tuple[str, str]] = []
        for section in self._cleaner.split_sections(document):
            pieces.extend((section.heading, text) for text in self._split_section(section.text))
        return [Chunk(chunk_id=f"{document.title}#{index}", document_title=document.title, section=heading,
                      text=text, url=document.url)
                for index, (heading, text) in enumerate(pieces)]

    def _split_section(self, text: str) -> List[str]:
        size, overlap = self._config.chunk_size_words, self._config.overlap_words
        chunks: List[List[str]] = []
        current: List[str] = []       # sentences of the chunk being built
        current_words = 0
        fresh_words = 0               # words in `current` that are not overlap from the previous chunk
        for sentence in self._sentences(text, size):
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > size:
                chunks.append(current)
                current = self._overlap_tail(current, min(overlap, size - sentence_words))
                current_words = sum(len(s.split()) for s in current)
                fresh_words = 0
            current.append(sentence)
            current_words += sentence_words
            fresh_words += sentence_words
        if fresh_words:
            if chunks and fresh_words < self._config.min_chunk_words:
                # A tiny tail is merged into the previous chunk instead of becoming a useless chunk
                tail = current[len(current) - self._count_fresh(current, fresh_words):]
                chunks[-1] = chunks[-1] + tail
            elif chunks or current_words >= self._config.min_chunk_words:
                chunks.append(current)
        return [" ".join(sentences) for sentences in chunks]

    @staticmethod
    def _count_fresh(sentences: List[str], fresh_words: int) -> int:
        """Number of trailing sentences that make up `fresh_words` words."""
        count, words = 0, 0
        for sentence in reversed(sentences):
            if words >= fresh_words:
                break
            words += len(sentence.split())
            count += 1
        return count

    @staticmethod
    def _overlap_tail(sentences: List[str], budget: int) -> List[str]:
        """Text from the end of a finished chunk, at most `budget` words, that starts the next chunk."""
        if budget <= 0:
            return []
        tail: List[str] = []
        words = 0
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if words + sentence_words > budget:
                break
            tail.insert(0, sentence)
            words += sentence_words
        if words < budget // 2:
            # Whole sentences do not fit: fall back to the last `budget` words (may start mid-sentence)
            all_words = " ".join(sentences).split()
            return [" ".join(all_words[-budget:])]
        return tail

    @staticmethod
    def _sentences(text: str, max_words: int) -> List[str]:
        """Splits text into sentences; a sentence longer than a whole chunk is cut by words."""
        sentences: List[str] = []
        for sentence in SENTENCE_END_PATTERN.split(text):
            words = sentence.split()
            for start in range(0, len(words), max_words):
                sentences.append(" ".join(words[start:start + max_words]))
        return [sentence for sentence in sentences if sentence]


@dataclass(frozen=True)
class OverlapStats:
    """Actual overlap between consecutive chunks of the same section."""
    mean_words: float
    share_without_overlap: float


def measure_overlap(chunks: Sequence[Chunk]) -> OverlapStats:
    """Counts, for each pair of neighbouring chunks, how many words end one and start the other."""
    overlaps: List[int] = []
    for previous, current in zip(chunks, chunks[1:]):
        if previous.document_title != current.document_title or previous.section != current.section:
            continue
        previous_words, current_words = previous.text.split(), current.text.split()
        longest = 0
        for length in range(min(len(previous_words), len(current_words)), 0, -1):
            if previous_words[-length:] == current_words[:length]:
                longest = length
                break
        overlaps.append(longest)
    if not overlaps:
        return OverlapStats(0.0, 0.0)
    return OverlapStats(mean_words=sum(overlaps) / len(overlaps),
                        share_without_overlap=sum(value == 0 for value in overlaps) / len(overlaps))
