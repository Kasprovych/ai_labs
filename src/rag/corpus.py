"""Corpus loading: downloads Wikipedia articles as plain text and caches them on disk."""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from src.config.lab4_config import CorpusConfig

logger = logging.getLogger(__name__)

# Wikipedia rate-limits anonymous clients: pause between requests and back off on HTTP 429
POLITE_DELAY_SECONDS = 1.0
RETRY_DELAY_SECONDS = 10.0
MAX_DOWNLOAD_ATTEMPTS = 5


@dataclass(frozen=True)
class Document:
    """One source document of the corpus (a Wikipedia article)."""
    title: str
    url: str
    revision_id: int
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class WikipediaCorpusLoader:
    """Fetches articles through the MediaWiki API (TextExtracts: plain text without wiki markup).

    Every article is stored as a JSON file, so later runs work offline and always use exactly the
    same revision of the text (reproducible experiments).
    """

    def __init__(self, config: CorpusConfig, cache_dir: Path) -> None:
        self._config = config
        self._cache_dir = cache_dir

    def load(self) -> List[Document]:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        documents = []
        for title in self._config.titles:
            path = self._cache_dir / f"{self._slug(title)}.json"
            if not path.exists():
                document = self._download(title)
                path.write_text(json.dumps(asdict(document), ensure_ascii=False, indent=1), encoding="utf-8")
                logger.info("Downloaded '%s' (%d words)", document.title, document.word_count)
            # **dict unpacks JSON fields into constructor arguments
            documents.append(Document(**json.loads(path.read_text(encoding="utf-8"))))
        logger.info("Corpus: %d documents, %d words in total", len(documents),
                    sum(document.word_count for document in documents))
        return documents

    def _download(self, title: str) -> Document:
        import requests  # imported lazily: not needed when the corpus is already cached

        api_url = f"https://{self._config.language}.wikipedia.org/w/api.php"
        params = {"action": "query", "format": "json", "redirects": 1, "titles": title,
                  "prop": "extracts|revisions|info", "explaintext": 1, "rvprop": "ids", "inprop": "url"}
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            response = requests.get(api_url, params=params, headers={"User-Agent": self._config.user_agent},
                                    timeout=60)
            if response.status_code != 429:  # 429 = Too Many Requests: wait and retry
                break
            time.sleep(RETRY_DELAY_SECONDS * attempt)
        response.raise_for_status()
        time.sleep(POLITE_DELAY_SECONDS)
        page = next(iter(response.json()["query"]["pages"].values()))
        if "missing" in page or not page.get("extract"):
            raise ValueError(f"Wikipedia article not found or empty: {title}")
        return Document(title=page["title"], url=page["fullurl"], revision_id=page["revisions"][0]["revid"],
                        text=page["extract"])

    @staticmethod
    def _slug(title: str) -> str:
        return re.sub(r"[^\w\-]+", "_", title).strip("_")
