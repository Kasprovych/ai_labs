"""Console helpers shared by the lab orchestrators (Labs 3 and 4): output mirroring, logging, banners."""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class Tee:
    """Duplicates writes to several streams (like the Unix `tee` command)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> None:
        for stream in self._streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


# @contextmanager turns a generator into a `with` block (Java analogy: try-with-resources).
@contextmanager
def mirrored_stdout(log_path: Path) -> Iterator[None]:
    """Mirrors everything printed inside the `with` block into `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stdout = sys.__stdout__


def setup_logging() -> None:
    """Configures structured console logging (same format as Labs 1-2)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)-24s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def print_banner(title: str) -> None:
    """Prints a styled CLI section separator."""
    border = "=" * 80
    print(f"\n{border}\n >>> {title.upper()}\n{border}\n")
