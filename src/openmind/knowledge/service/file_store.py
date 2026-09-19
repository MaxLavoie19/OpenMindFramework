import json
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

#: What marks a line as an entry forgotten rather than an entry.
FORGOTTEN = "forgotten"


class FileStore:
    """One kind of knowledge kept as JSON lines in one file, in the order it was written: the default `Store`.

    The file is only ever appended to: writing an entry whose id is already there writes it anew and the last line
    wins, and forgetting one writes a line saying so."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, entry: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def load(self) -> Iterator[dict[str, object]]:
        if not self._path.exists():
            return
        entries: dict[str, dict[str, object]] = {}
        with self._path.open(encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                entry = json.loads(text)
                if FORGOTTEN in entry and len(entry) == 1:
                    entries.pop(str(entry[FORGOTTEN]), None)
                    continue
                entries[str(entry["id"])] = entry
        yield from entries.values()

    def forget(self, entry_id: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({FORGOTTEN: entry_id}, separators=(",", ":")) + "\n")
        logger.info("Forgot %s in %s", entry_id, self._path.name)
