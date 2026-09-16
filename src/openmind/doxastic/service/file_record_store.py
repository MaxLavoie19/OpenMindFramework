import json
import logging
from collections.abc import Iterator
from pathlib import Path

from openmind.doxastic.mapper.record_json_mapper import RecordJsonMapper
from openmind.doxastic.model.record import Record

logger = logging.getLogger(__name__)

#: What marks a line as a record forgotten rather than a record.
FORGOTTEN = "forgotten"


class FileRecordStore:
    """Records kept as JSON lines in one file, word for word, in the order they were remembered.

    The file is only ever appended to: remembering a record whose id is already there writes it anew and the last line
    wins, and forgetting one writes a line saying so. A record's place is its byte offset, so reading one back is a seek
    and a line, however long the file has grown. Another store — SQLite, or anything else — takes its place behind
    `RecordStore` without the knowledge base knowing."""

    def __init__(self, path: Path, record_json_mapper: RecordJsonMapper | None = None) -> None:
        self._path = path
        self._mapper = RecordJsonMapper() if record_json_mapper is None else record_json_mapper

    def append(self, record: Record) -> object:
        """Writes the record at the end of the file and gives back its byte offset."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = self._mapper.to_line(record).encode("utf-8")
        with self._path.open("ab") as file:
            offset = file.tell()
            file.write(line + b"\n")
        return offset

    def read(self, place: object) -> Record:
        """The record written at that offset, word for word."""
        with self._path.open("rb") as file:
            file.seek(int(place))  # type: ignore[arg-type]
            line = file.readline()
        return self._mapper.from_line(line.decode("utf-8"))

    def load(self) -> Iterator[tuple[object, Record]]:
        """Every record still remembered, with its offset, in the order the ids were first written: a record written
        again gives its newest line, and a forgotten one is left out."""
        if not self._path.exists():
            return
        places: dict[str, int] = {}
        records: dict[str, Record] = {}
        with self._path.open("rb") as file:
            offset = 0
            for line in file:
                written_at, offset = offset, offset + len(line)
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                forgotten = self._forgotten(text)
                if forgotten is not None:
                    places.pop(forgotten, None)
                    records.pop(forgotten, None)
                    continue
                record = self._mapper.from_line(text)
                places[record.id] = written_at
                records[record.id] = record
        for record_id, record in records.items():
            yield places[record_id], record

    def forget(self, record_id: str) -> None:
        """Writes a line saying the record is forgotten; it is left out from then on, and the line it was on stays."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({FORGOTTEN: record_id}, separators=(",", ":")) + "\n")
        logger.info("Forgot record %s", record_id)

    def _forgotten(self, text: str) -> str | None:
        """The id a line says is forgotten, or None where the line is a record."""
        if f'"{FORGOTTEN}"' not in text:
            return None
        forgotten = json.loads(text).get(FORGOTTEN)
        return None if forgotten is None else str(forgotten)
