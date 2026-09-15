from pathlib import Path

from openmind.training.mapper.signal_library_json_mapper import SignalLibraryJsonMapper
from openmind.training.model.signal_library import SignalLibrary


class SignalLibraryRepository:
    """Saves signal libraries as JSON files and loads them back; saving again overwrites the same file."""

    def __init__(self, signal_library_json_mapper: SignalLibraryJsonMapper) -> None:
        self._signal_library_json_mapper = signal_library_json_mapper

    def write(self, library: SignalLibrary, path: Path) -> Path:
        """Writes the library at the path given, creating its folders, and returns the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._signal_library_json_mapper.to_json(library) + "\n", encoding="utf-8")
        return path

    def load(self, path: Path) -> SignalLibrary:
        return self._signal_library_json_mapper.from_json(path.read_text(encoding="utf-8"))
