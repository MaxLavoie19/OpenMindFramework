from pathlib import Path

from openmind.training.mapper.arm_library_json_mapper import ArmLibraryJsonMapper
from openmind.training.model.arm_library import ArmLibrary


class ArmLibraryRepository:
    """Saves arm libraries as JSON files and loads them back; saving again overwrites the same file."""

    def __init__(self, arm_library_json_mapper: ArmLibraryJsonMapper) -> None:
        self._arm_library_json_mapper = arm_library_json_mapper

    def write(self, library: ArmLibrary, path: Path) -> Path:
        """Writes the library at the path given, creating its folders, and returns the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._arm_library_json_mapper.to_json(library) + "\n", encoding="utf-8")
        return path

    def load(self, path: Path) -> ArmLibrary:
        return self._arm_library_json_mapper.from_json(path.read_text(encoding="utf-8"))
