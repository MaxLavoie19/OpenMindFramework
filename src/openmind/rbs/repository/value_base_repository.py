from datetime import datetime
from pathlib import Path

from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.model.value_base import ValueBase


class ValueBaseRepository:
    """Saves value bases as JSON files and loads them back."""

    def __init__(self, value_base_json_mapper: ValueBaseJsonMapper) -> None:
        self._value_base_json_mapper = value_base_json_mapper

    def save(self, value_base: ValueBase, directory: Path, created_at: datetime) -> Path:
        """Writes <directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json and returns its path."""
        path = directory / value_base.domain / f"{created_at:%Y-%m-%d_%H-%M-%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._value_base_json_mapper.to_json(value_base) + "\n", encoding="utf-8")
        return path

    def write(self, value_base: ValueBase, path: Path) -> Path:
        """Writes the value base at the path given, creating its folders, and returns the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._value_base_json_mapper.to_json(value_base) + "\n", encoding="utf-8")
        return path

    def export(self, text: str, path: Path) -> Path:
        """Writes a value base's text export, such as its explained rules in Markdown, at the path given, creating its
        folders, and returns the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, path: Path) -> ValueBase:
        return self._value_base_json_mapper.from_json(path.read_text(encoding="utf-8"))
