from datetime import datetime
from pathlib import Path

from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.model.rule_base import RuleBase


class RuleBaseRepository:
    """Saves rule bases as JSON files and loads them back."""

    def __init__(self, rule_base_json_mapper: RuleBaseJsonMapper) -> None:
        self._rule_base_json_mapper = rule_base_json_mapper

    def save(self, rule_base: RuleBase, directory: Path, created_at: datetime) -> Path:
        """Writes <directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json and returns its path."""
        path = directory / rule_base.domain / f"{created_at:%Y-%m-%d_%H-%M-%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._rule_base_json_mapper.to_json(rule_base) + "\n", encoding="utf-8")
        return path

    def load(self, path: Path) -> RuleBase:
        return self._rule_base_json_mapper.from_json(path.read_text(encoding="utf-8"))
