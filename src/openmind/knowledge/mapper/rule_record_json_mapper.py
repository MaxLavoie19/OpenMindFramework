import importlib
import json

from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.rule.model.python_rule import PythonRule


class RuleRecordJsonMapper:
    """Maps a rule the agent knows to one line of JSON and back.

    A rule written as source keeps its source. A rule the project gave as a function keeps the module it lives in and
    its name there, and loading imports it again, which is the same way a worker process finds it. A function that
    can't be found raises ValueError rather than coming back missing: a game model short of a rule is a different
    game."""

    def __init__(self, knowledge_json_mapper: KnowledgeJsonMapper | None = None) -> None:
        self._knowledge = KnowledgeJsonMapper() if knowledge_json_mapper is None else knowledge_json_mapper

    def to_line(self, rule: RuleRecord) -> str:
        """The rule as one line of JSON, without a line break of its own."""
        return json.dumps(self.to_data(rule), separators=(",", ":"))

    def from_line(self, line: str) -> RuleRecord:
        return self.from_data(json.loads(line))

    def to_data(self, rule: RuleRecord) -> dict[str, object]:
        written = rule.rule
        source = written.source if isinstance(written, PythonRule) else None
        return {
            "id": rule.id,
            "name": rule.name,
            "kind": rule.kind,
            "rule": source,
            "module": None if source is not None else getattr(written, "__module__", None),
            "function": None if source is not None else getattr(written, "__qualname__", None),
            "action": rule.action,
            "parameter": rule.parameter,
            "probability": rule.probability,
            "source": self._knowledge.source_to_data(rule.source),
            "open": rule.open,
            "tags": self._knowledge.tags_to_data(rule.tags),
        }

    def from_data(self, data: dict[str, object]) -> RuleRecord:
        return RuleRecord(
            str(data["name"]),
            str(data["kind"]),
            self._rule(data),  # type: ignore[arg-type]
            self._knowledge.source_from_data(data["source"]),  # type: ignore[arg-type]
            data.get("action"),  # type: ignore[arg-type]
            data.get("parameter"),  # type: ignore[arg-type]
            float(data.get("probability", 1.0)),  # type: ignore[arg-type]
            bool(data.get("open", False)),
            self._knowledge.tags_from_data(data.get("tags")),
            str(data.get("id", "")),
        )

    def _rule(self, data: dict[str, object]) -> object:
        """The rule itself: its source, or the function imported again from where it lives."""
        source = data.get("rule")
        if source is not None:
            return PythonRule(str(source))
        module_name, function_name = data.get("module"), data.get("function")
        if module_name is None or function_name is None:
            raise ValueError(f"Rule {data.get('name')!r} has neither source nor a function to find it by")
        found: object
        try:
            found = importlib.import_module(str(module_name))
            for part in str(function_name).split("."):
                found = getattr(found, part)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Rule {data.get('name')!r} is {module_name}.{function_name}, which isn't there: {error}") from error
        return found
