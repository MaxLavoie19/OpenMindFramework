import importlib
import json
from datetime import datetime

from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.rbs.model.python_rule import PythonRule


class RuleRecordJsonMapper:
    """Maps a rule the agent knows to one line of JSON and back.

    A rule written as source keeps its source. A rule the project gave as a function keeps the module it lives in and
    its name there, and loading imports it again, which is the same way a worker process finds it. A function that
    can't be found raises ValueError rather than coming back missing: a game model short of a rule is a different
    game."""

    def to_line(self, rule: RuleRecord) -> str:
        """The rule as one line of JSON, without a line break of its own."""
        return json.dumps(self.to_data(rule), separators=(",", ":"))

    def from_line(self, line: str) -> RuleRecord:
        return self.from_data(json.loads(line))

    def to_data(self, rule: RuleRecord) -> dict[str, object]:
        provenance = rule.provenance
        written = rule.rule
        source = written.source if isinstance(written, PythonRule) else None
        return {
            "id": rule.id,
            "name": rule.name,
            "kind": rule.kind,
            "source": source,
            "module": None if source is not None else getattr(written, "__module__", None),
            "function": None if source is not None else getattr(written, "__qualname__", None),
            "contexts": [[context, weight] for context, weight in rule.contexts],
            "action": rule.action,
            "parameter": rule.parameter,
            "probability": rule.probability,
            "from": provenance.source,
            "told": provenance.told,
            "at": provenance.at,
            "game": provenance.game,
            "round": provenance.round,
            "ply": provenance.ply,
            "when": None if provenance.when is None else provenance.when.isoformat(),
        }

    def from_data(self, data: dict[str, object]) -> RuleRecord:
        when = data.get("when")
        provenance = Provenance(
            str(data["from"]),
            data.get("told"),  # type: ignore[arg-type]
            data.get("at"),  # type: ignore[arg-type]
            data.get("game"),  # type: ignore[arg-type]
            data.get("round"),  # type: ignore[arg-type]
            data.get("ply"),  # type: ignore[arg-type]
            None if when is None else datetime.fromisoformat(str(when)),
        )
        contexts = tuple((str(context), float(weight)) for context, weight in data.get("contexts", ()))  # type: ignore[misc]
        return RuleRecord(
            str(data["name"]),
            str(data["kind"]),
            self._rule(data),
            provenance,
            contexts,
            data.get("action"),  # type: ignore[arg-type]
            data.get("parameter"),  # type: ignore[arg-type]
            float(data.get("probability", 1.0)),  # type: ignore[arg-type]
            str(data.get("id", "")),
        )

    def _rule(self, data: dict[str, object]) -> object:
        """The rule itself: its source, or the function imported again from where it lives."""
        source = data.get("source")
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
