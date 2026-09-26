import importlib
import json

from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.rule.mapper.clause_json_mapper import ClauseJsonMapper
from openmind.rule.mapper.consequence_json_mapper import ConsequenceJsonMapper
from openmind.rule.model.clause_rule import ClauseRule
from openmind.rule.model.consequence_rule import ConsequenceRule
from openmind.rule.model.domain_rule import DomainRule
from openmind.rule.model.python_rule import PythonRule


class RuleRecordJsonMapper:
    """Maps a rule the agent knows to one line of JSON and back.

    A rule written as source keeps its source. A rule OMF reasoned out keeps its clause, as the structure it is, so
    that reading it back gives something that can be resolved against and not only run. A rule the project gave as a
    function keeps the module it lives in and its name there, and loading imports it again, which is the same way a
    worker process finds it. A function that can't be found raises ValueError rather than coming back missing: a
    game model short of a rule is a different game."""

    def __init__(
        self,
        knowledge_json_mapper: KnowledgeJsonMapper | None = None,
        clause_json_mapper: ClauseJsonMapper | None = None,
        consequence_json_mapper: ConsequenceJsonMapper | None = None,
    ) -> None:
        self._knowledge = KnowledgeJsonMapper() if knowledge_json_mapper is None else knowledge_json_mapper
        self._clause = ClauseJsonMapper() if clause_json_mapper is None else clause_json_mapper
        self._consequence = (
            ConsequenceJsonMapper(self._clause) if consequence_json_mapper is None else consequence_json_mapper
        )

    def to_line(self, rule: RuleRecord) -> str:
        """The rule as one line of JSON, without a line break of its own."""
        return json.dumps(self.to_data(rule), separators=(",", ":"))

    def from_line(self, line: str) -> RuleRecord:
        return self.from_data(json.loads(line))

    def to_data(self, rule: RuleRecord) -> dict[str, object]:
        written = rule.rule
        source = written.source if isinstance(written, PythonRule) else None
        clause = self._clause.clause_to_data(written.clause) if isinstance(written, ClauseRule) else None
        does = (
            [self._consequence.to_data(one) for one in written.consequences]
            if isinstance(written, ConsequenceRule)
            else None
        )
        values = (
            [self._knowledge.value_to_data(one) for one in written.values]
            if isinstance(written, DomainRule)
            else None
        )
        named = source is None and clause is None and does is None and values is None
        return {
            "id": rule.id,
            "name": rule.name,
            "kind": rule.kind,
            "rule": source,
            "clause": clause,
            "does": does,
            "values": values,
            "module": getattr(written, "__module__", None) if named else None,
            "function": getattr(written, "__qualname__", None) if named else None,
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
        """The rule itself: its source, its clause, what it was learned an action does, the values a parameter
        may take, or the function imported again from where it lives.

        **Everything OMF learns is one of the middle three, and two of them were unwritable.** A game induced
        from what a learner worked out declares its legality as clauses, what its actions do as consequences
        and what its parameters range over as values; a mapper knowing only source and clause could write the
        first and lost the rest on the way to disk. The game declared fine and could not be read back, which
        is the worst place for that to be found — a whole run of learning, kept and unreadable."""
        source = data.get("rule")
        if source is not None:
            return PythonRule(str(source))
        clause = data.get("clause")
        if clause is not None:
            return ClauseRule(self._clause.clause_from_data(clause))  # type: ignore[arg-type]
        does = data.get("does")
        if does is not None:
            return ConsequenceRule(tuple(self._consequence.from_data(one) for one in does))  # type: ignore[arg-type]
        values = data.get("values")
        if values is not None:
            return DomainRule(tuple(self._knowledge.value_from_data(one) for one in values))  # type: ignore[arg-type]
        module_name, function_name = data.get("module"), data.get("function")
        if module_name is None or function_name is None:
            raise ValueError(
                f"Rule {data.get('name')!r} has no source, no clause, no consequences, no values and no "
                "function to find it by"
            )
        found: object
        try:
            found = importlib.import_module(str(module_name))
            for part in str(function_name).split("."):
                found = getattr(found, part)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Rule {data.get('name')!r} is {module_name}.{function_name}, which isn't there: {error}") from error
        return found
