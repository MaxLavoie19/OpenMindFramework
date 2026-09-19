from dataclasses import dataclass

from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, DEFINITIONS, EFFECTS, VALUES
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class RuleBasedSystem:
    """A rule-based system: one ruleset's rules, each with its weight there, OMF's main explainable model. It is data:
    the services that run it — the simulation, the heuristic evaluator — are given it. Which task it fills is its
    ruleset's; any other model may fill the same task.

    `context` is the name of the context it was built for, `context_id` its id; the ruleset may be one the context
    inherits from."""

    context: str
    context_id: str
    ruleset: Ruleset
    rules: tuple[tuple[RuleRecord, float], ...]

    def of(self, *kinds: str) -> tuple[RuleRecord, ...]:
        """Its rules of those kinds, in the order they were linked."""
        return tuple(rule for rule, _ in self.rules if rule.kind in kinds)

    def weight(self, rule: RuleRecord) -> float:
        """What the rule weighs in the ruleset; 0 for a rule it doesn't list."""
        return next((weight for held, weight in self.rules if held.id == rule.id), 0.0)

    def action_names(self) -> tuple[str, ...]:
        """The actions its constraint and values rules are about, in the order first linked."""
        return tuple(dict.fromkeys(rule.action for rule in self.of(CONSTRAINT, VALUES) if rule.action))

    def values(self, action: str) -> dict[str, Rule]:
        """Each parameter of the action with the rule giving its values."""
        return {str(rule.parameter): rule.rule for rule in self.of(VALUES) if rule.action == action and rule.parameter}

    def constraints(self, action: str) -> tuple[Rule, ...]:
        """The rules every legal action of that name satisfies."""
        return tuple(rule.rule for rule in self.of(CONSTRAINT) if rule.action == action)

    def effects(self, action: str | None) -> tuple[tuple[float, Rule], ...]:
        """What the action leads to, each effects rule with its chance; for None, what the actions taken at once lead
        to together."""
        return tuple((rule.probability, rule.rule) for rule in self.of(EFFECTS) if rule.action == action)

    def definitions(self, name: str) -> PythonRule | None:
        """The definitions script of that name, or the only one the ruleset has."""
        scripts = [rule for rule in self.of(DEFINITIONS) if isinstance(rule.rule, PythonRule)]
        for rule in scripts:
            if rule.name == name:
                return rule.rule  # type: ignore[return-value]
        return scripts[0].rule if len(scripts) == 1 else None  # type: ignore[return-value]
