import itertools
import logging
import math

from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.induction_settings import InductionSettings
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)


class RuleInducer:
    """Induces rules from search samples, with conditions built from state variables and action parameters."""

    def __init__(
        self,
        interpreter: Interpreter,
        variable_name_mapper: VariableNameMapper,
        rule_text_mapper: RuleTextMapper,
    ) -> None:
        self._interpreter = interpreter
        self._variable_name_mapper = variable_name_mapper
        self._rule_text_mapper = rule_text_mapper

    def induce(self, domain: str, samples: tuple[ActionSample, ...], settings: InductionSettings) -> RuleBase:
        counted = [sample for sample in samples if sample.visits >= settings.min_visits]
        rules: list[Rule] = []
        for name in dict.fromkeys(sample.action.name for sample in counted):
            group = [sample for sample in counted if sample.action.name == name]
            action_rules = self._induce_action(name, group, settings)
            logger.info("Induced %d rules for %s from %d samples", len(action_rules), name, len(group))
            if logger.isEnabledFor(logging.DEBUG):
                for rule in action_rules:
                    logger.debug("%s", self._rule_text_mapper.to_text(rule))
            rules.extend(action_rules)
        return RuleBase(domain, tuple(rules))

    def _induce_action(self, name: str, group: list[ActionSample], settings: InductionSettings) -> list[Rule]:
        weights = [sample.visits for sample in group]
        values = [sample.mean_payoff for sample in group]
        base = self._rule(name, (), frozenset(range(len(group))), weights, values)
        literals = [
            (literal, matching)
            for literal in self._literals(group)
            if (matching := self._matching(literal, group)) is not None
        ]
        rules = [base]
        frontier: list[tuple[Rule, frozenset[int], int]] = [(base, frozenset(range(len(group))), -1)]
        for _ in range(settings.max_conditions):
            next_frontier: list[tuple[Rule, frozenset[int], int]] = []
            for parent, parent_matching, last in frontier:
                for index in range(last + 1, len(literals)):
                    literal, literal_matching = literals[index]
                    matching = parent_matching & literal_matching
                    if not matching or matching == parent_matching:
                        continue
                    if sum(weights[sample] for sample in matching) < settings.min_rule_visits:
                        continue
                    rule = self._rule(name, (*parent.conditions, literal), matching, weights, values)
                    if abs(rule.expected_value - parent.expected_value) < settings.min_gain:
                        continue
                    rules.append(rule)
                    next_frontier.append((rule, matching, index))
            frontier = next_frontier
        return rules

    def _rule(
        self,
        name: str,
        conditions: tuple[Expression, ...],
        matching: frozenset[int],
        weights: list[int],
        values: list[float],
    ) -> Rule:
        visits = sum(weights[sample] for sample in matching)
        expected_value = math.fsum(weights[sample] * values[sample] for sample in matching) / visits
        return Rule(name, conditions, expected_value, visits)

    def _literals(self, group: list[ActionSample]) -> list[Expression]:
        """variable == value, indexed variable at the action's parameters == value, and parameter == value."""
        values_by_variable: dict[str, dict[Value, None]] = {}
        values_by_parameter: dict[str, dict[Value, None]] = {}
        for sample in group:
            for variable, value in sample.state.variables:
                values_by_variable.setdefault(variable, {})[value] = None
            for parameter, value in sample.action.parameters:
                values_by_parameter.setdefault(parameter, {})[value] = None

        literals: list[Expression] = [
            Equals(StateVariable(variable), Constant(value))
            for variable, values in values_by_variable.items()
            for value in values
        ]

        parameters = list(values_by_parameter)
        values_by_base: dict[str, dict[Value, None]] = {}
        for variable, values in values_by_variable.items():
            base, indices = self._variable_name_mapper.from_name(variable)
            if indices and len(indices) == len(parameters):
                values_by_base.setdefault(base, {}).update(values)
        literals.extend(
            Equals(StateVariable(base, tuple(ActionParameter(parameter) for parameter in order)), Constant(value))
            for base, values in values_by_base.items()
            for order in itertools.permutations(parameters)
            for value in values
        )

        literals.extend(
            Equals(ActionParameter(parameter), Constant(value))
            for parameter, values in values_by_parameter.items()
            for value in values
        )
        return literals

    def _matching(self, literal: Expression, group: list[ActionSample]) -> frozenset[int] | None:
        """The samples where the literal holds, or None when it can't be evaluated for every sample."""
        matching: set[int] = set()
        for index, sample in enumerate(group):
            try:
                if self._interpreter.evaluate(literal, sample.state, sample.action) is True:
                    matching.add(index)
            except KeyError:
                return None
        return frozenset(matching)
