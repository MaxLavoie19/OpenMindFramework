import math

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.state import State


class RuleValuer:
    """Values a domain's positions with a value base: each player's value is low + (high - low) × logistic(bias + the sum
    of each rule's weight times its term), the terms reading the state's variables and the consequence library's names
    with `me` being that player. A term raising KeyError, NameError, TypeError, AttributeError, ValueError or an
    arithmetic error, or giving something other than a finite number, leaves the position unvalued."""

    def __init__(
        self,
        value_base: ValueBase,
        domain: Domain,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
    ) -> None:
        self._value_base = value_base
        self._domain = domain
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library

    def value(self, state: State) -> tuple[float, ...] | None:
        """Each player's value, in the order of the players' names, or None when a term can't be evaluated."""
        base = self._value_base
        values: list[float] = []
        for player in self._domain.players.names:
            contributions = self._contributions(state, player)
            if contributions is None:
                return None
            score = base.bias + math.fsum(contribution for _, contribution in contributions)
            values.append(base.low + (base.high - base.low) * self._logistic(score))
        return tuple(values)

    def explain(self, state: State, player: str) -> tuple[tuple[ValueRule, float], ...] | None:
        """Each rule with what it adds to the player's score, its weight times its term, or None when a term can't be
        evaluated."""
        return self._contributions(state, player)

    def _contributions(self, state: State, player: str) -> tuple[tuple[ValueRule, float], ...] | None:
        names = self._consequence_library.names(self._domain, state, player)
        contributions: list[tuple[ValueRule, float]] = []
        for rule in self._value_base.rules:
            compiled = self._rule_compiler.compile_value(rule.term)
            try:
                value = self._rule_runner.value(compiled, state, None, names)
            except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                return None
            if not isinstance(value, bool | int | float | np.bool_ | np.number):
                return None
            term = float(value)  # type: ignore[arg-type]
            if not math.isfinite(term):
                return None
            contributions.append((rule, rule.weight * term))
        return tuple(contributions)

    def _logistic(self, score: float) -> float:
        if score >= 0.0:
            return 1.0 / (1.0 + math.exp(-score))
        exponential = math.exp(score)
        return exponential / (1.0 + exponential)
