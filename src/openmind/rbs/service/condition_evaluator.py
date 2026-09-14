from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.constant.consequence_constant import ACTION
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner


class ConditionEvaluator:
    """Evaluates a rule on action rows. On each row the rule reads the row's state, the row's action parameters by name,
    the row's action as `action`, and the consequence library's names."""

    def __init__(
        self, rule_compiler: RuleCompiler, rule_runner: RuleRunner, consequence_library: ConsequenceLibrary
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library

    def values(self, domain: Domain, rows: Sequence[ActionRow], rule: PythonRule) -> list[object] | None:
        """The rule's value on every row, or None when it raises KeyError, NameError or TypeError on one of them. An
        action parameter named `action` raises ValueError: generated rules reserve the name."""
        values: list[object] = []
        for row in rows:
            parameters: dict[str, object] = dict(row.action.parameters)
            if ACTION in parameters:
                raise ValueError(f"Action parameter {ACTION!r} of {row.action.name!r} clashes with generated rules")
            parameters[ACTION] = row.action
            compiled = self._rule_compiler.compile_value(rule, parameters)
            try:
                values.append(
                    self._rule_runner.value(
                        compiled, row.state, parameters, self._consequence_library.names(domain, row.state)
                    )
                )
            except (KeyError, NameError, TypeError):
                return None
        return values

    def mask(self, domain: Domain, rows: Sequence[ActionRow], rule: PythonRule) -> np.ndarray | None:
        """Where the rule gives True, or None when it can't be evaluated on every row."""
        values = self.values(domain, rows, rule)
        return None if values is None else np.array([value is True for value in values], dtype=bool)
