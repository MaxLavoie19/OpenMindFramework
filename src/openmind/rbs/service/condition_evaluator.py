from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.constant.consequence_constant import ACTION
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner


class ConditionEvaluator:
    """Evaluates rules on action rows. On each row a rule reads the row's state, the row's action parameters by name,
    the row's action as `action`, and the consequence library's names. Several rules at once are evaluated in the task
    runner's workers: the rows are split into slices, and each worker evaluates every rule on its slice."""

    def __init__(
        self,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        task_runner: TaskRunner,
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._task_runner = task_runner

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
        return self._to_mask(self.values(domain, rows, rule))

    def all_values(
        self, domain: Domain, rows: Sequence[ActionRow], rules: Sequence[PythonRule]
    ) -> list[list[object] | None]:
        """What values gives for each rule, in the rules' order, the rows split between the task runner's workers. In
        workers, the values travel back by pickling."""
        slices = self._task_runner.split(rows)
        if len(slices) <= 1 or not rules:
            return self.slice_values(domain, rows, rules)
        count = len(slices)
        results = self._task_runner.map(self.slice_values, [domain] * count, slices, [tuple(rules)] * count)
        merged: list[list[object] | None] = []
        for index in range(len(rules)):
            parts = [result[index] for result in results]
            merged.append(None if any(part is None for part in parts) else [value for part in parts for value in part])  # type: ignore[union-attr]
        return merged

    def slice_values(
        self, domain: Domain, rows: Sequence[ActionRow], rules: Sequence[PythonRule]
    ) -> list[list[object] | None]:
        """What values gives for each rule on these rows, in this process."""
        return [self.values(domain, rows, rule) for rule in rules]

    def masks(self, domain: Domain, rows: Sequence[ActionRow], rules: Sequence[PythonRule]) -> list[np.ndarray | None]:
        """What mask gives for each rule, in the rules' order, the rows split between the task runner's workers."""
        return [self._to_mask(values) for values in self.all_values(domain, rows, rules)]

    def _to_mask(self, values: list[object] | None) -> np.ndarray | None:
        return None if values is None else np.array([value is True for value in values], dtype=bool)
