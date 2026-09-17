from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from scipy.special import expit

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rule.model.python_rule import PythonRule


class OnlineValueFitter:
    """Moves a value base's weights one step toward what a finished game showed, on the loss the rule search fits: the
    mean logistic loss of the targets, scaled from the value base's lowest to its highest payoff, plus price × the sum
    of the weights' sizes, the bias unpriced. One proximal gradient step: each weight and the bias move against the
    gradient by the learning rate, then each weight shrinks toward 0 by learning rate × price, and a weight landing on 0
    drops its rule. A blank reading adds nothing, as it does when valuing. It adds no rule: new rules come from the
    search."""

    def step(
        self,
        value_base: ValueBase,
        terms: Mapping[PythonRule, np.ndarray | None],
        targets: np.ndarray,
        learning_rate: float,
        price: float,
    ) -> tuple[ValueBase, float] | None:
        """The value base after the step and the largest move of a weight or the bias; None when a rule's term couldn't
        be read on every row, or there is no row."""
        rules = value_base.rules
        columns = [terms.get(rule.term) for rule in rules]
        if len(targets) == 0 or any(column is None for column in columns):
            return None
        spread = value_base.high - value_base.low
        scaled = np.clip((targets - value_base.low) / spread, 0.0, 1.0) if spread > 0.0 else np.full(len(targets), 0.5)
        design = np.column_stack([np.nan_to_num(column, nan=0.0) for column in columns]) if rules else np.empty((len(targets), 0))  # type: ignore[arg-type]
        weights = np.array([rule.weight for rule in rules], dtype=float)
        errors = expit(design @ weights + value_base.bias) - scaled
        moved = weights - learning_rate * (design.T @ errors) / len(targets)
        shrunk = np.sign(moved) * np.maximum(np.abs(moved) - learning_rate * price, 0.0)
        bias = value_base.bias - learning_rate * float(np.mean(errors))
        largest = max([abs(bias - value_base.bias), *np.abs(shrunk - weights).tolist()])
        kept = tuple(ValueRule(rule.term, float(weight)) for rule, weight in zip(rules, shrunk, strict=True) if weight != 0.0)
        return replace(value_base, bias=bias, rules=kept), float(largest)
