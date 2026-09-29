from dataclasses import dataclass

import numpy as np

from openmind.inference.model.expression import Expression
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class RuleCandidate:
    """One term a fit kept, and everything a signal may read about it.

    **Every field is already computed by the sweep that produced it.** The readings are the column the search
    took over the training rows and the payoffs are what those rows led to, both of which the fitting needed
    anyway — so asking four signals what they make of a hundred candidates costs arithmetic over numbers in
    hand, not a second pass over the positions.

    A reading of NaN is blank: the term had nothing to say about that row. A reading of 0 is not blank — the
    term looked and found nothing, which is what a detector does nearly everywhere, and the difference between
    the two is the whole of why a mate detector is not a rare term."""

    expression: Expression
    rule: PythonRule
    #: What the fit gave it, on the standardised column.
    weight: float
    #: Its column over the training rows, blanks as NaN.
    readings: np.ndarray
    #: What those rows paid whoever played them.
    payoffs: np.ndarray

    def fires(self) -> np.ndarray:
        """Where the term had something to say: present, and not nought."""
        return ~np.isnan(self.readings) & (self.readings != 0.0)
