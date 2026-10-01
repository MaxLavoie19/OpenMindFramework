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
    #: What somebody who knows makes of each of those rows, blanks as NaN; empty where nobody was asked.
    #:
    #: **A second anchor beside the payoff, and never a replacement for it.** What a game paid is one number
    #: per game credited back across every decision in it; a teller gives one per position, which is thousands
    #: of graded observations where the payoff gives a handful of sparse ones. That is what it buys — evidence
    #: per row — and what it cannot buy is the right to decide, because a term that tracks a teller perfectly
    #: has tracked its blind spots too.
    #:
    #: Empty rather than nought where no teller was asked. Nought is a real evaluation — a level position — so
    #: a column of them would read as a teller that called every row even, and a signal would correlate against
    #: it and come back with nothing, silently.
    told: np.ndarray | None = None

    def fires(self) -> np.ndarray:
        """Where the term had something to say: present, and not nought."""
        return ~np.isnan(self.readings) & (self.readings != 0.0)

    def influence(self) -> float:
        """How loudly this term can speak at all, which is what the fit gave it.

        **A rule that cannot change a decision is not a rule worth buying.** Measured in a real fitted
        heuristic: two terms at 0.494 and five more between 0.00072 and 0.000036 — fourteen thousand times
        too small to reorder anything the first two had separated. They were bought all the same, because
        three of the four signals read only what a term says and never how loudly it is allowed to say it.

        **A fact rather than a threshold.** The weights are fitted on standardised columns, so they are
        already on one scale and comparable across terms; nothing here picks a cut-off. A signal that scales
        what it wants by this simply wants a term it cannot hear proportionally less, and spends on what it
        wanted most first — so a term fourteen thousand times quieter is bought fourteen thousand places
        later, which in practice is never."""
        return abs(float(self.weight))
