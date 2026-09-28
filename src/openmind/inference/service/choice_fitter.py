import logging
import math
from collections.abc import Sequence

import numpy as np

from openmind.inference.model.choice import Choice
from openmind.rbs.model.sparse_fit import SparseFit

logger = logging.getLogger(__name__)


class ChoiceFitter:
    """Fits what somebody prefers from what they chose, where nobody said what anything was worth.

    **A value is fitted to a target and a preference is fitted to a ranking, and the difference is the whole
    of why this is not the sparse fitter.** Every fit in OMF until now has had a column of answers: this
    position paid that, fit something that says so. Watching somebody play gives no such column. Nobody
    announces what a move was worth to them; they take one of the moves in front of them, and everything
    knowable is in the comparison.

    So the likelihood is over the candidates of each decision — the softmax — and it is maximised by the
    same proximal step the sparse fitter uses, because the reason for that step is unchanged: an absolute
    value has no derivative at zero, which is exactly where the useful behaviour is, and a smooth optimiser
    slides past it leaving every weight merely small rather than gone.

    **There is no bias.** A constant added to every candidate of a decision cancels in the softmax, so a bias
    is a direction the likelihood is flat along and fitting one is fitting noise. The sparse fitter has one
    because a value has a level; a preference does not.

    **The candidates are the evidence as much as the choice is.** A move taken from three says little, the
    same move taken from forty says a great deal, and a fit shown only what was played cannot tell them
    apart. A decision offering one thing is not a decision and is left out.

    What comes back is a `SparseFit` like any other, so what reads a fitted heuristic does not have to know
    which kind of fitting produced it."""

    def fit(
        self,
        choices: Sequence[Choice],
        price: float,
        max_steps: int,
        tolerance: float,
        start: SparseFit | None = None,
        costs: np.ndarray | None = None,
    ) -> SparseFit:
        """The weights that best explain what was chosen, most of them nought.

        `price` shrinks every weight toward zero and `costs` says what each column's shrinking costs, so a
        term that is dear to read is dear to keep — the same handle the sparse fitter offers. No decision
        that decides anything raises ValueError: there is nothing to learn from choices nobody made."""
        deciding = [one for one in choices if one.decides]
        if not deciding:
            raise ValueError("A preference is fitted to decisions, and none of these decided anything")
        width = int(deciding[0].candidates.shape[1])
        if any(int(one.candidates.shape[1]) != width for one in deciding):
            raise ValueError("Every decision has to be read in the same terms")
        step = self._step(deciding, width)
        shrink = step * price * (np.ones(width) if costs is None else np.asarray(costs, dtype=float))
        weights = np.zeros(width) if start is None else np.array(start.weights, dtype=float)
        previous = weights.copy()
        settled, steps = False, 0
        for steps in range(1, max_steps + 1):
            momentum = weights + (steps - 1) / (steps + 2) * (weights - previous)
            candidate = momentum - step * self._gradient(deciding, momentum)
            candidate = np.sign(candidate) * np.maximum(np.abs(candidate) - shrink, 0.0)
            previous, weights = weights, candidate
            if np.max(np.abs(weights - previous)) <= tolerance * max(1.0, float(np.max(np.abs(weights), initial=0.0))):
                settled = True
                break
        kept = int(np.count_nonzero(weights))
        logger.info(
            "Fitted a preference over %d decisions in %d terms: %d kept at price %g, %s after %d steps",
            len(deciding),
            width,
            kept,
            price,
            "settled" if settled else "not settled",
            steps,
        )
        return SparseFit(tuple(float(one) for one in weights), 0.0, steps, settled)

    def loss(self, choices: Sequence[Choice], weights: tuple[float, ...]) -> float:
        """How surprised these weights are by what was chosen, per decision, without the price.

        The mean negative log likelihood. A fit that always expected what happened scores nought; one that
        spread its expectation evenly over ten candidates scores the log of ten, which is what knowing
        nothing costs — so the number can be read against what ignorance would have cost rather than only
        against another fit."""
        deciding = [one for one in choices if one.decides]
        if not deciding:
            return 0.0
        held = np.asarray(weights, dtype=float)
        return float(np.mean([-self._logged(one, held) for one in deciding]))

    def knowing_nothing(self, choices: Sequence[Choice]) -> float:
        """What the same decisions cost a fit that has no opinion: the log of how many were on offer, meaned.

        Worth having beside the loss, because a loss on its own says nothing. Ten candidates and a loss of
        2.30 is a fit that learned exactly nothing, and the same loss where two were on offer is a fit that
        is worse than a coin."""
        deciding = [one for one in choices if one.decides]
        return float(np.mean([math.log(one.offered) for one in deciding])) if deciding else 0.0

    def _logged(self, choice: Choice, weights: np.ndarray) -> float:
        """The log chance this fit gave what was actually chosen."""
        scored = choice.candidates @ weights
        return float(scored[choice.taken] - self._total(scored))

    def _total(self, scored: np.ndarray) -> float:
        """The log of the summed exponentials, shifted by the largest so nothing overflows."""
        most = float(np.max(scored))
        return most + float(np.log(np.sum(np.exp(scored - most))))

    def _gradient(self, choices: Sequence[Choice], weights: np.ndarray) -> np.ndarray:
        """Which way the likelihood falls: what the fit expected minus what was taken, meaned over decisions.

        The softmax's gradient, and it reads as what it is — a term the fit expected more of than was
        actually chosen gets pulled down, and one it expected less of gets pulled up."""
        total = np.zeros(len(weights))
        for choice in choices:
            scored = choice.candidates @ weights
            chance = np.exp(scored - self._total(scored))
            total += choice.candidates.T @ chance - choice.candidates[choice.taken]
        return total / len(choices)

    def _step(self, choices: Sequence[Choice], width: int) -> float:
        """How far to move each time, small enough that the quadratic model holds.

        The softmax's curvature is bounded by a half of the largest squared length of a candidate row, so the
        inverse of that is a step the descent lemma allows. Taken over every candidate of every decision and
        not over the chosen ones alone, since the gradient reads them all."""
        largest = max(
            (float(np.max(np.sum(one.candidates**2, axis=1), initial=0.0)) for one in choices), default=0.0
        )
        return 1.0 if largest <= 0.0 else min(1.0, 2.0 / largest)
