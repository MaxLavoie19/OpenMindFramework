import logging
import math
import random
from collections.abc import Mapping, Sequence

from openmind.heuristic.model.rule_cost import RuleCost
from openmind.model.model.rule_candidate import RuleCandidate
from openmind.model.model.rule_signal import RuleSignal

logger = logging.getLogger(__name__)


class RulesetBuilder:
    """Candidate rulesets, each as much value as fits the time a decision has.

    **Signals are aspects, not authors.** One prizes certainty and another coverage, and a set built entirely
    on either is worse than one holding both: a narrow accurate rule is worth having and is worth little
    without generic rules beside it to speak when it does not. So a set is not one signal's opinion — it is a
    *mix*, and which mix synergises is not a thing anybody can work out in advance.

    **So several mixes are built and play decides between them.** Each is packed under the same time budget by
    what it is worth for what it costs, so one comes out as many cheap rules, another as a few dear ones,
    another as a handful of each. They are registered as models and drawn against one another by UCB, which is
    the machinery that already races heuristics — and *that* is where "the signal that gives the best
    heuristic for a problem weighs more" actually happens: in which blends keep winning, measured, rather than
    in a budget deciding who may buy what.

    **Ratings are made comparable by softmax, which is this project's answer wherever scales are arbitrary.**
    One signal returns a correlation, another a share of rows, another a squared contribution; nothing makes
    those commensurable on their face, and the same reading already turns a heuristic's scores into a
    preference over moves.

    It keeps nothing: built once, it is given the candidates, the signals and the budget on every call."""

    def built(
        self,
        candidates: Sequence[RuleCandidate],
        signals: Sequence[RuleSignal],
        seconds: float,
        costs: Mapping[str, RuleCost],
        how_many: int,
        rng: random.Random,
        caution: float = 1.0,
    ) -> list[tuple[Mapping[str, float], tuple[RuleCandidate, ...]]]:
        """That many candidate sets, each with the mix of signals it was built on.

        The first is always an even mix of every signal, so that a run which builds one set builds the
        sensible one; the rest are drawn flat over the signals, which is what gives the drawing something to
        tell apart. A mix is kept with its set because what is being raced is the mix, and a set that wins
        says which aspects mattered for this game.

        **What is not here yet is the mix learning from what won.** The mixes are drawn afresh every time and
        nothing carries between ponders, so a blend that wins is not leant on next time. That is a deliberate
        gap rather than an oversight."""
        if not candidates or not signals or how_many < 1:
            return []
        rated = {one.name: self._shares(one.rates(candidates)) for one in signals}
        found = []
        for at in range(how_many):
            mix = self._mixed([one.name for one in signals], rng, even=at == 0)
            worth = [
                math.fsum(mix[name] * rated[name][where] for name in mix) for where in range(len(candidates))
            ]
            found.append((mix, self._packed(candidates, worth, seconds, costs, caution)))
        logger.info(
            "Built %d candidate rulesets from %d candidates within %.4f seconds a valuing: %s rules",
            len(found), len(candidates), seconds, ", ".join(str(len(one)) for _, one in found),
        )
        return found

    def _packed(
        self,
        candidates: Sequence[RuleCandidate],
        worth: Sequence[float],
        seconds: float,
        costs: Mapping[str, RuleCost],
        caution: float,
    ) -> tuple[RuleCandidate, ...]:
        """As much of that worth as fits the budget: best value for the cost first, taken while it fits.

        **The same packing the reading does, and for the same reason.** A budget is a capacity and rules have
        costs, so what to take is a knapsack and the greedy answer is by ratio. A rule too dear to fit is
        passed over rather than ending the taking, so one expensive candidate early in the order does not cost
        the cheap ones behind it — which is how a set of many cheap rules and a set of a few dear ones both
        become reachable.

        A rule nobody has timed is priced at nothing and so is taken early, which is what gets it measured."""
        def priced(one: RuleCandidate) -> float:
            """What that candidate has been measured to cost. A rule is named for what it reads, which is how
            a cost gathered while reading is found again here."""
            named = getattr(one.rule, "source", "")
            return costs.get(named, RuleCost()).priced(caution)
        order = sorted(
            range(len(candidates)),
            key=lambda at: -(worth[at] / priced(candidates[at]) if priced(candidates[at]) > 0 else float("inf")),
        )
        taken, spent = [], 0.0
        for at in order:
            if worth[at] <= 0.0:
                continue
            cost = priced(candidates[at])
            if not taken or spent + cost <= seconds:
                taken.append(candidates[at])
                spent += cost
        return tuple(taken)

    def _mixed(self, names: Sequence[str], rng: random.Random, even: bool) -> dict[str, float]:
        """How much of each signal this set is built on, summing to one.

        Drawn flat over the signals rather than favouring any, because which aspects matter is the thing being
        found out. An even mix is offered first so that a run building few sets still builds the obvious one.
        """
        if even or len(names) == 1:
            return {name: 1.0 / len(names) for name in names}
        drawn = [rng.random() for _ in names]
        whole = math.fsum(drawn) or 1.0
        return {name: one / whole for name, one in zip(names, drawn, strict=True)}

    def _shares(self, rated: Sequence[float]) -> list[float]:
        """Those ratings as shares of one, so signals on different scales can be mixed.

        Softmax, shifted by the largest so nothing overflows — the same reading that already turns a
        heuristic's scores into a preference over moves. A signal that rated everything alike comes out even,
        which is what having no opinion should look like."""
        if not rated:
            return []
        most = max(rated)
        weighed = [math.exp(one - most) for one in rated]
        whole = math.fsum(weighed) or 1.0
        return [one / whole for one in weighed]
