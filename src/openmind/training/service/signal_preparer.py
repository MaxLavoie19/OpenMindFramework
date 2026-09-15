import logging
from dataclasses import replace

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import DEFAULT_HIGHEST_PAYOFF, DEFAULT_LOWEST_PAYOFF
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.training.constant.signal_constant import DEDUCED, DEFAULT_GOAL_LIMIT, DOUBLED
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord
from openmind.training.service.heuristic_deducer import HeuristicDeducer

logger = logging.getLogger(__name__)


class SignalPreparer:
    """Prepares a clueless agent before its first game: the signals deduced from the domain's rules, and the value bases
    round 1's arms follow. `deduced` weighs every deduced signal the same, 1 over their number, with a bias of 0 and
    payoffs from DEFAULT_LOWEST_PAYOFF to DEFAULT_HIGHEST_PAYOFF; each `deduced, <signal> doubled` is the same with that
    signal's weight doubled. Games then tell which weights should grow."""

    def __init__(self, heuristic_deducer: HeuristicDeducer) -> None:
        self._heuristic_deducer = heuristic_deducer

    def prepare(self, domain: Domain, goal_limit: int = DEFAULT_GOAL_LIMIT) -> SignalLibrary:
        """A library with a record for every deduced signal, never read yet, and the deduced value bases; goal distance
        looks for a win up to `goal_limit` moves ahead."""
        signals = self._heuristic_deducer.deduce(domain, goal_limit)
        weight = 1.0 / len(signals)
        base = ValueBase(
            domain.name,
            0.0,
            DEFAULT_LOWEST_PAYOFF,
            DEFAULT_HIGHEST_PAYOFF,
            tuple(ValueRule(signal.source, weight) for signal in signals if signal.source is not None),
        )
        variations = tuple(
            (
                f"{DEDUCED}, {signal.name} {DOUBLED}",
                replace(base, rules=tuple(replace(rule, weight=2 * weight) if at == index else rule for at, rule in enumerate(base.rules))),
            )
            for index, signal in enumerate(signals)
        )
        logger.info(
            "Prepared %d signals deduced from the rules: value base %s, every weight %s, and %d variations doubling one weight each",
            len(signals),
            DEDUCED,
            weight,
            len(variations),
        )
        return SignalLibrary(domain.name, tuple(SignalRecord(signal) for signal in signals), (), ((DEDUCED, base), *variations))
