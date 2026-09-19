import logging
from collections.abc import Sequence

from openmind.inference.model.deduction import Deduction
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class EndingWalker:
    """Walks a game back from its end with the position deducer: its positions deduced from the last one backward."""

    def __init__(self, position_deducer: PositionDeducer) -> None:
        self._position_deducer = position_deducer

    def walk_back(
        self, rbs: RuleBasedGame, states: Sequence[State], budget: DeductionBudget, limit: int
    ) -> tuple[Deduction, ...]:
        """The game's positions deduced from the last one backward, stopping after the first one not proven or at the
        limit."""
        deductions: list[Deduction] = []
        for state in reversed(states):
            if len(deductions) >= limit:
                break
            deduction = self._position_deducer.deduce(rbs, state, budget)
            deductions.append(deduction)
            if deduction.payoffs is None:
                break
        logger.info(
            "Walked back %d positions from the end, %d proven",
            len(deductions),
            sum(1 for deduction in deductions if deduction.payoffs is not None),
        )
        return tuple(deductions)
