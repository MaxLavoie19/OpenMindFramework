import logging
from collections.abc import Sequence

from openmind.agent.constant.game_record_constant import ACTIONS
from openmind.agent.model.domain import Domain
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class GameRecorder:
    """Reads what a domain says about a finished game, for the logs: why it ended, from its last state, and its record,
    from the initial state and the actions played. A rule that raises is logged as a warning and gives None."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def ending(self, domain: Domain, state: State) -> str | None:
        """Why the game ended, or None when the domain doesn't say or the rule gives nothing."""
        if domain.ending is None:
            return None
        try:
            value = self._rule_caller.value(domain.ending, state, None, None, domain.transitions.definitions)
        except Exception:  # noqa: BLE001 - a domain's rule is the project's code; the logs must survive it
            logger.warning("The %s ending rule raised", domain.name, exc_info=True)
            return None
        return None if value is None else str(value)

    def record(self, domain: Domain, actions: Sequence[Action]) -> str | None:
        """The game's record, or None when the domain doesn't say or the rule gives nothing."""
        if domain.record is None:
            return None
        try:
            value = self._rule_caller.value(
                domain.record, domain.initial_state, {ACTIONS: tuple(actions)}, None, domain.transitions.definitions
            )
        except Exception:  # noqa: BLE001 - a domain's rule is the project's code; the logs must survive it
            logger.warning("The %s record rule raised", domain.name, exc_info=True)
            return None
        return None if value is None else str(value)
