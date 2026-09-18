import logging
import random

from openmind.agent.factory.agent_factory import create_agent
from openmind.evaluation.constant.evaluation_constant import POSITION_ATTEMPTS
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ReferenceSearch:
    """Stands in for perfect play where exact search can't reach. Positions come from uniformly random games: each game
    gives one of its positions with a legal action, drawn uniformly, and duplicates are skipped. An action's value is
    the mean payoff for the player to act that a long unguided search gives it."""


    def positions(self, rbs: RuleBasedSystem, count: int, rng: random.Random) -> tuple[State, ...]:
        """Up to count distinct positions, from at most POSITION_ATTEMPTS random games per position wanted."""
        found: dict[State, None] = {}
        games = 0
        while len(found) < count and games < POSITION_ATTEMPTS * count:
            games += 1
            state, played = rbs.start(), []
            while actions := rbs.actions(state):
                played.append(state)
                outcomes = rbs.outcomes(state, rng.choice(actions)).outcomes
                (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[chance for _, chance in outcomes])
            if played:
                found.setdefault(rng.choice(played), None)
        logger.info("%s: %d reference positions from %d random games", rbs.context, len(found), games)
        return tuple(found)

    def action_values(
        self, rbs: RuleBasedSystem, state: State, iterations: int, seed: int
    ) -> tuple[tuple[Action, float], ...]:
        """Each legal action with its mean payoff in an unguided search of that many iterations, in the solver's order;
        an action the search never visited is worth 0."""
        statistics = {item.action: item.mean_payoff for item in create_agent(iterations, seed).search(rbs, state).statistics}
        return tuple((action, statistics.get(action, 0.0)) for action in rbs.actions(state))
