import logging
import random

from openmind.agent.model.domain import Domain
from openmind.agent.model.policy import Policy
from openmind.csp.service.solver import Solver
from openmind.evaluation.model.match_results import MatchResults
from openmind.predictor.service.predictor import Predictor
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class MatchRunner:
    """Plays series of games between two policies in a two-player domain."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader

    def series(
        self,
        domain: Domain,
        evaluated: Policy,
        opponent: Policy,
        opponent_name: str,
        games: int,
        rng: random.Random,
    ) -> MatchResults:
        """Plays the games, switching seats every game; results count from the evaluated policy's side."""
        if len(domain.players.names) != 2:
            raise ValueError(f"A series needs two players, not {len(domain.players.names)}")
        wins = draws = losses = 0
        for game in range(1, games + 1):
            seat = (game - 1) % 2
            payoffs = self._play(domain, (evaluated, opponent) if seat == 0 else (opponent, evaluated), rng)
            if payoffs[seat] > payoffs[1 - seat]:
                wins += 1
            elif payoffs[seat] == payoffs[1 - seat]:
                draws += 1
            else:
                losses += 1
            logger.debug(
                "Game %d against %s: evaluated agent plays %s, payoffs %s",
                game,
                opponent_name,
                domain.players.names[seat],
                " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, payoffs)),
            )
        return MatchResults(opponent_name, games, wins, draws, losses)

    def _play(
        self, domain: Domain, policies: tuple[Policy, Policy], rng: random.Random
    ) -> tuple[float, ...]:
        state = domain.initial_state
        while self._solver.solve(domain.problem, state):
            policy = policies[self._state_reader.player_to_act(state, domain.players)]
            outcomes = self._predictor.predict(domain.transitions, state, policy.choose(domain, state)).outcomes
            (state,) = rng.choices(
                [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
            )
        return self._state_reader.payoffs(state, domain.players)
