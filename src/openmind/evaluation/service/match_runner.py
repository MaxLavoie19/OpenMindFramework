import logging
import random

from openmind.agent.model.domain import Domain
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.csp.service.solver import Solver
from openmind.evaluation.model.match_results import MatchResults
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.service.predictor import Predictor
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class MatchRunner:
    """Plays series of games between two policies in a two-player domain. Every game draws two seeds up front: one its
    policies are created from, one for its outcomes. Games don't depend on each other, so they run in the task
    runner's workers and give the same results whatever the number of workers."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader, task_runner: TaskRunner) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._task_runner = task_runner

    def series(
        self,
        domain: Domain,
        evaluated: PolicyFactory,
        opponent: PolicyFactory,
        opponent_name: str,
        games: int,
        rng: random.Random,
    ) -> MatchResults:
        """Plays the games, switching seats every game; results count from the evaluated policy's side."""
        if len(domain.players.names) != 2:
            raise ValueError(f"A series needs two players, not {len(domain.players.names)}")
        seats = [(game - 1) % 2 for game in range(1, games + 1)]
        policy_seeds, outcome_seeds = [], []
        for _ in range(games):
            policy_seeds.append(rng.getrandbits(32))
            outcome_seeds.append(rng.getrandbits(32))
        results = self._task_runner.map(
            self.play_game, [domain] * games, [evaluated] * games, [opponent] * games, seats, policy_seeds, outcome_seeds
        )
        wins = draws = losses = 0
        for game, (seat, payoffs) in enumerate(zip(seats, results, strict=True), start=1):
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

    def play_game(
        self,
        domain: Domain,
        evaluated: PolicyFactory,
        opponent: PolicyFactory,
        seat: int,
        policy_seed: int,
        outcome_seed: int,
    ) -> tuple[float, ...]:
        """One game, the evaluated policy in the given seat; the final payoffs."""
        created = (evaluated(policy_seed), opponent(policy_seed))
        policies = created if seat == 0 else (created[1], created[0])
        rng = random.Random(outcome_seed)
        state = domain.initial_state
        while self._solver.solve(domain.problem, state):
            policy = policies[self._state_reader.player_to_act(state, domain.players)]
            outcomes = self._predictor.predict(domain.transitions, state, policy.choose(domain, state)).outcomes
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        return self._state_reader.payoffs(state, domain.players)
