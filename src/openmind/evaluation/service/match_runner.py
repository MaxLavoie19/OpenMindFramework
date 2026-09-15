import logging
import random

from openmind.agent.model.domain import Domain
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.csp.service.joint_solver import JointSolver
from openmind.csp.service.solver import Solver
from openmind.evaluation.model.match_results import MatchResults
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.observation.service.state_observer import StateObserver
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.service.predictor import Predictor
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class MatchRunner:
    """Plays series of games between two policies in a two-player domain. Every game draws two seeds up front: one its
    policies are created from, one for its outcomes. Games don't depend on each other, so they run in the task
    runner's workers and give the same results whatever the number of workers. In a domain with an observation, a policy
    is given only what its player sees. Where players act at once, every player to act chooses, given its player's name,
    and the actions are taken together. Each game is logged by the worker that plays it, as soon as it ends."""

    def __init__(
        self,
        solver: Solver,
        predictor: Predictor,
        state_reader: StateReader,
        task_runner: TaskRunner,
        state_observer: StateObserver | None = None,
        joint_solver: JointSolver | None = None,
    ) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._task_runner = task_runner
        self._state_observer = create_state_observer() if state_observer is None else state_observer
        self._joint_solver = JointSolver(solver, state_reader) if joint_solver is None else joint_solver

    def series(
        self,
        domain: Domain,
        evaluated: PolicyFactory,
        opponent: PolicyFactory,
        opponent_name: str,
        games: int,
        rng: random.Random,
    ) -> MatchResults:
        """Plays the games, switching seats every game; results count from the evaluated policy's side. A game that took
        its worker over the memory cap in a fresh worker too isn't counted."""
        if len(domain.players.names) != 2:
            raise ValueError(f"A series needs two players, not {len(domain.players.names)}")
        seats = [(game - 1) % 2 for game in range(1, games + 1)]
        policy_seeds, outcome_seeds = [], []
        for _ in range(games):
            policy_seeds.append(rng.getrandbits(32))
            outcome_seeds.append(rng.getrandbits(32))
        results = self._task_runner.map(
            self.play_game,
            [domain] * games,
            [evaluated] * games,
            [opponent] * games,
            seats,
            policy_seeds,
            outcome_seeds,
            droppable=True,
        )
        played = wins = draws = losses = 0
        for game, (seat, payoffs) in enumerate(zip(seats, results, strict=True), start=1):
            if isinstance(payoffs, DroppedCall):
                continue
            played += 1
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
        return MatchResults(opponent_name, played, wins, draws, losses)

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
        state, plies = domain.initial_state, 0
        names = domain.players.names
        while True:
            if self._state_reader.acts_at_once(state, domain.players):
                legal = self._joint_solver.legal(domain.problem, state, domain.players)
                if not legal:
                    break
                joint = JointAction(
                    tuple(
                        (names[index], policies[index].choose(domain, self._seen(domain, state, index), names[index]))  # type: ignore[call-arg]
                        for index, _ in legal
                    )
                )
                outcomes = self._predictor.predict_joint(domain.transitions, state, joint).outcomes
            else:
                if not self._solver.solve(domain.problem, state):
                    break
                player = self._state_reader.player_to_act(state, domain.players)
                seen = self._seen(domain, state, player)
                outcomes = self._predictor.predict(domain.transitions, state, policies[player].choose(domain, seen)).outcomes
            plies += 1
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        payoffs = self._state_reader.payoffs(state, domain.players)
        logger.info(
            "Game with seeds %d and %d finished in %d plies, the evaluated policy playing %s: payoffs %s",
            policy_seed,
            outcome_seed,
            plies,
            domain.players.names[seat],
            " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, payoffs)),
        )
        return payoffs

    def _seen(self, domain: Domain, state: State, player: int) -> State:
        if domain.observation is None:
            return state
        return self._state_observer.observe(domain.observation, state, domain.players.names[player])
