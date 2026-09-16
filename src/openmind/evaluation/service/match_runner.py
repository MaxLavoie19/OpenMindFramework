import logging
import random
from collections.abc import Callable

from openmind.agent.model.domain import Domain
from openmind.agent.model.policy import Policy
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.agent.service.game_recorder import GameRecorder
from openmind.agent.service.timekeeper import Timekeeper
from openmind.csp.service.joint_solver import JointSolver
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.csp.service.solver import Solver
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.model.match_results import MatchResults
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.observation.service.state_observer import StateObserver
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.service.predictor import Predictor
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class MatchRunner:
    """Plays series of games between two policies in a two-player domain. Every game draws two seeds up front: one its
    policies are created from, one for its outcomes. Games don't depend on each other, so they run in the task
    runner's workers and give the same results whatever the number of workers. In a domain with an observation, a policy
    is given only what its player sees. Where players act at once, every player to act chooses, given its player's name,
    and the actions are taken together. Each game is logged by the worker that plays it, as soon as it ends.

    Given a time control, games are played on a clock: each choice is timed and charged to its player's clock, given to
    the policy with the steps that player has played; where players act at once, each choice is charged to its own
    player's clock. A player whose time runs out doesn't play that move, and the domain's timeout rule ends the game,
    applied for every player whose time ran out, in the players' order."""

    def __init__(
        self,
        solver: Solver,
        predictor: Predictor,
        state_reader: StateReader,
        task_runner: TaskRunner,
        state_observer: StateObserver | None = None,
        joint_solver: JointSolver | None = None,
        game_recorder: GameRecorder | None = None,
        timekeeper: Timekeeper | None = None,
    ) -> None:
        self._game_recorder = (
            GameRecorder(create_rule_caller())
            if game_recorder is None
            else game_recorder
        )
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._task_runner = task_runner
        self._state_observer = create_state_observer() if state_observer is None else state_observer
        self._joint_solver = JointSolver(solver, state_reader) if joint_solver is None else joint_solver
        self._timekeeper = Timekeeper(create_rule_caller()) if timekeeper is None else timekeeper

    def series(
        self,
        domain: Domain,
        evaluated: PolicyFactory,
        opponent: PolicyFactory,
        opponent_name: str,
        games: int,
        rng: random.Random,
        time_control: TimeControl | None = None,
        on_game: Callable[[int, int, MatchGame], None] | None = None,
    ) -> MatchResults:
        """Plays the games, switching seats every game; results count from the evaluated policy's side. A game that took
        its worker over the memory cap in a fresh worker too isn't counted. `on_game` is given each game's index, the
        evaluated policy's seat and the game in this process as soon as it ends, in the order games end; a game not
        counted isn't given."""
        if len(domain.players.names) != 2:
            raise ValueError(f"A series needs two players, not {len(domain.players.names)}")
        seats = [(game - 1) % 2 for game in range(1, games + 1)]
        policy_seeds, outcome_seeds = [], []
        for _ in range(games):
            policy_seeds.append(rng.getrandbits(32))
            outcome_seeds.append(rng.getrandbits(32))
        def arguments_for(index: int) -> tuple[object, ...]:
            return domain, evaluated, opponent, seats[index], policy_seeds[index], outcome_seeds[index], time_control

        def on_result(index: int, game: MatchGame | DroppedCall) -> None:
            if on_game is not None and not isinstance(game, DroppedCall):
                on_game(index, seats[index], game)

        results = self._task_runner.stream(self.play_game, games, arguments_for, on_result, droppable=True)
        played = wins = draws = losses = wins_on_time = losses_on_time = 0
        for game, (seat, result) in enumerate(zip(seats, results, strict=True), start=1):
            if isinstance(result, DroppedCall):
                continue
            payoffs = result.payoffs
            on_time = result.flagged is not None
            played += 1
            if payoffs[seat] > payoffs[1 - seat]:
                wins += 1
                wins_on_time += on_time
            elif payoffs[seat] == payoffs[1 - seat]:
                draws += 1
            else:
                losses += 1
                losses_on_time += on_time
            logger.debug(
                "Game %d against %s: evaluated agent plays %s, payoffs %s",
                game,
                opponent_name,
                domain.players.names[seat],
                " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, payoffs)),
            )
        return MatchResults(opponent_name, played, wins, draws, losses, time_control, wins_on_time, losses_on_time)

    def play_game(
        self,
        domain: Domain,
        evaluated: PolicyFactory,
        opponent: PolicyFactory,
        seat: int,
        policy_seed: int,
        outcome_seed: int,
        time_control: TimeControl | None = None,
    ) -> MatchGame:
        """One game, the evaluated policy in the given seat. The player whose time ran out is the first in the players'
        order where several did at once."""
        created = (evaluated(policy_seed), opponent(policy_seed))
        policies = created if seat == 0 else (created[1], created[0])
        rng = random.Random(outcome_seed)
        state, plies, actions = domain.initial_state, 0, []
        names = domain.players.names
        clocks: list[Clock] = [] if time_control is None else list(self._timekeeper.clocks(domain, time_control))
        steps, flagged, seconds = [0] * len(names), [], []
        while True:
            if self._state_reader.acts_at_once(state, domain.players):
                legal = self._joint_solver.legal(domain.problem, state, domain.players)
                if not legal:
                    break
                choices = []
                for index, _ in legal:
                    seen = self._seen(domain, state, index)
                    if time_control is None:
                        choices.append((names[index], policies[index].choose(domain, seen, names[index])))  # type: ignore[call-arg]
                    else:
                        choices.append(
                            (names[index], self._timed(domain, policies[index], seen, index, clocks, steps, flagged, seconds, names[index]))
                        )
                if flagged:
                    for name in flagged:
                        state = self._timekeeper.flag(domain, state, name)
                    break
                outcomes = self._predictor.predict_joint(domain.transitions, state, JointAction(tuple(choices))).outcomes
            else:
                if not self._solver.solve(domain.problem, state):
                    break
                player = self._state_reader.player_to_act(state, domain.players)
                seen = self._seen(domain, state, player)
                if time_control is None:
                    action = policies[player].choose(domain, seen)
                else:
                    action = self._timed(domain, policies[player], seen, player, clocks, steps, flagged, seconds)
                    if flagged:
                        state = self._timekeeper.flag(domain, state, flagged[0])
                        break
                actions.append(action)
                outcomes = self._predictor.predict(domain.transitions, state, action).outcomes
            plies += 1
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        payoffs = self._state_reader.payoffs(state, domain.players)
        ending = f"{flagged[0]}'s flag" if flagged else self._game_recorder.ending(domain, state)
        clock_text = (
            ""
            if time_control is None
            else f" on {TimeControlTextMapper().to_text(time_control)}, clocks "
            + " ".join(f"{name}={clock.remaining:.2f}" for name, clock in zip(names, clocks))
        )
        logger.info(
            "Game with seeds %d and %d finished in %d plies%s, the evaluated policy playing %s: payoffs %s",
            policy_seed,
            outcome_seed,
            plies,
            ("" if ending is None else f" by {ending}") + clock_text,
            domain.players.names[seat],
            " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, payoffs)),
        )
        record = self._game_recorder.record(domain, actions) if actions else None
        if record is not None:
            logger.info("Game with seeds %d and %d record: %s", policy_seed, outcome_seed, record)
        return MatchGame(
            payoffs,
            flagged[0] if flagged else None,
            plies,
            ending,
            tuple(actions),
            policy_seed,
            outcome_seed,
            tuple(seconds),
            tuple(clocks),
        )

    def _timed(
        self,
        domain: Domain,
        policy: Policy,
        seen: State,
        player: int,
        clocks: list[Clock],
        steps: list[int],
        flagged: list[str],
        seconds: list[float],
        name: str | None = None,
    ) -> Action:
        """The policy's choice, timed and charged to the player's clock, its seconds added to `seconds`; a player whose
        time ran out joins `flagged`."""
        clock, played = clocks[player], steps[player]
        action, spent = self._timekeeper.timed(lambda: policy.choose(domain, seen, name, clock, played))
        clocks[player], steps[player] = clock.after(spent), played + 1
        seconds.append(spent)
        logger.debug(
            "Step %d: %s took %.2f seconds, %.1f left",
            played + 1,
            domain.players.names[player],
            spent,
            clocks[player].remaining,
        )
        if clocks[player].flagged:
            flagged.append(domain.players.names[player])
        return action

    def _seen(self, domain: Domain, state: State, player: int) -> State:
        if domain.observation is None:
            return state
        return self._state_observer.observe(domain.observation, state, domain.players.names[player])
