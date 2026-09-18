import logging
import random
from collections.abc import Callable

from openmind.agent.model.policy import Policy
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.agent.service.timekeeper import Timekeeper
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.model.match_results import MatchResults
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class MatchRunner:
    """Plays series of games between two policies in a two-player rbs. Every game draws two seeds up front: one its
    policies are created from, one for its outcomes. Games don't depend on each other, so they run in the task
    runner's workers and give the same results whatever the number of workers. A policy
    is given only what its player sees. Where players act at once, every player to act chooses, given its player's name,
    and the actions are taken together. Each game is logged by the worker that plays it, as soon as it ends.

    Given a time control, games are played on a clock: each choice is timed and charged to its player's clock, given to
    the policy with the steps that player has played; where players act at once, each choice is charged to its own
    player's clock. A player whose time runs out doesn't play that move, and the domain's timeout rule ends the game,
    applied for every player whose time ran out, in the players' order."""

    def __init__(
        self,
        state_reader: StateReader,
        task_runner: TaskRunner,
        timekeeper: Timekeeper | None = None,
    ) -> None:
        self._state_reader = state_reader
        self._task_runner = task_runner
        self._timekeeper = Timekeeper(create_rule_caller()) if timekeeper is None else timekeeper

    def series(
        self,
        rbs: RuleBasedSystem,
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
        if len(rbs.players().names) != 2:
            raise ValueError(f"A series needs two players, not {len(rbs.players().names)}")
        seats = [(game - 1) % 2 for game in range(1, games + 1)]
        policy_seeds, outcome_seeds = [], []
        for _ in range(games):
            policy_seeds.append(rng.getrandbits(32))
            outcome_seeds.append(rng.getrandbits(32))
        def arguments_for(index: int) -> tuple[object, ...]:
            return rbs, evaluated, opponent, seats[index], policy_seeds[index], outcome_seeds[index], time_control

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
                rbs.players().names[seat],
                " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, payoffs)),
            )
        return MatchResults(opponent_name, played, wins, draws, losses, time_control, wins_on_time, losses_on_time)

    def play_game(
        self,
        rbs: RuleBasedSystem,
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
        state, plies, actions = rbs.start(), 0, []
        names = rbs.players().names
        clocks: list[Clock] = [] if time_control is None else list(self._timekeeper.clocks(rbs, time_control))
        steps, flagged, seconds = [0] * len(names), [], []
        while True:
            if self._state_reader.acts_at_once(state, rbs.players()):
                legal = rbs.joint_actions(state)
                if not legal:
                    break
                choices = []
                for index, _ in legal:
                    seen = state
                    if time_control is None:
                        choices.append((names[index], policies[index].choose(rbs, seen, names[index])))  # type: ignore[call-arg]
                    else:
                        choices.append(
                            (names[index], self._timed(rbs, policies[index], seen, index, clocks, steps, flagged, seconds, names[index]))
                        )
                if flagged:
                    for name in flagged:
                        state = self._timekeeper.flag(rbs, state, name)
                    break
                outcomes = rbs.joint_outcomes(state, JointAction(tuple(choices))).outcomes
            else:
                if not rbs.actions(state):
                    break
                player = self._state_reader.player_to_act(state, rbs.players())
                seen = state
                if time_control is None:
                    action = policies[player].choose(rbs, seen)
                else:
                    action = self._timed(rbs, policies[player], seen, player, clocks, steps, flagged, seconds)
                    if flagged:
                        state = self._timekeeper.flag(rbs, state, flagged[0])
                        break
                actions.append(action)
                outcomes = rbs.outcomes(state, action).outcomes
            plies += 1
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        payoffs = self._state_reader.payoffs(state, rbs.players())
        ending = f"{flagged[0]}'s flag" if flagged else rbs.ended(state)
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
            rbs.players().names[seat],
            " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, payoffs)),
        )
        record = rbs.record(actions, flagged[0] if flagged else None, payoffs) if actions else None
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
        rbs: RuleBasedSystem,
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
        action, spent = self._timekeeper.timed(lambda: policy.choose(rbs, seen, name, clock, played))
        clocks[player], steps[player] = clock.after(spent), played + 1
        seconds.append(spent)
        logger.debug(
            "Step %d: %s took %.2f seconds, %.1f left",
            played + 1,
            rbs.players().names[player],
            spent,
            clocks[player].remaining,
        )
        if clocks[player].flagged:
            flagged.append(rbs.players().names[player])
        return action
