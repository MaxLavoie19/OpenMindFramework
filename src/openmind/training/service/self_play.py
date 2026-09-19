import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.service.agent import Agent
from openmind.agent.service.timekeeper import Timekeeper
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.training.constant.training_constant import SEED_RANGE
from openmind.training.model.played_game import PlayedGame
from openmind.training.service.arm_selector import ArmSelector
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class SelfPlay:
    """Lets an agent play a domain against itself and keeps each game's searches, positions and payoffs. Every game draws
    two seeds up front, one for its agents and one for its outcomes, so games don't depend on each other: they run in the
    task runner's workers and give the same games whatever the number of workers. Each game is logged by the worker that
    plays it, as soon as it ends.

    Arms, agents each following a signal's rules, can also play each other in a two-player rbs. A game's two arms are
    chosen only when a worker starts it, by UCB on their scores so far, every game finished before counting; so games
    between arms depend on the order games finish, and on the number of workers.

    Given a time control, games are played on a clock: each move is timed around the agent's search and charged to its
    player's clock, and a player whose time runs out doesn't play that move; the domain's timeout rule ends the game."""

    def __init__(
        self,
        state_reader: StateReader,
        task_runner: TaskRunner,
        timekeeper: Timekeeper | None = None,
    ) -> None:
        self._state_reader = state_reader
        self._task_runner = task_runner
        self._timekeeper = Timekeeper() if timekeeper is None else timekeeper

    def play(
        self,
        rbs: RuleBasedGame,
        agent_builder: AgentBuilder,
        games: int,
        rng: random.Random,
        keep_samples: bool = True,
        time_control: TimeControl | None = None,
        on_game: Callable[[int, PlayedGame], None] | None = None,
    ) -> tuple[PlayedGame, ...]:
        """Every game, in order; the builder needs its iterations and exploration set. Without keeping samples, the
        searches' samples are counted but not kept, which is all value training needs. A game that took its worker over
        the memory cap in a fresh worker too is left out. `on_game` is given each game's index and the game in this
        process as soon as it ends, in the order games end; a game left out isn't given."""
        agent_seeds, outcome_seeds = self._seeds(games, rng)

        def arguments_for(index: int) -> tuple[object, ...]:
            return rbs, agent_builder, agent_seeds[index], outcome_seeds[index], keep_samples, time_control

        def on_result(index: int, game: PlayedGame | DroppedCall) -> None:
            if on_game is not None and not isinstance(game, DroppedCall):
                on_game(index, game)

        played = self._task_runner.stream(self.play_game, games, arguments_for, on_result, droppable=True)
        return tuple(game for game in played if not isinstance(game, DroppedCall))

    def play_arms(
        self,
        rbs: RuleBasedGame,
        builders: Mapping[str, AgentBuilder],
        games: int,
        scores: Mapping[str, tuple[int, float]],
        selector: ArmSelector,
        exploration: float,
        rng: random.Random,
        keep_samples: bool = True,
        time_control: TimeControl | None = None,
        on_game: Callable[[int, PlayedGame], None] | None = None,
    ) -> tuple[PlayedGame, ...]:
        """Every game between two arms, in order. `on_game` is given each game's index and the game in this process as
        soon as it ends, before the game's result adds to the scores; a game left out isn't given. `scores` gives each arm's games and points before these games, which
        this round's results add to as they come; a game's first arm plays first in even games and second in odd ones.
        A domain without two players, or fewer than two arms, raises ValueError; a game dropped for memory is left out."""
        if len(rbs.players().names) != 2:
            raise ValueError(f"Games between arms need two players, not {len(rbs.players().names)}")
        arms = list(builders)
        agent_seeds, outcome_seeds = self._seeds(games, rng)
        live = {arm: scores.get(arm, (0, 0.0)) for arm in arms}
        pending = dict.fromkeys(arms, 0)
        pairs: dict[int, tuple[str, str]] = {}

        def arguments_for(index: int) -> tuple[object, ...]:
            first, second = selector.pair(live, pending, arms, exploration, rng)
            seated = (first, second) if index % 2 == 0 else (second, first)
            pairs[index] = seated
            for arm in seated:
                pending[arm] += 1
            return rbs, tuple(builders[arm] for arm in seated), seated, agent_seeds[index], outcome_seeds[index], keep_samples, time_control

        def on_result(index: int, game: PlayedGame | DroppedCall) -> None:
            seated = pairs[index]
            for arm in seated:
                pending[arm] -= 1
            if isinstance(game, DroppedCall):
                return
            if on_game is not None:
                on_game(index, game)
            for arm, points in zip(seated, self._points(game.payoffs), strict=True):
                played, total = live[arm]
                live[arm] = (played + 1, total + points)
            logger.info(
                "Arms game %d: %s against %s, payoffs %s; %s",
                index + 1,
                seated[0],
                seated[1],
                " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, game.payoffs)),
                ", ".join(f"{arm} scores {live[arm][1] / live[arm][0]:.3f} over {live[arm][0]} games" for arm in seated),
            )

        played = self._task_runner.stream(self.play_arm_game, games, arguments_for, on_result, droppable=True)
        return tuple(game for game in played if not isinstance(game, DroppedCall))

    def play_game(
        self,
        rbs: RuleBasedGame,
        agent_builder: AgentBuilder,
        agent_seed: int,
        outcome_seed: int,
        keep_samples: bool = True,
        time_control: TimeControl | None = None,
    ) -> PlayedGame:
        """One game: the samples of its searches when kept, its positions with the search's mean payoff in each, and the
        final payoffs."""
        agent = agent_builder.with_seed(agent_seed).build()
        game, sampled, last = self._game(rbs, (agent,) * len(rbs.players().names), outcome_seed, keep_samples, time_control)
        logger.info(
            "Self-play game with seeds %d and %d finished in %d plies%s: %d samples, payoffs %s",
            agent_seed,
            outcome_seed,
            len(game.states),
            self._ending(rbs, last, game),
            sampled,
            " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, game.payoffs)),
        )
        self._log_record(rbs, game, f"Self-play game with seeds {agent_seed} and {outcome_seed}")
        return replace(game, agent_seed=agent_seed, outcome_seed=outcome_seed, ending=self._ending_of(rbs, last, game))

    def play_arm_game(
        self,
        rbs: RuleBasedGame,
        builders: Sequence[AgentBuilder],
        arms: tuple[str, ...],
        agent_seed: int,
        outcome_seed: int,
        keep_samples: bool = True,
        time_control: TimeControl | None = None,
    ) -> PlayedGame:
        """One game where each player's own agent, built from its builder, searches the player's moves."""
        agents = tuple(builder.with_seed(agent_seed).build() for builder in builders)
        game, sampled, last = self._game(rbs, agents, outcome_seed, keep_samples, time_control)
        logger.info(
            "Self-play game with seeds %d and %d, %s, finished in %d plies%s: %d samples, payoffs %s",
            agent_seed,
            outcome_seed,
            " against ".join(arms),
            len(game.states),
            self._ending(rbs, last, game),
            sampled,
            " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, game.payoffs)),
        )
        self._log_record(rbs, game, f"Self-play game with seeds {agent_seed} and {outcome_seed}")
        return replace(
            game, arms=arms, agent_seed=agent_seed, outcome_seed=outcome_seed, ending=self._ending_of(rbs, last, game)
        )

    def _game(
        self,
        rbs: RuleBasedGame,
        agents: Sequence[Agent],
        outcome_seed: int,
        keep_samples: bool,
        time_control: TimeControl | None = None,
    ) -> tuple[PlayedGame, int, State]:
        """The game each player's agent plays, how many samples its searches made, and its last state."""
        rng = random.Random(outcome_seed)
        state, samples, sampled, states, search_values, actions = rbs.start(), [], 0, [], [], []
        names = rbs.players().names
        clocks: list[Clock] = [] if time_control is None else list(self._timekeeper.clocks(rbs, time_control))
        steps, seconds, budgets, flagged = [0] * len(names), [], [], None
        while rbs.actions(state):
            player = rbs.players().names.index(rbs.acting_player(state))
            agent = agents[player]
            if time_control is None:
                result = agent.search(rbs, state)
            else:
                clock, played = clocks[player], steps[player]
                result, spent = self._timekeeper.timed(lambda: agent.search(rbs, state, None, clock, played))
                clocks[player], steps[player] = (clock if clock.flagged else clock.after(spent)), played + 1
                seconds.append(spent)
                budgets.append(result.budget)
                logger.debug(
                    "Step %d: %s took %.2f seconds of a %s second budget, %.1f left",
                    len(seconds),
                    names[player],
                    spent,
                    "no" if result.budget is None else f"{result.budget:.2f}",
                    clocks[player].remaining,
                )
                if clocks[player].flagged and not clock.flagged:
                    flagged = names[player]
                    logger.info("%s's time ran out; what that does is the game's own rule", flagged)
            sampled += len(result.samples)
            if keep_samples:
                samples.extend(result.samples)
            visits = sum(item.visits for item in result.statistics)
            states.append(state)
            actions.append(result.chosen)
            search_values.append(math.fsum(item.visits * item.mean_payoff for item in result.statistics) / visits)
            outcomes = rbs.outcomes(state, result.chosen).outcomes
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        payoffs = self._state_reader.payoffs(state, rbs.players())
        game = PlayedGame(
            tuple(samples),
            tuple(states),
            tuple(search_values),
            payoffs,
            (),
            tuple(actions),
            time_control,
            tuple(seconds),
            tuple(budgets),
            tuple(clocks),
            flagged,
        )
        return game, sampled, state

    def _ending_of(self, rbs: RuleBasedGame, state: State, game: PlayedGame) -> str | None:
        """Why the game ended: the player whose time ran out, or what the domain says."""
        return f"{game.flagged}'s flag" if game.flagged is not None else rbs.ended(state)

    def _ending(self, rbs: RuleBasedGame, state: State, game: PlayedGame) -> str:
        """` by <why the game ended>` when the domain says or a player's time ran out, nothing otherwise; on a clock,
        then ` on <time control>, clocks <player>=<seconds left> ...`."""
        ending = self._ending_of(rbs, state, game)
        text = "" if ending is None else f" by {ending}"
        if game.time_control is None:
            return text
        clocks = " ".join(f"{name}={clock.remaining:.2f}" for name, clock in zip(rbs.players().names, game.clocks))
        return f"{text} on {TimeControlTextMapper().to_text(game.time_control)}, clocks {clocks}"

    def records(self, rbs: RuleBasedGame, games: Sequence[PlayedGame]) -> tuple[str, ...]:
        """Each game's record, in the games' order; a game the domain doesn't record, or whose record rule gives nothing,
        is left out."""
        found = (rbs.record(game.actions, game.payoffs) for game in games)
        return tuple(record for record in found if record is not None)

    def _log_record(self, rbs: RuleBasedGame, game: PlayedGame, game_name: str) -> None:
        record = rbs.record(game.actions, game.payoffs)
        if record is not None:
            logger.info("%s record: %s", game_name, record)

    def _seeds(self, games: int, rng: random.Random) -> tuple[list[int], list[int]]:
        agent_seeds, outcome_seeds = [], []
        for _ in range(games):
            agent_seeds.append(rng.randrange(SEED_RANGE))
            outcome_seeds.append(rng.randrange(SEED_RANGE))
        return agent_seeds, outcome_seeds

    def _points(self, payoffs: Sequence[float]) -> tuple[float, float]:
        """Each player's points for the game: 1 for a win, 0.5 for a draw, 0 for a loss."""
        if payoffs[0] == payoffs[1]:
            return 0.5, 0.5
        return (1.0, 0.0) if payoffs[0] > payoffs[1] else (0.0, 1.0)
