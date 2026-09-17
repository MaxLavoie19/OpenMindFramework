import argparse
import logging
import random
from datetime import datetime
from functools import partial
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import DEFAULT_ITERATIONS, DEFAULT_UNFINISHED_PAYOFF, EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.model.domain import Domain
from openmind.agent.service.agent import Agent
from openmind.agent.service.timekeeper import Timekeeper
from openmind.csp.factory.csp_factory import create_solver
from openmind.entrypoint.clock_options import add_clock_options
from openmind.entrypoint.search_options import add_selection_options
from openmind.mcts.constant.mcts_constant import UNIFORM_PRIOR
from openmind.mcts.service.uniform_prior import UniformPrior
from openmind.entrypoint.constant.entrypoint_constant import LOG_FORMAT
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Plays a domain in the terminal: humans pick legal actions by number, the agent searches, until none is legal. On a
    clock, every player's choice is timed, a human's thinking at the prompt included, and a player whose time runs out
    loses the move and the game as the domain's timeout rule says."""
    parser = argparse.ArgumentParser(prog="openmind-play", description="Play a domain in the terminal.")
    parser.add_argument("domain", help="domain to play, such as tictactoe")
    parser.add_argument(
        "--agent", action="append", default=[], metavar="PLAYER", help="a player the agent controls; repeat for several"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"MCTS iterations per agent move (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for the agent's search")
    parser.add_argument(
        "--rollout-limit",
        type=_non_negative,
        default=None,
        help="actions an agent's rollout plays at most before every player gets the unfinished payoff (default: no limit)",
    )
    parser.add_argument(
        "--unfinished-payoff",
        type=float,
        default=DEFAULT_UNFINISHED_PAYOFF,
        help=f"each player's payoff for a rollout stopped at the limit (default: {DEFAULT_UNFINISHED_PAYOFF}, a draw in "
        "games paying 1, 0.5 and 0)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the game log (default: INFO)",
    )
    parser.add_argument(
        "--log-directory", default="data/log/play", help="where game logs are saved (default: data/log/play)"
    )
    add_clock_options(parser)
    add_selection_options(parser)
    arguments = parser.parse_args(argv)
    if arguments.prior != UNIFORM_PRIOR:
        parser.error(f"--prior {arguments.prior} needs rules the agent doesn't have when playing; use uniform")
    domain = create_domain(arguments.domain)
    if arguments.time_control is not None and domain.timeout is None:
        parser.error(f"{domain.name} can't be played on a clock: it has no timeout rule")
    unknown = [player for player in arguments.agent if player not in domain.players.names]
    if unknown:
        parser.error(f"unknown player {', '.join(unknown)}; {domain.name} players: {', '.join(domain.players.names)}")
    unfinished_payoff = None if arguments.rollout_limit is None else arguments.unfinished_payoff
    builder = (
        AgentBuilder()
        .with_iterations(arguments.iterations)
        .with_exploration(EXPLORATION)
        .with_seed(arguments.seed)
        .with_rollout_limit(arguments.rollout_limit, unfinished_payoff)
        .with_selection(arguments.selection, arguments.puct_exploration)
        .with_prior(UniformPrior())
    )
    if arguments.time_control is not None:
        builder.with_time_budget_estimator(
            PlainTimeBudgetEstimator(arguments.expected_steps, arguments.time_control.base_seconds * arguments.time_reserve)
        )
    agent = builder.build() if arguments.agent else None

    directory = Path(arguments.log_directory) / domain.name
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    try:
        _play(domain, frozenset(arguments.agent), agent, arguments.time_control)
    except EOFError:
        print()
        logger.info("Input ended before the game was over")
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _play(
    domain: Domain,
    agent_players: frozenset[str],
    agent: Agent | None,
    time_control: TimeControl | None = None,
    timekeeper: Timekeeper | None = None,
) -> None:
    action_text = ActionTextMapper()
    solver = create_solver()
    predictor = create_predictor()
    state_text, state_reader, state_observer = GridTextMapper(VariableNameMapper()), StateReader(), create_state_observer()
    keeper = Timekeeper(create_rule_caller()) if timekeeper is None else timekeeper
    names = domain.players.names
    clocks: dict[str, Clock] = {} if time_control is None else dict(zip(names, keeper.clocks(domain, time_control), strict=True))
    steps = dict.fromkeys(names, 0)

    logger.info("Playing %s", domain.name)
    state = domain.initial_state
    while actions := solver.solve(domain.problem, state):
        player = str(state_reader.value(state, domain.players.to_act))
        seen = state if domain.observation is None else state_observer.observe(domain.observation, state, player)
        print(state_text.to_text(seen))
        print()
        if clocks:
            print("  ".join(f"{name} {_clock_text(clock)}" for name, clock in clocks.items()))
        if not clocks:
            action = _choose(domain, player, seen, actions, agent_players, agent, None, 0)
        else:
            clock, played = clocks[player], steps[player]
            action, spent = keeper.timed(partial(_choose, domain, player, seen, actions, agent_players, agent, clock, played))
            clocks[player], steps[player] = clock.after(spent), played + 1
            logger.info("%s took %.2f seconds, %.1f left", player, spent, clocks[player].remaining)
            if clocks[player].flagged:
                state = keeper.flag(domain, state, player)
                print(f"{player}'s time ran out")
                print()
                print(state_text.to_text(state))
                logger.info("%s's time ran out: game over", player)
                return
        logger.info("Chose %s", action_text.to_text(action))
        outcomes = predictor.predict(domain.transitions, state, action).outcomes
        (state,) = random.choices(
            [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
        )
        print()
    print(state_text.to_text(state))
    logger.info("No legal action left: game over")


def _choose(
    domain: Domain,
    player: str,
    seen: State,
    actions: tuple[Action, ...],
    agent_players: frozenset[str],
    agent: Agent | None,
    clock: Clock | None,
    steps_played: int,
) -> Action:
    """The agent's choice for its players, given their clock; a human's, picked by number."""
    if agent is not None and player in agent_players:
        action = agent.choose(domain, seen, None, clock, steps_played)
        print(f"{player} chose {ActionTextMapper().to_text(action)}")
        return action
    for number, candidate in enumerate(actions, start=1):
        print(f"{number}. {ActionTextMapper().to_text(candidate)}")
    return actions[_read_choice(len(actions)) - 1]


def _clock_text(clock: Clock) -> str:
    """A clock's time left as minutes and seconds, such as 2:41.3."""
    minutes, seconds = divmod(max(clock.remaining, 0.0), 60.0)
    return f"{int(minutes)}:{seconds:04.1f}"


def _non_negative(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, not {number}")
    return number


def _read_choice(count: int) -> int:
    while True:
        try:
            number = int(input("Action number: "))
        except ValueError:
            number = 0
        if 1 <= number <= count:
            return number
        print(f"Enter a number from 1 to {count}.")


if __name__ == "__main__":
    main()
