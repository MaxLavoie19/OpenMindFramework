import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

from openmind.agent.constant.agent_constant import DEFAULT_ITERATIONS, DEFAULT_UNFINISHED_PAYOFF
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.model.domain import Domain
from openmind.agent.service.agent import Agent
from openmind.csp.factory.csp_factory import create_solver
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Plays a domain in the terminal: humans pick legal actions by number, the agent searches, until none is legal."""
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
    arguments = parser.parse_args(argv)
    domain = create_domain(arguments.domain)
    unknown = [player for player in arguments.agent if player not in domain.players.names]
    if unknown:
        parser.error(f"unknown player {', '.join(unknown)}; {domain.name} players: {', '.join(domain.players.names)}")
    unfinished_payoff = None if arguments.rollout_limit is None else arguments.unfinished_payoff
    agent = (
        create_agent(arguments.iterations, arguments.seed, arguments.rollout_limit, unfinished_payoff)
        if arguments.agent
        else None
    )

    directory = Path(arguments.log_directory) / domain.name
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s %(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    try:
        _play(domain, frozenset(arguments.agent), agent)
    except EOFError:
        print()
        logger.info("Input ended before the game was over")
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _play(domain: Domain, agent_players: frozenset[str], agent: Agent | None) -> None:
    action_text = ActionTextMapper()
    solver = create_solver()
    predictor = create_predictor()
    state_text, state_reader, state_observer = GridTextMapper(VariableNameMapper()), StateReader(), create_state_observer()

    logger.info("Playing %s", domain.name)
    state = domain.initial_state
    while actions := solver.solve(domain.problem, state):
        player = state_reader.value(state, domain.players.to_act)
        seen = state if domain.observation is None else state_observer.observe(domain.observation, state, str(player))
        print(state_text.to_text(seen))
        print()
        if agent is not None and player in agent_players:
            action = agent.choose(domain, seen)
            print(f"{player} chose {action_text.to_text(action)}")
        else:
            for number, candidate in enumerate(actions, start=1):
                print(f"{number}. {action_text.to_text(candidate)}")
            action = actions[_read_choice(len(actions)) - 1]
        logger.info("Chose %s", action_text.to_text(action))
        outcomes = predictor.predict(domain.transitions, state, action).outcomes
        (state,) = random.choices(
            [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
        )
        print()
    print(state_text.to_text(state))
    logger.info("No legal action left: game over")


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
