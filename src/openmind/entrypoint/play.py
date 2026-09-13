import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Plays a domain in the terminal: players pick legal actions by number until none is left."""
    parser = argparse.ArgumentParser(prog="openmind-play", description="Play a domain in the terminal.")
    parser.add_argument("domain", help="domain to play, such as tictactoe")
    parser.add_argument(
        "--log-directory", default="data/log/play", help="where game logs are saved (default: data/log/play)"
    )
    arguments = parser.parse_args(argv)
    domain = create_domain(arguments.domain)

    directory = Path(arguments.log_directory) / domain.name
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s %(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        _play(domain)
    except EOFError:
        print()
        logger.info("Input ended before the game was over")
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _play(domain: Domain) -> None:
    names = VariableNameMapper()
    interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
    solver = Solver(interpreter, expression_text, action_text)
    predictor = Predictor(interpreter, names, expression_text, action_text)
    state_text = StateTextMapper()

    logger.info("Playing %s", domain.name)
    state = domain.initial_state
    while actions := solver.solve(domain.problem, state):
        print(state_text.to_text(state))
        print()
        for number, action in enumerate(actions, start=1):
            print(f"{number}. {action_text.to_text(action)}")
        action = actions[_read_choice(len(actions)) - 1]
        logger.info("Chose %s", action_text.to_text(action))
        outcomes = predictor.predict(domain.transitions, state, action).outcomes
        (state,) = random.choices(
            [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
        )
        print()
    print(state_text.to_text(state))
    logger.info("No legal action left: game over")


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
