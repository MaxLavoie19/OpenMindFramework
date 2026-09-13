import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

from openmind.agent.constant.sudoku_constant import NAME as SUDOKU
from openmind.agent.constant.sudoku_constant import SEPARATOR
from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.domain import Domain
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository
from openmind.csp.constant.solver_constant import DEFAULT_SOLUTION_LIMIT
from openmind.csp.factory.csp_factory import create_solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Solves domains' constraint problems from their initial states with the CSP alone. A domain prints each solution's
    outcome and a summary; a sudoku collection prints a summary per puzzle, then its totals."""
    parser = argparse.ArgumentParser(prog="openmind-solve", description="Solve domains' constraint problems with the CSP.")
    parser.add_argument(
        "domains",
        nargs="+",
        metavar="DOMAIN",
        help="domain to solve: sudoku, a sudoku collection such as sudoku/top95, or one of its puzzles such as "
        "sudoku/top95/7",
    )
    parser.add_argument(
        "--limit",
        type=_limit,
        default=DEFAULT_SOLUTION_LIMIT,
        help=f"most solutions to find (default: {DEFAULT_SOLUTION_LIMIT}, enough to tell whether a solution is unique)",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the log (default: DEBUG)",
    )
    parser.add_argument(
        "--log-directory", default="data/log/solve", help="where logs are saved (default: data/log/solve)"
    )
    parser.add_argument(
        "--puzzle-directory",
        default="data/sudoku",
        help="where sudoku collections are read from (default: data/sudoku)",
    )
    arguments = parser.parse_args(argv)
    repository = SudokuPuzzleRepository(SudokuCollectionMapper())
    runs = [_resolve(parser, repository, Path(arguments.puzzle_directory), domain) for domain in arguments.domains]

    solver = create_solver()
    names = VariableNameMapper()
    predictor = Predictor(Interpreter(names), names, ExpressionTextMapper(names), ActionTextMapper())
    state_text = StateTextMapper()
    for name, domains, is_collection in runs:
        directory = Path(arguments.log_directory) / name
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s %(message)s"))
        root = logging.getLogger()
        level = root.level
        root.addHandler(handler)
        root.setLevel(arguments.log_level)
        try:
            if is_collection:
                logger.info("Solving %s", name)
            solutions = assignments = dead_ends = pruned_values = 0
            total_seconds = 0.0
            for domain in domains:
                logger.info("Solving %s", domain.name)
                started = time.perf_counter()
                actions, statistics = solver.solve_with_statistics(domain.problem, domain.initial_state, arguments.limit)
                seconds = time.perf_counter() - started
                if not is_collection:
                    for number, action in enumerate(actions, start=1):
                        outcomes = predictor.predict(domain.transitions, domain.initial_state, action).outcomes
                        for outcome, probability in outcomes:
                            print(f"Solution {number} (probability {probability}):")
                            print(state_text.to_text(outcome))
                limit_reached = " (limit reached)" if len(actions) == arguments.limit else ""
                _report(
                    f"{domain.name}: {len(actions)} solution(s), {statistics.assignments} assignments, "
                    f"{statistics.dead_ends} dead ends, {statistics.pruned_values} values pruned, "
                    f"{seconds:.4f} seconds{limit_reached}"
                )
                solutions += len(actions)
                assignments += statistics.assignments
                dead_ends += statistics.dead_ends
                pruned_values += statistics.pruned_values
                total_seconds += seconds
            if is_collection:
                _report(
                    f"{name}: {len(domains)} puzzle(s), {solutions} solution(s), {assignments} assignments, "
                    f"{dead_ends} dead ends, {pruned_values} values pruned, {total_seconds:.4f} seconds"
                )
        finally:
            root.setLevel(level)
            root.removeHandler(handler)
            handler.close()


def _resolve(
    parser: argparse.ArgumentParser, repository: SudokuPuzzleRepository, directory: Path, argument: str
) -> tuple[str, list[Domain], bool]:
    """A DOMAIN argument's name, its domains, and whether they are a sudoku collection's puzzles."""
    parts = argument.split(SEPARATOR)
    if len(parts) == 1:
        domain = create_domain(argument)
        return domain.name, [domain], False
    if parts[0] != SUDOKU or len(parts) > 3:
        parser.error(
            f"{argument!r} is not a domain, a sudoku collection (sudoku/<collection>) or a sudoku puzzle "
            "(sudoku/<collection>/<number>)"
        )
    collection = parts[1]
    collections = repository.collections(directory)
    if collection not in collections:
        parser.error(
            f"no sudoku collection {collection!r} in {directory}; collections found: {', '.join(collections) or 'none'}"
        )
    puzzles = repository.load(directory, collection)
    if len(parts) == 2:
        return SEPARATOR.join((SUDOKU, collection)), [_puzzle_domain(puzzle) for puzzle in puzzles], True
    number = parts[2]
    if not number.isdecimal() or not 1 <= int(number) <= len(puzzles):
        parser.error(f"{argument!r}: {collection} has puzzles 1 to {len(puzzles)}")
    domain = _puzzle_domain(puzzles[int(number) - 1])
    return domain.name, [domain], False


def _puzzle_domain(puzzle: SudokuPuzzle) -> Domain:
    return create_sudoku_domain(SEPARATOR.join((SUDOKU, puzzle.collection, str(puzzle.number))), puzzle.grid)


def _report(summary: str) -> None:
    logger.info("%s", summary)
    print(summary)


def _limit(text: str) -> int:
    try:
        limit = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if limit < 1:
        raise argparse.ArgumentTypeError(f"the limit needs to be at least 1, not {limit}")
    return limit


if __name__ == "__main__":
    main()
