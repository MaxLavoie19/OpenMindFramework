import argparse
import logging
import time
from pathlib import Path

from openmind.entrypoint.debug_options import add_debug_option, start_debugging
from openmind.agent.constant.sudoku_constant import NAME as SUDOKU
from openmind.agent.constant.sudoku_constant import SEPARATOR
from openmind.agent.factory.game_factory import create_game
from openmind.agent.factory.sudoku_factory import declare_sudoku
from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository
from openmind.csp.constant.solver_constant import DEFAULT_SOLUTION_LIMIT
from openmind.entrypoint.clock_options import add_knowledge_option
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.mapper.grid_text_mapper import GridTextMapper

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
    add_knowledge_option(parser)
    add_debug_option(parser)
    arguments = parser.parse_args(argv)
    repository = SudokuPuzzleRepository(SudokuCollectionMapper())
    knowledge_base = create_knowledge_base("sudoku", arguments.knowledge)
    runs = [
        _resolve(parser, repository, Path(arguments.puzzle_directory), name, knowledge_base) for name in arguments.domains
    ]

    state_text = GridTextMapper()
    for name, domains, is_collection in runs:
        debugger = start_debugging(
            arguments, f"solve {name}", Path(arguments.log_directory) / name, arguments.log_level, knowledge_base
        )
        try:
            if is_collection:
                logger.info("Solving %s", name)
            solutions = assignments = dead_ends = pruned_values = 0
            total_seconds = 0.0
            for rbs in domains:
                logger.info("Solving %s", rbs.context)
                started = time.perf_counter()
                actions, statistics = rbs.actions_with_statistics(rbs.start(), arguments.limit)
                seconds = time.perf_counter() - started
                if not is_collection:
                    for number, action in enumerate(actions, start=1):
                        outcomes = rbs.outcomes(rbs.start(), action).outcomes
                        for outcome, probability in outcomes:
                            print(f"Solution {number} (probability {probability}):")
                            print(state_text.to_text(outcome))
                limit_reached = " (limit reached)" if len(actions) == arguments.limit else ""
                _report(
                    f"{rbs.context}: {len(actions)} solution(s), {statistics.assignments} assignments, "
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
            debugger.stop()


def _resolve(
    parser: argparse.ArgumentParser,
    repository: SudokuPuzzleRepository,
    directory: Path,
    argument: str,
    knowledge_base: KnowledgeBase,
) -> tuple[str, list[RuleBasedSystem], bool]:
    """A GAME argument's name, the games it names, and whether they are a sudoku collection's puzzles."""
    parts = argument.split(SEPARATOR)
    if parts[0] != SUDOKU or len(parts) == 1:
        rbs = create_game(argument, knowledge_base)
        return rbs.context, [rbs], False
    if len(parts) > 3:
        parser.error(f"{argument!r} is not a sudoku collection (sudoku/<collection>) or puzzle (sudoku/<collection>/<number>)")
    collection = parts[1]
    collections = repository.collections(directory)
    if collection not in collections:
        parser.error(
            f"no sudoku collection {collection!r} in {directory}; collections found: {', '.join(collections) or 'none'}"
        )
    puzzles = repository.load(directory, collection)
    if len(parts) == 2:
        return (
            SEPARATOR.join((SUDOKU, collection)),
            [_puzzle_game(puzzle, knowledge_base) for puzzle in puzzles],
            True,
        )
    number = parts[2]
    if not number.isdecimal() or not 1 <= int(number) <= len(puzzles):
        parser.error(f"{argument!r}: {collection} has puzzles 1 to {len(puzzles)}")
    rbs = _puzzle_game(puzzles[int(number) - 1], knowledge_base)
    return rbs.context, [rbs], False


def _puzzle_game(puzzle: SudokuPuzzle, knowledge_base: KnowledgeBase) -> RuleBasedSystem:
    """The RBS for one puzzle of a collection, its rules declared under its own context."""
    context = declare_sudoku(knowledge_base, SEPARATOR.join((SUDOKU, puzzle.collection, str(puzzle.number))), puzzle.grid)
    return create_rule_based_system(knowledge_base, context)


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
