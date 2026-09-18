import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.doxastic.constant.rule_kind_constant import MOVE, POSITION
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.game_factory import create_game
from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.clock_options import add_clock_options, add_knowledge_option
from openmind.entrypoint.constant.entrypoint_constant import LOG_FORMAT
from openmind.entrypoint.rollout_options import checked_unfinished_payoff
from openmind.entrypoint.search_options import add_selection_options
from openmind.evaluation.constant.evaluation_constant import (
    ALL_POSITIONS,
    DEFAULT_BUDGETS,
    DEFAULT_GAMES,
    DEFAULT_ITERATIONS,
    DEFAULT_POSITIONS,
    DEFAULT_SEED,
)
from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.mapper.report_json_mapper import ReportJsonMapper
from openmind.evaluation.mapper.report_text_mapper import ReportTextMapper
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.repository.report_repository import ReportRepository
from openmind.mcts.constant.mcts_constant import RATER_PRIOR, VALUE_PRIOR
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Measures how well the agent plays a rbs, prints the report and its summary, and saves the report."""
    parser = argparse.ArgumentParser(prog="openmind-evaluate", description="Measure how well the agent plays a rbs.")
    parser.add_argument("domain", help="domain to evaluate, such as tictactoe")
    parser.add_argument(
        "--games", type=int, default=DEFAULT_GAMES, help=f"games per baseline series (default: {DEFAULT_GAMES})"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"MCTS iterations per move in the baseline series (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--positions",
        type=_positions,
        default=DEFAULT_POSITIONS,
        help=f"positions sampled to measure agreement with perfect play: a number, 0 to skip, or {ALL_POSITIONS} "
        f"(default: {DEFAULT_POSITIONS})",
    )
    parser.add_argument(
        "--budgets",
        type=_budgets,
        default=DEFAULT_BUDGETS,
        help=f"comma-separated iteration budgets for agreement (default: {','.join(map(str, DEFAULT_BUDGETS))})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--reference-iterations",
        type=_iterations,
        default=None,
        help="stand in for perfect play with unguided searches of this many iterations on positions of random games, "
        "for domains exact search can't reach (default: exact search)",
    )
    parser.add_argument(
        "--heuristics",
        default=None,
        help="the context whose position and move rules the evaluated agent plays, such as a round of training "
        "(default: none, rollouts play to the end)",
    )
    parser.add_argument(
        "--rollouts",
        default="guided",
        choices=("guided", "unguided"),
        help="whether the guided agent's rollouts follow the rules' ratings, or only its tree's nodes are rated "
        "(default: guided)",
    )
    parser.add_argument(
        "--rollout-actions",
        type=_non_negative,
        default=0,
        help="rollout actions the valuing agent plays before valuing a position (default: 0)",
    )
    parser.add_argument(
        "--rollout-limit",
        type=_non_negative,
        default=None,
        help="actions every agent's rollout plays at most before every player gets the unfinished payoff "
        "(default: no limit)",
    )
    parser.add_argument(
        "--unfinished-payoff",
        type=float,
        default=None,
        help="each player's payoff for a rollout stopped at the limit; needed with --rollout-limit",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"worker processes games and searches run in (default: {DEFAULT_WORKERS}, half the logical CPUs)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the log (default: INFO)",
    )
    parser.add_argument(
        "--log-directory", default="data/log/evaluate", help="where logs are saved (default: data/log/evaluate)"
    )
    parser.add_argument(
        "--report-directory", default="data/evaluation", help="where reports are saved (default: data/evaluation)"
    )
    add_clock_options(parser)
    add_knowledge_option(parser)
    add_selection_options(parser)
    arguments = parser.parse_args(argv)
    if arguments.prior in (RATER_PRIOR, VALUE_PRIOR) and arguments.heuristics is None:
        parser.error(f"--prior {arguments.prior} needs --heuristics")
    knowledge_base = create_knowledge_base(arguments.domain.split("/")[0], arguments.knowledge)
    rbs = create_game(arguments.domain, knowledge_base)
    if arguments.time_control is not None and not rbs.timed():
        parser.error(f"{rbs.context} can't be played on a clock: it has no timeout rule")
    settings = EvaluationSettings(
        arguments.games,
        arguments.iterations,
        arguments.positions,
        arguments.budgets,
        arguments.seed,
        arguments.reference_iterations,
        arguments.rollouts == "guided",
        arguments.rollout_actions,
        arguments.rollout_limit,
        checked_unfinished_payoff(parser, arguments),
        arguments.time_control,
        arguments.expected_steps,
        arguments.time_reserve,
        arguments.selection,
        arguments.puct_exploration,
        arguments.prior,
        arguments.prior_temperature,
    )

    directory = Path(arguments.log_directory) / rbs.context
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    try:
        agent_builder = AgentBuilder().with_exploration(EXPLORATION)
        rules_file = values_file = None
        rater = valuer = None
        if arguments.heuristics is not None:
            heuristics = create_rule_based_system(knowledge_base, arguments.heuristics)
            kinds = {rule.kind for rule in heuristics.rules}
            if MOVE in kinds:
                rater, rules_file = heuristics, arguments.heuristics
                agent_builder.with_guidance(rater)
                logger.info("Evaluating with the move rules of %s", arguments.heuristics)
            if POSITION in kinds:
                valuer, values_file = heuristics, arguments.heuristics
                agent_builder.with_valuation(valuer)
                logger.info("Evaluating with the position rules of %s", arguments.heuristics)
        logger.info("Running games and searches in %d worker processes", arguments.workers)
        report = create_evaluator(arguments.workers, GameMemory(knowledge_base)).evaluate(
            rbs, agent_builder, settings, rules_file, rater, values_file, valuer
        )
        path = ReportRepository(ReportJsonMapper()).save(report, Path(arguments.report_directory))
        logger.info("Saved report %s", path)
        print(path.read_text(encoding="utf-8"), end="")
        print(ReportTextMapper().to_text(report))
        print(f"Saved report {path}")
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _iterations(text: str) -> int:
    try:
        iterations = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if iterations < 1:
        raise argparse.ArgumentTypeError(f"needs at least 1 iteration, not {iterations}")
    return iterations


def _non_negative(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, not {number}")
    return number


def _positions(text: str) -> int | None:
    if text == ALL_POSITIONS:
        return None
    try:
        positions = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or {ALL_POSITIONS!r}, not {text!r}") from None
    if positions < 0:
        raise argparse.ArgumentTypeError(f"positions can't be negative, not {positions}")
    return positions


def _budgets(text: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, not {text!r}") from None
    if min(budgets) < 1:
        raise argparse.ArgumentTypeError(f"every budget needs at least 1 iteration, not {text!r}")
    return budgets


if __name__ == "__main__":
    main()
