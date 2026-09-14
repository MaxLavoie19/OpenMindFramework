import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.factory.domain_factory import create_domain
from openmind.evaluation.constant.evaluation_constant import ALL_POSITIONS
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository
from openmind.training.constant.training_constant import (
    DEFAULT_MARGIN,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    DEFAULT_SELECTION_CONFIDENCE,
    DEFAULT_SELECTION_ITERATIONS,
)
from openmind.training.factory.training_factory import create_rule_selector
from openmind.training.mapper.selection_report_json_mapper import SelectionReportJsonMapper
from openmind.training.mapper.selection_report_text_mapper import SelectionReportTextMapper
from openmind.training.model.selection_settings import SelectionSettings
from openmind.training.repository.selection_report_repository import SelectionReportRepository

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Selects the smallest set of a rule base's rules that plays no worse than all of them, saves it and the report,
    and prints the summary. The report is saved after every decision, so a long selection can be read while it runs."""
    parser = argparse.ArgumentParser(
        prog="openmind-select", description="Select the rules that play no worse than all of them."
    )
    parser.add_argument("domain", help="domain the rules play, such as tictactoe")
    parser.add_argument(
        "--rules", type=Path, required=True, help="candidate rule base, such as one from openmind-distill --explore"
    )
    parser.add_argument(
        "--positions",
        type=_positions,
        default=None,
        help=f"positions rule sets are compared on: a number, or {ALL_POSITIONS} (default: {ALL_POSITIONS})",
    )
    parser.add_argument(
        "--reference-iterations",
        type=_positive,
        default=None,
        help="stand in for perfect play with unguided searches of this many iterations on positions of random games, "
        "for domains exact search can't reach; needs a number of positions (default: exact search)",
    )
    parser.add_argument(
        "--iterations",
        type=_positive,
        default=DEFAULT_SELECTION_ITERATIONS,
        help=f"MCTS iterations of the guided searches compared (default: {DEFAULT_SELECTION_ITERATIONS})",
    )
    parser.add_argument(
        "--rollouts",
        default="guided",
        choices=("guided", "unguided"),
        help="whether the guided searches' rollouts follow the ratings (default: guided)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"largest rise in mean regret a removed rule may cause (default: {DEFAULT_MARGIN})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_SELECTION_CONFIDENCE,
        help=f"confidence of the upper bound on that rise (default: {DEFAULT_SELECTION_CONFIDENCE})",
    )
    parser.add_argument(
        "--resamples",
        type=_positive,
        default=DEFAULT_RESAMPLES,
        help=f"bootstrap resamples per comparison (default: {DEFAULT_RESAMPLES})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="stop trying removals after this many hours, then confirm what is selected (default: no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"worker processes the searches run in (default: {DEFAULT_WORKERS}, half the logical CPUs)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the log (default: INFO)",
    )
    parser.add_argument(
        "--log-directory", default="data/log/select", help="where logs are saved (default: data/log/select)"
    )
    parser.add_argument(
        "--rules-directory", default="data/rbs", help="where the selected rule base is saved (default: data/rbs)"
    )
    parser.add_argument(
        "--report-directory", default="data/selection", help="where reports are saved (default: data/selection)"
    )
    arguments = parser.parse_args(argv)
    domain = create_domain(arguments.domain)
    settings = SelectionSettings(
        arguments.positions,
        arguments.reference_iterations,
        arguments.iterations,
        arguments.rollouts == "guided",
        arguments.margin,
        arguments.confidence,
        arguments.resamples,
        arguments.seed,
        arguments.max_hours,
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
        rule_bases = RuleBaseRepository(RuleBaseJsonMapper())
        candidates = rule_bases.load(arguments.rules)
        reports = SelectionReportRepository(SelectionReportJsonMapper(RuleTextMapper()))
        report_directory = Path(arguments.report_directory)
        logger.info(
            "Selecting from %d rules in %s, searching in %d worker processes",
            len(candidates.rules),
            arguments.rules,
            arguments.workers,
        )
        report = create_rule_selector(arguments.workers).select(
            domain, candidates, str(arguments.rules), settings, lambda progress: reports.save(progress, report_directory)
        )
        rules_path = rule_bases.save(report.selected, Path(arguments.rules_directory), report.created_at)
        report_path = reports.save(report, report_directory)
        print(SelectionReportTextMapper(RuleTextMapper()).to_text(report))
        print(f"Saved selected rules {rules_path}")
        print(f"Saved selection report {report_path}")
        logger.info("Saved selected rules %s", rules_path)
        logger.info("Saved selection report %s", report_path)
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _positive(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"needs at least 1, not {number}")
    return number


def _positions(text: str) -> int | None:
    return None if text == ALL_POSITIONS else _positive(text)


if __name__ == "__main__":
    main()
