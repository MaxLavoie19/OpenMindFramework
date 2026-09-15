import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.constant.agent_constant import DEFAULT_UNFINISHED_PAYOFF
from openmind.agent.factory.domain_factory import create_domain
from openmind.inference.constant.inference_constant import (
    DEFAULT_DEDUCTION_SECONDS,
    DEFAULT_HIGHEST_PAYOFF,
    DEFAULT_SEARCH_MEMORY,
    DEFAULT_SEARCH_SECONDS,
)
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.rbs.constant.explanation_constant import DEFAULT_EXPLANATIONS_DIRECTORY
from openmind.rbs.constant.value_constant import DEFAULT_MAX_STEPS, DEFAULT_PRICES, DEFAULT_TOLERANCE
from openmind.rbs.factory.rbs_factory import create_rule_explainer
from openmind.rbs.mapper.value_base_explanation_mapper import ValueBaseExplanationMapper
from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.repository.value_base_repository import ValueBaseRepository
from openmind.rbs.service.ollama_language_model import OllamaLanguageModel
from openmind.training.constant.training_constant import (
    DEFAULT_EVALUATION_GAMES,
    DEFAULT_SEED,
    DEFAULT_TRAINING_GAMES,
    DEFAULT_TRAINING_HELD_OUT_GAMES,
    DEFAULT_TRAINING_ITERATIONS,
    DEFAULT_TRAINING_ROLLOUT_ACTIONS,
    DEFAULT_TRAINING_ROUNDS,
    SEARCH_TARGET,
    VALUE_TARGETS,
)
from openmind.training.factory.training_factory import create_value_training_loop
from openmind.training.mapper.training_report_json_mapper import TrainingReportJsonMapper
from openmind.training.mapper.training_report_text_mapper import TrainingReportTextMapper
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.model.value_training_settings import ValueTrainingSettings
from openmind.training.repository.training_report_repository import TrainingReportRepository

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Trains value rules round after round from self-play, saves every round's value base and the report as each round
    ends, and prints the rounds."""
    parser = argparse.ArgumentParser(
        prog="openmind-train-values", description="Train value rules round after round from self-play."
    )
    parser.add_argument("domain", help="domain to train value rules for, such as tictactoe")
    options: list[tuple[str, type, object, str]] = [
        ("--games", int, DEFAULT_TRAINING_GAMES, "self-play games per round to fit value rules on"),
        ("--held-out-games", int, DEFAULT_TRAINING_HELD_OUT_GAMES, "self-play games per round to choose a fit and measure it on"),
        ("--iterations", int, DEFAULT_TRAINING_ITERATIONS, "MCTS iterations per move, in self-play and in each round's games"),
        ("--seed", int, DEFAULT_SEED, "random seed; round k uses the seed plus k"),
        ("--seconds", float, DEFAULT_SEARCH_SECONDS, "seconds the expression search runs per round"),
        ("--memory", float, DEFAULT_SEARCH_MEMORY / 1024**3, "GB the expression search's process holds at most, workers each holding an even share"),
        ("--max-steps", int, DEFAULT_MAX_STEPS, "steps a fit takes at most"),
        ("--tolerance", float, DEFAULT_TOLERANCE, "weight change below which a fit has settled"),
        ("--workers", int, DEFAULT_WORKERS, "worker processes self-play games, term evaluations and games run in"),
    ]
    for flag, kind, default, meaning in options:
        parser.add_argument(flag, type=kind, default=default, help=f"{meaning} (default: {default})")
    parser.add_argument(
        "--rounds",
        type=_positive,
        default=DEFAULT_TRAINING_ROUNDS,
        help=f"rounds of self-play and fitting (default: {DEFAULT_TRAINING_ROUNDS})",
    )
    parser.add_argument(
        "--start",
        type=Path,
        default=None,
        help="value rules round 1's self-play values positions with (default: none, plain MCTS self-play)",
    )
    parser.add_argument(
        "--target",
        default=SEARCH_TARGET,
        choices=VALUE_TARGETS,
        help=f"what a position is valued at: the game's final payoffs (outcome) or the search's mean payoff (search) "
        f"(default: {SEARCH_TARGET})",
    )
    parser.add_argument(
        "--prices",
        type=_prices,
        default=DEFAULT_PRICES,
        help=f"comma-separated L1 prices swept (default: {','.join(map(str, DEFAULT_PRICES))})",
    )
    parser.add_argument(
        "--candidates",
        type=_non_negative,
        default=None,
        help="candidates the expression search tries at most per round (default: no limit)",
    )
    parser.add_argument(
        "--rollout-actions",
        type=_non_negative,
        default=DEFAULT_TRAINING_ROLLOUT_ACTIONS,
        help="rollout actions played before a position is valued with value rules "
        f"(default: {DEFAULT_TRAINING_ROLLOUT_ACTIONS})",
    )
    parser.add_argument(
        "--rollout-limit",
        type=_non_negative,
        default=None,
        help="actions a rollout plays at most before every player gets the unfinished payoff, for every agent "
        "(default: no limit)",
    )
    parser.add_argument(
        "--unfinished-payoff",
        type=float,
        default=DEFAULT_UNFINISHED_PAYOFF,
        help=f"each player's payoff for a rollout stopped at the limit (default: {DEFAULT_UNFINISHED_PAYOFF})",
    )
    parser.add_argument(
        "--evaluation-games",
        type=_non_negative,
        default=DEFAULT_EVALUATION_GAMES,
        help="games against each opponent after every round; 0 plays none "
        f"(default: {DEFAULT_EVALUATION_GAMES})",
    )
    _add_deduction_options(parser)
    _add_worker_memory_option(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="lowest level saved in the log (default: INFO)",
    )
    parser.add_argument(
        "--log-directory",
        default="data/log/train-values",
        help="where logs are saved (default: data/log/train-values)",
    )
    parser.add_argument(
        "--values-directory", default="data/values", help="where each round's value base is saved (default: data/values)"
    )
    parser.add_argument(
        "--report-directory", default="data/training", help="where training reports are saved (default: data/training)"
    )
    parser.add_argument(
        "--explainer-url",
        default=None,
        help="an Ollama server that explains each round's rules in sentences, such as http://127.0.0.1:11434 "
        "(default: none, literal readings only)",
    )
    parser.add_argument(
        "--explainer-model", default=None, help="the Ollama model explaining the rules, such as qwen3:8b (default: none)"
    )
    parser.add_argument(
        "--explanations-directory",
        default=DEFAULT_EXPLANATIONS_DIRECTORY,
        help=f"where the model's sentences are cached (default: {DEFAULT_EXPLANATIONS_DIRECTORY})",
    )
    arguments = parser.parse_args(argv)
    if (arguments.explainer_url is None) != (arguments.explainer_model is None):
        parser.error("--explainer-url and --explainer-model go together")
    domain = create_domain(arguments.domain)
    value_bases = ValueBaseRepository(ValueBaseJsonMapper())
    start = None if arguments.start is None else value_bases.load(arguments.start)
    if start is not None and start.domain != domain.name:
        parser.error(f"--start holds value rules for {start.domain}, not {domain.name}")
    values = ValueSettings(
        arguments.prices,
        arguments.max_steps,
        arguments.tolerance,
        arguments.seconds,
        int(arguments.memory * 1024**3),
        arguments.candidates,
    )
    deduction, pondering = _deduction_settings(parser, arguments)
    settings = ValueTrainingSettings(
        arguments.rounds,
        ValueDistillationSettings(
            arguments.games,
            arguments.held_out_games,
            arguments.iterations,
            arguments.seed,
            arguments.target,
            values,
            pondering,
        ),
        arguments.rollout_actions,
        arguments.rollout_limit,
        None if arguments.rollout_limit is None else arguments.unfinished_payoff,
        arguments.evaluation_games,
        None if arguments.start is None else str(arguments.start),
        deduction,
    )

    directory = Path(arguments.log_directory) / domain.name
    memory_cap = _memory_cap(parser, arguments, directory)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s %(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    try:
        reports = TrainingReportRepository(TrainingReportJsonMapper())
        explainer, explanation_mapper = create_rule_explainer(), ValueBaseExplanationMapper()
        language_model = (
            None
            if arguments.explainer_url is None
            else OllamaLanguageModel(arguments.explainer_url, arguments.explainer_model)
        )
        round_files: dict[int, Path] = {}
        report_files: list[Path] = []

        def save(report: TrainingReport) -> None:
            for item in report.rounds:
                if item.number not in round_files:
                    run = Path(arguments.values_directory) / report.domain / f"{report.created_at:%Y-%m-%d_%H-%M-%S}"
                    round_files[item.number] = value_bases.write(item.value_base, run / f"round-{item.number}.json")
                    logger.info("Saved round %d values %s", item.number, round_files[item.number])
                    explanations = explainer.explain(
                        item.value_base, domain, language_model, Path(arguments.explanations_directory)
                    )
                    exported = value_bases.export(
                        explanation_mapper.to_markdown(item.value_base, explanations), run / f"round-{item.number}.md"
                    )
                    logger.info("Exported round %d rules %s", item.number, exported)
            report_files.append(reports.save(report, Path(arguments.report_directory)))
            logger.info("Saved training report %s", report_files[-1])

        logger.info(
            "Training in %d worker processes, each holding at most %d bytes; memory diagnoses in %s",
            arguments.workers,
            memory_cap.worker_bytes,
            memory_cap.diagnosis_directory,
        )
        report = create_value_training_loop(arguments.workers, memory_cap).train(domain, start, settings, save)
        print(TrainingReportTextMapper().to_text(report))
        for number, path in round_files.items():
            print(f"Saved round {number} values {path}")
        print(f"Saved training report {report_files[-1]}")
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _add_deduction_options(parser: argparse.ArgumentParser) -> None:
    """The options of the deduction agents fall back on and of pondering, shared with openmind-distill-values."""
    parser.add_argument(
        "--deduction-plies",
        type=_non_negative,
        default=0,
        help="actions ahead a deduction of one position looks at most, when an agent's rules have no clue there and when "
        "pondering; 0 never deduces (default: 0)",
    )
    parser.add_argument(
        "--deduction-seconds",
        type=float,
        default=DEFAULT_DEDUCTION_SECONDS,
        help=f"seconds a deduction of one position runs at most (default: {DEFAULT_DEDUCTION_SECONDS})",
    )
    parser.add_argument(
        "--highest-payoff",
        type=float,
        default=DEFAULT_HIGHEST_PAYOFF,
        help=f"the highest payoff a player can get, which a deduction needs no comparison past (default: "
        f"{DEFAULT_HIGHEST_PAYOFF})",
    )
    parser.add_argument(
        "--ponder-positions",
        type=_non_negative,
        default=0,
        help="positions the rules missed most, deduced before fitting; needs --deduction-plies (default: 0)",
    )


def _add_worker_memory_option(parser: argparse.ArgumentParser) -> None:
    """The option capping each worker's memory, shared with openmind-distill-values."""
    parser.add_argument(
        "--worker-memory",
        type=float,
        default=None,
        help="GB each worker process holds at most: over it a worker clears its caches, and a worker that stays over is "
        "ended, its call run again in a fresh worker, and a call over it twice dropped or its search stopped (default: "
        "the expression search's memory shared between the workers)",
    )


def _memory_cap(parser: argparse.ArgumentParser, arguments: argparse.Namespace, log_directory: Path) -> MemoryCap:
    """The workers' memory cap, diagnoses written under the log directory; this process's caches are cleared above the
    expression search's memory."""
    if arguments.memory <= 0.0:
        parser.error(f"--memory needs more than 0, not {arguments.memory}")
    if arguments.worker_memory is not None and arguments.worker_memory <= 0.0:
        parser.error(f"--worker-memory needs more than 0, not {arguments.worker_memory}")
    gigabytes = arguments.memory / max(1, arguments.workers) if arguments.worker_memory is None else arguments.worker_memory
    process_memory_guard().limit(int(arguments.memory * 1024**3))
    return MemoryCap(int(gigabytes * 1024**3), log_directory / "memory")


def _deduction_settings(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> tuple[DeductionBudget | None, PonderingSettings | None]:
    if arguments.ponder_positions and not arguments.deduction_plies:
        parser.error("--ponder-positions needs --deduction-plies")
    if arguments.deduction_plies and arguments.deduction_seconds <= 0.0:
        parser.error(f"--deduction-seconds needs more than 0, not {arguments.deduction_seconds}")
    if not arguments.deduction_plies:
        return None, None
    budget = DeductionBudget(arguments.deduction_plies, arguments.deduction_seconds, arguments.highest_payoff)
    return budget, PonderingSettings(arguments.ponder_positions, budget) if arguments.ponder_positions else None


def _positive(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"needs at least 1, not {number}")
    return number


def _non_negative(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, not {number}")
    return number


def _prices(text: str) -> tuple[float, ...]:
    try:
        prices = tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, not {text!r}") from None
    if min(prices) < 0.0:
        raise argparse.ArgumentTypeError(f"prices can't be negative, not {text!r}")
    return prices


if __name__ == "__main__":
    main()
