import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.factory.game_factory import create_game
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.clock_options import add_clock_options, add_knowledge_option
from openmind.entrypoint.constant.entrypoint_constant import LOG_FORMAT
from openmind.entrypoint.rollout_options import add_rollout_limit_options, checked_unfinished_payoff
from openmind.entrypoint.search_options import add_selection_options
from openmind.inference.constant.inference_constant import (
    DEFAULT_DEDUCTION_SECONDS,
    DEFAULT_HIGHEST_PAYOFF,
    DEFAULT_SEARCH_MEMORY,
)
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.mcts.constant.mcts_constant import RATER_PRIOR
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.training.constant.continuous_constant import DEFAULT_ARM_EXPLORATION
from openmind.training.constant.training_constant import (
    DEFAULT_SEED,
    DEFAULT_TRAINING_ITERATIONS,
    DEFAULT_TRAINING_ROLLOUT_ACTIONS,
)
from openmind.training.factory.training_factory import create_continuous_trainer
from openmind.training.mapper.arm_library_json_mapper import ArmLibraryJsonMapper
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.repository.arm_library_repository import ArmLibraryRepository

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Plays games continuously between arms, one after another, remembering every game and every position a decisive
    game's walk back proved."""
    parser = argparse.ArgumentParser(
        prog="openmind-train-values", description="Play games continuously between arms, remembering every game."
    )
    parser.add_argument("domain", help="domain to play, such as chess")
    parser.add_argument(
        "--games", type=_positive, default=None, help="games to play, then stop (default: until stopped)"
    )
    options: list[tuple[str, type, object, str]] = [
        ("--iterations", int, DEFAULT_TRAINING_ITERATIONS, "MCTS iterations per move without a clock; on a clock each move's budget replaces them"),
        ("--seed", int, DEFAULT_SEED, "random seed"),
        ("--memory", float, DEFAULT_SEARCH_MEMORY / 1024**3, "GB this process holds before clearing its caches, shared evenly between the workers unless --worker-memory says otherwise"),
        ("--workers", int, DEFAULT_WORKERS, "worker processes games are played and studied in"),
    ]
    for flag, kind, default, meaning in options:
        parser.add_argument(flag, type=kind, default=default, help=f"{meaning} (default: {default})")
    parser.add_argument(
        "--rollout-actions",
        type=_non_negative,
        default=DEFAULT_TRAINING_ROLLOUT_ACTIONS,
        help=f"rollout actions played before a position is valued with value rules (default: {DEFAULT_TRAINING_ROLLOUT_ACTIONS})",
    )
    add_rollout_limit_options(parser)
    _add_deduction_options(parser)
    parser.add_argument(
        "--ponder-endings",
        type=_non_negative,
        default=0,
        help="positions of each decisive game deduced at most, walking back from its end until a position isn't proven; "
        "needs --deduction-plies (default: 0)",
    )
    _add_worker_memory_option(parser)
    parser.add_argument(
        "--arm-exploration",
        type=float,
        default=DEFAULT_ARM_EXPLORATION,
        help=f"UCB1's exploration weight when choosing which arms play each other (default: {DEFAULT_ARM_EXPLORATION})",
    )
    parser.add_argument(
        "--arm-library",
        type=Path,
        default=None,
        help="the value bases the arms play with (default: none, games played without value rules)",
    )
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
    add_clock_options(parser, "--training-time-control")
    add_knowledge_option(parser)
    add_selection_options(parser)
    arguments = parser.parse_args(argv)
    if arguments.prior == RATER_PRIOR:
        parser.error("--prior rater needs rules that rate actions, which training doesn't have; use value or uniform")
    if arguments.arm_exploration < 0.0:
        parser.error(f"--arm-exploration needs 0 or more, not {arguments.arm_exploration}")
    unfinished_payoff = checked_unfinished_payoff(parser, arguments)
    knowledge_base = create_knowledge_base(arguments.domain.split("/")[0], arguments.knowledge)
    rbs = create_game(arguments.domain, knowledge_base)
    if arguments.training_time_control is not None and not rbs.timed():
        parser.error(f"{rbs.context} can't be played on a clock: it has no timeout rule")
    library = None if arguments.arm_library is None else ArmLibraryRepository(ArmLibraryJsonMapper()).load(arguments.arm_library)
    if library is not None and library.domain != rbs.context:
        parser.error(f"--arm-library holds arms for {library.domain}, not {rbs.context}")
    deduction = _deduction_settings(parser, arguments)
    if arguments.ponder_endings and deduction is None:
        parser.error("--ponder-endings needs --deduction-plies")
    settings = ContinuousTrainingSettings(
        arguments.games,
        arguments.iterations,
        arguments.seed,
        arguments.arm_exploration,
        arguments.rollout_actions,
        arguments.rollout_limit,
        unfinished_payoff,
        deduction,
        arguments.ponder_endings,
        arguments.training_time_control,
        arguments.expected_steps,
        arguments.time_reserve,
        arguments.selection,
        arguments.puct_exploration,
        arguments.prior,
        arguments.prior_temperature,
    )

    directory = Path(arguments.log_directory) / rbs.context
    memory_cap = _memory_cap(parser, arguments, directory)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    try:
        logger.info(
            "Playing in %d worker processes, each holding at most %d bytes; memory diagnoses in %s; arms from %s",
            arguments.workers,
            memory_cap.worker_bytes,
            memory_cap.diagnosis_directory,
            arguments.arm_library or "no arm library, games played without position rules",
        )
        create_continuous_trainer(knowledge_base, arguments.workers, memory_cap).train(rbs, library, settings)
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _add_deduction_options(parser: argparse.ArgumentParser) -> None:
    """The options of the deduction agents fall back on, shared with openmind-distill-values."""
    parser.add_argument(
        "--deduction-plies",
        type=_non_negative,
        default=0,
        help="actions ahead a deduction of one position looks at most, when an agent's rules have no clue there, and when "
        "walking a game back in openmind-train-values; 0 never deduces (default: 0)",
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


def _deduction_settings(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> DeductionBudget | None:
    if arguments.deduction_plies and arguments.deduction_seconds <= 0.0:
        parser.error(f"--deduction-seconds needs more than 0, not {arguments.deduction_seconds}")
    if not arguments.deduction_plies:
        return None
    return DeductionBudget(arguments.deduction_plies, arguments.deduction_seconds, arguments.highest_payoff)


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


if __name__ == "__main__":
    main()
