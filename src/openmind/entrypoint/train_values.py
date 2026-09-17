import argparse
import logging
from datetime import datetime
from pathlib import Path

from openmind.agent.constant.agent_constant import DEFAULT_UNFINISHED_PAYOFF
from openmind.agent.factory.domain_factory import create_domain
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.clock_options import add_clock_options, add_knowledge_option
from openmind.entrypoint.constant.entrypoint_constant import LOG_FORMAT
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
from openmind.rbs.constant.value_constant import DEFAULT_MAX_STEPS, DEFAULT_PRICES, DEFAULT_TOLERANCE
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.constant.continuous_constant import DEFAULT_LEARNING_RATE, DEFAULT_RULE_SEARCH_SECONDS
from openmind.training.constant.signal_constant import (
    DEFAULT_ARM_EXPLORATION,
    DEFAULT_ARMS,
    DEFAULT_GOAL_LIMIT,
    DEFAULT_SIGNAL_HORIZON,
)
from openmind.training.constant.training_constant import (
    DEFAULT_SEED,
    DEFAULT_TRAINING_ITERATIONS,
    DEFAULT_TRAINING_ROLLOUT_ACTIONS,
)
from openmind.training.factory.training_factory import create_continuous_trainer
from openmind.training.mapper.signal_library_json_mapper import SignalLibraryJsonMapper
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.repository.signal_library_repository import SignalLibraryRepository

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Trains value rules continuously: games between arms one after another, learning from each game as it ends, a rule
    search after each decisive game, and the signal library saved after every game."""
    parser = argparse.ArgumentParser(
        prog="openmind-train-values", description="Train value rules continuously, learning from every game as it ends."
    )
    parser.add_argument("domain", help="domain to train value rules for, such as chess")
    parser.add_argument(
        "--games", type=_positive, default=None, help="games to play, then stop (default: until stopped)"
    )
    options: list[tuple[str, type, object, str]] = [
        ("--iterations", int, DEFAULT_TRAINING_ITERATIONS, "MCTS iterations per move; with a clock, the cap of each move"),
        ("--seed", int, DEFAULT_SEED, "random seed"),
        ("--seconds", float, DEFAULT_RULE_SEARCH_SECONDS, "seconds the rule search after each decisive game runs at most"),
        ("--memory", float, DEFAULT_SEARCH_MEMORY / 1024**3, "GB the rule search's process holds at most, workers each holding an even share"),
        ("--max-steps", int, DEFAULT_MAX_STEPS, "steps a fit takes at most"),
        ("--tolerance", float, DEFAULT_TOLERANCE, "weight change below which a fit has settled"),
        ("--workers", int, DEFAULT_WORKERS, "worker processes games are played and studied in, and the rule search evaluates terms in"),
    ]
    for flag, kind, default, meaning in options:
        parser.add_argument(flag, type=kind, default=default, help=f"{meaning} (default: {default})")
    parser.add_argument(
        "--prices",
        type=_prices,
        default=DEFAULT_PRICES,
        help=f"comma-separated L1 prices swept by each search; the middle one prices each game's weight step (default: {','.join(map(str, DEFAULT_PRICES))})",
    )
    parser.add_argument(
        "--candidates",
        type=_non_negative,
        default=None,
        help="candidates each rule search tries at most (default: no limit)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"how far each weight moves toward what a finished game showed (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--rollout-actions",
        type=_non_negative,
        default=DEFAULT_TRAINING_ROLLOUT_ACTIONS,
        help=f"rollout actions played before a position is valued with value rules (default: {DEFAULT_TRAINING_ROLLOUT_ACTIONS})",
    )
    parser.add_argument(
        "--rollout-limit",
        type=_non_negative,
        default=None,
        help="actions a rollout plays at most before every player gets the unfinished payoff (default: no limit)",
    )
    parser.add_argument(
        "--unfinished-payoff",
        type=float,
        default=DEFAULT_UNFINISHED_PAYOFF,
        help=f"each player's payoff for a rollout stopped at the limit (default: {DEFAULT_UNFINISHED_PAYOFF})",
    )
    _add_deduction_options(parser)
    _add_worker_memory_option(parser)
    parser.add_argument(
        "--arms",
        type=_non_negative,
        default=DEFAULT_ARMS,
        help=f"signals followed at most, those with the best records, besides winning and the aggregations (default: {DEFAULT_ARMS})",
    )
    parser.add_argument(
        "--signal-horizon",
        type=_non_negative,
        default=DEFAULT_SIGNAL_HORIZON,
        help=f"plies later a position's signals are read for its targets (default: {DEFAULT_SIGNAL_HORIZON})",
    )
    parser.add_argument(
        "--arm-exploration",
        type=float,
        default=DEFAULT_ARM_EXPLORATION,
        help=f"UCB1's exploration weight when choosing which arms play each other (default: {DEFAULT_ARM_EXPLORATION})",
    )
    parser.add_argument(
        "--goal-limit",
        type=_non_negative,
        default=DEFAULT_GOAL_LIMIT,
        help=f"moves ahead the deduced goal distance looks for a win, 1 or more (default: {DEFAULT_GOAL_LIMIT})",
    )
    parser.add_argument(
        "--signal-library",
        type=Path,
        default=None,
        help="the signal library to start from, such as an earlier run's (default: the signals deduced from the rules)",
    )
    parser.add_argument(
        "--signals-directory",
        default="data/signals",
        help="where the signal library is saved after every game (default: data/signals)",
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
    if arguments.goal_limit < 1:
        parser.error(f"--goal-limit needs 1 or more, not {arguments.goal_limit}")
    if arguments.learning_rate < 0.0:
        parser.error(f"--learning-rate needs 0 or more, not {arguments.learning_rate}")
    domain = create_domain(arguments.domain)
    if arguments.training_time_control is not None and domain.timeout is None:
        parser.error(f"{domain.name} can't be played on a clock: it has no timeout rule")
    libraries = SignalLibraryRepository(SignalLibraryJsonMapper())
    library = None if arguments.signal_library is None else libraries.load(arguments.signal_library)
    if library is not None and library.domain != domain.name:
        parser.error(f"--signal-library holds signals for {library.domain}, not {domain.name}")
    deduction, _ = _deduction_settings(parser, arguments)
    settings = ContinuousTrainingSettings(
        arguments.games,
        arguments.iterations,
        arguments.seed,
        ValueSettings(
            arguments.prices,
            arguments.max_steps,
            arguments.tolerance,
            arguments.seconds,
            int(arguments.memory * 1024**3),
            arguments.candidates,
        ),
        SignalSettings(arguments.arms, arguments.signal_horizon, arguments.arm_exploration, arguments.goal_limit),
        arguments.rollout_actions,
        arguments.rollout_limit,
        None if arguments.rollout_limit is None else arguments.unfinished_payoff,
        deduction,
        arguments.ponder_positions,
        arguments.ponder_endings,
        arguments.learning_rate,
        arguments.training_time_control,
        arguments.expected_steps,
        arguments.selection,
        arguments.puct_exploration,
        arguments.prior,
        arguments.prior_temperature,
    )

    directory = Path(arguments.log_directory) / domain.name
    memory_cap = _memory_cap(parser, arguments, directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    handler = logging.FileHandler(directory / f"{started:%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(arguments.log_level)
    path = Path(arguments.signals_directory) / domain.name / f"{started:%Y-%m-%d_%H-%M-%S}.json"
    try:
        logger.info(
            "Training in %d worker processes, each holding at most %d bytes; memory diagnoses in %s",
            arguments.workers,
            memory_cap.worker_bytes,
            memory_cap.diagnosis_directory,
        )
        logger.info(
            "Following signals: at most %d besides winning and the aggregations, read %d plies later, goal distance "
            "looking up to %d moves ahead; starting from %s; the library saved after every game to %s",
            arguments.arms,
            arguments.signal_horizon,
            arguments.goal_limit,
            arguments.signal_library or "the signals deduced from the rules",
            path,
        )

        def save(saved: SignalLibrary) -> None:
            libraries.write(saved, path)

        knowledge_base = create_knowledge_base(domain.name, arguments.knowledge)
        library = create_continuous_trainer(knowledge_base, arguments.workers, memory_cap).train(domain, library, settings, save)
        logger.info("Saved signal library %s", path)
        print(f"Saved signal library {path}")
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
    parser.add_argument(
        "--ponder-endings",
        type=_non_negative,
        default=0,
        help="positions of decisive training games deduced at most per round, walking back from each game's end until a "
        "position isn't proven; needs --deduction-plies (default: 0)",
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
    pondered = arguments.ponder_positions or arguments.ponder_endings
    if pondered and not arguments.deduction_plies:
        parser.error("--ponder-positions and --ponder-endings need --deduction-plies")
    if arguments.deduction_plies and arguments.deduction_seconds <= 0.0:
        parser.error(f"--deduction-seconds needs more than 0, not {arguments.deduction_seconds}")
    if not arguments.deduction_plies:
        return None, None
    budget = DeductionBudget(arguments.deduction_plies, arguments.deduction_seconds, arguments.highest_payoff)
    return budget, PonderingSettings(arguments.ponder_positions, budget, arguments.ponder_endings) if pondered else None


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
