import argparse
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.game_factory import create_game
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.clock_options import add_clock_options, add_knowledge_option
from openmind.entrypoint.constant.entrypoint_constant import LOG_FORMAT
from openmind.entrypoint.rollout_options import add_rollout_limit_options, checked_unfinished_payoff
from openmind.entrypoint.search_options import add_selection_options
from openmind.entrypoint.train_values import (
    _add_deduction_options,
    _add_worker_memory_option,
    _deduction_settings,
    _memory_cap,
)
from openmind.inference.constant.inference_constant import DEFAULT_SEARCH_MEMORY, DEFAULT_SEARCH_SECONDS
from openmind.mcts.constant.mcts_constant import UNIFORM_PRIOR
from openmind.mcts.service.uniform_prior import UniformPrior
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS
from openmind.rbs.constant.value_constant import DEFAULT_MAX_STEPS, DEFAULT_PRICES, DEFAULT_TOLERANCE
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.constant.training_constant import (
    DEFAULT_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_VALUE_GAMES,
    DEFAULT_VALUE_HELD_OUT_GAMES,
    OUTCOME_TARGET,
    VALUE_TARGETS,
)
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.value_distillation_settings import ValueDistillationSettings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Fits value rules on the positions of self-play games, chooses among the fits on held-out games, prints the rules
    with their measures, and saves the value base."""
    parser = argparse.ArgumentParser(
        prog="openmind-distill-values", description="Fit rules that value positions from self-play."
    )
    parser.add_argument("domain", help="domain to distill value rules for, such as tictactoe")
    options: list[tuple[str, type, object, str]] = [
        ("--games", int, DEFAULT_VALUE_GAMES, "self-play games to fit value rules on"),
        ("--held-out-games", int, DEFAULT_VALUE_HELD_OUT_GAMES, "self-play games to choose a fit and measure it on"),
        ("--iterations", int, DEFAULT_ITERATIONS, "MCTS iterations per self-play move"),
        ("--seed", int, DEFAULT_SEED, "random seed"),
        ("--seconds", float, DEFAULT_SEARCH_SECONDS, "seconds the expression search runs"),
        ("--memory", float, DEFAULT_SEARCH_MEMORY / 1024**3, "GB the expression search's process holds at most, workers each holding an even share"),
        ("--max-steps", int, DEFAULT_MAX_STEPS, "steps a fit takes at most"),
        ("--tolerance", float, DEFAULT_TOLERANCE, "weight change below which a fit has settled"),
        ("--workers", int, DEFAULT_WORKERS, "worker processes self-play games and term evaluations run in"),
    ]
    for flag, kind, default, meaning in options:
        parser.add_argument(flag, type=kind, default=default, help=f"{meaning} (default: {default})")
    parser.add_argument(
        "--target",
        default=OUTCOME_TARGET,
        choices=VALUE_TARGETS,
        help="what a position is valued at: the game's final payoff for each player (outcome), or the search's mean "
        f"payoff for the player to act (search) (default: {OUTCOME_TARGET})",
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
        help="candidates the expression search tries at most (default: no limit)",
    )
    add_rollout_limit_options(parser)
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
        default="data/log/distill-values",
        help="where logs are saved (default: data/log/distill-values)",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="the context the fitted position rules are declared under, a variant of the game carrying its rules "
        "(default: <domain> distilled)",
    )
    add_clock_options(parser, "--training-time-control")
    add_knowledge_option(parser)
    add_selection_options(parser)
    arguments = parser.parse_args(argv)
    if arguments.prior != UNIFORM_PRIOR:
        parser.error(f"--prior {arguments.prior} needs rules the self-play agent doesn't have; use uniform")
    unfinished_payoff = checked_unfinished_payoff(parser, arguments)
    knowledge_base = create_knowledge_base(arguments.domain.split("/")[0], arguments.knowledge)
    rbs = create_game(arguments.domain, knowledge_base)
    if arguments.training_time_control is not None and not rbs.timed():
        parser.error(f"{rbs.context} can't be played on a clock: it has no timeout rule")
    values = ValueSettings(
        arguments.prices,
        arguments.max_steps,
        arguments.tolerance,
        arguments.seconds,
        int(arguments.memory * 1024**3),
        arguments.candidates,
    )
    deduction = _deduction_settings(parser, arguments)
    settings = ValueDistillationSettings(
        arguments.games,
        arguments.held_out_games,
        arguments.iterations,
        arguments.seed,
        arguments.target,
        values,
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
            "Running self-play and term evaluations in %d worker processes, each holding at most %d bytes; memory "
            "diagnoses in %s",
            arguments.workers,
            memory_cap.worker_bytes,
            memory_cap.diagnosis_directory,
        )
        logger.info(
            "Valuing positions at the %s target; prices %s; searching expressions for %s seconds within %d bytes, "
            "trying %s candidates",
            settings.target,
            ", ".join(map(str, values.prices)),
            values.seconds,
            values.memory_bytes,
            "any number of" if values.candidates is None else f"at most {values.candidates}",
        )
        agent_builder = AgentBuilder().with_exploration(EXPLORATION).with_deduction(deduction)
        agent_builder.with_selection(settings.selection, settings.puct_exploration).with_prior(UniformPrior())
        if deduction is not None:
            logger.info(
                "Self-play deduces the positions without rules within %d plies and %s seconds",
                deduction.plies,
                deduction.seconds,
            )
        if arguments.rollout_limit is not None:
            agent_builder.with_rollout_limit(arguments.rollout_limit, unfinished_payoff)
            logger.info(
                "Self-play rollouts stop after %d actions, every player getting %s",
                arguments.rollout_limit,
                unfinished_payoff,
            )
        declarer = RuleDeclarer(knowledge_base, arguments.context or f"{rbs.context} distilled")
        declarer.inherits(rbs.context)
        result = create_value_distiller(
            knowledge_base, arguments.workers, memory_cap, GameMemory(knowledge_base)
        ).distill(rbs, agent_builder, settings, declarer)
        for rule in result.rules:
            print(f"{rule.weight(result.context):+.6g} × {rule.name}")
        chosen = "nothing to fit" if result.chosen is None else f"chosen at price {result.chosen.price}"
        print(
            f"Position rules: {len(result.rules)} of {len(result.candidates)} candidate terms, {chosen}; declared "
            f"under {result.context}"
        )
        if result.fits:
            header = ("price", "terms kept", "steps", "settled", "training loss", "held-out loss")
            rows = [
                (
                    str(fit.price),
                    str(fit.terms_kept),
                    str(fit.steps),
                    "yes" if fit.settled else "no",
                    f"{fit.training_loss:.6f}",
                    "none" if fit.held_out_loss is None else f"{fit.held_out_loss:.6f}",
                )
                for fit in result.fits
            ]
            for line in _table(header, rows):
                print(line)
        print(
            f"Rows: {result.training_rows} for training, {result.held_out_rows} held out, valued at the "
            f"{settings.target} target; mean absolute error on held-out rows: {result.held_out_error}"
        )
        print(f"Declared under {result.context}")
        logger.info("Declared the position rules under %s", result.context)
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
    return ["  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in (header, *rows)]


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
