import argparse
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import DEFAULT_UNFINISHED_PAYOFF, EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.entrypoint.train_values import (
    _add_deduction_options,
    _add_worker_memory_option,
    _deduction_settings,
    _memory_cap,
)
from openmind.inference.constant.inference_constant import DEFAULT_SEARCH_MEMORY, DEFAULT_SEARCH_SECONDS
from openmind.parallel.constant.parallel_constant import DEFAULT_WORKERS
from openmind.rbs.constant.value_constant import DEFAULT_MAX_STEPS, DEFAULT_PRICES, DEFAULT_TOLERANCE
from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.mapper.value_rule_text_mapper import ValueRuleTextMapper
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.repository.value_base_repository import ValueBaseRepository
from openmind.training.constant.training_constant import (
    DEFAULT_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_VALUE_GAMES,
    DEFAULT_VALUE_HELD_OUT_GAMES,
    OUTCOME_TARGET,
    SIGNALS_TARGET,
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
    parser.add_argument(
        "--rollout-limit",
        type=_non_negative,
        default=None,
        help="actions a self-play rollout plays at most before every player gets the unfinished payoff (default: no "
        "limit)",
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
        "--values-directory", default="data/values", help="where value bases are saved (default: data/values)"
    )
    arguments = parser.parse_args(argv)
    if arguments.target == SIGNALS_TARGET:
        parser.error("--target signals needs openmind-train-values, which keeps the signal library from round to round")
    domain = create_domain(arguments.domain)
    values = ValueSettings(
        arguments.prices,
        arguments.max_steps,
        arguments.tolerance,
        arguments.seconds,
        int(arguments.memory * 1024**3),
        arguments.candidates,
    )
    deduction, pondering = _deduction_settings(parser, arguments)
    settings = ValueDistillationSettings(
        arguments.games, arguments.held_out_games, arguments.iterations, arguments.seed, arguments.target, values, pondering
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
        if deduction is not None:
            logger.info(
                "Self-play deduces the positions without rules within %d plies and %s seconds; pondering %d positions",
                deduction.plies,
                deduction.seconds,
                0 if pondering is None else pondering.positions,
            )
        if arguments.rollout_limit is not None:
            agent_builder.with_rollout_limit(arguments.rollout_limit, arguments.unfinished_payoff)
            logger.info(
                "Self-play rollouts stop after %d actions, every player getting %s",
                arguments.rollout_limit,
                arguments.unfinished_payoff,
            )
        result = create_value_distiller(arguments.workers, memory_cap).distill(domain, agent_builder, settings)
        path = ValueBaseRepository(ValueBaseJsonMapper()).save(
            result.value_base, Path(arguments.values_directory), datetime.now()
        )
        base, rule_text = result.value_base, ValueRuleTextMapper()
        print(f"bias {base.bias:+.6g}")
        for rule in base.rules:
            print(rule_text.to_text(rule))
        chosen = "nothing to fit" if result.chosen is None else f"chosen at price {result.chosen.price}"
        print(
            f"Value rules: {len(base.rules)} of {len(result.candidates)} candidate terms, {chosen}; payoffs from "
            f"{base.low} to {base.high}"
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
        if result.pondering is not None:
            summary = result.pondering
            print(
                f"Pondered {summary.positions} positions: {summary.proven} proven; {summary.seeds} seeds, "
                f"{summary.seeds_kept} kept by the search, {summary.seeds_in_rules} in the value rules"
            )
        print(f"Saved values {path}")
        logger.info("Saved values %s", path)
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
