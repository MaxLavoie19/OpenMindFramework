"""The options entry points share for playing on a clock and for the knowledge base games are remembered in."""

import argparse

from openmind.knowledge.constant.knowledge_constant import KNOWLEDGE_DIRECTORY
from openmind.timing.constant.timing_constant import DEFAULT_EXPECTED_STEPS, DEFAULT_TIME_RESERVE
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.time_control import TimeControl


def add_clock_options(parser: argparse.ArgumentParser, flag: str = "--time-control") -> None:
    """The time control games are played on, under the given flag, and the steps the time budget estimator expects."""
    parser.add_argument(
        flag,
        type=time_control,
        default=None,
        metavar="MINUTES+SECONDS",
        help="play on a clock, as chess writes a time control: 3+2 is 3 minutes and 2 seconds a move; --iterations then "
        "no longer count, each move's budget replacing them (default: no clock, the iterations are the budget)",
    )
    parser.add_argument(
        "--expected-steps",
        type=expected_steps,
        default=DEFAULT_EXPECTED_STEPS,
        help="steps a player expects to be left at any point of a game on a clock, the time left being shared between "
        f"them (default: {DEFAULT_EXPECTED_STEPS})",
    )
    parser.add_argument(
        "--time-reserve",
        type=time_reserve,
        default=DEFAULT_TIME_RESERVE,
        help="the share of a player's base time kept in reserve on a clock: below it, a player only plays random moves, so "
        f"its clock never runs out; from 0 to below 1 (default: {DEFAULT_TIME_RESERVE})",
    )


def add_knowledge_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--knowledge",
        default=KNOWLEDGE_DIRECTORY,
        help=f"where the knowledge base remembers every game, under <domain>/ (default: {KNOWLEDGE_DIRECTORY})",
    )


def time_control(text: str) -> TimeControl:
    try:
        return TimeControlTextMapper().from_text(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def time_reserve(text: str) -> float:
    try:
        share = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if not 0.0 <= share < 1.0:
        raise argparse.ArgumentTypeError(f"needs a share from 0 to below 1, not {share}")
    return share


def expected_steps(text: str) -> int:
    try:
        steps = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if steps < 1:
        raise argparse.ArgumentTypeError(f"needs at least 1 step, not {steps}")
    return steps
