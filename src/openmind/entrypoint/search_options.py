"""The options entry points share for how an agent's search selects the actions it follows."""

import argparse

from openmind.mcts.constant.mcts_constant import (
    DEFAULT_PRIOR_TEMPERATURE,
    DEFAULT_PUCT_EXPLORATION,
    PRIORS,
    SELECTIONS,
    UCB1,
    UNIFORM_PRIOR,
)


def add_selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default=UCB1,
        help="how a tried node picks the action to follow: ucb1 tries every legal action once first; puct follows "
        f"Q + c · P · √N / (1 + n) with a prior P (default: {UCB1})",
    )
    parser.add_argument(
        "--prior",
        choices=PRIORS,
        default=UNIFORM_PRIOR,
        help="the prior PUCT follows: every action alike, the agent's rules' ratings, or its value rules' values of each "
        f"action's outcomes (default: {UNIFORM_PRIOR})",
    )
    parser.add_argument(
        "--puct-exploration",
        type=non_negative_float,
        default=DEFAULT_PUCT_EXPLORATION,
        help=f"PUCT's exploration weight c (default: {DEFAULT_PUCT_EXPLORATION})",
    )
    parser.add_argument(
        "--prior-temperature",
        type=positive_float,
        default=DEFAULT_PRIOR_TEMPERATURE,
        help="the temperature of the softmax turning ratings or values into a prior; lower follows the best more closely "
        f"(default: {DEFAULT_PRIOR_TEMPERATURE})",
    )


def non_negative_float(text: str) -> float:
    number = _float(text)
    if number < 0.0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, not {number}")
    return number


def positive_float(text: str) -> float:
    number = _float(text)
    if number <= 0.0:
        raise argparse.ArgumentTypeError(f"expected more than 0, not {number}")
    return number


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
