import argparse


def add_rollout_limit_options(parser: argparse.ArgumentParser) -> None:
    """The rollout limit and the payoff a rollout stopped there gives, for the entry points that train."""
    parser.add_argument(
        "--rollout-limit",
        type=_non_negative,
        default=None,
        help="actions a rollout plays at most before every player gets the unfinished payoff (default: no limit)",
    )
    parser.add_argument(
        "--unfinished-payoff",
        type=float,
        default=None,
        help="each player's payoff for a rollout stopped at the limit; needed with --rollout-limit",
    )


def checked_unfinished_payoff(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> float | None:
    """The unfinished payoff with a rollout limit; a limit without one, or one without a limit, is rejected."""
    if arguments.rollout_limit is not None and arguments.unfinished_payoff is None:
        parser.error("--rollout-limit needs --unfinished-payoff")
    if arguments.rollout_limit is None and arguments.unfinished_payoff is not None:
        parser.error("--unfinished-payoff needs --rollout-limit")
    return arguments.unfinished_payoff


def _non_negative(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, not {number}")
    return number
