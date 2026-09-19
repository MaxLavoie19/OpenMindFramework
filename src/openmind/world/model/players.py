from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Players:
    """The players, in order; the model saying who acts — a Scalar naming the one player to act, or, where players act
    at once, a Map flagging each player acting; and the Map holding each player's payoff by name."""

    names: tuple[str, ...]
    to_act: str
    payoff: str
