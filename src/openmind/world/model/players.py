from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Players:
    """The players, the variable naming the player to act, and each player's payoff variable in the order of names."""

    names: tuple[str, ...]
    to_act: str
    payoffs: tuple[str, ...]
