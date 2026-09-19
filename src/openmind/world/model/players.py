from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Players:
    """The players, in order, and the Map holding each player's payoff by name, which an end state fills. OMF doesn't
    know whose turn it is: all players play at the same time, all the time, and the game's constraints leave a player
    no action outside their turn."""

    names: tuple[str, ...]
    payoff: str
