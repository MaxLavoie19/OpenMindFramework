from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameListing:
    """A decisive game as the list of games shows it, without its positions: its record id in the knowledge base, its
    label, when it ended, each player with the model it played in the order of the players' names, the payoffs, why it
    ended when the domain says, and its plies."""

    id: str
    label: str
    ended: str
    players: tuple[tuple[str, str], ...]
    payoffs: tuple[float, ...]
    ending: str | None
    plies: int
