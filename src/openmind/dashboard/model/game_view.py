from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameView:
    """A game as a page shows it: its label and when it ended; each player with the model it played, in the order of the
    players' names; the payoffs and why it ended, when the domain says; its record, such as a chess game's PGN; its moves
    as text; and one picture per position, from the start to the last move's, each an SVG image or, for a domain that
    doesn't draw its positions, the position as text, `pictured` telling which."""

    label: str
    ended: str
    players: tuple[tuple[str, str], ...]
    payoffs: tuple[float, ...]
    ending: str | None
    record: str | None
    moves: tuple[str, ...]
    pictures: tuple[str, ...]
    pictured: bool
