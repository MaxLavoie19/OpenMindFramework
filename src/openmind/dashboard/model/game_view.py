from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameView:
    """A game as a page shows it: its label and when it ended; each player with the model it played, in the order of the
    players' names; the payoffs and why it ended, when the domain says; its record, such as a chess game's PGN; its moves
    as text; one picture per position, from the start to the last move's, each an SVG image or, for a domain that
    doesn't draw its positions, the position as text, `pictured` telling which; and its record id in the knowledge base
    with those of the games just before and just after it (None at either end of the list).

    `heuristics` is what each side actually judged with: the player, the name of its model, and every rule of that
    model with its weight, in the order the model lists them. A name says which heuristic won and nothing about
    why; the rules are the why, and they are already in the record — a model is remembered by everything needed to
    build it again, and for a rule-based one that is its rules and their weights. A player that judged with
    nothing has no rules here, which is a thing to know rather than a blank."""

    label: str
    ended: str
    players: tuple[tuple[str, str], ...]
    payoffs: tuple[float, ...]
    ending: str | None
    record: str | None
    moves: tuple[str, ...]
    pictures: tuple[str, ...]
    pictured: bool
    id: str = ""
    previous_id: str | None = None
    next_id: str | None = None
    heuristics: tuple[tuple[str, str, tuple[tuple[str, float], ...]], ...] = ()
