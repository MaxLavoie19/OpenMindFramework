from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelfPlaySettings:
    """What self-play may spend. `games` is how many to play, `seconds` what a player may spend on one step, and
    `steps` how many steps a game may take before it is cut short, None for as many as the game takes.

    **A game whose rules end it is left to run.** Chess ends itself: the fifty-move rule and threefold repetition see
    to that, and games played with no heuristic at all still reach checkmate in a hundred moves or so. Cutting such a
    game short doesn't save time, it throws away the only thing the game had to say — what it paid. `steps` is for a
    game that genuinely never ends, such as a repeated one with no last round, and it is the caller's to set rather
    than a limit OMF keeps to itself."""

    games: int = 1
    seconds: float = 1.0
    steps: int | None = None
    seed: int = 0
