from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrisonersDilemmaVariant:
    """A variant of the repeated prisoner's dilemma: how many rounds are played, None when no last round is known, the
    chance the game ends after each round, and whether both players choose at once instead of one after the other with
    the first choice hidden."""

    name: str
    rounds: int | None
    ending_chance: float
    simultaneous: bool = False
