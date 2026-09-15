from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrisonersDilemmaVariant:
    """A variant of the repeated prisoner's dilemma: how many rounds are played, None when no last round is known, and
    the chance the game ends after each round."""

    name: str
    rounds: int | None
    ending_chance: float
