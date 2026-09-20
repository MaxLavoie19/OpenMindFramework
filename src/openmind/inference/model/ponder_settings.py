from dataclasses import dataclass

from openmind.rbs.model.value_settings import ValueSettings


@dataclass(frozen=True, slots=True)
class PonderSettings:
    """What pondering may spend and how far it may reason.

    `positions` is how many positions it gathers by playing the game; `held_out` how many of them it keeps back, so
    the rules it settles on are chosen on positions they were not fitted on. `plies` and `deduction_seconds` are what
    one position may be reasoned about, and `relaxations` whether a game's relaxations are tried where the game
    itself taught nothing. `values` is how the rules are generated and fitted."""

    values: ValueSettings
    positions: int = 200
    held_out: int = 50
    plies: int = 4
    deduction_seconds: float = 1.0
    relaxations: bool = True
    seed: int = 0
