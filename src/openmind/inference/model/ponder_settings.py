from dataclasses import dataclass

from openmind.rbs.model.value_settings import ValueSettings


@dataclass(frozen=True, slots=True)
class PonderSettings:
    """What pondering may spend and how far it may reason.

    `positions` is how many positions it gathers by playing the game; `held_out` how many of them it keeps back, so
    the rules it settles on are chosen on positions they were not fitted on. `plies` and `deduction_seconds` are what
    one position may be reasoned about, and `relaxations` whether a game's relaxations are tried where the game
    itself taught nothing. `values` is how the rules are generated and fitted.

    `worth_positions` is how many of the fitted-on positions the rules are asked what things are worth over, None
    for all of them. Asking costs a legal-move generation per thing per position, so it is the one part of
    pondering that grows with the board — and it is a budget rather than a cap chosen here, because how much of
    it is worth paying for depends on the game."""

    values: ValueSettings
    positions: int = 200
    held_out: int = 50
    #: How many walks to measure steadiness over, and how long each may run. A walk is cheap and short
    #: walks say little, so both are here to be set rather than assumed; no walks at all measures nothing
    #: and orders nothing, which is what a caller that does not want this asks for.
    walks: int = 3
    walk_steps: int = 40
    #: How many of the steadiest terms to let the search take up, None for all of them.
    #:
    #: A budget and not a threshold. How steady is steady enough has no answer that travels between games, and
    #: how many generations there are to spend is something the caller knows and this cannot. A term that never
    #: varied at all is dropped whatever this says, which is arithmetic rather than a judgement: a column the
    #: same everywhere tells no position from another, and the fit has its own constant already.
    steadiest: int | None = None
    plies: int = 4
    deduction_seconds: float = 1.0
    relaxations: bool = True
    seed: int = 0
    worth_positions: int | None = None
