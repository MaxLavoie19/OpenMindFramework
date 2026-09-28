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
    #: How many walks to measure steadiness over, and how long each may run. No walks measures nothing,
    #: orders nothing and drops nothing, and that is the default because measuring it made things worse.
    #:
    #: **Off because it was measured, not because it was untried.** Over three seeds of tic-tac-toe, held-out
    #: loss went from 0.0105 measuring nothing, to 0.0357 ordering by steadiness, to 0.1252 keeping only the
    #: four steadiest — worse every seed and worse the more it was allowed to do. The reason is the sample:
    #: a walk of tic-tac-toe is about nine positions from the opening while the fit runs on eighty gathered
    #: ones, so a term that never varies *along a walk* can vary plenty over the positions being fitted, and
    #: dropping it throws away something the fit wanted. Measuring the spread over the gathered positions and
    #: only the step between neighbours over the walks would separate "carries nothing" from "cannot be
    #: steered by"; until that is done and measured, this stays where a caller has to ask for it.
    walks: int = 0
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
