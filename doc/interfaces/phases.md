# Interfaces: phases

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

A turn is one action in chess and several in most games: roll then move, draw then discard, bid then play then
score. OMF offers the data structures a game is made of — scalar, list, grid, map, record — and a phase machine
belongs beside them, so that a game with phases is declared rather than improvised by each integrator.

## What it is and is not

**It is structure, not rules.** A grid says a game has cells addressed by coordinates; it does not say what may
stand on them or how anything moves. A machine says a turn passes through phases and which actions belong to
each; it does not say when an action is legal. That stays a constraint to be learned, and the phase is one more
reading a constraint may be conditioned on.

The line matters because it is the line this whole project is about. Declaring "a pawn promotes on the last rank"
would be handing over a rule. Declaring "there is a promoting phase, and the action available in it is
`promote`" is handing over the same sort of thing as "a move has an origin and a destination" — the shape of
what can be asked, not the answer.

**Transitions are not declared.** Which phase follows is something an action *does*, and what an action does is
the predictor's to learn: the phase is a scalar the state holds, and moving to the next phase is a `Told` change
like any other. A game that declared its transitions would be declaring half its effects, and OMF would learn
the other half — which is worse than either doing the whole thing.

## What it buys

**Promotion stops being a special case.** It was set aside as "a third parameter that exists only under a
condition" — a conditional variable, awkward in any formulation. As a phase it is an ordinary action of one
parameter, and the phase does the conditioning. Nothing conditional remains.

**A phase is a real condition, unlike a postcode.** A constraint conditioned on the castling rights is keeping a
tag that says which position it was fitted to; a constraint conditioned on the phase is saying something that
holds wherever that phase holds. They look alike to the machinery — both are scalars every candidate in a
position shares — and this is what makes them tellable apart: a phase is declared as one, so a reading of it is
known to be about the game rather than about the position.

**The games on the roadmap become representable.** Liar's dice is roll then bid then challenge; cheat is play
then doubt; the bargaining games are offer then accept or counter. None of them fits one action kind per turn.

## Models

```python
@dataclass(frozen=True, slots=True)
class Phase:
    """One part of a turn, and what may be attempted in it.

    `actions` are the actions that exist here at all — the candidate space, as a parameter's kind is the space
    its values come from. Which of them is legal is a constraint, and the same action may appear in several
    phases."""

    name: str
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Machine:
    """The phases a game's turns pass through: a data model like a grid, held by a state.

    It says which phase a position is in and what may be attempted there. It does not say what follows: that is
    a change an action makes, learned as every other change is."""

    phases: tuple[Phase, ...]
    at: str

    def phase(self, name: str) -> Phase: ...
    def acting(self) -> Phase: ...          # the phase it is in
    def moved_to(self, name: str) -> "Machine": ...   # the same machine in another phase
```

`Machine` joins `DataModel`, so `State.of(phase=Machine(...))` is how a game carries one, and the readings say
`phase(bidding)` the way they say `turn(white)`.

## What changes elsewhere

**The schema** says which actions belong to which phase, so the domains laid out for a position are the domains
of the actions available *there*. Today it declares one action and the loop assumes it.

**The evidence** becomes per phase: a position offers the candidates of its phase, and the moves the game lists
are of that phase. `Evidence` already carries the action's name, so this is the caller asking the machine which
one rather than assuming.

**The chess side** gains a promoting phase and can then stop collapsing four promotions into one move.

## Open

- **Whether a phase may belong to a player.** Simultaneous games have all players acting in one phase, and games
  like bidding have them acting in turn within one. Whether that is the machine's business or the constraints'
  is not decided.
- **Whether a machine should say which phases can follow which**, without saying when. It would bound what a
  predictor has to consider and is one step further from structure toward rules.
