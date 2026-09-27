from openmind.statement.model.term import Constant, Functor, Number, Term
from openmind.structure.model.value import Value

#: How one part of a change is drawn from the action it follows, by the name the term is built under.
#:
#: A change hands back squares and values, and a prediction has to say where they came from. Naming them
#: outright would tie a rule to one position — the square a pawn was taken on this time. Drawn from the action,
#: the same rule holds wherever the action is played: a piece taken in passing stands at the row of the source
#: and the column of the target, whichever squares those are today.
#:
#: **These are terms and not a vocabulary of their own.** They were nine dataclasses, and a reading's argument
#: was something else again, so the same square had two spellings and nothing knew they were one square. Said
#: as functors they are the language clauses are already written in: they unify, they are written down by the
#: mapper that writes down any term, and a search that can build a term can build one of these. The engine's
#: own note on `Functor` asked for exactly this — *"a function of terms standing for a term: the square one
#: step on from another, the player whose turn follows."*

#: One of the places a parameter points at, by the name the game gave it.
#:
#: The general form of `ROW` and `COLUMN`, which were written when a cell was two coordinates by assumption. A
#: game declares what its parameters' places are called, so a game whose action names one place or five is said
#: the same way.
PLACE = "place"

#: One of the places a parameter points at, moved on by what another parameter says.
#:
#: **Where a move said as a step ends up.** While an action named the square it went to, where a thing landed
#: was a place a parameter pointed at and `PLACE` said it. Said as how far it goes, the landing is neither
#: parameter — it is one of them plus the other — and without a way to say so the predictor can draw where a
#: move *starts* and not where it *ends*. No drawing being agreed, it reports no move at all, which is what it
#: did: `told turn`, `told halfmove clock`, and nothing about the piece.
#:
#: The same gap the readings had, where `lands on` fills it. Nothing here is a board: it is one number a
#: parameter holds added to another, which is as meaningful for a bet raised by a bid or a clock moved on by a
#: count.
STEPPED = "stepped"

#: The row of the cell a parameter points at, where the parameter names a place rather than holding its parts.
#:
#: Not the same as `PLACE` with "row", although the docstrings once implied it: `PLACE` takes the named part of
#: whatever the parameter holds, and this resolves a name the game gave a square — "e4" — through the grid's
#: own aliases. A game whose parameters hold records needs the first; a game whose parameters hold square names
#: needs the second.
ROW = "row"

#: The column of that cell, and the same distinction applies.
COLUMN = "column"

#: What a model holds where a parameter points, before the action.
STANDING = "standing"

#: A parameter of the action as it was given: which piece a pawn is to become, which side to castle.
ASKED = "asked"

#: A value that is the same whatever the action: whose turn it becomes, an emptied square.
ALWAYS = "always"

#: The player whose action this is not. In a game of two, the one about to act.
OTHER = "other"

#: What a scalar read before, and so much more — a clock counting, a tally kept.
#:
#: Every other way of drawing a part names something the position already holds. This one does arithmetic on
#: it, which nothing else in OMF's vocabulary does, and it is here because a count that goes up cannot be said
#: any other way.
MORE = "more"

#: Every way there is, so anything reading them by name has one list to read rather than its own.
DRAWINGS: tuple[str, ...] = (PLACE, STEPPED, ROW, COLUMN, STANDING, ASKED, ALWAYS, OTHER, MORE)

#: What a part of a change is drawn as. A term, because that is what everything else that says where something
#: is is made of.
type Drawn = Term


def place(parameter: str, named: str) -> Functor:
    """The place of that parameter the game calls that."""
    return Functor(PLACE, (Constant(parameter), Constant(named)))


def stepped(parameter: str, named: str, by: str) -> Functor:
    """That place of that parameter, moved on by what another parameter says."""
    return Functor(STEPPED, (Constant(parameter), Constant(named), Constant(by)))


def row(parameter: str) -> Functor:
    """The row of the cell that parameter names."""
    return Functor(ROW, (Constant(parameter),))


def column(parameter: str) -> Functor:
    """The column of the cell that parameter names."""
    return Functor(COLUMN, (Constant(parameter),))


def standing(model: str, parameter: str) -> Functor:
    """What that model holds where that parameter points, before the action."""
    return Functor(STANDING, (Constant(model), Constant(parameter)))


def asked(parameter: str) -> Functor:
    """That parameter of the action, as it was given."""
    return Functor(ASKED, (Constant(parameter),))


def always(value: Value) -> Functor:
    """That value, whatever the action."""
    return Functor(ALWAYS, (Constant(value),))


def other() -> Functor:
    """The player whose action this is not."""
    return Functor(OTHER, ())


def more(model: str, by: int = 1) -> Functor:
    """What that scalar read before, and that much more."""
    return Functor(MORE, (Constant(model), Number(by)))


def drawing(term: Drawn) -> str:
    """Which way of drawing that is, or empty where the term is not one of them."""
    return term.name if isinstance(term, Functor) and term.name in DRAWINGS else ""


def part(term: Drawn, at: int) -> Value:
    """That argument of the drawing, as the plain value it stands for."""
    held = term.arguments[at] if isinstance(term, Functor) and at < len(term.arguments) else None
    if isinstance(held, Constant):
        return held.name
    return held.value if isinstance(held, Number) else None


def said(term: Drawn) -> str:
    """That drawing in words, for anything showing one to somebody."""
    which = drawing(term)
    if which == PLACE:
        return f"the {part(term, 1)} of {part(term, 0)}"
    if which == STEPPED:
        return f"the {part(term, 1)} of {part(term, 0)}, stepped by {part(term, 2)}"
    if which in (ROW, COLUMN):
        return f"the {which} of {part(term, 0)}"
    if which == STANDING:
        return f"what {part(term, 0)} holds at {part(term, 1)}"
    if which == ASKED:
        return f"the {part(term, 0)} asked for"
    if which == ALWAYS:
        return repr(part(term, 0))
    if which == OTHER:
        return "the player not acting"
    if which == MORE:
        return f"{part(term, 0)} and {part(term, 1)} more"
    return str(term)
