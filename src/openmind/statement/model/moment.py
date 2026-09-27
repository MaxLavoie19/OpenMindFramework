from openmind.statement.model.term import Constant, Functor, Number, Term

#: The position as it stands, which is when a reading is read unless it says otherwise.
NOW = "now"

#: A time, counted. A turn-based game's clock ticks once per joint action; a real-time game's does not wait for
#: anybody, which is the whole reason a moment is a time and not an action.
#:
#: **Indexed by time and not by what happened, and the literature is unanimous about why.** The situation
#: calculus says outright that its actions "are instantaneous, have no duration, and have immediate and
#: permanent effects" — all three false of a held key, a projectile in flight or a cooldown — and the Game
#: Description Language's `next` means *the next turn*, which a real-time game has not got. The event calculus
#: was built as the alternative and assumes "an explicit linear time-structure, independent of any events".
AT = "at"

#: The time just after something happened, where a game counts its time in things happening rather than in
#: units. `after(A)` is `at(T + 1)` for a game whose clock ticks once per action, and it is kept as its own way
#: of saying so because a turn-based game should not have to invent a clock to say "next".
AFTER = "after"

#: Something happening at a time: an action somebody took, or something that simply happened.
#:
#: **An event need not be anybody's action**, and this is what that buys even for chess. A clock running out, a
#: pawn promoting, a pawn taken in passing — these are things that happen, and having to say them as parts of
#: somebody's move is what `Consequence.order` has been working around. In the event calculus the same
#: predicate carries "both action events performed by agents and external events outside any agent's control".
HAPPENS = "happens"

#: Every way of saying when, so anything reading them by name reads one list rather than keeping its own.
MOMENTS: tuple[str, ...] = (NOW, AT, AFTER)


def now() -> Functor:
    """The position as it stands."""
    return Functor(NOW, ())


def at(when: int | float | Term) -> Functor:
    """That time."""
    return Functor(AT, (when if isinstance(when, Constant | Functor | Number) else Number(when),))


def after(what: Term) -> Functor:
    """The time just after that happened."""
    return Functor(AFTER, (what,))


def happens(what: Term, when: Term) -> Functor:
    """That thing happening then, whether or not anybody did it."""
    return Functor(HAPPENS, (what, when))


def moment(term: Term | None) -> str:
    """Which way of saying when that is, or empty where the term is not one of them.

    None is `now`: a reading that does not say when it was read was read of the position as it stands, which is
    every reading there has ever been. Saying so is what lets moments arrive without every existing clause
    having to be rewritten to mention one."""
    if term is None:
        return NOW
    return term.name if isinstance(term, Functor) and term.name in MOMENTS else ""


def said(term: Term | None) -> str:
    """That moment in words, for anything showing one to somebody."""
    which = moment(term)
    if which == NOW:
        return "now"
    if which == AT and isinstance(term, Functor):
        return f"at {_plainly(term.arguments[0])}"
    if which == AFTER and isinstance(term, Functor):
        return f"once {_plainly(term.arguments[0])} has happened"
    return str(term)


def _plainly(term: Term) -> object:
    """A term as the plain thing it stands for, where it stands for one."""
    if isinstance(term, Constant):
        return term.name
    return term.value if isinstance(term, Number) else term
