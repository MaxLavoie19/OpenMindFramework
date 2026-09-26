from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Phase:
    """One part of a turn, and what may be attempted in it.

    A turn is one action in chess and several in most games: roll then move, draw then discard, bid then play.
    A phase says which actions exist while it holds — the candidates there are to consider — as a parameter's
    kind says which values there are to consider.

    **It says what may be attempted and never what is allowed.** Which of a phase's actions is legal is a
    constraint, learned like every other, and the phase is one more reading such a constraint may be conditioned
    on. Declaring "a pawn promotes on the last rank" would be handing over a rule; declaring that there is a
    promoting phase whose action is `promote` is handing over the same sort of thing as "a move has an origin
    and a destination" — the shape of what can be asked, not the answer."""

    name: str
    actions: tuple[str, ...] = ()

    def offers(self, action: str) -> bool:
        return action in self.actions
