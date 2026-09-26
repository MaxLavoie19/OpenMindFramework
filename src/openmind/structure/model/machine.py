from dataclasses import dataclass

from openmind.structure.model.phase import Phase


@dataclass(frozen=True, slots=True)
class Machine:
    """The phases a game's turns pass through, and which one a position is in.

    A data model like a grid or a scalar: a state holds one, and the readings say `phase(bidding)` the way they
    say `turn(white)`. Every change gives a new machine, so states holding one stay comparable and usable as
    search keys.

    **What follows what is not here.** Moving to the next phase is something an action *does*, and what an action
    does is learned: the phase a state is in is a value, and changing it is a `Told` change like any other. A game
    that declared its transitions would be declaring half its effects while OMF worked out the rest, which is
    worse than either doing the whole of it.

    **A phase is a condition, where a position's accidents are not.** A constraint conditioned on the castling
    rights is keeping a tag saying which position it was fitted to; a constraint conditioned on the phase says
    something that holds wherever that phase holds. The two look alike to a learner — both are values every
    candidate in a position shares — and a phase being declared as one is what tells them apart."""

    phases: tuple[Phase, ...]
    at: str

    def __post_init__(self) -> None:
        if not any(one.name == self.at for one in self.phases):
            raise ValueError(f"A machine cannot be in {self.at!r}, which is not one of its phases")

    @property
    def acting(self) -> Phase:
        """The phase it is in."""
        return self.phase(self.at)

    def phase(self, name: str) -> Phase:
        for one in self.phases:
            if one.name == name:
                return one
        raise KeyError(f"The machine has no phase called {name!r}")

    def moved_to(self, name: str) -> "Machine":
        """The same machine in another phase. A phase it has not got raises, since a game moving somewhere that
        does not exist is a game with a bug in it rather than a game in an unknown state."""
        self.phase(name)
        return Machine(self.phases, name)

    def offers(self) -> tuple[str, ...]:
        """The actions that exist here, which is what a position's candidates are laid out from."""
        return self.acting.actions
