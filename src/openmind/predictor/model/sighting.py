from dataclasses import dataclass

from openmind.predictor.model.watched import Watched


@dataclass(frozen=True, slots=True)
class Sighting:
    """One turn seen, and what the game called it.

    **Two things that have to stay in step, kept as one.** What a turn did and what it was named are learned by
    different learners from the same moment, and while they travelled as two lists side by side, staying aligned
    was something every caller had to remember to do. Crossing a process boundary is where a pairing kept by
    convention becomes a pairing kept by a type.

    **A turn may be more than one doing, and `also` is the rest of them.** A game of several phases lets a
    player do more than one thing before the turn passes — chess writes `e8=Q` for a move and a promotion — and
    a name is given to the *turn*, not to the first of its actions. Each doing is still learned under its own
    action, which is what keeps a move's rules the rules of moving; what they share is the name.

    `watched` is the first doing, so everything that only ever asks what a move did goes on asking exactly that.

    `said` is the game's own name for it and may be empty: a game need not have a notation, and nothing here
    requires one."""

    watched: Watched
    said: str = ""
    also: tuple[Watched, ...] = ()

    @property
    def doings(self) -> tuple[Watched, ...]:
        """Everything that turn did, in the order it was done."""
        return (self.watched, *self.also)
