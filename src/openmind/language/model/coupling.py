from dataclasses import dataclass

from openmind.language.model.role import Role
from openmind.language.model.shape import Shape
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Coupling:
    """One thing a symbol of a game's notation says about what happened.

    `symbol` is the character standing in the place, `where` the role that place plays, `part` what it speaks of
    and `value` what it says.

    **Nothing here is declared and the pairs are measured.** A symbol belongs to a part when knowing it stands
    there tells us about that part, in bits, over the sightings it stands in — and a symbol that tells us
    nothing is not a symbol however often it turns up. That is what makes `B` a kind and `b` a place without
    anyone saying that capitals mean pieces: two characters in two roles, found separately.

    **That guard holds one way only, deliberately.** The converse — a symbol that tells us everything is not a
    symbol unless it turns up often enough — was true of the code for as long as bits were averaged over the
    sightings a symbol is *absent* from, and nothing ever wanted it. How rare a word is and what it means are
    two questions, and only the second decides whether it is a word.

    **A role and not a position.** The same character says one thing in the opening place of one shape and
    another in the opening place of another, and a notation's length moves its places about — so a saying pinned
    to a count from the start of the string is two sayings averaged into one that is wrong about both. Measured
    against chess, that average was right ninety-five times in a hundred where each of its halves is right
    always."""

    symbol: str
    part: str
    value: Value
    where: Role | None = None
    telling: float = 0.0
    #: How often it is right where it applies — of the sightings with this symbol in this role, the share whose
    #: part really is this value.
    #:
    #: **Told apart from `telling` because they answer different questions.** Bits say a symbol is worth attending
    #: to; they never say it is always right. A symbol carrying two thirds of a bit and right four times in five
    #: is worth knowing and ruinous as a filter, since one reading in five then refuses the very happening that
    #: occurred. Narrowing by a saying needs this one; ranking what to attend to needs the other.
    surely: float = 1.0
    #: How many sightings it was measured on, so that pooling two sayings into one role weighs them by what each
    #: was worth rather than treating a saying seen four times as the equal of one seen four thousand.
    seen: int = 0

    def said_by(self, shape: Shape | None, said: str) -> bool:
        """Whether that notation has this symbol in this role."""
        if self.where is None:
            return self.symbol in said
        return self.where.filled_by(shape, said) == self.symbol

    @property
    def readable(self) -> str:
        where = "anywhere" if self.where is None else self.where.readable
        return f"{self.symbol!r} at {where} says {self.part} is {self.value!r} ({self.telling:.2f} bits)"
