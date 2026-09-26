from dataclasses import dataclass, field

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Evidence:
    """One position and what the game showed of it: which actions of one kind it allows there.

    This is the whole of what the learner is given. It is not told which of its rules were wrong and nothing here
    is a correction — a position and the actions listed in it is the input, and everything concluded is concluded
    from that.

    Every action not listed is refused, which is what makes a position evidence rather than a hint: a list of
    twenty legal moves out of four thousand candidates says twenty things about what is allowed and three thousand
    nine hundred and eighty about what is not, and the second is where nearly all the information is."""

    where: State
    action: str
    legal: tuple[Action, ...]

    #: The same actions gathered for looking up, worked out once when the evidence is made.
    #:
    #: Every candidate in the domain is asked whether it was listed, thousands of times per position, and walking
    #: the list for each turns a question into a search.
    listed: frozenset[Action] = field(init=False, compare=False, hash=False, repr=False, default=frozenset())

    def __post_init__(self) -> None:
        object.__setattr__(self, "listed", frozenset(self.legal))

    def allows(self, action: Action) -> bool:
        """Whether the game listed that action here."""
        return action in self.listed
