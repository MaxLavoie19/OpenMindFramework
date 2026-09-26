from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Row:
    """The row of the cell a parameter points at."""

    parameter: str

    def __str__(self) -> str:
        return f"the row of {self.parameter}"


@dataclass(frozen=True, slots=True)
class Column:
    """The column of the cell a parameter points at."""

    parameter: str

    def __str__(self) -> str:
        return f"the column of {self.parameter}"


@dataclass(frozen=True, slots=True)
class Standing:
    """What a model holds where a parameter points, before the action."""

    model: str
    parameter: str

    def __str__(self) -> str:
        return f"what {self.model} holds at {self.parameter}"


@dataclass(frozen=True, slots=True)
class Asked:
    """A parameter of the action as it was given: which piece a pawn is to become, which side to castle."""

    parameter: str

    def __str__(self) -> str:
        return f"the {self.parameter} asked for"


@dataclass(frozen=True, slots=True)
class Always:
    """A value that is the same whatever the action: whose turn it becomes, an emptied square."""

    value: Value

    def __str__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True, slots=True)
class Other:
    """The player whose action this is not. In a game of two, the one about to act."""

    def __str__(self) -> str:
        return "the player not acting"


@dataclass(frozen=True, slots=True)
class More:
    """What a scalar read before, and one more — a clock counting, a tally kept.

    Every other way of drawing a part names something the position already holds. This one does arithmetic on it,
    which nothing else in OMF's vocabulary does, and it is here because a count that goes up cannot be said any
    other way."""

    model: str
    by: int = 1

    def __str__(self) -> str:
        return f"{self.model} and {self.by} more"


#: How one part of a change is drawn from the action it follows.
#:
#: A change hands back squares and values, and a prediction has to say where they came from. Naming them outright
#: would tie a rule to one position — the square a pawn was taken on this time. Drawn from the action, the same
#: rule holds wherever the action is played: a piece taken in passing stands at the row of the source and the
#: column of the target, whichever squares those are today.
Drawn = Row | Column | Standing | Asked | Always | Other | More
