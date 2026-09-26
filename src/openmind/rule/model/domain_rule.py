from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class DomainRule:
    """The values a parameter can take, kept as the values rather than as source that would rebuild them.

    **A domain written as source is a domain that has to be parsed, and one day will not be.** Written with
    `repr`, a parameter ranging over strings or numbers reads back perfectly and one ranging over anything a
    game builds — a cell of a row and a column, a card of a suit and a rank — reads back as source naming a
    class the rule's namespace has never heard of, and fails at the moment somebody tries to play. A game of
    parts is exactly the kind of game OMF is for, so that is not an edge.

    The third time this answer has been right, after a learned constraint and a learned effect: what is stored
    is the thing, and nothing has to be turned into anything. Here it is the plainest of the three — there is
    nothing to run, so there is nothing to run it with."""

    values: tuple[Value, ...] = ()

    @property
    def readable(self) -> str:
        return f"{len(self.values)} values" if len(self.values) > 6 else repr(list(self.values))
