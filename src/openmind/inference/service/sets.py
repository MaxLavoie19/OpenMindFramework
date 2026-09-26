import logging
from collections.abc import Collection, Sequence

from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: What a set's terms are called wherever they are used — as readings, as conditions, as questions.
ELEMENT_OF, CONTAINS, SIZE = "element of", "contains", "size"


class Sets:
    """Belonging, holding and how many — the things a collection can be asked.

    **The theory that is not yet for the rules, and is honest about it.** Where sets were wanted was in OMF's
    own work: what a move must be told apart from, how large the vocabulary a constraint is written in is, what
    a set of rules leaves standing. Every one of those is a set and every one was counted by hand or enumerated
    because nothing could name it.

    Whether a *learned rule* wants any of this is another question and the answer today is probably not. Chess's
    collections are its piece types, its players, and the domains of `x` and `y`; a rule saying "the mover is
    one of the sliders" needs "the sliders" to be a declared collection, and no game declares one. A term nothing
    uses is not free either — under `DescriptionLength` the vocabulary's size prices every rule in the set — so
    offering these to the search before anything wants them would tax every constraint for nothing.

    **So this is built to be asked, not to be searched.** Its terms answer questions; they are not yet offered
    as readings a body may be built from. When a game declares a collection worth reasoning about, that changes,
    and the change is one flag.

    Nothing here knows what a game is. A collection is anything holding many of one kind, and belonging is
    belonging."""

    def element_of(self, one: Value, collection: Collection[Value]) -> bool:
        """Whether that thing is one of those.

        By what it *is* and not by where it sits. Two things equal to each other are the same member, which is
        what makes a collection a set rather than a list with pretensions."""
        return one in collection

    def contains(self, collection: Collection[Value], one: Value) -> bool:
        """The same question from the collection's side.

        Both directions exist because a rule reads in one direction and a reader in the other, and making
        somebody turn the sentence round to ask is making them do the theory's work."""
        return self.element_of(one, collection)

    def size(self, collection: Collection[Value]) -> int:
        """How many distinct things it holds.

        Distinct, because that is what a set is. A caller wanting how many *times* something was seen is
        counting, which is `Statistics`' work and not this."""
        return len(set(collection))

    def within(self, collection: Collection[Value], of: Collection[Value]) -> bool:
        """Whether everything in the one is in the other."""
        return set(collection) <= set(of)

    def shared(self, collection: Collection[Value], of: Collection[Value]) -> tuple[Value, ...]:
        """What both hold, in the order the first holds them.

        Ordered by the first rather than arbitrarily, because an answer that comes back differently on two runs
        of the same question is an answer nobody can pin a test to."""
        held = set(of)
        return tuple(dict.fromkeys(one for one in collection if one in held))

    def besides(self, collection: Collection[Value], of: Collection[Value]) -> tuple[Value, ...]:
        """What the first holds and the second does not, in the order the first holds them.

        **This is the shape of nearly every question OMF asks about its own sets**: which candidates survive the
        constraints, which of a move's namings were not chosen, which refused cases nothing accounts for."""
        held = set(of)
        return tuple(dict.fromkeys(one for one in collection if one not in held))

    def claims(self, held: object) -> bool:
        """Whether this theory speaks about that declared structure.

        Anything holding many things of one kind, which is what a collection is — and a string is not one,
        however much Python insists it is iterable. A prior and not a fence: a term may be offered wherever its
        places typecheck, and this only says where it is likely to pay."""
        return isinstance(held, (Sequence, Collection)) and not isinstance(held, (str, bytes))
