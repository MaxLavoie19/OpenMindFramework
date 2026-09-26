from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Sort:
    """One sort of character, found by the company it keeps.

    Not a category anyone named. Characters that appear after the same things and before the same things are
    doing the same job, whatever that job turns out to be — and two characters that look alike to us but keep
    different company are two sorts, which is how a notation's capitals separate from its lower case without
    anyone saying that capitals mean anything.

    Its name is its members, so a log says what a sort is rather than which number it was given."""

    characters: frozenset[str]

    @property
    def name(self) -> str:
        return "".join(sorted(self.characters))

    def __contains__(self, character: str) -> bool:
        return character in self.characters
