from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class RuleSet:
    """The rules deciding what one kind of thing may do, held as pointers rather than as rules.

    **Pointers, so that what several movers share is one rule and not six.** A bishop and a rook both refuse a
    move onto one of their owner's own things; written into each set that is two rules that happen to read alike,
    and nothing could tell them from two rules that merely resemble each other. Pointed at, it is one rule with
    two sets naming it, and how widely a rule holds is something you can read off rather than something anybody
    declared.

    **A rule in here carries no dispatch of its own.** It says "not unequal offsets", never "a bishop may not
    move unequal offsets". Were the dispatch inside the rule, a white bishop's rule and a black bishop's would be
    different rules however identical their content, nothing could be shared, and the pointer count would stop
    meaning anything. Whose rule it is, is the set it is in.

    `of` is what the game calls the thing these are the rules of — a piece, a phase, whatever it declares. It is
    the value a candidate's first parameter is constrained to, so which set applies to a candidate is settled
    inside the constraint problem rather than by something dispatching outside it."""

    of: Value
    rules: tuple[str, ...] = ()

    def with_rule(self, pointer: str) -> "RuleSet":
        """The same set, also pointing at that rule. Pointing twice is pointing once."""
        return self if pointer in self.rules else RuleSet(self.of, (*self.rules, pointer))

    def without(self, pointer: str) -> "RuleSet":
        """The same set, no longer pointing at that rule."""
        return RuleSet(self.of, tuple(one for one in self.rules if one != pointer))

    def __len__(self) -> int:
        return len(self.rules)

    def __contains__(self, pointer: str) -> bool:
        return pointer in self.rules
