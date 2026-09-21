from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Fact:
    """Something concluded about a game, and what it was concluded from.

    A fact is not a rule: a rule says what is allowed, a fact says what follows. `kind` is what sort of thing is
    being said, `about` what it is said of, and `held` the number where the saying has one. `from_rules` and
    `from_facts` are what it rests on, so anything concluded can be traced back to the rules it came out of — which
    is what lets the knowledge base say how sure it is and why."""

    kind: str
    about: tuple[Value, ...]
    held: float | None = None
    from_rules: tuple[str, ...] = ()
    from_facts: tuple["Fact", ...] = ()

    @property
    def readable(self) -> str:
        said = f"{self.kind} {' '.join(repr(one) for one in self.about)}"
        if self.held is not None:
            said += f" = {self.held:g}"
        return said

    @property
    def rests_on(self) -> str:
        """What it was concluded from, in words."""
        held = [*self.from_rules, *(one.readable for one in self.from_facts)]
        return " and ".join(held) or "the rules themselves"
