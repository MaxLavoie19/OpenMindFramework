from dataclasses import dataclass

from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.tags import Tags
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Belief:
    """A variable as some holder takes it to be, in one context.

    `holder` is whose belief it is, as a chain: `()` the agent's own, `("black",)` what the agent believes black
    believes, `("black", "white")` a level deeper. `certainty` is how strongly it is held: 100 % unless said otherwise;
    it should depend on the evidence, but may lack a proper justification. `accuracy` depends on the mechanism that
    produced it, and `precision` is how narrow the value is, such as the spread of an estimate; either is None where it
    isn't known or doesn't apply. The evidence is optional, and kept whole: the evidence for `value` and the evidence
    for other values stay apart. `id` is empty until the knowledge base keeps the belief."""

    variable: str
    context: str
    value: Value
    holder: tuple[str, ...] = ()
    certainty: float = 1.0
    accuracy: float | None = None
    precision: float | None = None
    evidence: tuple[Evidence, ...] = ()
    tags: Tags = ()
    id: str = ""

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        """What identifies the belief: its variable, its context and whose it is."""
        return self.variable, self.context, self.holder

    @property
    def supporting(self) -> tuple[Evidence, ...]:
        """The evidence for the value held."""
        return tuple(evidence for evidence in self.evidence if evidence.value == self.value)

    @property
    def opposing(self) -> tuple[Evidence, ...]:
        """The evidence for any other value."""
        return tuple(evidence for evidence in self.evidence if evidence.value != self.value)
