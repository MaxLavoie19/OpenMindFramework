from dataclasses import dataclass

from openmind.knowledge.model.source import Source
from openmind.knowledge.model.tags import Tags
from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """A rule the agent knows, kept the way everything else it knows is kept.

    `kind` says what the rule is a rule about (see `constant/rule_kind_constant.py`): the rules of a game, which an
    application declares, or a heuristic, which inference and fitting produce. `rule` is the rule itself, Python source
    or one of the application's own functions.

    `contexts` are the contexts the rule is relevant to, each with its weight there: the same rule can be heavy in one
    game and light in another, and a context it has no weight in is one it says nothing about. `action` is the action a
    constraint or an effects rule belongs to, and `parameter` the parameter a values rule gives. `probability` is the
    chance an effects rule's outcome happens, one outcome of an action among several; every other kind always holds.

    `source` is where the rule came from: declared by an application, inferred, fitted. A rule an application declares
    is frozen unless `open`; one that OMF produced itself, or one declared open, may be revised. `id` is empty until the
    knowledge base takes the rule."""

    name: str
    kind: str
    rule: Rule
    source: Source
    contexts: tuple[tuple[str, float], ...] = ()
    action: str | None = None
    parameter: str | None = None
    probability: float = 1.0
    open: bool = False
    tags: Tags = ()
    id: str = ""

    def weight(self, context: str) -> float:
        """How much the rule weighs in that context; 0 where it says nothing about it."""
        for name, weight in self.contexts:
            if name == context:
                return weight
        return 0.0

    def relevant(self, context: str) -> bool:
        """Whether the rule bears on that context at all."""
        return any(name == context for name, _ in self.contexts)
