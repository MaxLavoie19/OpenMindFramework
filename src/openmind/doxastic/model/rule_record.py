from dataclasses import dataclass

from openmind.doxastic.model.provenance import Provenance
from openmind.rbs.model.rule import Rule


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """A rule the agent knows, kept the way everything else it knows is kept.

    `kind` says what the rule is a rule about (see `constant/rule_kind_constant.py`): the rules of a game, which a game
    project declares, or a heuristic, which the inference engine and fitting produce. `rule` is the rule itself, Python
    source or one of the project's own functions.

    `contexts` are the contexts the rule is relevant to, each with its weight there: the same rule can be heavy in one
    game and light in another, and a context it has no weight in is one it says nothing about. `action` is the action a
    constraint or an effects rule belongs to, and `parameter` the parameter a values rule gives.

    `probability` is the chance an effects rule's outcome happens, one outcome of an action among several; every other
    kind of rule always holds.

    `provenance` is where the rule came from — declared by a project, proved by the inference engine, counted over
    games, or assumed. `id` is empty until the knowledge base takes the rule, which gives it one."""

    name: str
    kind: str
    rule: Rule
    provenance: Provenance
    contexts: tuple[tuple[str, float], ...] = ()
    action: str | None = None
    parameter: str | None = None
    probability: float = 1.0
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
