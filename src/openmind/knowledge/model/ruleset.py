from dataclasses import dataclass

from openmind.knowledge.model.ruleset_link import RulesetLink
from openmind.knowledge.model.source import Source
from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Ruleset:
    """A part of a context holding the rules for one purpose: the game's simulation, the position value heuristic, the
    move value heuristic, one tactic's, one known agent's play style. It is one model of its `task`, which any other
    model may replace.

    It lists its rules by id, each with its weight here; a rule can be in several rulesets. `context` is the id of the
    context it belongs to. A ruleset an application declared is frozen unless `open`, and a frozen ruleset lists only
    frozen rules. `id` is empty until the knowledge base keeps it."""

    name: str
    context: str
    task: str
    source: Source
    links: tuple[RulesetLink, ...] = ()
    open: bool = False
    tags: Tags = ()
    id: str = ""

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(link.rule for link in self.links)

    def weight(self, rule_id: str) -> float | None:
        """What the rule weighs here, or None where the ruleset doesn't list it."""
        for link in self.links:
            if link.rule == rule_id:
                return link.weight
        return None
