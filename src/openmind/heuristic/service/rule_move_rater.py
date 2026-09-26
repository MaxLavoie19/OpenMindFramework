from openmind.heuristic.model.node import Node
from openmind.heuristic.service.rule_heuristic import RuleHeuristic
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action


class RuleMoveRater:
    """The move value task filled by a rule-based system: what its move rules say each action is worth to the
    player taking it.

    It is the port's shape over `RuleHeuristic`, as `RulePositionValuer` is for the other task. `RuleHeuristic`
    is a service of six methods that runs a heuristic ruleset; the port is one question. Handing the service
    where the port is asked for type-checks structurally, because one of the six happens to carry the port's
    signature — which is an accident of arity, not a design, and it made the two tasks look unlike each other
    for no reason. One of them had a wrapper and the other leaned on the coincidence.

    **And rules are one family of several.** Rating moves is a job a network may well do better than a weighted
    sum of readings, and the design says so already: a model records its family, `Outfitter` loads the families
    it knows and leaves the rest to whatever does know them, and games decide which was right. A rater named
    for its family is a rater a second family can sit beside. Leaning on the coincidence pointed the other
    way — it made the rules service *be* the port, so a network could only have arrived by displacing it.

    It keeps nothing: built once, it is given the model with every call."""

    def __init__(self, rule_heuristic: RuleHeuristic) -> None:
        self._heuristic = rule_heuristic

    def rate(
        self, model: RuleBasedSystem, node: Node, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]:
        """What each action is worth to the player taking it, in the actions' order; None for one the model's
        rules say nothing about."""
        return self._heuristic.rate(model, node, actions, player)
