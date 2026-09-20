from openmind.heuristic.model.node import Node
from openmind.heuristic.service.rule_heuristic import RuleHeuristic
from openmind.rbs.model.rule_based_system import RuleBasedSystem


class RulePositionValuer:
    """The position value task filled by a rule-based system: what its position rules say a state is worth to each
    player.

    It is the port's shape over `RuleHeuristic`, which needs to be told the players; they are the node's game's, so
    nothing here is kept."""

    def __init__(self, rule_heuristic: RuleHeuristic) -> None:
        self._heuristic = rule_heuristic

    def values(self, model: RuleBasedSystem, node: Node) -> tuple[float, ...] | None:
        """What the position is worth to each player, in the players' order; None where no rule could be read."""
        players = node.game.players().names  # type: ignore[union-attr]
        return self._heuristic.values(model, node, players)
