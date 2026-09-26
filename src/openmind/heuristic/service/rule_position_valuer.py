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

    def describe(self, model: RuleBasedSystem) -> str:
        """Everything it judges with, word for word: every rule with its weight.

        A game remembers what each side played with by asking the service that ran it, so that what won can be
        read afterwards rather than only named. Two models describe alike when they judge alike, which is also
        what gives them the same id — a heuristic that changed a weight is a different heuristic and says so."""
        return self._heuristic.describe(model)
