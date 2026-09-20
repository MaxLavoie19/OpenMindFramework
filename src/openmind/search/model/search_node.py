from dataclasses import dataclass, field

from openmind.heuristic.model.node import Node
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State


@dataclass(slots=True)
class ActionStatistics:
    """One action of one player at one node: what its model rated it before anything was explored, how often it was
    taken, and what it brought the player taking it, summed over those visits.

    `strategy` is the regret-matching probability summed over the times the node was selected, which is what the
    average strategy is read off where several players act at once."""

    action: Action
    prior: float = 0.0
    visits: int = 0
    value: float = 0.0
    strategy: float = 0.0

    @property
    def mean(self) -> float:
        """What taking it has brought on average; 0 where it has never been taken."""
        return self.value / self.visits if self.visits else 0.0


@dataclass(slots=True)
class SearchNode:
    """A node of the tree: the position, what exploring it found, and where the players' actions led.

    `actions` holds each acting player's actions with their statistics; a node with none is a position the game is
    over in. `children` holds where a joint action led, one child per outcome it can have, so a game of chance
    branches as the game itself does. `values` is what the leaves below were worth to each player, summed over the
    visits."""

    node: Node
    visits: int = 0
    values: tuple[float, ...] = ()
    actions: dict[str, tuple[ActionStatistics, ...]] = field(default_factory=dict)
    children: dict[tuple[JointAction, State], "SearchNode"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def over(self) -> bool:
        """Whether the game is over here: nobody has an action left."""
        return self.expanded and not self.actions

    def statistics(self, player: str, action: Action) -> ActionStatistics | None:
        """That player's statistics for that action here, or None where it has none."""
        return next((held for held in self.actions.get(player, ()) if held.action == action), None)

    def mean(self, player: int) -> float:
        """What this position has been worth to that player on average; 0 before anything was valued."""
        return self.values[player] / self.visits if self.visits and player < len(self.values) else 0.0
