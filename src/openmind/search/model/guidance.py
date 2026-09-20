from collections.abc import Mapping
from dataclasses import dataclass, field

from openmind.heuristic.model.move_rater import MoveRater
from openmind.heuristic.model.position_valuer import PositionValuer
from openmind.knowledge.model.policy import Policy
from openmind.search.model.hypotheses import Hypotheses

#: A model and the stateless service running it: what every task is filled by.
type Filled[Port] = tuple[object, Port]


@dataclass(frozen=True, slots=True)
class Guidance:
    """What a planner runs with: the models the caller chose for each task it needs.

    `agents` holds the other agents' models, by name: an agent model is the move value heuristic trained to predict
    that agent's play, so playing a named opponent calls for theirs and playing a stranger for a generic one.
    `hypotheses` is what fills the hypothesis task where the game hides something."""

    player: str
    position_value: Filled[PositionValuer[object]] | None = None
    move_value: Filled[MoveRater[object]] | None = None
    agents: Mapping[str, Filled[MoveRater[object]]] = field(default_factory=dict)
    hypotheses: Filled[Hypotheses[object]] | None = None
    policies: tuple[Policy, ...] = ()
