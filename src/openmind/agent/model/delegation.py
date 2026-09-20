from dataclasses import dataclass

from openmind.budget.model.budget import Budget


@dataclass(frozen=True, slots=True)
class Delegation:
    """What a parent hands a child level: which level, what it is after, what it may spend, and who it plays as.

    The seconds are either carved out of the parent's own budget or run alongside it (`Budget.carve`,
    `Budget.alongside`); the parent decides which, and what it chose is visible in what it has left.

    `context` is the child's context id and `goal` a goal id in it: what the child pursues is the parent's to set, so a
    child can be told to coach where its parent plays to win. `player` is who the child plays as, empty where its level
    has no players of its own."""

    context: str
    goal: str
    budget: Budget
    player: str = ""
