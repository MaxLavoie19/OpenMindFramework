from dataclasses import dataclass

from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Policy:
    """A way of playing tuned to a subset of the actions: the agent picks how it wants to play, then picks the action
    that serves that best.

    A policy performs two tasks, and `position_value` and `move_value` are the ids of the models performing them: the
    heuristic valuing the states it is after, and the one valuing the moves that serve it. One model can perform both,
    such as a network with a head for each. `sub_goal` says what it is after, in the words whoever made it used; a
    policy is manufactured — an agent told to learn freeze, fight and flight — or learned by policy optimization.

    The default policy has no name and no models of its own: its candidates are the legal actions. `id` is empty until
    the knowledge base keeps it."""

    name: str
    context: str
    sub_goal: str = ""
    position_value: str = ""
    move_value: str = ""
    tags: Tags = ()
    id: str = ""
