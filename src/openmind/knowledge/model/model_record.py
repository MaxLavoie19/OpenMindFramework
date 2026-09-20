from dataclasses import dataclass

from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """One way to perform a task, kept the way everything else the agent knows is kept: a ruleset, a lookup table, a
    decision tree, a network, or whatever an integrator registers.

    `tasks` are what it performs (see `constant/task_constant.py`), more than one where one model performs several —
    a network with a head for a policy's position value and another for its move value. `context` is the id of the
    context it performs them in,
    and `family` what kind of model it is. `mechanism` is the id of the mechanism its readings are sourced by, so what
    it says is scored like any other evidence and its accuracy is measured. `location` is whatever loads it: a
    ruleset's id, a path under `data/`, a URI. Its measured accuracy and processing time are beliefs about it, not
    fields here. `id` is empty until the knowledge base keeps it."""

    name: str
    tasks: tuple[str, ...]
    context: str
    family: str
    mechanism: str
    location: str = ""
    tags: Tags = ()
    id: str = ""
