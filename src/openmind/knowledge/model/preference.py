from dataclasses import dataclass

from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Preference:
    """What a goal weighs for whoever holds it. Preferences are kept like everything else the agent produces, so it can
    look back on its own choices and say what it preferred.

    `holder` is whose preference it is: empty for the agent's own, `("black",)` for what it takes another agent's to
    be, nested as beliefs are. `role` is the role it applies in — player, coach — empty for any role."""

    goal: str
    weight: float
    holder: tuple[str, ...] = ()
    role: str = ""
    tags: Tags = ()
    id: str = ""
