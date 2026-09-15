from dataclasses import dataclass

from openmind.rhetoric.model.ethos import Ethos
from openmind.rhetoric.model.pathos import Pathos


@dataclass(frozen=True, slots=True)
class Speaker:
    """A speaker: who they are (effective ethos), how they show themselves to each audience member (projective ethos,
    by member), and how they perceive each audience member (projective pathos, by member)."""

    name: str
    effective_ethos: Ethos
    projective_ethos: tuple[tuple[str, Ethos], ...]
    projective_pathos: tuple[tuple[str, Pathos], ...]
