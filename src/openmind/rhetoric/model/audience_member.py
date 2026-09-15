from dataclasses import dataclass

from openmind.rhetoric.model.pathos import Pathos


@dataclass(frozen=True, slots=True)
class AudienceMember:
    """An audience member and who they really are, their effective pathos, which no speaker sees."""

    name: str
    effective_pathos: Pathos
