from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Justification:
    """Where a belief's support comes from, traced back through its sources' `rests_on`: the anchors it reaches (direct
    experiences, frozen rules), the ids met again on the way, which give no support, and how many of its pieces of
    evidence reach an anchor each by a path of its own."""

    belief_id: str
    anchors: tuple[str, ...]
    circular: tuple[str, ...]
    independent_supports: int
    justified: tuple[int, ...]

    @property
    def anchored(self) -> bool:
        return bool(self.anchors)
