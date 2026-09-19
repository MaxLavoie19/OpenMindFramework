from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Conflict:
    """Knowledge that doesn't cohere, in one context: a belief whose justified evidence names different values, or a
    belief an anchor contradicts. `ids` names what is in conflict; `kind` says how (see `epistemology_constant`)."""

    id: str
    variable: str
    context: str
    ids: tuple[str, ...]
    kind: str
