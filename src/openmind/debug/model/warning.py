from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Warning:  # noqa: A001 - the design's name for it; it shadows the builtin only where imported
    """Something a developer should look at: conflicting rules, an observation a frozen rule can't explain, or any OMF
    WARNING log line. `ids` names the rules, experiences or beliefs involved."""

    kind: str
    message: str
    context: str | None = None
    ids: tuple[str, ...] = ()
