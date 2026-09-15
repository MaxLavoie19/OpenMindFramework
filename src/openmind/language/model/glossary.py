from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Glossary:
    """How a domain names what its variables' indices mean, by base: for chess, {"piece": {(5, 4): "d5"}}. An index
    without a label stays its numbers."""

    labels: dict[str, dict[tuple[object, ...], str]] = field(default_factory=dict)
