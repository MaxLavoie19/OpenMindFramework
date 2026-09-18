from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArmLibrary:
    """The contexts a game's arms play with, by arm name: each arm plays the position rules the knowledge base holds
    for its own context, so an arm is a set of rules like any other."""

    domain: str
    contexts: tuple[tuple[str, str], ...] = ()
