from dataclasses import dataclass

from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Context:
    """A compartment of knowledge, and a level in the hierarchy, named for people and linked by its id. `parent` is the
    id of the level it sits in, whose knowledge it doesn't share; `inherits` holds the ids of the contexts it took rules
    from on purpose, such as a variant from its game, which copies their rules less those it leaves."""

    id: str
    name: str
    parent: str | None = None
    inherits: tuple[str, ...] = ()
    tags: Tags = ()
