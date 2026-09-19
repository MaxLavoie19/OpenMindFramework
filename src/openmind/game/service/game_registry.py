import logging
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points

from openmind.game.constant.game_constant import DOMAIN_ENTRY_POINTS, VARIANT_SEPARATOR
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

#: What a project registers under the openmind.domains entry points: a function declaring its game into a knowledge
#: base from the whole name, such as "chess" or "chess/960", and giving back the context it was declared under; it
#: raises ValueError for a variant it doesn't know.
type GameDeclaration = Callable[[str, KnowledgeBase], str]


class GameRegistry:
    """The games installed projects register under the openmind.domains entry points, OMF's own example games among
    them. A game is found by the part of its name before "/", and its declaration gets the whole name."""

    def __init__(self, games: Mapping[str, GameDeclaration] | None = None) -> None:
        self._games = dict(games) if games is not None else None

    def names(self) -> tuple[str, ...]:
        """Every game registered, sorted."""
        return tuple(sorted(self._registered()))

    def declare(self, name: str, knowledge_base: KnowledgeBase) -> str:
        """Declares the game, or one of its variants, into the knowledge base and gives back the context it was declared
        under. An unknown game raises ValueError naming the games registered."""
        game = name.partition(VARIANT_SEPARATOR)[0]
        registered = self._registered()
        if game not in registered:
            raise ValueError(f"Unknown game {name!r}; known games: {', '.join(sorted(registered)) or 'none'}")
        context = registered[game](name, knowledge_base)
        logger.debug("Declared %s as context %s", name, context)
        return context

    def _registered(self) -> dict[str, GameDeclaration]:
        if self._games is not None:
            return self._games
        return {point.name: point.load() for point in entry_points(group=DOMAIN_ENTRY_POINTS)}
