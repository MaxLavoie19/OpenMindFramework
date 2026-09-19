from collections.abc import Callable

from openmind.knowledge.service.knowledge_base import KnowledgeBase

#: What an installed project registers under the openmind.domains entry points: a function that declares its game's
#: rules into a knowledge base, from the whole game name, such as "chess" or "chess/960", and gives back the context
#: they were declared under; it raises ValueError for a name it doesn't know.
type GameRules = Callable[[str, KnowledgeBase], str]
