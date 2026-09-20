import logging
import random
from collections.abc import Sequence

from openmind.inference.model.expression import Expression
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.knowledge.constant.knowledge_constant import INFERENCE
from openmind.knowledge.constant.rule_kind_constant import POSITION
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.service.model_registry import ModelRegistry
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: How a proposed heuristic's ruleset is named.
PROPOSED = "proposed heuristic {number}"


class HeuristicProposer:
    """Proposes heuristics from a game's rules, before it has been played.

    A game states what it is made of: the models a position holds, the values they take, what a player may do and what
    the game pays. That is a vocabulary, and every way of reading a position out of it is a candidate for what a
    position is worth. Nothing here knows which readings matter — that a queen is worth more than a knight, that the
    centre is worth taking — and nothing here is told.

    What it does is make candidates: a few readings drawn together, each weighed, as one heuristic. Which of them is
    worth anything is not decided here and cannot be: it is decided by playing them against each other, where a
    heuristic earns its rank by what its games paid. Drawing the combinations at random is what a bootstrap can do
    with no data at all; combining them by what earlier games showed is for when there are games to show it.

    It fits nothing. A fit needs positions whose worth is already known, and at a bootstrap none are."""

    def __init__(self, expression_generator: ExpressionGenerator, model_registry: ModelRegistry) -> None:
        self._generator = expression_generator
        self._registry = model_registry

    def readings(self, game: RuleBasedGame, positions: Sequence[State] = ()) -> tuple[Expression, ...]:
        """Every way of reading a position the game's own vocabulary allows: what each of its models holds, how much
        of it, and what its players can do. Read from the position the game starts at, and from any others given."""
        vocabulary = self._generator.vocabulary(game, (game.start(), *positions))
        return self._generator.leaves(vocabulary)

    def propose(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        heuristics: int,
        readings: int = 3,
        positions: Sequence[State] = (),
        seed: int = 0,
    ) -> tuple[ModelRecord, ...]:
        """That many heuristics, each of a few readings drawn from the game's vocabulary with a weight each, declared
        in the game's context and registered as models of the position value task.

        A game whose vocabulary offers no reading gives none: there is nothing to say about its positions."""
        available = self.readings(game, positions)
        if not available:
            logger.info("%s offers no reading to make a heuristic of", game.context)
            return ()
        rng = random.Random(seed)
        proposed = tuple(
            self._declared(knowledge_base, game, number, self._drawn(available, readings, rng), rng)
            for number in range(1, heuristics + 1)
        )
        logger.info(
            "Proposed %d heuristics of %s from its %d readings, %d readings each",
            len(proposed),
            game.context,
            len(available),
            readings,
        )
        return proposed

    def _drawn(self, available: Sequence[Expression], readings: int, rng: random.Random) -> tuple[Expression, ...]:
        """A few readings, no two the same. A game with fewer readings than asked for gives what it has."""
        return tuple(rng.sample(list(available), min(readings, len(available))))

    def _declared(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        number: int,
        readings: Sequence[Expression],
        rng: random.Random,
    ) -> ModelRecord:
        """The heuristic as a ruleset of its own: each reading a rule, each weighed between -1 and 1, since a reading
        can be as telling against a player as for it and nothing here knows which."""
        source = Source(knowledge_base.ensure_mechanism(INFERENCE).id, (("method", "proposed"), ("context", game.context)))
        name = PROPOSED.format(number=number)
        ruleset = knowledge_base.ruleset(Ruleset(name, game.context_id, POSITION_VALUE, source, open=True))
        for reading in readings:
            rule = knowledge_base.declare(
                RuleRecord(self._generator.source(reading).source, POSITION, self._generator.source(reading), source, open=True)
            )
            ruleset = knowledge_base.link(ruleset.id, rule.id, rng.uniform(-1.0, 1.0))
        return self._registry.register_ruleset(knowledge_base, ruleset, name)
