import logging
from dataclasses import replace

from openmind.knowledge.constant.knowledge_constant import SIMULATION
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

#: How a relaxation that drops a constraint is named.
WITHOUT = "{context} without {rule}"


class GameRelaxer:
    """Makes a relaxation of a game: the same game with fewer constraints, where the rules allow more than they do.

    A relaxation is not something applied to a game — it is a game of its own, with its own context. Its simulation
    ruleset is an open copy of the game's, less the constraint it drops; the game's rules themselves stay as they are.
    An RBS built for that context is then a game like any other, so the solver, the predictor and deduction work in it
    unchanged. Dropping the constraint that keeps a player to their own turn gives the relaxation where that player can
    act now.

    What is proved in a relaxation holds there; here it is a hint, which is what the `relaxed` kind of record is for."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    def relaxations(self, context: str) -> tuple[str, ...]:
        """The relaxations the game's own rules allow: each constraint dropped. Wider values for a parameter aren't among
        them, since the rules don't say what wider would mean; a project declares such a variant itself."""
        simulation = self._simulation(context)
        if simulation is None:
            return ()
        return tuple(
            WITHOUT.format(context=context, rule=rule.name)
            for rule, _ in self._knowledge_base.ruleset_rules(simulation.id, (CONSTRAINT,))
        )

    def relax(self, context: str, relaxation: str) -> str:
        """Makes the relaxation a context of its own and gives back its name. A name that isn't one of the game's own
        relaxations raises ValueError."""
        if relaxation not in self.relaxations(context):
            raise ValueError(f"{relaxation!r} isn't a relaxation of {context}: {', '.join(self.relaxations(context))}")
        simulation = self._simulation(context)
        assert simulation is not None
        dropped = relaxation.removeprefix(f"{context} without ")
        game = self._knowledge_base.ensure_context(context).id
        relaxed = self._knowledge_base.ensure_context(relaxation)
        self._knowledge_base.context(replace(relaxed, inherits=(game,)))
        standing = self._knowledge_base.ruleset_named(relaxed.id, simulation.name)
        copy = standing or self._knowledge_base.copy_ruleset(simulation.id, simulation.name, relaxed.id)
        for rule, _ in self._knowledge_base.ruleset_rules(copy.id, (CONSTRAINT,)):
            if rule.name == dropped:
                copy = self._knowledge_base.unlink(copy.id, rule.id)
        logger.info("Relaxed %s into %s: %d rules", context, relaxation, len(copy.links))
        return relaxation

    def _simulation(self, context: str) -> Ruleset | None:
        game = self._knowledge_base.ensure_context(context).id
        return next(iter(self._knowledge_base.rulesets(game, SIMULATION)), None)
