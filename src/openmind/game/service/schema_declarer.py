import logging
from collections.abc import Mapping

from openmind.knowledge.model.rule_record import RuleRecord
from openmind.game.service.game_declarer import GameDeclarer
from openmind.rule.model.domain_rule import DomainRule
from openmind.structure.model.schema import Schema
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

class SchemaDeclarer:
    """What a schema says an action's parameters range over, written down as the rules a game is assembled from.

    **The learning side has always known this and never said it where a game could hear.** A schema says a move
    takes a square and two whole numbers; `Schema.domains` resolves that into what each one can actually be, and
    the constraint learner generates its candidates over exactly those domains. A rule-based game needs the same
    thing as a `VALUES` rule per parameter, because the solver enumerates from the domains and the constraints
    only ever narrow what was enumerated. With none, there is nothing to narrow and no move can be listed at all.

    So nothing here is worked out. It is a translation, and it is the first thing to carry a domain across the
    line between what OMF learns over and what OMF plays: `Schema` is imported by the predictor, the inference
    engine and the structures, and by nothing that assembles a game.

    **The domain is kept rather than computed.** A value rule may read the position, and this one need not:
    what a parameter ranges over before any constraint narrows it does not depend on the position it is asked
    about. Kept as the values, it needs no function a worker has to find — the same reason a learned constraint
    is a clause rather than a closure — and it survives a game whose values are things it builds, which writing
    them as source did not: a cell of a row and a column reads back as source naming a class no rule has heard
    of, and fails the moment somebody plays.

    It keeps nothing: built once, it is given the schema and the declarer on every call."""

    def declare(self, declarer: GameDeclarer, schema: Schema, action: str) -> tuple[RuleRecord, ...]:
        """One `VALUES` rule per parameter of that action, declared into the declarer's ruleset.

        A parameter whose domain is empty is declared anyway and says so: a game whose numbers could not be
        bounded — one with no grid to bound them by — has a parameter nothing can enumerate, and a rule giving
        nothing is the honest way to carry that to whatever tries to play it."""
        domains = schema.domains(action)
        declared = tuple(
            declarer.values(action, parameter, DomainRule(tuple(values)))
            for parameter, values in domains.items()
        )
        logger.info(
            "What %s can take, from the schema: %s",
            action,
            "; ".join(f"{parameter} over {len(values)}" for parameter, values in domains.items()) or "no parameters",
        )
        return declared

    def domains(self, schema: Schema, action: str) -> Mapping[str, tuple[Value, ...]]:
        """What that action's parameters range over, for a caller wanting the values rather than the rules."""
        return schema.domains(action)
