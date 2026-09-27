import logging
from collections.abc import Sequence

from openmind.game.service.game_declarer import GameDeclarer
from openmind.game.service.schema_declarer import SchemaDeclarer
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.statement.model.clause import Clause
from openmind.rule.model.clause_rule import ClauseRule
from openmind.statement.model.consequence import Consequence
from openmind.rule.model.consequence_rule import ConsequenceRule
from openmind.rule.service.rule_caller import RuleCaller
from openmind.structure.model.schema import Schema
from openmind.world.model.players import Players
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class InducedDeclarer:
    """A game declared out of what was learned of one, rather than out of what somebody wrote about it.

    **The counterpart to a project's own declaration, and the point of all of it.** A project declares a game by
    handing OMF its start, its players, what its actions may take, what makes one legal and what it leads to.
    Everything in that list is now something OMF can work out for itself: the constraints come from watching what
    a game refuses, what an action does from watching it done, what a parameter may take from the schema it was
    shown, and what a game pays from the position it was shown paying it. Declared together they are a game, and
    a game can be played.

    **No listing, ever.** A game may declare the legal actions outright as a fast path, and the constraints then
    go unused — which for an induced game would mean playing the declared game while believing it was playing the
    learned one. `GameRelaxer` drops it for the same reason.

    **And no ending.** A game is over where its ending rule says so *or where nobody can act*, and an induced
    game should end the second way: a position its own rules leave no move in is over as far as it knows. Saying
    otherwise would be borrowing a judgement it has not earned. What the game paid is not an ending rule either —
    it is what an effects rule fills, learned like everything else here.

    It keeps nothing: built once, it is given the knowledge base and the parts on every call."""

    def __init__(self, schema_declarer: SchemaDeclarer | None = None, rule_caller: RuleCaller | None = None) -> None:
        self._schema = SchemaDeclarer() if schema_declarer is None else schema_declarer
        # What checks a rule can be found by a worker process. A clause and a consequence pass it, being data;
        # anything written as a closure over what was learned does not, which is the check doing its job.
        self._caller = rule_caller

    def declare(
        self,
        knowledge_base: KnowledgeBase,
        context: str,
        action: str,
        starts_at: State,
        played_by: Players,
        schema: Schema,
        refused: Sequence[Clause] = (),
        does: Sequence[Consequence] = (),
    ) -> str:
        """Everything learned of a game, declared as that game, and the context it went into.

        `refused` may be empty where the constraints were listed as they were learned, which is how the
        constraint learner writes them — it declares each as it holds it and takes out the ones it no longer
        holds, so handing them again here would say the same thing twice. What it cannot do is leave the game
        without the rest, which is what this is for."""
        declarer = GameDeclarer(knowledge_base, context, rule_caller=self._caller, open=True)
        declared: list[RuleRecord] = [declarer.starts_at(starts_at), declarer.played_by(played_by)]
        declared.extend(self._schema.declare(declarer, schema, action))
        declared.extend(
            declarer.constraint(action, number, ClauseRule(clause))
            for number, clause in enumerate(refused, start=1)
        )
        if does:
            declared.append(declarer.leads_to(action, ConsequenceRule(tuple(does))))
        logger.info(
            "Declared %s out of what was learned: %d rules, %d of them constraints, and %s",
            context,
            len(declared),
            len(refused),
            f"{len(does)} things a {action} does" if does else f"nothing yet about what a {action} does",
        )
        return declarer.done()
