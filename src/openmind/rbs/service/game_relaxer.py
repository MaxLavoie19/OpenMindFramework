import logging
from dataclasses import replace

from openmind.doxastic.constant.doxastic_constant import ASSUMED
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT, EFFECTS, VALUES
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.constant.rule_based_constant import RULES_DEFINITIONS
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.world.model.players import Players

logger = logging.getLogger(__name__)

#: The action a player takes to hand the turn over where the rules let them pass.
PASS = "pass"

#: How a relaxation that drops a constraint is named, and how one that lets a player pass is.
WITHOUT = "{context} without {rule}"
PASSING = "{context} where a player may pass"


class GameRelaxer:
    """Makes a relaxation of a game: the same game with fewer constraints, where the rules allow more than they do.

    A relaxation is not something applied to a game — it is a game of its own, a variant with its own context. Relaxing
    declares nothing new except where a rule has to be invented: every rule of the game gains a weight in the relaxed
    context, except the constraint the relaxation drops. An RBS built for that context is then a game like any other,
    so the solver, the predictor and deduction work in it unchanged, and the game relaxed is left as it was.

    What is proved in a relaxation holds there; here it is a hint, which is what the `relaxed` kind of record is for."""

    def __init__(self, knowledge_base: KnowledgeBase, rule_caller: RuleCaller) -> None:
        self._knowledge_base = knowledge_base
        self._rule_caller = rule_caller

    def relaxations(self, context: str) -> tuple[str, ...]:
        """The relaxations the game's own rules allow: each constraint dropped, and, where one player acts at a time,
        every player also being able to pass. Wider values for a parameter aren't among them, since the rules don't say
        what wider would mean; a project declares such a variant itself."""
        rules = self._knowledge_base.rules(context)
        found = [
            WITHOUT.format(context=context, rule=rule.name)
            for rule in rules
            if rule.kind == CONSTRAINT
        ]
        if self._hands_over(context):
            found.append(PASSING.format(context=context))
        return tuple(found)

    def relax(self, context: str, relaxation: str) -> str:
        """Declares the relaxation as a context of its own and gives back its name. A name that isn't one of the game's
        own relaxations raises ValueError."""
        if relaxation not in self.relaxations(context):
            raise ValueError(f"{relaxation!r} isn't a relaxation of {context}: {', '.join(self.relaxations(context))}")
        passing = relaxation == PASSING.format(context=context)
        dropped = None if passing else relaxation.removeprefix(f"{context} without ")
        kept = 0
        for rule in self._knowledge_base.rules(context):
            if rule.kind == CONSTRAINT and rule.name == dropped:
                continue
            self._knowledge_base.declare(
                replace(rule, contexts=(*rule.contexts, (relaxation, rule.weight(context))))
            )
            kept += 1
        if passing:
            kept += self._passing(context, relaxation)
        logger.info("Relaxed %s into %s: %d rules", context, relaxation, kept)
        return relaxation

    def _passing(self, context: str, relaxation: str) -> int:
        """The rules a passing player needs, declared in the relaxation alone: the constraints that read no parameter,
        so a player can pass while the game goes on and not once it is over, and effects handing the turn over."""
        rules = self._knowledge_base.rules(context)
        definitions = next(
            (rule.rule for rule in rules if rule.name == RULES_DEFINITIONS and isinstance(rule.rule, PythonRule)), None
        )
        names_by_action: dict[str, list[str]] = {}
        for rule in rules:
            if rule.kind == VALUES and rule.action and rule.parameter:
                names_by_action.setdefault(rule.action, []).append(rule.parameter)
        free: list[RuleRecord] = []
        for rule in rules:
            if rule.kind != CONSTRAINT or not rule.action:
                continue
            prepared = self._rule_caller.prepare(rule.rule, names_by_action.get(rule.action, []), definitions)
            if not prepared.arguments and all(rule.rule != kept.rule for kept in free):
                free.append(rule)
        declared = 0
        for number, rule in enumerate(free, start=1):
            self._knowledge_base.declare(
                RuleRecord(
                    f"{PASS} is legal, {number}",
                    CONSTRAINT,
                    rule.rule,
                    Provenance(ASSUMED, relaxation),
                    ((relaxation, rule.weight(context)),),
                    PASS,
                )
            )
            declared += 1
        self._knowledge_base.declare(
            RuleRecord(
                f"what {PASS} leads to",
                EFFECTS,
                self._handover(self._players(context)),
                Provenance(ASSUMED, relaxation),
                ((relaxation, 1.0),),
                PASS,
            )
        )
        return declared + 1

    def _handover(self, players: Players) -> PythonRule:
        """Effects that give the turn to the next player, in the players' order."""
        following = {name: players.names[(at + 1) % len(players.names)] for at, name in enumerate(players.names)}
        return PythonRule(f"{players.to_act} = {following!r}[{players.to_act}]")

    def _players(self, context: str) -> Players:
        return create_rule_based_system(self._knowledge_base, context).players()

    def _hands_over(self, context: str) -> bool:
        """Whether a variable names the player to act, without which no player can pass."""
        rbs = create_rule_based_system(self._knowledge_base, context)
        try:
            return rbs.players().to_act in {name for name, _ in rbs.start().variables}
        except ValueError:
            return False
