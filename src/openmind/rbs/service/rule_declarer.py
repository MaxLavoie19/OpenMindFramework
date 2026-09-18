import logging
from collections.abc import Collection
from dataclasses import replace

from openmind.doxastic.constant.doxastic_constant import COUNTED, TOLD
from openmind.doxastic.constant.rule_kind_constant import (
    CONSTRAINT,
    MOVE,
    POSITION,
    DEFINITIONS,
    EFFECTS,
    EMPTY,
    ENDING,
    INITIAL,
    PICTURE,
    PLAYERS,
    RECORD,
    TIMEOUT,
    VALUES,
)
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.constant.rule_based_constant import EFFECTS_DEFINITIONS, RULES_DEFINITIONS
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.model.rule import Rule
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)

#: What a rule that always holds carries as its chance.
CERTAIN = 1.0


class RuleDeclarer:
    """How a game project writes its rules into the knowledge base, which is where every rule the agent knows lives.

    A game is not built and handed over; it is declared, rule by rule, under the name of the context it belongs to.
    Whatever plays it then retrieves those rules (see `rbs/service/rule_based_system.py`) rather than being given a
    game. They are `told`: the project said so, and the agent has no reason of its own to doubt it.

    A rule already declared under the same name and kind for that context is declared anew rather than added beside it,
    so declaring a game twice leaves the knowledge base as it was."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        context: str,
        weight: float = 1.0,
        rule_caller: RuleCaller | None = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._context = context
        self._weight = weight
        self._caller = rule_caller
        self._declared = 0

    @property
    def context(self) -> str:
        """The context the rules are declared under: a game, or a variant of one."""
        return self._context

    def rule(
        self,
        name: str,
        kind: str,
        rule: Rule,
        action: str | None = None,
        parameter: str | None = None,
        probability: float = CERTAIN,
    ) -> RuleRecord:
        """Declares one rule of the game. A rule written as a function no worker process could find is refused."""
        if self._caller is not None:
            self._caller.check(rule)
        standing = self._standing(name, kind)
        self._declared += 1
        return self._knowledge_base.declare(
            RuleRecord(
                name,
                kind,
                rule,
                Provenance(TOLD, self._context),
                self._contexts(standing, self._weight),
                action,
                parameter,
                probability,
                "" if standing is None else standing.id,
            )
        )

    def starts_at(self, state: State) -> RuleRecord:
        """Where the game starts."""
        return self.rule("where the game starts", INITIAL, PythonRule(repr(state.variables)))

    def played_by(self, players: Players) -> RuleRecord:
        """Who plays, in what order, the variable naming the player to act and each player's payoff variable."""
        return self.rule("who plays", PLAYERS, PythonRule(repr((players.names, players.to_act, players.payoffs))))

    def empty(self, base: str, value: Value) -> RuleRecord:
        """What a cell of that grid holds when nothing is on it, as the game defines what can be on its board."""
        return self.rule(f"an empty {base}", EMPTY, PythonRule(repr(value)), parameter=base)

    def definitions(self, rule: PythonRule, effects: bool = False) -> RuleRecord:
        """The script whose names the context's rules see, or the one its effects see."""
        return self.rule(EFFECTS_DEFINITIONS if effects else RULES_DEFINITIONS, DEFINITIONS, rule)

    def constraint(self, action: str, number: int, rule: Rule) -> RuleRecord:
        """One of the rules every legal action of that name satisfies."""
        return self.rule(f"{action} is legal, {number}", CONSTRAINT, rule, action)

    def constraints(self, action: str, *rules: Rule) -> tuple[RuleRecord, ...]:
        """Every rule a legal action of that name satisfies, numbered in the order they are given."""
        return tuple(self.constraint(action, number, rule) for number, rule in enumerate(rules, start=1))

    def values(self, action: str, parameter: str, rule: Rule) -> RuleRecord:
        """The values that parameter of the action can take: a rule reading the state, or one reading nothing where
        they are the same in every position."""
        return self.rule(f"what {parameter} can be in {action}", VALUES, rule, action, parameter)

    def leads_to(self, action: str, rule: Rule, probability: float = CERTAIN, number: int | None = None) -> RuleRecord:
        """What the action leads to, and how often, where it has more than one outcome."""
        name = f"what {action} leads to"
        return self.rule(
            name if number is None else f"{name}, {number}", EFFECTS, rule, action, probability=probability
        )

    def together(self, rule: Rule, probability: float = CERTAIN, number: int | None = None) -> RuleRecord:
        """What the players' choices, made at once, lead to together."""
        name = "what the players' choices together lead to"
        return self.rule(name if number is None else f"{name}, {number}", EFFECTS, rule, probability=probability)

    def ending(self, rule: Rule) -> RuleRecord:
        """Why a finished game ended, for the logs."""
        return self.rule("why the game ended", ENDING, rule)

    def record(self, rule: Rule) -> RuleRecord:
        """The game's record, from where it starts and the actions played."""
        return self.rule("the game's record", RECORD, rule)

    def timeout(self, rule: Rule) -> RuleRecord:
        """What a player's clock running out does to the payoffs, without which the game can't be played on a clock."""
        return self.rule("what running out of time does", TIMEOUT, rule)

    def picture(self, rule: Rule) -> RuleRecord:
        """The position as an SVG image, for pages showing games."""
        return self.rule("the position as a picture", PICTURE, rule)

    def inherits(self, context: str, leaving: Collection[tuple[str, str]] = ()) -> int:
        """Makes this context a variant of another: every rule that game holds gains a weight here, at the weight it
        has there, except those left, each named by its kind and its name. What is declared afterwards — heuristics, or
        a rule of its own in place of one left — belongs to the variant alone, so a round of training, an arm and a
        relaxation are all games in their own right."""
        left = set(leaving)
        taken = 0
        for rule in self._knowledge_base.rules(context):
            if rule.relevant(self._context) or (rule.kind, rule.name) in left:
                continue
            self._knowledge_base.declare(
                replace(rule, contexts=(*rule.contexts, (self._context, rule.weight(context))))
            )
            taken += 1
        logger.debug("%s inherits %d rules from %s", self._context, taken, context)
        return taken

    def position(self, name: str, rule: Rule, weight: float) -> RuleRecord:
        """A position heuristic: what a position is worth to `me`, at that weight in this context. Counted, not told:
        it was fitted over many games rather than said by the project."""
        return self._heuristic(name, POSITION, rule, weight)

    def move(self, name: str, rule: Rule, weight: float) -> RuleRecord:
        """A move heuristic: what a move is worth to the player to act, at that weight in this context."""
        return self._heuristic(name, MOVE, rule, weight)

    def _heuristic(self, name: str, kind: str, rule: Rule, weight: float) -> RuleRecord:
        standing = self._standing(name, kind)
        self._declared += 1
        return self._knowledge_base.declare(
            RuleRecord(
                name,
                kind,
                rule,
                Provenance(COUNTED, self._context),
                self._contexts(standing, weight),
                id="" if standing is None else standing.id,
            )
        )

    def _contexts(self, standing: RuleRecord | None, weight: float) -> tuple[tuple[str, float], ...]:
        """The rule's weight here, beside whatever weights it already has elsewhere: declaring a game again changes
        what its rules weigh in it, never what they weigh in the variants, rounds and arms that inherited them."""
        elsewhere = () if standing is None else tuple(item for item in standing.contexts if item[0] != self._context)
        return ((self._context, weight), *elsewhere)

    def done(self) -> str:
        """Says what was declared and gives back the context, so a factory can end on it."""
        logger.info("Declared %d rules of %s", self._declared, self._context)
        return self._context

    def _standing(self, name: str, kind: str) -> RuleRecord | None:
        """The rule already declared under that name and kind for the context, or None where there is none."""
        for rule in self._knowledge_base.rules(self._context, (kind,)):
            if rule.name == name:
                return rule
        return None
