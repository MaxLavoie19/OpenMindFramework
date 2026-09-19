import logging
from collections.abc import Collection
from dataclasses import replace

from openmind.knowledge.constant.knowledge_constant import DECLARATION, SIMULATION
from openmind.knowledge.constant.rule_kind_constant import (
    CONSTRAINT,
    COOLDOWN,
    DEFINITIONS,
    DURATION,
    EFFECTS,
    ENDING,
    INITIAL,
    PICTURE,
    PLAYERS,
    RECORD,
    VALUES,
)
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.ruleset_link import RulesetLink
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.constant.rule_constant import EFFECTS_DEFINITIONS, RULES_DEFINITIONS
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.players import Players
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: What a rule that always holds carries as its chance.
CERTAIN = 1.0


class GameDeclarer:
    """How a game project writes its game into the knowledge base: rule by rule, into the simulation ruleset of the
    game's context. OMF only simulates the game; the integrator runs it.

    The rules and the ruleset are frozen unless declared open: an application's rules are never revised by OMF, which
    warns instead where an observation conflicts with one. A rule already declared under the same name and kind in the
    ruleset is declared anew rather than added beside it, so declaring a game twice leaves the knowledge base as it
    was.

    OMF knows nothing of turns, phases, draws, clocks or boards: they are the game's own variables, actions and rules.
    All players play at the same time, all the time; in a game played in turns, the constraints leave a player no
    action outside their turn."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        context: str,
        ruleset: str = SIMULATION,
        open: bool = False,
        rule_caller: RuleCaller | None = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._context = context
        self._context_id = knowledge_base.ensure_context(context).id
        self._open = open
        self._caller = rule_caller
        self._declared = 0
        mechanism = knowledge_base.ensure_mechanism(DECLARATION).id
        standing = knowledge_base.ruleset_named(self._context_id, ruleset)
        self._ruleset = (
            standing
            if standing is not None and standing.open == open
            else knowledge_base.ruleset(
                Ruleset(
                    ruleset,
                    self._context_id,
                    SIMULATION,
                    Source(mechanism, (("context", context),)),
                    () if standing is None else standing.links,
                    open,
                    id="" if standing is None else standing.id,
                )
            )
        )

    @property
    def context(self) -> str:
        """The context the rules are declared under: a game, or a variant of one."""
        return self._context

    @property
    def ruleset(self) -> Ruleset:
        """The simulation ruleset the rules are declared into, as it stands now."""
        return self._ruleset

    def rule(
        self,
        name: str,
        kind: str,
        rule: Rule,
        action: str | None = None,
        parameter: str | None = None,
        probability: float = CERTAIN,
        open: bool = False,
    ) -> RuleRecord:
        """Declares one rule of the game into the ruleset, frozen unless open. A rule written as a function no worker
        process could find is refused."""
        if self._caller is not None:
            self._caller.check(rule)
        standing = self._standing(name, kind)
        if standing is not None and standing.source.parameter("context") != self._context:
            self._ruleset = self._knowledge_base.unlink(self._ruleset.id, standing.id)
            standing = None
        self._declared += 1
        declared = self._knowledge_base.declare(
            RuleRecord(
                name,
                kind,
                rule,
                Source(self._knowledge_base.ensure_mechanism(DECLARATION).id, (("context", self._context),)),
                action,
                parameter,
                probability,
                open,
                id="" if standing is None else standing.id,
            )
        )
        if standing is None:
            self._ruleset = self._knowledge_base.link(self._ruleset.id, declared.id)
        return declared

    def starts_at(self, state: State) -> RuleRecord:
        """Where the game starts."""
        return self.rule("where the game starts", INITIAL, PythonRule(repr(state.models)))

    def played_by(self, players: Players) -> RuleRecord:
        """Who plays, and the Map holding each player's payoff, which an end state fills."""
        return self.rule("who plays", PLAYERS, PythonRule(repr((players.names, players.payoff))))

    def definitions(self, rule: PythonRule, effects: bool = False) -> RuleRecord:
        """The script whose names the game's rules see, or the one its effects see."""
        return self.rule(EFFECTS_DEFINITIONS if effects else RULES_DEFINITIONS, DEFINITIONS, rule)

    def values(self, action: str, parameter: str, rule: Rule) -> RuleRecord:
        """The values that parameter of the action can take: a rule reading the state, or one reading nothing where
        they are the same in every state."""
        return self.rule(f"what {parameter} can be in {action}", VALUES, rule, action, parameter)

    def constraint(self, action: str, number: int, rule: Rule) -> RuleRecord:
        """One of the rules every legal action of that name satisfies. The constraints read `player`, the player the
        legal actions are solved for, which is how a game played in turns leaves a player no action outside their
        turn."""
        return self.rule(f"{action} is legal, {number}", CONSTRAINT, rule, action)

    def constraints(self, action: str, *rules: Rule) -> tuple[RuleRecord, ...]:
        """Every rule a legal action of that name satisfies, numbered in the order they are given."""
        return tuple(self.constraint(action, number, rule) for number, rule in enumerate(rules, start=1))

    def leads_to(self, action: str, rule: Rule, probability: float = CERTAIN, number: int | None = None) -> RuleRecord:
        """What the action leads to, and how often, where it has more than one outcome."""
        name = f"what {action} leads to"
        return self.rule(name if number is None else f"{name}, {number}", EFFECTS, rule, action, probability=probability)

    def together(self, rule: Rule, probability: float = CERTAIN, number: int | None = None) -> RuleRecord:
        """What the players' actions, taken at once, lead to together, after each one's own effects."""
        name = "what the players' actions together lead to"
        return self.rule(name if number is None else f"{name}, {number}", EFFECTS, rule, probability=probability)

    def ending(self, rule: Rule) -> RuleRecord:
        """Whether the game is over, and why: a reason, or None while it goes on."""
        return self.rule("why the game ended", ENDING, rule)

    def lasts(self, action: str, rule: Rule) -> RuleRecord:
        """How long the action takes to perform, in seconds, from the state and its parameters."""
        return self.rule(f"how long {action} takes", DURATION, rule, action)

    def cools_down(self, action: str, rule: Rule) -> RuleRecord:
        """How long before the action is available again, in seconds, from the state and its parameters."""
        return self.rule(f"how long before {action} is available again", COOLDOWN, rule, action)

    def record(self, rule: Rule) -> RuleRecord:
        """The game's record, from where it starts and the actions played. An encoder of the game rather than a rule of
        its simulation, it moves to the codec step."""
        return self.rule("the game's record", RECORD, rule)

    def picture(self, rule: Rule) -> RuleRecord:
        """The state as an SVG image, for pages showing games. An encoder of the game, it moves to the codec step."""
        return self.rule("the state as a picture", PICTURE, rule)

    def variant_of(self, game: str, leaving: Collection[tuple[str, str]] = ()) -> Ruleset:
        """Makes this context a variant of another game: its simulation ruleset lists every rule of the game's, except
        those left, each named by its kind and its name. What is declared afterwards belongs to the variant alone.
        `Context.inherits` records where it came from."""
        left = set(leaving)
        game_id = self._knowledge_base.ensure_context(game).id
        original = self._knowledge_base.ruleset_named(game_id, self._ruleset.name)
        if original is None:
            raise ValueError(f"{game} has no {self._ruleset.name} ruleset to make a variant of")
        present = set(self._ruleset.rule_ids)
        taken = tuple(
            RulesetLink(rule.id, weight)
            for rule, weight in self._knowledge_base.ruleset_rules(original.id)
            if rule.id not in present and (rule.kind, rule.name) not in left
        )
        self._ruleset = self._knowledge_base.ruleset(replace(self._ruleset, links=(*self._ruleset.links, *taken)))
        known = self._knowledge_base.context_by_id(self._context_id) or self._knowledge_base.ensure_context(self._context)
        if game_id not in known.inherits:
            self._knowledge_base.context(replace(known, inherits=(*known.inherits, game_id)))
        logger.debug("%s is a variant of %s: it takes %d of its rules", self._context, game, len(taken))
        return self._ruleset

    def done(self) -> str:
        """Says what was declared and gives back the context, so a declaring function can end on it."""
        logger.info(
            "Declared %d rules of %s into ruleset %s%s",
            self._declared,
            self._context,
            self._knowledge_base.readable_ruleset(self._ruleset.id),
            ", open" if self._open else "",
        )
        return self._context

    def _standing(self, name: str, kind: str) -> RuleRecord | None:
        """The rule already declared under that name and kind in the ruleset, or None where there is none."""
        for rule, _ in self._knowledge_base.ruleset_rules(self._ruleset.id, (kind,)):
            if rule.name == name:
                return rule
        return None
