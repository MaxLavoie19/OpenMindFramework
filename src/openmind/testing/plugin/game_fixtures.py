from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from openmind.rbs.factory.rbs_factory import create_game
from openmind.game.service.game_declarer import GameDeclarer
from openmind.knowledge.constant.knowledge_constant import INFERENCE, MOVE_VALUE, POSITION_VALUE, SIMULATION
from openmind.knowledge.constant.rule_kind_constant import MOVE, POSITION
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.source import Source
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.players import Players
from openmind.world.model.state import State


def simulation_rules(knowledge: KnowledgeBase, context: str, kinds: tuple[str, ...] = ()) -> list[RuleRecord]:
    """The rules of a context's simulation ruleset, of those kinds or of any kind, in the order they were linked."""
    known = knowledge.context_named(context)
    ruleset = None if known is None else knowledge.ruleset_named(known.id, SIMULATION)
    return [] if ruleset is None else [rule for rule, _ in knowledge.ruleset_rules(ruleset.id, kinds)]


@pytest.fixture
def knowledge(tmp_path: Path) -> KnowledgeBase:
    """A knowledge base of the test's own, so nothing it declares or remembers outlives it."""
    return create_knowledge_base("tests", tmp_path)


@pytest.fixture
def game(knowledge: KnowledgeBase) -> Callable[[str], RuleBasedGame]:
    """Declares a game's rules into the test's knowledge base and gives its RBS: `game("tictactoe")`. Every game
    declared shares the one knowledge base, as they would in a running agent."""

    def declared(name: str) -> RuleBasedGame:
        return create_game(name, knowledge)

    return declared


type Outcomes = Sequence[tuple[float, Rule]]


@pytest.fixture
def declared(knowledge: KnowledgeBase) -> Callable[..., RuleBasedGame]:
    """Declares a small game of the test's own, rule by rule, as a project would, and gives its RBS.

    `legal` is each action's constraint rules, `outcomes` each action's effects rules with the chance of each, and
    `parameters` each action's parameters with the rule giving their values. `together` is what the players' choices,
    made at once, lead to."""

    def declare(
        state: State,
        legal: Mapping[str, Sequence[Rule]] | None = None,
        outcomes: Mapping[str, Outcomes] | None = None,
        players: Players = Players(("me",), "payoff"),
        parameters: Mapping[str, Mapping[str, Rule]] | None = None,
        definitions: PythonRule | None = None,
        together: Outcomes = (),
        context: str = "a game",
        ending: Rule | None = None,
    ) -> RuleBasedGame:
        declarer = GameDeclarer(knowledge, context)
        declarer.starts_at(state)
        declarer.played_by(players)
        if definitions is not None:
            declarer.definitions(definitions)
            declarer.definitions(definitions, effects=True)
        for action, named in (parameters or {}).items():
            for parameter, rule in named.items():
                declarer.values(action, parameter, rule)
        for action, constraints in (legal or {}).items():
            declarer.constraints(action, *constraints)
        for action, branches in (outcomes or {}).items():
            for number, (chance, rule) in enumerate(branches, start=1):
                declarer.leads_to(action, rule, chance, number if len(branches) > 1 else None)
        for number, (chance, rule) in enumerate(together, start=1):
            declarer.together(rule, chance, number if len(together) > 1 else None)
        if ending is not None:
            declarer.ending(ending)
        return create_rule_based_game(knowledge, declarer.done())

    return declare


@pytest.fixture
def heuristic(knowledge: KnowledgeBase) -> Callable[..., RuleRecord]:
    """Links a heuristic rule into a context's position value ruleset, or its move value ruleset for a move rule, at
    that weight, as a producer of heuristics does: `heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)`. A
    rule of the same name already there is declared anew and its weight set anew."""

    def link(context: str, name: str, rule: Rule, weight: float, kind: str = POSITION) -> RuleRecord:
        task = MOVE_VALUE if kind == MOVE else POSITION_VALUE
        context_id = knowledge.ensure_context(context).id
        source = Source(knowledge.ensure_mechanism(INFERENCE).id, (("method", "fit"),))
        ruleset = knowledge.ruleset_named(context_id, task) or knowledge.ruleset(Ruleset(task, context_id, task, source))
        standing = next((held for held, _ in knowledge.ruleset_rules(ruleset.id) if held.name == name), None)
        declared = knowledge.declare(RuleRecord(name, kind, rule, source, id="" if standing is None else standing.id))
        knowledge.link(ruleset.id, declared.id, weight)
        return declared

    return link
