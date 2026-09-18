from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from openmind.agent.factory.game_factory import create_game
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.model.rule import Rule
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value


@pytest.fixture
def knowledge(tmp_path: Path) -> KnowledgeBase:
    """A knowledge base of the test's own, so nothing it declares or remembers outlives it."""
    return create_knowledge_base("tests", tmp_path)


@pytest.fixture
def game(knowledge: KnowledgeBase) -> Callable[[str], RuleBasedSystem]:
    """Declares a game's rules into the test's knowledge base and gives its RBS: `game("tictactoe")`. Every game
    declared shares the one knowledge base, as they would in a running agent."""

    def declared(name: str) -> RuleBasedSystem:
        return create_game(name, knowledge)

    return declared


type Outcomes = Sequence[tuple[float, Rule]]


@pytest.fixture
def declared(knowledge: KnowledgeBase) -> Callable[..., RuleBasedSystem]:
    """Declares a small game of the test's own, rule by rule, as a project would, and gives its RBS.

    `legal` is each action's constraint rules, `outcomes` each action's effects rules with the chance of each, and
    `parameters` each action's parameters with the rule giving their values. `together` is what the players' choices,
    made at once, lead to."""

    def declare(
        state: State,
        legal: Mapping[str, Sequence[Rule]] | None = None,
        outcomes: Mapping[str, Outcomes] | None = None,
        players: Players = Players(("me",), "turn", ("payoff",)),
        parameters: Mapping[str, Mapping[str, Rule]] | None = None,
        definitions: PythonRule | None = None,
        empties: Mapping[str, Value] | None = None,
        together: Outcomes = (),
        context: str = "a game",
        ending: Rule | None = None,
        timeout: Rule | None = None,
    ) -> RuleBasedSystem:
        declarer = RuleDeclarer(knowledge, context)
        declarer.starts_at(state)
        declarer.played_by(players)
        if definitions is not None:
            declarer.definitions(definitions)
            declarer.definitions(definitions, effects=True)
        for base, empty in (empties or {}).items():
            declarer.empty(base, empty)
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
        if timeout is not None:
            declarer.timeout(timeout)
        return create_rule_based_system(knowledge, declarer.done())

    return declare
