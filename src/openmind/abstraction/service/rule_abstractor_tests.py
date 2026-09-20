from collections.abc import Callable

from openmind.abstraction.factory.abstraction_factory import create_rule_abstractor
from openmind.knowledge.constant.rule_kind_constant import ABSTRACTION
from openmind.knowledge.constant.task_constant import ABSTRACTION as ABSTRACTION_TASK
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system, find_rule_based_system
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.world.model.state import State

type Declared = Callable[..., RuleBasedGame]
type Linked = Callable[..., RuleRecord]

STATE = State.of(turn="X", resources=12, secret="where the mines are")


def _abstraction(linked: Linked, name: str, rule: Rule) -> None:
    linked("a game", name, rule, 1.0, kind=ABSTRACTION, task=ABSTRACTION_TASK)


def test_a_level_sees_the_models_its_abstraction_rules_give_and_none_of_the_rest(
    declared: Declared, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    game = declared(STATE)
    _abstraction(heuristic, "it sees whose turn it is", PythonRule("{'turn': turn}"))
    _abstraction(heuristic, "it sees what it has", PythonRule("{'resources': resources}"))

    seen = create_rule_abstractor().abstract(
        create_rule_based_system(knowledge, "a game", ABSTRACTION_TASK), game.node(STATE), "macromanagement"
    )

    assert seen == State.of(turn="X", resources=12)


def test_a_rule_that_can_t_be_read_here_leaves_the_others_to_say_what_the_level_sees(
    declared: Declared, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    game = declared(STATE)
    _abstraction(heuristic, "it sees what isn't there", PythonRule("{'armies': armies}"))
    _abstraction(heuristic, "it sees whose turn it is", PythonRule("{'turn': turn}"))

    seen = create_rule_abstractor().abstract(
        create_rule_based_system(knowledge, "a game", ABSTRACTION_TASK), game.node(STATE), "macromanagement"
    )

    assert seen == State.of(turn="X")


def test_a_ruleset_whose_rules_all_say_nothing_abstracts_nothing(
    declared: Declared, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    game = declared(STATE)
    _abstraction(heuristic, "it sees what isn't there", PythonRule("{'armies': armies}"))

    seen = create_rule_abstractor().abstract(
        create_rule_based_system(knowledge, "a game", ABSTRACTION_TASK), game.node(STATE), "macromanagement"
    )

    assert seen is None


def test_a_context_with_no_abstraction_ruleset_has_no_rule_based_system_to_abstract_with(
    declared: Declared, knowledge: KnowledgeBase
) -> None:
    declared(STATE)

    assert find_rule_based_system(knowledge, "a game", ABSTRACTION_TASK) is None
