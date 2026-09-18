import logging

import pytest

from openmind.doxastic.constant.doxastic_constant import COUNTED, TOLD
from openmind.doxastic.constant.rule_kind_constant import (
    CONSTRAINT,
    DEFINITIONS,
    EFFECTS,
    EMPTY,
    ENDING,
    GAME_KINDS,
    INITIAL,
    MOVE,
    PICTURE,
    PLAYERS,
    POSITION,
    RECORD,
    TIMEOUT,
    VALUES,
)
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.constant.rule_based_constant import EFFECTS_DEFINITIONS, RULES_DEFINITIONS
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.world.model.players import Players
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

SWITCH = State((("light", "off"), ("settings", ("off", "on")), ("payoff", None), ("turn", "me")))


def declare_switch(knowledge: KnowledgeBase, weight: float = 1.0) -> RuleDeclarer:
    """A one-player game declared rule by rule, with a rule of every kind a game can have: a light switched to one of
    the settings the state offers."""
    declarer = RuleDeclarer(knowledge, "switch", weight)
    declarer.starts_at(SWITCH)
    declarer.played_by(Players(("me",), "turn", ("payoff",)))
    declarer.empty("light", "off")
    declarer.definitions(PythonRule("BRIGHT = 'on'"))
    declarer.definitions(PythonRule("DIM = 'off'"), effects=True)
    declarer.values("switch", "setting", PythonRule("[s for s in settings if s != light]"))
    declarer.constraints("switch", PythonRule("setting != light"), PythonRule("payoff['me'] is None"))
    declarer.leads_to("switch", PythonRule("light = setting"), 0.7, 1)
    declarer.leads_to("switch", PythonRule("light = 'off'"), 0.3, 2)
    declarer.ending(PythonRule("'switched' if light == BRIGHT else None"))
    declarer.record(PythonRule("' '.join(str(a) for a in actions)"))
    declarer.timeout(PythonRule("payoff['me'] = 0.0"))
    declarer.picture(PythonRule("'<svg/>'"))
    return declarer


def test_every_rule_is_declared_under_the_context_as_the_project_told_it(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    rules = knowledge.rules("switch")

    assert rules
    assert {rule.kind for rule in rules} <= set(GAME_KINDS)
    assert all(rule.provenance.source == TOLD and rule.provenance.told == "switch" for rule in rules)
    assert all(rule.contexts == (("switch", 1.0),) for rule in rules)


def test_a_rule_of_every_kind_is_declared_as_the_kind_it_is(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    assert [rule.kind for rule in knowledge.rules("switch")] == [
        INITIAL,
        PLAYERS,
        EMPTY,
        DEFINITIONS,
        DEFINITIONS,
        VALUES,
        CONSTRAINT,
        CONSTRAINT,
        EFFECTS,
        EFFECTS,
        ENDING,
        RECORD,
        TIMEOUT,
        PICTURE,
    ]


def test_the_two_definitions_scripts_are_told_apart_by_name(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    scripts = {rule.name: rule.rule for rule in knowledge.rules("switch", (DEFINITIONS,))}

    assert scripts[RULES_DEFINITIONS] == PythonRule("BRIGHT = 'on'")
    assert scripts[EFFECTS_DEFINITIONS] == PythonRule("DIM = 'off'")


def test_a_constraint_says_which_action_it_makes_legal_and_a_values_rule_its_parameter(
    knowledge: KnowledgeBase,
) -> None:
    declare_switch(knowledge).done()

    ((values,),) = (knowledge.rules("switch", (VALUES,)),)

    assert [rule.action for rule in knowledge.rules("switch", (CONSTRAINT,))] == ["switch", "switch"]
    assert (values.action, values.parameter, values.name) == ("switch", "setting", "what setting can be in switch")


def test_an_action_with_several_outcomes_declares_each_with_its_chance(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    outcomes = knowledge.rules("switch", (EFFECTS,))

    assert [rule.name for rule in outcomes] == ["what switch leads to, 1", "what switch leads to, 2"]
    assert sorted(rule.probability for rule in outcomes) == [0.3, 0.7]


def test_declaring_the_same_game_again_leaves_the_knowledge_base_as_it_was(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    first = knowledge.rules("switch")

    declare_switch(knowledge).done()

    assert [rule.id for rule in knowledge.rules("switch")] == [rule.id for rule in first]


def test_a_context_declared_at_another_weight_carries_it(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge, weight=0.6).done()

    assert all(rule.weight("switch") == 0.6 for rule in knowledge.rules("switch"))


def test_a_heuristic_is_counted_rather_than_told(knowledge: KnowledgeBase) -> None:
    declarer = RuleDeclarer(knowledge, "switch")

    declarer.position("the light is on", PythonRule("light == 'on'"), 0.8)
    declarer.move("switching costs nothing", PythonRule("1.0"), -0.2)

    ((position,), (move,)) = (knowledge.rules("switch", (POSITION,)), knowledge.rules("switch", (MOVE,)))
    assert (position.provenance.source, position.weight("switch")) == (COUNTED, 0.8)
    assert (move.kind, move.weight("switch")) == (MOVE, -0.2)


def test_what_was_declared_is_logged(caplog: pytest.LogCaptureFixture, knowledge: KnowledgeBase) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")

    declare_switch(knowledge).done()

    assert any("Declared 14 rules of switch" == message for message in caplog.messages)


def test_declaring_the_game_again_leaves_its_variants_whole(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    variant = RuleDeclarer(knowledge, "switch at night")
    variant.inherits("switch")

    declare_switch(knowledge).done()

    assert [rule.name for rule in knowledge.rules("switch at night")] == [rule.name for rule in knowledge.rules("switch")]


def test_a_variant_can_leave_out_a_rule_and_declare_its_own_in_its_place(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    variant = RuleDeclarer(knowledge, "switch anywhere")

    variant.inherits("switch", leaving=((VALUES, "what setting can be in switch"),))
    variant.values("switch", "setting", PythonRule("('off', 'on', 'dim')"))

    ((own,),) = (knowledge.rules("switch anywhere", (VALUES,)),)
    ((base,),) = (knowledge.rules("switch", (VALUES,)),)
    assert own.rule == PythonRule("('off', 'on', 'dim')")
    assert base.rule == PythonRule("[s for s in settings if s != light]")
    assert own.id != base.id
