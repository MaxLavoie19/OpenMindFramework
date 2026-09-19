import logging

import pytest

from openmind.game.service.game_declarer import GameDeclarer
from openmind.knowledge.constant.knowledge_constant import DECLARATION
from openmind.knowledge.constant.task_constant import SIMULATION
from openmind.knowledge.constant.rule_kind_constant import (
    CONSTRAINT,
    COOLDOWN,
    DEFINITIONS,
    DURATION,
    EFFECTS,
    ENDING,
    GAME_KINDS,
    INITIAL,
    PICTURE,
    PLAYERS,
    RECORD,
    VALUES,
)
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.constant.rule_constant import EFFECTS_DEFINITIONS, RULES_DEFINITIONS
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.list import List
from openmind.world.model.players import Players
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

SWITCH = State.of(light="off", settings=List(("off", "on")), payoff=None)


def declare_switch(knowledge: KnowledgeBase, open: bool = False) -> GameDeclarer:
    """A one-player game declared rule by rule, with a rule of every kind a game can have: a light switched to one of
    the settings the state offers."""
    declarer = GameDeclarer(knowledge, "switch", open=open)
    declarer.starts_at(SWITCH)
    declarer.played_by(Players(("me",), "payoff"))
    declarer.definitions(PythonRule("BRIGHT = 'on'"))
    declarer.definitions(PythonRule("DIM = 'off'"), effects=True)
    declarer.values("switch", "setting", PythonRule("[s for s in settings if s != light]"))
    declarer.constraints("switch", PythonRule("setting != light"), PythonRule("payoff is None"))
    declarer.leads_to("switch", PythonRule("light = setting"), 0.7, 1)
    declarer.leads_to("switch", PythonRule("light = 'off'"), 0.3, 2)
    declarer.ending(PythonRule("'switched' if light == BRIGHT else None"))
    declarer.lasts("switch", PythonRule("0.5"))
    declarer.cools_down("switch", PythonRule("2.0"))
    declarer.record(PythonRule("' '.join(str(a) for a in actions)"))
    declarer.picture(PythonRule("'<svg/>'"))
    return declarer


def rules(knowledge: KnowledgeBase, context: str = "switch", kinds: tuple[str, ...] = ()) -> list[RuleRecord]:
    ruleset = knowledge.ruleset_named(knowledge.context_named(context).id, SIMULATION)  # type: ignore[union-attr]
    return [rule for rule, _ in knowledge.ruleset_rules(ruleset.id, kinds)]  # type: ignore[union-attr]


def test_every_rule_is_declared_into_the_context_s_simulation_ruleset_frozen(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    ruleset = knowledge.ruleset_named(knowledge.context_named("switch").id, SIMULATION)  # type: ignore[union-attr]
    declared = rules(knowledge)
    assert ruleset is not None and ruleset.task == SIMULATION and knowledge.frozen_ruleset(ruleset)
    assert {rule.kind for rule in declared} <= set(GAME_KINDS)
    declaration = knowledge.mechanism_named(DECLARATION).id  # type: ignore[union-attr]
    assert all(rule.source.mechanism == declaration and rule.source.parameter("context") == "switch" for rule in declared)
    assert all(knowledge.frozen(rule) for rule in declared)


def test_a_rule_of_every_kind_is_declared_as_the_kind_it_is(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    assert [rule.kind for rule in rules(knowledge)] == [
        INITIAL,
        PLAYERS,
        DEFINITIONS,
        DEFINITIONS,
        VALUES,
        CONSTRAINT,
        CONSTRAINT,
        EFFECTS,
        EFFECTS,
        ENDING,
        DURATION,
        COOLDOWN,
        RECORD,
        PICTURE,
    ]


def test_the_two_definitions_scripts_are_told_apart_by_name(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    scripts = {rule.name: rule.rule for rule in rules(knowledge, kinds=(DEFINITIONS,))}

    assert scripts[RULES_DEFINITIONS] == PythonRule("BRIGHT = 'on'")
    assert scripts[EFFECTS_DEFINITIONS] == PythonRule("DIM = 'off'")


def test_a_constraint_says_which_action_it_makes_legal_and_a_values_rule_its_parameter(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    (values,) = rules(knowledge, kinds=(VALUES,))

    assert [rule.action for rule in rules(knowledge, kinds=(CONSTRAINT,))] == ["switch", "switch"]
    assert (values.action, values.parameter, values.name) == ("switch", "setting", "what setting can be in switch")


def test_an_action_with_several_outcomes_declares_each_with_its_chance(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    outcomes = rules(knowledge, kinds=(EFFECTS,))

    assert [rule.name for rule in outcomes] == ["what switch leads to, 1", "what switch leads to, 2"]
    assert sorted(rule.probability for rule in outcomes) == [0.3, 0.7]


def test_a_duration_and_a_cooldown_are_declared_for_their_action(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    ((lasting,), (cooling,)) = rules(knowledge, kinds=(DURATION,)), rules(knowledge, kinds=(COOLDOWN,))

    assert (lasting.action, lasting.rule) == ("switch", PythonRule("0.5"))
    assert (cooling.action, cooling.rule) == ("switch", PythonRule("2.0"))


def test_declaring_the_same_game_again_leaves_the_knowledge_base_as_it_was(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    first = rules(knowledge)

    declare_switch(knowledge).done()

    assert [rule.id for rule in rules(knowledge)] == [rule.id for rule in first]
    assert len(knowledge.rulesets(knowledge.context_named("switch").id)) == 1  # type: ignore[union-attr]


def test_a_ruleset_declared_open_is_open_and_its_rules_can_be_open_too(knowledge: KnowledgeBase) -> None:
    declarer = declare_switch(knowledge, open=True)

    house = declarer.rule("house rule", CONSTRAINT, PythonRule("True"), "switch", open=True)

    assert not knowledge.frozen_ruleset(declarer.ruleset)
    assert not knowledge.frozen(house) and house.id in declarer.ruleset.rule_ids


def test_an_open_rule_is_refused_in_a_frozen_ruleset_with_a_warning(
    caplog: pytest.LogCaptureFixture, knowledge: KnowledgeBase
) -> None:
    declarer = declare_switch(knowledge)

    house = declarer.rule("house rule", CONSTRAINT, PythonRule("True"), "switch", open=True)

    assert house.id not in declarer.ruleset.rule_ids
    assert any(record.levelname == "WARNING" and "is frozen" in record.getMessage() for record in caplog.records)


def test_what_was_declared_is_logged(caplog: pytest.LogCaptureFixture, knowledge: KnowledgeBase) -> None:
    caplog.set_level(logging.INFO, logger="openmind.game")

    declare_switch(knowledge).done()

    assert any(message.startswith("Declared 14 rules of switch into ruleset simulation (ruleset-") for message in caplog.messages)


def test_a_variant_lists_its_game_s_rules_and_its_context_records_the_link(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()

    GameDeclarer(knowledge, "switch at night").variant_of("switch")

    night = knowledge.context_named("switch at night")
    assert night is not None and night.inherits == (knowledge.context_named("switch").id,)  # type: ignore[union-attr]
    assert [rule.id for rule in rules(knowledge, "switch at night")] == [rule.id for rule in rules(knowledge)]


def test_a_variant_can_leave_out_a_rule_and_declare_its_own_in_its_place(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    variant = GameDeclarer(knowledge, "switch anywhere")

    variant.variant_of("switch", leaving=((VALUES, "what setting can be in switch"),))
    variant.values("switch", "setting", PythonRule("('off', 'on', 'dim')"))

    (own,) = rules(knowledge, "switch anywhere", (VALUES,))
    (base,) = rules(knowledge, kinds=(VALUES,))
    assert own.rule == PythonRule("('off', 'on', 'dim')")
    assert base.rule == PythonRule("[s for s in settings if s != light]")
    assert own.id != base.id


def test_declaring_the_game_again_leaves_its_variants_whole(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    GameDeclarer(knowledge, "switch at night").variant_of("switch")

    declare_switch(knowledge).done()

    assert [rule.id for rule in rules(knowledge, "switch at night")] == [rule.id for rule in rules(knowledge)]


def test_a_variant_declaring_a_rule_it_took_from_its_game_declares_its_own_in_its_place(knowledge: KnowledgeBase) -> None:
    declare_switch(knowledge).done()
    variant = GameDeclarer(knowledge, "switch alone")
    variant.variant_of("switch")

    variant.played_by(Players(("me",), "score"))

    (own,) = rules(knowledge, "switch alone", (PLAYERS,))
    (base,) = rules(knowledge, kinds=(PLAYERS,))
    assert own.id != base.id
    assert base.rule == PythonRule(repr((("me",), "payoff")))
