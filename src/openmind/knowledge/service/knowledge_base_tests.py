import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from openmind.knowledge.constant.knowledge_constant import DECLARATION, DIRECT_EXPERIENCE, DONE, INFERENCE, PENDING
from openmind.knowledge.constant.task_constant import POSITION_VALUE, SIMULATION
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, MOVE, POSITION
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.context import Context
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.opinion import Opinion
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.ruleset_link import RulesetLink
from openmind.knowledge.model.source import Source
from openmind.knowledge.model.task import Task
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule

pytestmark = pytest.mark.log_level("INFO")

MICROPHONE = Source("microphone", (("device", "table mic"),))


def new_base(tmp_path: Path) -> KnowledgeBase:
    return create_knowledge_base("cheat", tmp_path)


def said(text: str, *tags: tuple[str, object]) -> DirectExperience:
    return DirectExperience("heard", "cheat", text, MICROPHONE, tags=tags)  # type: ignore[arg-type]


def test_a_direct_experience_is_kept_word_for_word_with_its_source_an_id_and_a_time(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    kept = base.experience(said("three kings", ("keyword", "claim")))

    assert kept.id.startswith("experience-") and (kept.value, kept.source) == ("three kings", MICROPHONE)
    assert kept.at is not None
    assert base.experienced(kept.id) == kept


def test_direct_experiences_are_retrieved_by_context_and_by_every_tag_asked_for(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.experience(said("three kings", ("keyword", "claim"), ("player", "black")))
    base.experience(said("two queens", ("keyword", "claim"), ("player", "white")))
    base.experience(DirectExperience("seen", "chess", "e4", Source("camera")))

    assert [kept.value for kept in base.experiences(tags=(("keyword", "claim"),))] == ["three kings", "two queens"]
    assert [kept.value for kept in base.experiences(tags=(("keyword", "claim"), ("player", "white")))] == ["two queens"]
    assert [kept.value for kept in base.experiences("chess")] == ["e4"]


def test_a_belief_holds_any_value_at_full_certainty_unless_said_otherwise_and_needs_no_evidence(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    count = base.believe(Belief("kings played", "cheat", 3))
    bluffing = base.believe(Belief("black is bluffing", "cheat", True, certainty=0.6))

    assert count.id.startswith("belief-") and (count.value, count.certainty, count.evidence) == (3, 1.0, ())
    assert bluffing.id != count.id
    assert (bluffing.value, bluffing.certainty) == (True, 0.6)


def test_believing_the_same_variable_again_updates_the_belief_under_its_id(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    first = base.believe(Belief("kings played", "cheat", 3))

    again = base.believe(Belief("kings played", "cheat", 4))

    assert again.id == first.id
    assert base.belief("kings played", "cheat") == again
    assert base.beliefs("cheat") == (again,)


def test_what_another_player_believes_is_a_belief_of_its_own(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    own = base.believe(Belief("black is bluffing", "cheat", True))
    theirs = base.believe(Belief("black is bluffing", "cheat", False, holder=("white",)))

    assert own.id != theirs.id
    assert base.belief("black is bluffing", "cheat", ("white",)) == theirs
    assert base.beliefs("cheat", holder=()) == (own,)


def test_the_evidence_for_the_value_held_and_the_evidence_for_other_values_are_kept_apart(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    heard = base.experience(said("three kings"))
    counted = Evidence(True, 0.8, Source(base.ensure_mechanism(INFERENCE).id, (("method", "counting"), ("kings seen", 2))))
    claimed = Evidence(False, 0.3, Source(base.ensure_mechanism(DIRECT_EXPERIENCE).id, rests_on=(heard.id,)))

    belief = base.believe(Belief("black is bluffing", "cheat", True, certainty=0.8, evidence=(counted, claimed)))

    assert (belief.supporting, belief.opposing) == ((counted,), (claimed,))


def test_beliefs_are_retrieved_by_every_tag_asked_for(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    base.believe(Belief("black is bluffing", "cheat", True, tags=(("topic", "bluffs"), ("player", "black"))))
    base.believe(Belief("white is bluffing", "cheat", False, tags=(("topic", "bluffs"), ("player", "white"))))

    assert [belief.variable for belief in base.beliefs(tags=(("topic", "bluffs"),))] == ["black is bluffing", "white is bluffing"]
    assert [belief.variable for belief in base.beliefs(tags=(("player", "white"),))] == ["white is bluffing"]


def test_an_opinion_is_the_agent_s_own_with_its_reasons(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    cold = base.hold(Opinion("the cold", "life", "don't like"))
    winter_is_cold = base.believe(Belief("winter is cold", "life", True))

    winter = base.hold(
        Opinion("winter", "life", "don't like", reasons=(Source("reasoning", rests_on=(cold.id, winter_is_cold.id)),))
    )

    assert winter.id.startswith("opinion-") and winter.value == "don't like"
    assert winter.reasons[0].rests_on == (cold.id, winter_is_cold.id)
    assert base.opinion("winter", "life") == winter
    assert base.hold(replace(winter, value="like", id="")).id == winter.id


def test_a_task_is_kept_with_its_value_and_expected_time_and_retrieved_by_status(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    worth = Belief("utility gained", "cheat", 0.4)
    time = Belief("expected time", "cheat", 120.0)

    task = base.task(Task("review game 12", "cheat", (worth,), time))
    base.task(replace(task, status=DONE))
    base.task(Task("study bluffs", "cheat", (worth,), time))

    assert task.id.startswith("task-")
    assert [kept.name for kept in base.tasks(status=PENDING)] == ["study bluffs"]
    assert [kept.name for kept in base.tasks(status=DONE)] == ["review game 12"]


def test_a_context_has_a_parent_level_and_contexts_it_inherits_from(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    life = base.ensure_context("life")
    chess = base.ensure_context("chess", life.id)
    variant = base.context(Context("context-960", "chess/960", life.id, (chess.id,)))

    assert chess.id.startswith("context-") and base.ensure_context("chess") == chess
    assert base.context_named("chess/960") == variant and base.context_by_id(variant.id) == variant
    assert [context.name for context in base.contexts()] == ["life", "chess", "chess/960"]
    assert base.readable_context(chess.id) == f"chess ({chess.id})"


def test_a_mechanism_has_a_name_and_an_id_and_is_registered_the_first_time_it_is_named(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    reader = base.ensure_mechanism("fork detector", 0.9)

    assert reader.id.startswith("mechanism-") and reader.declared_accuracy == 0.9
    assert base.ensure_mechanism("fork detector") == reader
    assert base.mechanism_named("fork detector") == reader and base.mechanism_by_id(reader.id) == reader
    assert new_base(tmp_path).mechanism_named("fork detector") == reader


def test_everything_kept_is_still_there_for_a_base_opened_again_and_ids_carry_on(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.knowledge")
    base = new_base(tmp_path)
    heard = base.experience(said("three kings"))
    belief = base.believe(Belief("kings played", "cheat", 3, tags=(("topic", "counts"),)))
    opinion = base.hold(Opinion("the table", "cheat", "noisy", at=datetime(2026, 9, 18, 12, 0)))
    task = base.task(Task("study bluffs", "cheat", (), Belief("expected time", "cheat", 60.0)))
    variant = base.context(Context("context-two-decks", "cheat/two decks", inherits=("context-cheat",)))

    opened = new_base(tmp_path)

    assert opened.experienced(heard.id) == heard
    assert opened.belief("kings played", "cheat") == belief
    assert opened.opinion("the table", "cheat") == opinion
    assert opened.tasks() == (task,)
    assert opened.contexts() == (variant,)
    assert any(
        "Knowledge of cheat: 1 direct experiences, 1 beliefs, 1 opinions, 1 tasks, 1 contexts, 0 mechanisms, 0 rules, "
        "0 rulesets, 0 models"
        == message
        for message in caplog.messages
    )


def new_rule(
    name: str, kind: str = POSITION, source: str = "True", mechanism: str = "mechanism-inference", open_: bool = False
) -> RuleRecord:
    return RuleRecord(name, kind, PythonRule(source), Source(mechanism), open=open_)


def new_ruleset(base: KnowledgeBase, name: str, *rules: RuleRecord, task: str = POSITION_VALUE, declared: bool = False) -> Ruleset:
    mechanism = base.ensure_mechanism(DECLARATION if declared else INFERENCE).id
    context = base.ensure_context("cheat").id
    return base.ruleset(Ruleset(name, context, task, Source(mechanism), tuple(RulesetLink(rule.id) for rule in rules)))


def test_a_declared_rule_is_given_an_id_and_a_time_and_is_found_again_by_it(tmp_path: Path) -> None:
    base = new_base(tmp_path)

    declared = base.declare(new_rule("a player holding no spade never leads one"))

    assert declared.id.startswith("rule-")
    assert declared.source.at is not None
    assert base.rule(declared.id) == declared
    assert base.rule("rule-unknown") is None


def test_a_ruleset_belongs_to_a_context_and_lists_its_rules_each_with_its_weight_there(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    mobility = base.declare(new_rule("more moves is better placed", POSITION, "1"))
    material = base.declare(new_rule("more cards is worse placed", POSITION, "2"))
    ruleset = new_ruleset(base, "position value")

    base.link(ruleset.id, mobility.id, 0.3)
    base.link(ruleset.id, material.id, 0.9)

    assert ruleset.id.startswith("ruleset-")
    assert base.ruleset_named(base.ensure_context("cheat").id, "position value").id == ruleset.id  # type: ignore[union-attr]
    assert [(rule.name, weight) for rule, weight in base.ruleset_rules(ruleset.id)] == [
        ("more moves is better placed", 0.3),
        ("more cards is worse placed", 0.9),
    ]


def test_the_same_rule_weighs_differently_in_each_ruleset_listing_it(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    mobility = base.declare(new_rule("more moves is better placed", POSITION, "1"))
    charge = new_ruleset(base, "charge")
    retreat = new_ruleset(base, "retreat")

    base.link(charge.id, mobility.id, 0.2)
    base.link(retreat.id, mobility.id, 0.8)

    assert base.ruleset_rules(charge.id) == ((mobility, 0.2),)
    assert base.ruleset_rules(retreat.id) == ((mobility, 0.8),)


def test_linking_a_listed_rule_again_changes_only_its_weight(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    mobility = base.declare(new_rule("more moves is better placed", POSITION, "1"))
    ruleset = new_ruleset(base, "position value", mobility)

    base.link(ruleset.id, mobility.id, 0.9)

    assert base.ruleset_rules(ruleset.id) == ((mobility, 0.9),)


def test_a_ruleset_s_rules_are_retrieved_by_kind_and_by_tag_where_asked_for(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    constraint = base.declare(new_rule("a cell is played only when it is empty", CONSTRAINT))
    mobility = base.declare(replace(new_rule("more moves is better placed", POSITION), tags=(("topic", "mobility"),)))
    move = base.declare(new_rule("a move taking a piece is worth looking at", MOVE))
    ruleset = new_ruleset(base, "everything", constraint, mobility, move)

    assert [rule.kind for rule, _ in base.ruleset_rules(ruleset.id, (CONSTRAINT,))] == [CONSTRAINT]
    assert {rule.kind for rule, _ in base.ruleset_rules(ruleset.id, (POSITION, MOVE))} == {POSITION, MOVE}
    assert [rule.name for rule, _ in base.ruleset_rules(ruleset.id, tags=(("topic", "mobility"),))] == [mobility.name]


def test_rulesets_are_retrieved_by_context_and_by_task(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    simulation = new_ruleset(base, "simulation", task=SIMULATION)
    position = new_ruleset(base, "position value")

    assert base.rulesets(base.ensure_context("cheat").id) == (simulation, position)
    assert base.rulesets(task=SIMULATION) == (simulation,)
    assert base.rulesets(base.ensure_context("chess").id) == ()


def test_a_rule_an_application_declared_is_frozen_and_its_revision_refused_with_a_warning(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    base = new_base(tmp_path)
    declared = base.declare(
        new_rule("a pawn takes diagonally", CONSTRAINT, "True", mechanism=base.ensure_mechanism(DECLARATION).id)
    )

    kept = base.revise(declared.id, replace(declared, rule=PythonRule("False")))

    assert kept == declared and base.rule(declared.id) == declared
    assert any(
        record.levelname == "WARNING" and "is frozen" in record.getMessage() for record in caplog.records
    )


def test_a_rule_declared_open_or_produced_by_omf_can_be_revised(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    opened = base.declare(
        new_rule("house rule", CONSTRAINT, "True", mechanism=base.ensure_mechanism(DECLARATION).id, open_=True)
    )
    inferred = base.declare(new_rule("mobility counts", POSITION, "1"))

    base.revise(opened.id, replace(opened, rule=PythonRule("False")))
    base.revise(inferred.id, replace(inferred, rule=PythonRule("2")))

    assert base.rule(opened.id).rule == PythonRule("False")  # type: ignore[union-attr]
    assert base.rule(inferred.id).rule == PythonRule("2")  # type: ignore[union-attr]


def test_a_frozen_ruleset_refuses_an_open_rule_with_a_warning(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    base = new_base(tmp_path)
    frozen_rule = base.declare(new_rule("a pawn takes diagonally", CONSTRAINT, mechanism=base.ensure_mechanism(DECLARATION).id))
    open_rule = base.declare(new_rule("mobility counts", POSITION, "1"))
    simulation = new_ruleset(base, "simulation", frozen_rule, task=SIMULATION, declared=True)

    kept = base.link(simulation.id, open_rule.id)

    assert kept.rule_ids == (frozen_rule.id,)
    assert base.ruleset_by_id(simulation.id).rule_ids == (frozen_rule.id,)  # type: ignore[union-attr]
    assert any(record.levelname == "WARNING" and "is frozen" in record.getMessage() for record in caplog.records)


def test_a_copy_of_a_frozen_ruleset_is_open_and_lists_the_same_rules_at_the_same_weights(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    frozen_rule = base.declare(new_rule("a pawn takes diagonally", CONSTRAINT, mechanism=base.ensure_mechanism(DECLARATION).id))
    simulation = new_ruleset(base, "simulation", frozen_rule, task=SIMULATION, declared=True)

    copy = base.copy_ruleset(simulation.id, "simulation with en passant")

    assert base.frozen_ruleset(simulation) and not base.frozen_ruleset(copy)
    assert (copy.links, copy.task, copy.context) == (simulation.links, SIMULATION, simulation.context)
    assert copy.source.rests_on == (simulation.id,)


def test_revising_a_frozen_rule_in_an_open_ruleset_revises_a_copy_there_only(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    frozen_rule = base.declare(new_rule("a pawn takes diagonally", CONSTRAINT, mechanism=base.ensure_mechanism(DECLARATION).id))
    simulation = new_ruleset(base, "simulation", frozen_rule, task=SIMULATION, declared=True)
    heuristic = base.link(new_ruleset(base, "charge").id, frozen_rule.id, 0.5)

    revised = base.revise_in(heuristic.id, frozen_rule.id, replace(frozen_rule, rule=PythonRule("False")))

    assert revised.id != frozen_rule.id and revised.open and revised.source.rests_on == (frozen_rule.id,)
    assert base.rule(frozen_rule.id) == frozen_rule
    assert base.ruleset_rules(heuristic.id) == ((revised, 0.5),)
    assert base.ruleset_rules(simulation.id) == ((frozen_rule, 1.0),)


def test_revising_an_open_rule_in_an_open_ruleset_revises_it_in_place(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    mobility = base.declare(new_rule("mobility counts", POSITION, "1"))
    heuristic = new_ruleset(base, "position value", mobility)

    revised = base.revise_in(heuristic.id, mobility.id, replace(mobility, rule=PythonRule("2")))

    assert revised.id == mobility.id and base.rule(mobility.id).rule == PythonRule("2")  # type: ignore[union-attr]


def test_a_frozen_ruleset_refuses_a_revision_with_a_warning(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    base = new_base(tmp_path)
    frozen_rule = base.declare(new_rule("a pawn takes diagonally", CONSTRAINT, mechanism=base.ensure_mechanism(DECLARATION).id))
    simulation = new_ruleset(base, "simulation", frozen_rule, task=SIMULATION, declared=True)

    kept = base.revise_in(simulation.id, frozen_rule.id, replace(frozen_rule, rule=PythonRule("False")))

    assert kept == frozen_rule and base.ruleset_rules(simulation.id) == ((frozen_rule, 1.0),)
    assert any(record.levelname == "WARNING" and "revise it in a copy" in record.getMessage() for record in caplog.records)


def test_the_rules_and_rulesets_declared_come_back_when_the_base_is_opened_again(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    constraint = base.declare(new_rule("a cell is played only when it is empty", CONSTRAINT, open_=True))
    mobility = base.declare(new_rule("more moves is better placed", POSITION))
    ruleset = base.link(new_ruleset(base, "position value", constraint).id, mobility.id, 0.4)

    opened = new_base(tmp_path)

    assert opened.ruleset_by_id(ruleset.id) == ruleset
    assert [(rule.name, rule.open, weight) for rule, weight in opened.ruleset_rules(ruleset.id)] == [
        ("a cell is played only when it is empty", True, 1.0),
        ("more moves is better placed", False, 0.4),
    ]


def test_an_undeclared_rule_is_retrieved_no_more_here_or_in_the_base_opened_again(tmp_path: Path) -> None:
    base = new_base(tmp_path)
    kept = base.declare(new_rule("kept", POSITION, "1"))
    dropped = base.declare(new_rule("dropped", POSITION, "2"))
    ruleset = new_ruleset(base, "position value", kept, dropped)

    base.undeclare(dropped.id)

    assert [rule.name for rule, _ in base.ruleset_rules(ruleset.id)] == [kept.name]
    assert [rule.name for rule, _ in new_base(tmp_path).ruleset_rules(ruleset.id)] == [kept.name]
