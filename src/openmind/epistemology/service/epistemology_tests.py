from pathlib import Path

import pytest

from openmind.epistemology.constant.epistemology_constant import AGAINST_AN_ANCHOR, VALUES_DISAGREE
from openmind.epistemology.factory.epistemology_factory import create_epistemology
from openmind.epistemology.service.accuracy_scorer import AccuracyScorer
from openmind.epistemology.service.justifier import Justifier
from openmind.knowledge.constant.knowledge_constant import DECLARATION
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule

pytestmark = pytest.mark.log_level("INFO")


def base(tmp_path: Path) -> tuple[KnowledgeBase, str]:
    knowledge = create_knowledge_base("cheat", tmp_path)
    return knowledge, knowledge.ensure_context("cheat").id


def heard(knowledge: KnowledgeBase, context: str, variable: str, value: object) -> str:
    microphone = knowledge.ensure_mechanism("table microphone").id
    return knowledge.experience(DirectExperience(variable, context, value, Source(microphone))).id  # type: ignore[arg-type]


def test_support_traced_back_reaches_direct_experiences_and_frozen_rules_through_other_beliefs(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    claim = heard(knowledge, cheat, "claim", "three kings")
    rule = knowledge.declare(
        RuleRecord("four of a kind at most", CONSTRAINT, PythonRule("True"), Source(knowledge.ensure_mechanism(DECLARATION).id))
    )
    reader = knowledge.ensure_mechanism("claim reader").id
    kings = knowledge.believe(Belief("kings claimed", cheat, 3, evidence=(Evidence(3, 1.0, Source(reader, rests_on=(claim,))),)))
    counting = knowledge.ensure_mechanism("counting").id
    bluff = Belief("black is bluffing", cheat, True, evidence=(Evidence(True, 0.7, Source(counting, rests_on=(kings.id, rule.id))),))

    justification = Justifier().justify(knowledge, knowledge.believe(bluff))

    assert set(justification.anchors) == {claim, rule.id} and justification.justified == (0,)


def test_a_circle_of_beliefs_resting_on_each_other_reaches_no_anchor(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    guessing = knowledge.ensure_mechanism("guessing").id
    first = knowledge.believe(Belief("a", cheat, True))
    second = knowledge.believe(Belief("b", cheat, True, evidence=(Evidence(True, 0.9, Source(guessing, rests_on=(first.id,))),)))
    first = knowledge.believe(Belief("a", cheat, True, evidence=(Evidence(True, 0.9, Source(guessing, rests_on=(second.id,))),)))

    justification = Justifier().justify(knowledge, first)

    assert (justification.anchors, justification.justified) == ((), ())
    assert first.id in justification.circular


def test_evidence_that_reaches_no_anchor_leaves_the_belief_as_given(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    guessing = knowledge.ensure_mechanism("guessing").id
    belief = Belief("black is bluffing", cheat, True, certainty=0.9, evidence=(Evidence(False, 1.0, Source(guessing)),))

    reviewed = create_epistemology().review(knowledge, belief)

    assert (reviewed.value, reviewed.certainty) == (True, 0.9)


def test_without_measured_accuracy_certainty_falls_back_on_fuzzy_logic_over_strengths(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    claim = heard(knowledge, cheat, "claim", "three kings")
    reader = knowledge.ensure_mechanism("claim reader").id
    evidence = (
        Evidence(True, 0.8, Source(reader, rests_on=(claim,))),
        Evidence(False, 0.3, Source(reader, rests_on=(claim,))),
    )

    reviewed = create_epistemology().review(knowledge, Belief("black is bluffing", cheat, False, evidence=evidence))

    assert reviewed.value is True and reviewed.certainty == pytest.approx(0.7)


def test_with_known_accuracy_certainty_is_bayesian_from_the_caller_s_value(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    claim = heard(knowledge, cheat, "claim", "three kings")
    reader = knowledge.ensure_mechanism("claim reader", declared_accuracy=0.9).id
    evidence = (Evidence(True, 1.0, Source(reader, rests_on=(claim,))),)

    reviewed = create_epistemology().review(knowledge, Belief("black is bluffing", cheat, True, certainty=0.5, evidence=evidence))

    # Prior: True 0.5, anything else 0.5. Likelihoods: 0.9 for True, 0.1 for anything else.
    assert reviewed.value is True and reviewed.certainty == pytest.approx(0.45 / (0.45 + 0.05))


def test_a_mechanism_s_accuracy_is_scored_when_an_anchor_settles_the_variable(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    reader = knowledge.ensure_mechanism("claim reader", declared_accuracy=0.5).id
    source = Source(reader)
    knowledge.believe(Belief("black is bluffing", cheat, True, evidence=(Evidence(True, 1.0, source),)))
    revealed = heard(knowledge, cheat, "black is bluffing", False)
    scorer = AccuracyScorer()

    scorer.settle(knowledge, "black is bluffing", cheat, revealed)

    assert scorer.accuracy(knowledge, reader, cheat) == pytest.approx((0 + 0.5 * 2) / (1 + 2))


def test_numbers_combine_as_gaussians_once_their_mechanisms_spread_is_measured(tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    counter = knowledge.ensure_mechanism("card counter").id
    scorer = AccuracyScorer()
    knowledge.believe(Belief("kings left", cheat, 3, evidence=(Evidence(3, 1.0, Source(counter)),)))
    scorer.settle(knowledge, "kings left", cheat, heard(knowledge, cheat, "kings left", 1))
    claim = heard(knowledge, cheat, "claim", "two kings")
    evidence = (Evidence(2.0, 1.0, Source(counter, rests_on=(claim,))), Evidence(4.0, 1.0, Source(counter, rests_on=(claim,))))

    reviewed = create_epistemology().review(knowledge, Belief("kings in the pile", cheat, 0.0, evidence=evidence))

    assert reviewed.value == pytest.approx(3.0)
    assert reviewed.precision == pytest.approx(2.0 / 2**0.5)


def test_conflicts_are_warnings_and_tasks(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    knowledge, cheat = base(tmp_path)
    claim = heard(knowledge, cheat, "black is bluffing", False)
    reader = knowledge.ensure_mechanism("claim reader").id
    knowledge.believe(
        Belief(
            "black is bluffing",
            cheat,
            True,
            evidence=(Evidence(True, 0.9, Source(reader, rests_on=(claim,))), Evidence(False, 0.4, Source(reader, rests_on=(claim,)))),
        )
    )
    worth, time = (Belief("utility gained", cheat, 0.2),), Belief("expected time", cheat, 30.0)

    conflicts = create_epistemology().audit(knowledge, cheat, worth, time)

    assert {conflict.kind for conflict in conflicts} == {VALUES_DISAGREE, AGAINST_AN_ANCHOR}
    assert len(knowledge.tasks(tags=(("keyword", "conflict"),))) == len(conflicts)
    assert any(record.levelname == "WARNING" and "Conflict in cheat" in record.getMessage() for record in caplog.records)
