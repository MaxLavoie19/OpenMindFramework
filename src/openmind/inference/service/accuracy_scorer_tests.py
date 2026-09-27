from pathlib import Path

import pytest

from openmind.inference.constant.certainty_constant import ACCURACY, RIGHT, SCORED, SPREAD
from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.source import Source

VARIABLE = "who wins"


def scoring(tmp_path: Path, declared: float | None = None):
    knowledge = create_knowledge_base("scoring", tmp_path)
    context = knowledge.ensure_context("a game").id
    guesser = knowledge.ensure_mechanism("the guesser", declared)
    return AccuracyScorer(), knowledge, context, guesser


def said(knowledge, context, mechanism, value, variable=VARIABLE):
    """One mechanism's evidence for a value, as a belief the scorer will later find."""
    knowledge.believe(
        Belief(variable, context, value, evidence=(Evidence(value, 1.0, Source(mechanism.id)),))
    )


def turned_out(knowledge, context, value, variable=VARIABLE):
    return knowledge.experience(DirectExperience(variable, context, value, Source("the game itself")))


def test_a_mechanism_that_said_what_turned_out_true_is_scored_right(tmp_path: Path) -> None:
    scorer, knowledge, context, guesser = scoring(tmp_path)
    said(knowledge, context, guesser, "white")

    scorer.settle(knowledge, VARIABLE, context, turned_out(knowledge, context, "white").id)

    belief = knowledge.belief(ACCURACY.format(mechanism=guesser.id), context)
    assert belief is not None
    assert dict(belief.tags)[RIGHT] == 1 and dict(belief.tags)[SCORED] == 1


def test_a_mechanism_that_said_something_else_is_scored_wrong(tmp_path: Path) -> None:
    scorer, knowledge, context, guesser = scoring(tmp_path)
    said(knowledge, context, guesser, "black")

    scorer.settle(knowledge, VARIABLE, context, turned_out(knowledge, context, "white").id)

    belief = knowledge.belief(ACCURACY.format(mechanism=guesser.id), context)
    assert belief is not None and dict(belief.tags)[RIGHT] == 0


def test_one_case_does_not_make_a_mechanism_certain(tmp_path: Path) -> None:
    """It starts from its declared accuracy as if it had been scored twice, so a mechanism right once is not
    believed always right — and one believed impossible would never be tried again."""
    scorer, knowledge, context, guesser = scoring(tmp_path)
    said(knowledge, context, guesser, "white")

    scorer.settle(knowledge, VARIABLE, context, turned_out(knowledge, context, "white").id)

    accuracy = float(knowledge.belief(ACCURACY.format(mechanism=guesser.id), context).value)
    assert 0.5 < accuracy < 1.0
    assert accuracy == pytest.approx((1 + 0.5 * 2) / (1 + 2))


def test_a_declared_accuracy_is_what_it_leans_on_before_anything_is_counted(tmp_path: Path) -> None:
    """A mechanism whose maker said it is usually right starts there rather than at the middle."""
    scorer, knowledge, context, guesser = scoring(tmp_path, declared=0.9)
    said(knowledge, context, guesser, "black")

    scorer.settle(knowledge, VARIABLE, context, turned_out(knowledge, context, "white").id)

    accuracy = float(knowledge.belief(ACCURACY.format(mechanism=guesser.id), context).value)
    assert accuracy == pytest.approx((0 + 0.9 * 2) / (1 + 2))


def test_a_mechanism_never_scored_is_taken_at_what_it_declared(tmp_path: Path) -> None:
    scorer, knowledge, context, guesser = scoring(tmp_path, declared=0.8)

    assert scorer.accuracy(knowledge, guesser.id, context) == 0.8


def test_a_mechanism_never_scored_and_declaring_nothing_has_no_accuracy(tmp_path: Path) -> None:
    """Not known, and not a low one. A model never measured has no accuracy."""
    scorer, knowledge, context, guesser = scoring(tmp_path)

    assert scorer.accuracy(knowledge, guesser.id, context) is None
    assert scorer.spread(knowledge, guesser.id, context) is None


def test_what_was_measured_beats_what_was_declared(tmp_path: Path) -> None:
    scorer, knowledge, context, guesser = scoring(tmp_path, declared=0.9)
    for _ in range(3):
        said(knowledge, context, guesser, "black")
        scorer.settle(knowledge, VARIABLE, context, turned_out(knowledge, context, "white").id)

    assert scorer.accuracy(knowledge, guesser.id, context) < 0.9


def test_a_number_is_scored_by_how_far_off_it_was_rather_than_right_or_wrong(tmp_path: Path) -> None:
    """Saying 0.9 where 1.0 turned out true is not the same mistake as saying 0.0, and the share got right
    cannot tell them apart."""
    scorer, knowledge, context, guesser = scoring(tmp_path)
    said(knowledge, context, guesser, 0.6, variable="the score")

    scorer.settle(knowledge, "the score", context, turned_out(knowledge, context, 1.0, "the score").id)

    spread = scorer.spread(knowledge, guesser.id, context)
    assert spread == pytest.approx(0.4)
    assert knowledge.belief(ACCURACY.format(mechanism=guesser.id), context) is None, "not scored right or wrong"


def test_the_spread_is_the_root_of_the_mean_squared_error_over_every_case(tmp_path: Path) -> None:
    scorer, knowledge, context, guesser = scoring(tmp_path)
    for value in (0.0, 1.0):
        said(knowledge, context, guesser, value, variable="the score")
        scorer.settle(knowledge, "the score", context, turned_out(knowledge, context, 0.5, "the score").id)

    assert scorer.spread(knowledge, guesser.id, context) == pytest.approx(0.5)
    assert dict(knowledge.belief(SPREAD.format(mechanism=guesser.id), context).tags)[SCORED] == 2


def test_the_anchor_does_not_score_the_evidence_that_rests_on_it(tmp_path: Path) -> None:
    """A mechanism marked right for having repeated what settled the question has measured nothing."""
    scorer, knowledge, context, guesser = scoring(tmp_path)
    anchor = turned_out(knowledge, context, "white")
    knowledge.believe(
        Belief(VARIABLE, context, "white", evidence=(Evidence("white", 1.0, Source(guesser.id, rests_on=(anchor.id,))),))
    )

    assert scorer.settle(knowledge, VARIABLE, context, anchor.id) == ()
    assert knowledge.belief(ACCURACY.format(mechanism=guesser.id), context) is None


def test_settling_by_an_anchor_that_is_not_there_says_so(tmp_path: Path) -> None:
    scorer, knowledge, context, _ = scoring(tmp_path)

    with pytest.raises(KeyError, match="No direct experience"):
        scorer.settle(knowledge, VARIABLE, context, "nothing-like-this")
