"""What a judging leaves behind, so that what it measured can be read again."""

from pathlib import Path

from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.training.model.agreement import Agreement
from openmind.training.model.teller_agreement import TellerAgreement
from openmind.training.service.judging_record import (
    DECIDED, JUDGED, JUDGINGS, MASS, TOLD, TOLD_OF, JudgingRecord,
)


def a_store(tmp_path):
    knowledge = create_knowledge_base("chess", tmp_path)
    return knowledge, knowledge.ensure_context("chess").id


def tags_of(knowledge, context, variable):
    return dict((knowledge.belief(variable, context)).tags)


def test_what_a_judging_found_is_added_to_what_came_before(tmp_path):
    """**A judging is a handful of games, and a handful says little.** A heuristic measured over eighty
    decisions has said almost nothing about itself; the same one over eight thousand has. So the counts add,
    and how many judgings they came from is kept beside them so nobody reads one long look as many."""
    knowledge, context = a_store(tmp_path)
    named = {"a heuristic": "position value, fit 1"}
    record = JudgingRecord()

    record.remember(knowledge, context, named, (Agreement("a heuristic", 0.4, 10, 90, 0, 0.1),))
    record.remember(knowledge, context, named, (Agreement("a heuristic", 0.6, 20, 80, 1, 0.2),))

    held = tags_of(knowledge, context, JUDGED.format(model="position value, fit 1"))
    assert held[MASS] == 1.0 and held[DECIDED] == 30
    assert held[JUDGINGS] == 2, "so one long look is not read as many"


def test_a_heuristic_the_judging_does_not_name_is_left_alone(tmp_path):
    """A judging calls a heuristic one thing and the store calls it another, and only the store's name can be
    looked up later. One it cannot translate is skipped rather than written under a name nothing will find."""
    knowledge, context = a_store(tmp_path)
    record = JudgingRecord()

    record.remember(knowledge, context, {}, (Agreement("a stranger", 0.4, 10, 90, 0, 0.1),))

    assert knowledge.belief(JUDGED.format(model="a stranger"), context) is None


def test_the_teller_s_number_is_the_latest_and_not_a_total(tmp_path):
    """A rank agreement is a correlation, and correlations do not sum. What is worth keeping beside it is how
    many positions it was taken over, since agreement over nine and over nine thousand are different claims."""
    knowledge, context = a_store(tmp_path)
    named = {"a heuristic": "position value, fit 1"}
    record = JudgingRecord()

    record.told(knowledge, context, named, (TellerAgreement("a heuristic", 0.5, 100, 0, 0, 100),))
    record.told(knowledge, context, named, (TellerAgreement("a heuristic", 0.8, 200, 0, 0, 200),))

    belief = knowledge.belief(TOLD.format(model="position value, fit 1"), context)
    assert belief.value == 0.8, "the latest, not 1.3 and not an average nobody asked for"
    assert dict(belief.tags)[TOLD_OF] == 200


def test_a_heuristic_the_teller_could_not_measure_is_not_written_down(tmp_path):
    """No opinion about a position is not a wrong opinion. Written at nought it would read as a heuristic that
    tracks the teller not at all, which is a measurement, where the truth is that there is none."""
    knowledge, context = a_store(tmp_path)
    record = JudgingRecord()

    record.told(
        knowledge, context, {"a heuristic": "silent"}, (TellerAgreement("a heuristic", 0.0, 0, 50, 0, 50),)
    )

    assert knowledge.belief(TOLD.format(model="silent"), context) is None
