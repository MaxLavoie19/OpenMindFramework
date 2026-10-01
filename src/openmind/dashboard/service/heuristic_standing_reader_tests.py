"""Reading back what every measure made of a heuristic."""

from openmind.dashboard.service.heuristic_standing_reader import HeuristicStandingReader
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.model.service.model_retirement import RETIRED
from openmind.model.service.rule_budget import VOUCHED_FOR, VOUCHING
from openmind.training.model.agreement import Agreement
from openmind.training.model.teller_agreement import TellerAgreement
from openmind.training.service.judging_record import JudgingRecord


def a_store(tmp_path):
    knowledge = create_knowledge_base("chess", tmp_path)
    return knowledge, knowledge.ensure_context("chess").id


def test_nothing_judged_yet_reads_as_nothing_rather_than_as_an_error(tmp_path):
    """A page opened before the first judging should say so, not fail."""
    assert HeuristicStandingReader().standings(tmp_path, "chess") == ()


def test_each_measure_comes_back_on_its_own(tmp_path):
    """**Each measure stands apart and none is folded into another.** What games paid is the anchor and what a
    teller makes of a position is a claim beside it — a heuristic may do well on one and badly on the other,
    and that is the case somebody opens the page to find."""
    knowledge, context = a_store(tmp_path)
    named = {"it": "position value, fit 1"}
    JudgingRecord().remember(knowledge, context, named, (Agreement("it", 0.5, 20, 80, 2, 0.2),))
    JudgingRecord().told(knowledge, context, named, (TellerAgreement("it", 0.61, 300, 10, 0, 310),))

    found = HeuristicStandingReader().standings(tmp_path, "chess")

    assert len(found) == 1
    assert found[0].name == "position value, fit 1"
    assert found[0].mass == 0.5 and found[0].offered == 0.2
    assert round(found[0].worth, 6) == 0.3, "worked out from its two halves, not stored beside them"
    assert found[0].decided == 20 and found[0].declined == 80 and found[0].undecided == 2
    assert found[0].tracks == 0.61 and found[0].told == 300


def test_a_heuristic_the_teller_never_saw_says_so_rather_than_reading_as_nought(tmp_path):
    """Nought is a real agreement — a heuristic that tracks the teller not at all — so a heuristic it was never
    asked about must not read as one."""
    knowledge, context = a_store(tmp_path)
    JudgingRecord().remember(knowledge, context, {"it": "untold"}, (Agreement("it", 0.5, 20, 80, 0, 0.2),))

    found = HeuristicStandingReader().standings(tmp_path, "chess")

    assert found[0].tracks is None


def test_who_vouched_for_it_comes_back_beside_what_it_was_worth(tmp_path):
    """The signals spend during a ponder on a guess about which rules will earn their place, and what they
    backed is the thing a page is read to argue with."""
    knowledge, context = a_store(tmp_path)
    JudgingRecord().remember(knowledge, context, {"it": "backed"}, (Agreement("it", 0.5, 20, 80, 0, 0.2),))
    knowledge.believe(
        Belief(
            VOUCHING.format(signal="went with winning", heuristic="backed"), context, 1.0,
            tags=((VOUCHED_FOR, "backed"),),
        )
    )

    found = HeuristicStandingReader().standings(tmp_path, "chess")

    assert found[0].vouched == ("went with winning",)


def test_a_retired_heuristic_is_shown_as_retired_rather_than_dropped(tmp_path):
    """It is out of the pool and its record is why. A page that hides it loses the evidence for the decision."""
    knowledge, context = a_store(tmp_path)
    JudgingRecord().remember(knowledge, context, {"it": "gone"}, (Agreement("it", 0.1, 20, 80, 0, 0.4),))
    knowledge.believe(Belief(RETIRED.format(model="gone"), context, True))

    found = HeuristicStandingReader().standings(tmp_path, "chess")

    assert found[0].retired is True
    assert round(found[0].worth, 6) == -0.3, "below nought, which is what retired it"


def test_the_ones_worth_most_come_first(tmp_path):
    """A page of forty candidates is read to find which are working."""
    knowledge, context = a_store(tmp_path)
    for name, mass in (("poor", 0.1), ("good", 0.9), ("middling", 0.5)):
        JudgingRecord().remember(knowledge, context, {name: name}, (Agreement(name, mass, 10, 0, 0, 0.2),))

    found = HeuristicStandingReader().standings(tmp_path, "chess")

    assert [one.name for one in found] == ["good", "middling", "poor"]


def test_coverage_is_read_beside_the_score_and_never_folded_into_it(tmp_path):
    """A detector that speaks on a twentieth of the decisions and is right every time is the thing this project
    keeps saying it wants, and averaging what it says over the decisions it declined hides it."""
    knowledge, context = a_store(tmp_path)
    JudgingRecord().remember(knowledge, context, {"it": "a detector"}, (Agreement("it", 0.5, 5, 95, 0, 0.1),))

    found = HeuristicStandingReader().standings(tmp_path, "chess")

    assert found[0].speaks == "5 of 100"
    assert round(found[0].coverage, 2) == 0.05
    assert round(found[0].worth, 6) == 0.4, "undimmed by how rarely it spoke"
