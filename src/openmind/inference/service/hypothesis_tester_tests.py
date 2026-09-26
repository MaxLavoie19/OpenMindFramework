from itertools import combinations

from openmind.inference.model.case_index import CaseIndex
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.hypothesis_table import HypothesisTable
from openmind.inference.service.hypothesis_tester import HypothesisTester
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.inference.service.refusal_learner_tests import (
    DOMAINS,
    Piece,
    a_position,
    evidence_where,
)


def a_board():
    """One white pawn on a two by two board, and the moves the game allows from it."""
    return evidence_where(a_position(a2=Piece("white", "pawn")), "a2")


def read():
    readings = CandidateReadings()
    evidence = a_board()
    cases = readings.cases(evidence, DOMAINS)
    refused = [one for one in cases if one.holds]
    guard = CaseIndex(tuple(one for one in cases if not one.holds))
    return readings, refused, guard


def a_tester():
    return HypothesisTester(RefusalLearner(CandidateReadings()))


def test_ranges_cut_the_work_up_without_losing_or_repeating_any():
    """A range is a piece of work handed to somebody else, so what it means must not depend on who holds it. Two
    callers naming the same range mean the same bodies, and the ranges together mean all of them."""
    readings, refused, guard = read()
    tester = a_tester()
    offered = readings.tied(refused[0].literals)

    whole = tester.tried(offered, 2, refused, guard)
    parts = [tester.tried(offered, 2, refused, guard, start, start + 5) for start in range(0, len(whole), 5)]

    assert [one.clause for one in whole] == [one.clause for part in parts for one in part]
    assert len({one.key for one in whole}) == len(whole), "no body is tried twice within a size"


def test_how_many_there_are_is_worked_out_and_not_counted():
    """Cutting the work into ranges means knowing how much there is before doing any of it — and counting by
    enumerating is the very thing being divided up."""
    readings, refused, _ = read()
    offered = readings.tied(refused[0].literals)

    for size in range(1, 4):
        assert a_tester().how_many(offered, size) == len(list(combinations(offered, size)))


def test_a_body_that_turns_away_a_legal_move_is_reported_and_not_measured():
    """Nothing will ever use it, so working out what it refuses is work spent on an answer nobody reads. It comes
    back all the same: what it is now good for is stopping the next case reaching the same body from paying for
    the same test."""
    readings, refused, guard = read()
    tester = a_tester()
    offered = readings.tied(refused[0].literals)

    tried = tester.tried(offered, 1, refused, guard)
    slipping = [one for one in tried if one.slips]

    assert slipping, "a single reading of a case is broad enough to catch something legal"
    assert all(not one.refusing for one in slipping)
    assert all(not one.useful for one in slipping)


def test_the_table_picks_what_the_learner_s_own_search_picks():
    """**The measurement that has to come before any other.** Splitting a search across processes is worth
    nothing if it answers a different question, and "faster" is easy to get by accident when the answer has
    quietly changed.

    Searched by size, the briefest body that turns nothing legal away wins; among those of that size, the one
    accounting for most of what is still unexplained. Gathered into a table the same rule has to give the same
    body, from the same readings and the same cases."""
    readings, refused, guard = read()
    learner = RefusalLearner(CandidateReadings())
    tester, table = HypothesisTester(learner), HypothesisTable()
    seed = refused[0]
    offered = readings.tied(seed.literals)

    for size in range(1, 5):
        table.tell(tester.tried(offered, size, refused, guard))
        if table.useful():
            break
    briefest = min(one.size for one in table.covering(refused.index(seed)))
    picked = max(
        (one for one in table.covering(refused.index(seed)) if one.size == briefest),
        key=lambda one: len(one.refusing),
    )

    alone = learner._briefest(seed, refused, guard, learner.clock() + 30.0)  # noqa: SLF001

    assert picked.clause.body == alone.body
    assert len(picked.refusing) == sum(1 for one in refused if learner.covers(alone, one))


def a_run(table):
    """What the learner ends up with over one position, with and without something to remember by."""
    from openmind.inference.model.inference_budget import InferenceBudget

    readings = CandidateReadings()
    evidence = a_board()
    cases = readings.cases(evidence, DOMAINS)
    learner = RefusalLearner(readings)
    return learner, cases, learner.learn(cases, InferenceBudget(seconds=30.0), table=table)


def test_remembering_what_was_tried_keeps_the_two_promises_that_matter():
    """Whatever else the table changes, it may not change these: nothing the game allows is turned away, and
    what the game refuses is accounted for. A search that is cheaper and answers a different question is not
    cheaper, and "faster" is easy to get by accident once the answer is allowed to drift."""
    learner, cases, learned = a_run(HypothesisTable())

    assert learned
    assert not [one for one in cases if not one.holds and learner.refuses(learned, one)]
    assert not [one for one in cases if one.holds and not learner.refuses(learned, one)]


def test_what_is_remembered_is_reached_for_before_anything_is_searched():
    """The saving, said as a test. Once a table has been filled by one position, learning the same position
    again searches nothing: every case is answered by something already tried."""
    table = HypothesisTable()
    a_run(table)
    before = table.tried

    a_run(table)

    assert table.tried == before, "the second time, nothing new was tried"
