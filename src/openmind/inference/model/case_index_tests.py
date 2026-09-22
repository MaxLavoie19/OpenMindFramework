from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.example import Example
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant


def reading(name: str, *values: object) -> Literal:
    return Literal(name, tuple(Constant(one) for one in values))


def case(kind: str, distance: int) -> Example:
    return Example((reading("at", "source", kind), reading("steps", "source", "target", distance)), False)


#: Fourteen cases: two kinds of thing, each seen going every distance from one to seven.
CASES = tuple(case(kind, distance) for kind in ("walker", "runner") for distance in range(1, 8))


def test_it_narrows_by_whichever_reading_the_fewest_cases_were_read_as_having() -> None:
    """Seven cases were read as a walker and two as going one step, so going one step is what rules out most."""
    narrowed = CaseIndex(CASES).narrowed([reading("at", "source", "walker"), reading("steps", "source", "target", 1)])

    assert len(narrowed) == 2


def test_no_case_holding_every_one_of_those_readings_is_left_out() -> None:
    """The whole of its correctness. Narrowing may offer cases that turn out not to match; it may never hide one
    that does, because the caller never looks at what it was not offered."""
    wanted = [reading("at", "source", "runner"), reading("steps", "source", "target", 3)]

    narrowed = CaseIndex(CASES).narrowed(wanted)

    assert [one for one in CASES if all(held in one.held for held in wanted)]
    assert all(one in narrowed for one in CASES if all(held in one.held for held in wanted))


def test_a_reading_no_case_was_ever_read_as_having_narrows_to_none_of_them() -> None:
    assert CaseIndex(CASES).narrowed([reading("at", "source", "flyer")]) == ()


def test_with_nothing_settled_to_narrow_by_every_case_is_offered() -> None:
    """A clause whose every reading is open rules nothing out, so there is nothing to narrow by. It is also the
    clause that meets its first contradiction almost at once, so offering everything costs nothing."""
    assert CaseIndex(CASES).narrowed([]) == CASES
