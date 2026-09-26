from openmind.structure.model.domain import Domain, Numbers
from openmind.structure.model.kind import Kind
from openmind.structure.model.schema import GridKind


def test_an_unbounded_set_is_not_listable_and_hands_out_nothing():
    """A game declares whole numbers and says nothing about how far. Nothing may enumerate that without
    narrowing it first, and asking is how a caller finds out it has to."""
    assert not Domain.WHOLE.listable
    assert Domain.WHOLE.domain == ()
    assert not Domain.WHOLE_POSITIVE.listable


def test_narrowing_never_widens():
    """Where it was already bounded more tightly, the tighter bound stands. Otherwise a narrowing could hand back
    values the game never said were there."""
    held = Domain.WHOLE_POSITIVE.within(-5, 3)

    assert held.least == 1
    assert held.most == 3
    assert held.domain == (1, 2, 3)


def test_positive_begins_where_there_is_a_number_to_begin_at():
    """Among whole numbers that is one. Among the reals there is no next number above zero, so the bound is zero
    and it is taken in — a difference a single bound cannot hide."""
    assert Domain.WHOLE_POSITIVE.least == 1
    assert Domain.REAL_POSITIVE.least == 0
    assert not Domain.REAL.whole


def test_a_grid_says_how_far_an_offset_could_carry_anything():
    """No offset outside the widest span can land on the grid from any cell, so nothing outside is refused for a
    reason the inside does not already show. The window comes from the structure, not from a number anyone
    chose."""
    grid = GridKind(holds=Kind("thing", ("a",)), shape=(8, 8))

    held = grid.reaches(Domain.WHOLE)

    assert held.least == -7
    assert held.most == 7
    assert held.listable
    assert len(held.domain) == 15


def test_the_window_still_offers_moves_that_leave_the_grid():
    """Which is why it is a window and not a cap. Whether an offset leaves the grid depends on where the mover
    stands, so from a corner every negative offset goes off it — and the rule against that keeps its evidence."""
    grid = GridKind(holds=Kind("thing", ("a",)), shape=(8, 8))

    offsets = grid.reaches(Domain.WHOLE).domain
    from_a_corner = [one for one in offsets if 1 + one < 1 or 1 + one > 8]

    assert from_a_corner
    assert all(one < 0 for one in from_a_corner)


def test_a_grid_that_is_not_square_offers_a_few_no_cell_could_use():
    """Nothing here knows which parameter is which axis, so the widest span is taken. What that offers over is
    refused like anything else."""
    grid = GridKind(holds=Kind("thing", ("a",)), shape=(3, 8))

    assert grid.reaches(Domain.WHOLE).most == 7


def test_numbers_say_what_they_are():
    assert Numbers().readable == "whole numbers"
    assert Domain.WHOLE_POSITIVE.readable == "whole numbers from 1"
    assert Numbers(whole=False, least=-1, most=1).readable == "numbers from -1 to 1"
