from openmind.statement.model.change import CHANGES, LOSING, Moved, Placed, Removed, Told


def test_each_kind_says_where_whatever_stood_there_stops_standing_there():
    """**More than one thing has to agree about this, and prose is not a place to keep an agreement.** What
    each kind means was written in its docstring and read separately by everything that cared: by what applies
    a change to a board, by what says it in words, and by what asks whether something was taken. The last
    asked about removals alone — and a predictor describing a capture as an arrival, which makes the same
    board out of every position so nothing preferred either, became one in which nothing could be taken."""
    assert Removed("grid", (1, 1)).losing == (1, 1), "what stood there, gone"
    assert Placed("grid", (1, 2), "a thing").losing == (1, 2), "put down over what was there"
    assert Moved("grid", (1, 1), (2, 2)).losing == (2, 2), "the target is arrived over"
    assert Told("turn", "black").losing is None, "nothing stands on a scalar"


def test_what_moves_away_is_not_lost():
    """`Moved` empties the place it came from, and a king walking off a square has not been captured."""
    assert Moved("grid", (1, 1), (2, 2)).losing != (1, 1)


def test_which_kinds_can_lose_something_is_derived_rather_than_listed_again():
    """A second copy of an agreement is how the two halves came to disagree in the first place."""
    assert LOSING == tuple(one.__name__ for one in CHANGES if one.LOSES)
    assert set(LOSING) == {"Placed", "Removed", "Moved"}


def test_every_kind_there_is_says_whether_it_loses_anything():
    """A kind added later is a kind everything reading this knows about, rather than one somebody has to
    remember to add to a list somewhere else."""
    for one in CHANGES:
        assert hasattr(one, "LOSES"), f"{one.__name__} does not say"
