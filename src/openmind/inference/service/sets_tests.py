from openmind.inference.service.sets import Sets


def test_belonging_is_by_what_a_thing_is_and_not_where_it_sits():
    """Which is what makes a collection a set rather than a list with pretensions."""
    held = Sets()

    assert held.element_of("rook", ("pawn", "rook", "king"))
    assert held.contains(("pawn", "rook"), "rook")
    assert not held.element_of("queen", ("pawn", "rook"))


def test_how_many_means_how_many_distinct():
    """A caller wanting how many *times* something was seen is counting, which is another theory's work."""
    assert Sets().size(("pawn", "pawn", "rook")) == 2


def test_what_one_holds_and_another_does_not_keeps_the_first_s_order():
    """**The shape of nearly every question OMF asks about its own sets** — which candidates survive the
    constraints, which of a move's namings were not chosen, which refused cases nothing accounts for.

    Ordered by the first and not arbitrarily, because an answer that comes back differently on two runs of the
    same question is an answer nobody can pin a test to."""
    held = Sets()

    assert held.besides((3, 1, 2, 1), (2,)) == (3, 1)
    assert held.shared((3, 1, 2), (2, 3)) == (3, 2)


def test_everything_in_one_being_in_the_other():
    held = Sets()

    assert held.within((1, 2), (1, 2, 3))
    assert not held.within((1, 4), (1, 2, 3))


def test_a_word_is_not_a_collection_however_iterable_python_thinks_it_is():
    """A claim is a prior about where a theory is likely to pay, and a theory claiming every string would be
    claiming nearly everything — which is the same as claiming nothing."""
    held = Sets()

    assert held.claims([1, 2, 3])
    assert held.claims(("a", "b"))
    assert not held.claims("abc")
    assert not held.claims(7)
