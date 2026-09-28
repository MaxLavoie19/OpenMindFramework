from openmind.inference.service.stability import UNVARYING, Stability


def test_a_term_that_never_moves_is_worth_nothing_to_steer_by() -> None:
    """However steady, a constant says nothing about which position is which, so it steers nothing. Said as
    nothing rather than as a division nobody guarded."""
    walk = [[(5.0,), (5.0,), (5.0,), (5.0,)]]

    assert Stability().of(walk) == (UNVARYING,)


def test_a_term_that_differs_across_a_game_and_little_between_neighbours_is_the_steady_one() -> None:
    """The whole measure, in one comparison: two terms with the same spread over the walk, one crawling and
    one leaping. What is wanted is the one a move can be chosen for its effect on."""
    # The first climbs a step at a time; the second takes the same values in an order that jumps.
    crawling = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    leaping = [0.0, 5.0, 1.0, 4.0, 2.0, 3.0]
    walk = [[(crawling[at], leaping[at]) for at in range(len(crawling))]]

    steady = Stability().of(walk)

    assert steady[0] > steady[1], "the one that crawls is the one to steer by"


def test_the_step_between_two_games_is_not_a_step_anything_took() -> None:
    """A game ends and the next begins somewhere else entirely. Counted as a move, that jump would make every
    term look wilder than it is — so walks are kept apart."""
    together = [[(0.0,), (1.0,), (2.0,), (90.0,), (91.0,), (92.0,)]]
    apart = [[(0.0,), (1.0,), (2.0,)], [(90.0,), (91.0,), (92.0,)]]

    assert Stability().of(apart)[0] > Stability().of(together)[0]


def test_every_term_comes_back_ordered_and_none_is_dropped() -> None:
    """Ordering and not choosing: what is measured here decides where the search looks first and never what
    it is allowed to look at."""
    walk = [[(0.0, 0.0, 9.0), (1.0, 7.0, 9.0), (2.0, 0.0, 9.0), (3.0, 7.0, 9.0)]]
    terms = ("crawling", "leaping", "constant")

    ordered = Stability().steadiest(walk, terms)

    assert set(ordered) == set(terms), "every term given comes back"
    assert ordered[0] == "crawling"
    assert ordered[-1] == "constant", "the one that steers nothing is reached last"


def test_nothing_walked_measures_nothing() -> None:
    assert Stability().of([]) == ()
    assert Stability().of([[]]) == ()
    assert Stability().steadiest([], ("a", "b")) == ("a", "b"), "with nothing measured the order is left alone"


def test_a_single_position_has_no_step_to_measure() -> None:
    """One position is a walk of no steps, so nothing moved between neighbours and nothing varied across it."""
    assert Stability().of([[(3.0, 4.0)]]) == (UNVARYING, UNVARYING)


def test_a_term_that_never_varied_is_dropped_whatever_the_budget_says() -> None:
    """Arithmetic and not an opinion: a column the same everywhere tells no position from another, and the
    fit has its own constant already."""
    walk = [[(0.0, 9.0), (1.0, 9.0), (2.0, 9.0), (3.0, 9.0)]]

    kept, dropped = Stability().kept(walk, ("varies", "constant"))

    assert kept == ("varies",)
    assert dropped == ("constant",)


def test_the_budget_keeps_the_steadiest_and_drops_the_rest() -> None:
    """How steady is steady enough has no answer that travels between games, so what is asked for is a count
    the caller chose and not a level this picked."""
    crawling = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    leaping = [0.0, 5.0, 1.0, 4.0, 2.0, 3.0]
    walk = [[(crawling[at], leaping[at]) for at in range(len(crawling))]]

    kept, dropped = Stability().kept(walk, ("crawling", "leaping"), keeping=1)

    assert kept == ("crawling",)
    assert dropped == ("leaping",), "kept nothing against it but the budget"


def test_no_budget_keeps_everything_that_varies() -> None:
    """Off unless asked for: the only thing dropped without a budget is what carries nothing."""
    walk = [[(0.0, 5.0, 9.0), (1.0, 0.0, 9.0), (2.0, 5.0, 9.0)]]

    kept, dropped = Stability().kept(walk, ("a", "b", "constant"))

    assert set(kept) == {"a", "b"}
    assert dropped == ("constant",)


def test_a_budget_of_none_of_them_drops_all_but_nothing_is_lost() -> None:
    """Every term is in one side or the other, whatever the budget, so a caller can always account for them."""
    walk = [[(0.0, 5.0), (1.0, 0.0), (2.0, 5.0)]]

    kept, dropped = Stability().kept(walk, ("a", "b"), keeping=0)

    assert kept == ()
    assert set(dropped) == {"a", "b"}


def test_terms_that_could_not_be_measured_are_all_kept() -> None:
    """Measuring nothing is no grounds for dropping anything."""
    assert Stability().kept([], ("a", "b")) == (("a", "b"), ())
