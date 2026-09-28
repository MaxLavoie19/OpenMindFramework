from openmind.inference.service.teaching_set import TeachingSet


def test_one_position_testing_every_rule_is_the_whole_teaching_set() -> None:
    """Fewest, because a teacher that sends everything has taught nothing about what matters."""
    firing = {"a": [0, 1, 2], "b": [0, 3], "c": [0, 4]}

    chosen, untestable = TeachingSet().covering(firing)

    assert chosen == (0,)
    assert untestable == ()


def test_every_rule_gets_somewhere_to_be_tested() -> None:
    """A rule that fires nowhere in what it was sent cannot be checked, however wrong it is."""
    firing = {"a": [0], "b": [1], "c": [2]}

    chosen, _ = TeachingSet().covering(firing)

    assert set(chosen) == {0, 1, 2}


def test_a_rule_nothing_can_test_is_named_rather_than_dropped() -> None:
    """It may be right; it may be the overfitted coincidence believed for weeks because nobody visited a
    position that disagreed. Either way the honest thing is to say so."""
    firing = {"a": [0, 1], "never fires": []}

    chosen, untestable = TeachingSet().covering(firing)

    assert untestable == ("never fires",)
    assert chosen, "the rest are still covered"


def test_asking_for_more_tests_sends_more_positions() -> None:
    """One test shows whether a rule is ever wrong; more shows whether it is wrong often, and which is
    wanted is the teacher's business."""
    firing = {"a": [0, 1, 2, 3]}
    teaching = TeachingSet()

    assert len(teaching.covering(firing, least=1)[0]) == 1
    assert len(teaching.covering(firing, least=3)[0]) == 3


def test_a_rule_cannot_be_asked_for_more_tests_than_it_has_positions() -> None:
    """Asking for three tests of a rule that fires twice takes both and stops, rather than looping."""
    firing = {"a": [0, 1]}

    chosen, _ = TeachingSet().covering(firing, least=5)

    assert set(chosen) == {0, 1}


def test_the_positions_come_back_most_useful_first() -> None:
    """So a teacher with less room than the whole set can send a prefix and still have spent it on the rules
    least covered."""
    firing = {"a": [0, 9], "b": [0, 9], "c": [0], "d": [7]}

    chosen, _ = TeachingSet().covering(firing)

    assert chosen[0] == 0, "the position testing three rules goes first"
    assert set(chosen) == {0, 7}


def test_what_a_prefix_would_cost_can_be_read_back() -> None:
    """A teacher sending fewer positions than were chosen needs to know which rules that leaves untested."""
    firing = {"a": [0], "b": [1], "c": [1]}
    teaching = TeachingSet()

    chosen, _ = teaching.covering(firing)
    tested = teaching.tested(firing, chosen[:1])

    assert sum(1 for count in tested.values() if count == 0) == 1, "one rule loses its only test"
    assert teaching.tested(firing, chosen) == {"a": 1, "b": 1, "c": 1}


def test_teaching_nothing_sends_nothing() -> None:
    assert TeachingSet().covering({}) == ((), ())
