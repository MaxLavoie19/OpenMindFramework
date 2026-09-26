from openmind.language.service.coupling_learner import CouplingLearner
from openmind.language.service.decoder import Decoder
from openmind.language.service.coupling_learner_tests import a_grammar, a_position, carrying, seen


def couplings():
    return CouplingLearner().learn(seen(), a_grammar(), least=4)


def could_happen(state):
    """Every carrying there could be on the board, taking or not: the whole space, before anything refuses one."""
    cells = state.model("grid").coordinates()
    return tuple(
        carrying(source, target, taking)
        for source in cells
        for target in cells
        if source != target
        for taking in (False, True)
    )


def test_a_notation_is_read_as_the_happenings_it_could_name():
    """Reading narrows the space by what the symbols say, and what is left is what the rules have still to
    settle."""
    state = a_position(t12="hopper")
    among = could_happen(state)

    found = Decoder().read(state, "H3", couplings(), among, a_grammar())

    assert found
    assert len(found) < len(among)
    assert all(one.said(state).get("from sort") == "hopper" for one in found)


def test_what_is_left_over_is_the_measure():
    """Exactly one means what is known agreed with the notation; several means it is too loose; none means it is
    too tight. Those are the two errors constraints are scored by, arriving from a record and not an oracle."""
    state = a_position(t12="hopper")
    among = could_happen(state)
    held = Decoder()

    assert held.scored(state, "H3", couplings(), among, a_grammar()) in ("read", "too many")
    assert held.scored(state, "H3", couplings(), (), a_grammar()) == "none"


def test_the_same_rules_write_the_notation_as_read_it():
    """Writing is the sterner direction: putting the shorter notation rather than the longer means knowing what
    is recoverable, which is knowing the rules."""
    state = a_position(t12="hopper")

    said = Decoder().write(state, carrying((1, 2), (3, 2)), couplings(), a_grammar())

    assert said == "H3"


def test_a_notation_saying_something_impossible_here_reads_as_nothing():
    """Which is the news, not a failure: the rules refuse what the game played, so something is missing."""
    state = a_position(t12="hopper")

    found = Decoder().read(state, "W3", couplings(), could_happen(state), a_grammar())

    assert not found


def test_a_notation_of_no_known_shape_is_spoken_for_by_nothing():
    """The honest answer where we have not seen its like: nothing narrows, rather than everything refusing."""
    state = a_position(t12="hopper")
    among = could_happen(state)

    found = Decoder().read(state, "H3x", couplings(), among, a_grammar())

    assert len(found) == len(among)


def test_writing_picks_the_saying_that_tells_the_move_from_the_others():
    """**The last of the three pairings, and the one the encoder named itself as missing.** A game says the
    least that still identifies what happened, and what counts as identifying it is a fact about what the rules
    allow — never about the notation. Chess writes `Nf3` where one knight can reach f3 and `Nbd2` where two can.

    Each saying that fits is read back, and the briefest that comes to this happening and no other is the one.
    Here the rules leave three things standing and the saying tells this one from the rest."""
    state = a_position(t12="hopper")
    held = Decoder()
    happening = carrying((1, 2), (3, 2))
    among = (happening, carrying((1, 2), (2, 2)), carrying((1, 2), (4, 2)))

    written = held.write(state, happening, couplings(), a_grammar(), among)

    assert held.read(state, written, couplings(), among, a_grammar()) == (happening,)


def test_where_nothing_tells_it_apart_something_is_still_written():
    """The notation genuinely cannot say it, and writing something ambiguous says that better than writing
    nothing: read back it comes to too many, which is the fact.

    In this notation nothing ever can, and the reason is worth knowing — the longer shape has a place saying
    whether anything was taken, and within that shape everything was, so the part carries no bits and no symbol
    is ever recorded for it. A place nothing speaks for cannot be filled, so the shape can never be written.
    That is [open-questions.md](doc/open-questions.md) 27 arriving somewhere it was not expected: settled there
    for reading, and it makes a shape unwritable."""
    state = a_position(t12="hopper")
    among = could_happen(state)
    held = Decoder()
    happening = carrying((1, 2), (3, 2))

    written = held.write(state, happening, couplings(), a_grammar(), among)

    assert written, "something is written even where it cannot be told apart"
    assert len(held.read(state, written, couplings(), among, a_grammar())) > 1
    assert written == held.write(state, happening, couplings(), a_grammar())
