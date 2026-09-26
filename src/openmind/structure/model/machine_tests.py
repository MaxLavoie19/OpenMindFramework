import pytest

from openmind.structure.model.machine import Machine
from openmind.structure.model.phase import Phase

ROLLING = Phase("rolling", ("roll",))
MOVING = Phase("moving", ("move", "pass"))


def a_machine(at="rolling"):
    return Machine((ROLLING, MOVING), at)


def test_a_position_offers_the_actions_of_the_phase_it_is_in():
    """What a phase offers is the candidate space there, as a parameter's kind is the space its values come from.
    Which of them is legal is a constraint and not this."""
    assert a_machine("rolling").offers() == ("roll",)
    assert a_machine("moving").offers() == ("move", "pass")


def test_moving_to_another_phase_gives_another_machine():
    """States holding one have to stay comparable and usable as search keys, so nothing changes in place."""
    held = a_machine("rolling")

    moved = held.moved_to("moving")

    assert moved.at == "moving"
    assert held.at == "rolling"


def test_a_phase_it_has_not_got_raises_rather_than_being_an_unknown_state():
    """A game moving somewhere that does not exist is a game with a bug in it, and saying so where it happens is
    worth more than a position nobody can read."""
    with pytest.raises(ValueError):
        Machine((ROLLING,), "moving")
    with pytest.raises(KeyError):
        a_machine().moved_to("scoring")


def test_what_follows_what_is_not_declared():
    """Moving to the next phase is something an action does, and what an action does is learned. A machine that
    said its transitions would be declaring half a game's effects while OMF worked out the rest."""
    assert not hasattr(a_machine(), "transitions")
    assert not any("then" in field for field in Machine.__slots__)
