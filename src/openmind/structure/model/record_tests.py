from dataclasses import dataclass

import pytest

from openmind.structure.model.record import Record


@dataclass(frozen=True, slots=True)
class Stamp(Record):
    colour: str
    type: str


def test_a_record_knows_what_the_game_calls_it():
    assert Stamp("white", "queen").kind == "Stamp"


def test_its_parts_come_back_in_the_order_the_game_declared_them():
    assert Stamp("white", "queen").parts == (("colour", "white"), ("type", "queen"))


def test_a_declared_record_can_be_made_again_from_its_parts():
    assert Record.of("Stamp", {"colour": "black", "type": "rook"}) == Stamp("black", "rook")


def test_a_record_no_game_declared_is_refused_by_name():
    with pytest.raises(KeyError, match="No game has declared a record called 'Nothing'"):
        Record.of("Nothing", {})


def test_a_part_named_for_something_a_record_already_calls_its_own_is_refused():
    """Refused where the game declares it, because the alternative is silence. A part called `kind` shadows the
    property saying what kind of record this is, so a piece whose type is a queen writes itself down as a record
    called `queen` and comes back much later, somewhere else, as a KeyError about a record no game declared."""
    with pytest.raises(TypeError, match="a name a record already uses for itself"):

        @dataclass(frozen=True, slots=True)
        class Confused(Record):
            colour: str
            kind: str


def test_the_other_names_a_record_uses_for_itself_are_refused_too():
    for taken in ("parts", "_kinds"):
        with pytest.raises(TypeError, match="a name a record already uses for itself"):
            type(f"Clashing_{taken}", (Record,), {"__annotations__": {taken: str}})


def test_a_part_named_anything_else_is_fine():
    @dataclass(frozen=True, slots=True)
    class Plain(Record):
        colour: str
        type: str

    assert Plain("white", "queen").kind == "Plain"
    assert Record.of("Plain", {"colour": "white", "type": "queen"}) == Plain("white", "queen")
