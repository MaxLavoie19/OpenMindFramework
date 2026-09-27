from dataclasses import dataclass

import pytest

from openmind.structure.mapper.value_json_mapper import ValueJsonMapper
from openmind.structure.model.record import Record
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Marker(Record):
    colour: str
    type: str


@dataclass(frozen=True, slots=True)
class Holder(Record):
    colour: str
    holds: Value = None


def test_a_record_goes_out_as_its_parts_and_comes_back_as_itself():
    mapper = ValueJsonMapper()
    held = Marker("white", "queen")

    assert mapper.from_data(mapper.to_data(held)) == held


def test_a_record_inside_a_record_goes_the_same_way():
    """What a game puts on a square is a value like any other, and a square is one too."""
    mapper = ValueJsonMapper()
    held = Holder("dark", Marker("black", "knight"))

    assert mapper.from_data(mapper.to_data(held)) == held


def test_names_numbers_nothing_and_lists_of_them_pass_through():
    mapper = ValueJsonMapper()

    for one in ("white", 3, 2.5, True, None):
        assert mapper.from_data(mapper.to_data(one)) == one
    assert mapper.from_data(mapper.to_data(["white", 3, None])) == ["white", 3, None]


def test_a_value_that_is_neither_is_refused_by_name():
    """Refused here rather than at the moment of writing, where what went wrong is a line of JSON and no longer
    a value anybody can name."""
    mapper = ValueJsonMapper()

    with pytest.raises(TypeError, match="not one of OMF's values"):
        mapper.to_data(object())
