from openmind.rule.mapper.clause_json_mapper import ClauseJsonMapper
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Variable

import pytest


def a_clause() -> Clause:
    return Clause(
        (
            Literal("worth at least", (Variable("Thing", "thing"), Number(14))),
            Literal("reaches", (Variable("Thing", "thing"), Number(14)), True),
        ),
        0.8,
        "what a thing affords is what it is worth",
    )


def test_a_clause_comes_back_as_the_clause_it_was() -> None:
    mapper = ClauseJsonMapper()
    clause = a_clause()

    assert mapper.clause_from_data(mapper.clause_to_data(clause)) == clause


def test_a_variable_keeps_its_sort_across_the_round_trip() -> None:
    mapper = ClauseJsonMapper()

    came_back = mapper.term_to_term(mapper.term_to_data(Variable("Thing", "player")))

    assert came_back == Variable("Thing", "player")


def test_a_constant_named_like_a_variable_does_not_come_back_as_one() -> None:
    mapper = ClauseJsonMapper()

    came_back = mapper.term_to_term(mapper.term_to_data(Constant("variable")))

    assert came_back == Constant("variable")


def test_a_constant_holding_nothing_comes_back_holding_nothing() -> None:
    mapper = ClauseJsonMapper()

    came_back = mapper.term_to_term(mapper.term_to_data(Constant(None)))

    assert came_back == Constant(None)


def test_a_number_does_not_come_back_as_a_constant() -> None:
    mapper = ClauseJsonMapper()

    came_back = mapper.term_to_term(mapper.term_to_data(Number(14)))

    assert came_back == Number(14)


def test_a_function_comes_back_with_everything_it_was_of() -> None:
    mapper = ClauseJsonMapper()
    term = Functor("next", (Variable("Place"), Constant("first")))

    assert mapper.term_to_term(mapper.term_to_data(term)) == term


def test_a_term_saying_nothing_about_which_kind_it_is_is_refused() -> None:
    with pytest.raises(ValueError):
        ClauseJsonMapper().term_to_term({"name": "neither"})


def test_a_constant_holding_a_record_is_written_down_and_read_back() -> None:
    """A constant's value is whatever the game's records are made of, and that need not be a string. Written
    raw it fails at the point of writing, which killed a training run at its twelfth position after the same
    fault had been fixed for consequences and left here."""
    from dataclasses import dataclass

    from openmind.structure.model.record import Record

    @dataclass(frozen=True, slots=True)
    class Held(Record):
        colour: str
        type: str

    mapper = ClauseJsonMapper()
    one = Constant(Held("black", "rook"))

    said = mapper.term_to_data(one)

    assert said == {"constant": {"record": "Held", "parts": {"colour": "black", "type": "rook"}}}
    assert mapper.term_to_term(said) == one


def test_a_constant_holding_a_plain_value_is_untouched() -> None:
    """The ordinary case stays ordinary: nothing is wrapped that does not need wrapping."""
    mapper = ClauseJsonMapper()

    assert mapper.term_to_data(Constant("grid")) == {"constant": "grid"}
    assert mapper.term_to_term({"constant": "grid"}) == Constant("grid")
    assert mapper.term_to_term({"constant": None}) == Constant(None)
