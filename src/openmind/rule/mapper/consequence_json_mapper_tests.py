import json

import pytest

from openmind.rule.mapper.consequence_json_mapper import ConsequenceJsonMapper
from openmind.rule.model.consequence import Consequence
from openmind.rule.model.drawn import Always, Asked, More, Other, Place, Standing, Stepped
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant


def a_consequence(**held) -> Consequence:
    fields = {
        "change": "Moved",
        "action": "move",
        "model": "grid",
        "where": (Place("self", "row"), Place("self", "column")),
        "onto": (Stepped("self", "row", "x"), Stepped("self", "column", "y")),
        "value": None,
        "when": (),
        "order": 2,
        "settled": True,
    }
    return Consequence(**{**fields, **held})


def written(consequence: Consequence) -> Consequence:
    """That consequence through JSON and back, as it would be if a later run read it."""
    mapper = ConsequenceJsonMapper()
    return mapper.from_data(json.loads(json.dumps(mapper.to_data(consequence))))


def test_what_an_action_does_survives_being_written_down():
    """Rules are kept, facts are kept, models are kept, and what the predictor learned was dropped when the
    process ended. So every run began by working out again what a move does."""
    one = a_consequence()

    assert written(one) == one


@pytest.mark.parametrize(
    "drawn",
    [Place("self", "row"), Stepped("self", "row", "x"), Standing("grid", "self"),
     Asked("promotion"), Always("a mark"), Always(None), Other(), More("clock", 2)],
)
def test_every_way_of_drawing_a_part_is_written_down_as_the_way_it_is(drawn):
    """`Always('row')` and a place called `row` must not come back as one another, so a drawing carries the name
    of its kind and not only its fields."""
    one = a_consequence(where=(drawn,), onto=(), value=drawn)

    assert written(one) == one


def test_the_conditions_it_happens_under_are_kept_as_the_logic_they_are():
    """They are the same kind of thing as the conditions under which an action is refused — learned by the same
    machinery — so they are written down the same way and come back reasonable-with rather than as text."""
    taking = Clause((Literal("refused", ()), Literal("turn", (Constant("white"),)).denied))
    one = a_consequence(when=(taking,))

    assert written(one).when == (taking,)


def test_a_drawing_nobody_knows_says_so_rather_than_coming_back_as_something_else():
    with pytest.raises(ValueError, match="No way of drawing"):
        ConsequenceJsonMapper().drawn_from_data({"drawn": "Sideways", "parameter": "self"})


def test_that_a_drawing_was_not_settled_is_kept_too():
    """Thin evidence looking like a confident error is what sends somebody hunting a fault that is not there,
    and that is as true of a consequence read back as of one just learned."""
    assert written(a_consequence(settled=False)).settled is False
