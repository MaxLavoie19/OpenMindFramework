import json

import pytest

from openmind.expression.mapper.expression_json_mapper import ExpressionJsonMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable


@pytest.mark.parametrize(
    "expression",
    [
        Constant("X"),
        Constant(None),
        Constant(0.5),
        Constant(True),
        StateVariable("turn"),
        StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))),
        ActionParameter("row"),
        Equals(StateVariable("cell(2,2)"), Constant(None)),
        Not(Constant(False)),
        AllOf((Constant(True), Equals(ActionParameter("row"), Constant(2)))),
        AnyOf((Constant(False), Constant(True))),
    ],
)
def test_round_trip_through_json_text(expression: Expression) -> None:
    mapper = ExpressionJsonMapper()

    assert mapper.from_data(json.loads(json.dumps(mapper.to_data(expression)))) == expression


def test_to_data_names_each_expression_type() -> None:
    expression = Equals(StateVariable("cell", (ActionParameter("row"), Constant(2))), Constant(None))

    assert ExpressionJsonMapper().to_data(expression) == {
        "equals": [
            {"state_variable": "cell", "indices": [{"action_parameter": "row"}, {"constant": 2}]},
            {"constant": None},
        ]
    }


def test_unknown_data_raises() -> None:
    with pytest.raises(ValueError, match="plus"):
        ExpressionJsonMapper().from_data({"plus": [1, 2]})
