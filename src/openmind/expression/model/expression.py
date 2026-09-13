from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable

type Expression = Constant | StateVariable | ActionParameter | Equals | Not | AllOf | AnyOf | AllDifferent
