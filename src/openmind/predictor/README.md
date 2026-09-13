# predictor

## Purpose

Transitions: what performing an action does to a state. Given a domain's transition model, a state and an action, the
predictor gives the outcome probability distribution: each possible new state with its probability. Transitions also
end the game and set payoffs, since a payoff is a predicted outcome of a transition to a win or a draw.

## Content

| File | What it is |
|---|---|
| `model/assign.py` | `Assign(target, value)`: sets an existing state variable to the value of an expression |
| `model/when.py` | `When(condition, then, otherwise)`: applies `then` when the condition is true, otherwise `otherwise` |
| `model/effect.py` | `Effect`: `Assign` or `When` |
| `model/branch.py` | `Branch(probability, effects)`: one possible outcome of an action |
| `model/transition.py` | `Transition(action, branches)`: what performing an action does |
| `model/transition_model.py` | `TransitionModel(transitions)`: every transition of a domain |
| `model/outcome_distribution.py` | `OutcomeDistribution(outcomes)`: each outcome (new state) with its probability |
| `builder/transition_model_builder.py` | `TransitionModelBuilder`: collects transitions; rejects a repeated action or probabilities that don't sum to 1 |
| `service/predictor.py` | `Predictor`: applies a transition's branches to give the outcome distribution |

## Usage

```python
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.constant import Constant
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

hit = Branch(0.25, (Assign(StateVariable("score"), Constant(1)),))
miss = Branch(0.75, ())
model = TransitionModelBuilder().with_transition("shoot", (hit, miss)).build()

names = VariableNameMapper()
predictor = Predictor(Interpreter(names), names, ExpressionTextMapper(names), ActionTextMapper())
predictor.predict(model, State((("score", 0),)), Action("shoot", ()))
# OutcomeDistribution(outcomes=((State(variables=(('score', 1),)), 0.25), (State(variables=(('score', 0),)), 0.75)))
```

## How transitions apply

- Each branch starts from the given state and applies its effects in order; each effect sees the state left by the
  ones before it.
- `Assign` changes only variables that already exist; an unknown variable raises `KeyError`. A state never gains or
  loses variables.
- A `When` condition must evaluate to `True` or `False`; any other value raises `TypeError`.
- Each branch gives its own entry, even when two branches produce the same state.
- Predicting an action that has no transition raises `KeyError`.
- Branches are discrete; continuous distributions come with the first domain that needs them.

## Logs

Logger `openmind.predictor.service.predictor`:

- `DEBUG Set cell(1,1) = 'X'`
- `DEBUG When turn == 'X': true` (or `false`)
- `INFO place(col=1, row=1) gives 1 outcome(s) with probabilities [1.0]`

Actions and conditions are written by `ActionTextMapper` and `ExpressionTextMapper`.

## Notes

- Tests: `builder/transition_model_builder_tests.py`, `service/predictor_tests.py`; integration:
  `test/integration/tictactoe_transitions_tests.py`.
