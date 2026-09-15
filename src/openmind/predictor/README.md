# predictor

## Purpose

Transitions: what performing an action does to a state. Given a domain's transition model, a state and an action, the
predictor gives the outcome probability distribution: each possible new state with its probability. Transitions also
end the game and set payoffs, since a payoff is a predicted outcome of a transition to a win or a draw. Games of chance
are branches with probabilities below 1.

## Content

| File | What it is |
|---|---|
| `model/branch.py` | `Branch(probability, effects)`: one possible outcome of an action; `effects` is a Python script (`PythonRule`) whose assignments to state variables make the new state, or the project's own function `(state, **parameters) -> State` (`EffectsRule`, see `rule/README.md`) |
| `model/transition.py` | `Transition(action, branches)`: what performing an action does |
| `model/transition_model.py` | `TransitionModel(transitions, definitions=None, resolution=None)`: every transition of a domain, the definitions their effects see, and, where players act at once, the resolution branches run after every player's action |
| `model/outcome_distribution.py` | `OutcomeDistribution(outcomes)`: each outcome (new state) with its probability |
| `builder/transition_model_builder.py` | `TransitionModelBuilder`: collects transitions, definitions and the resolution (`with_resolution(branches)`); rejects a repeated action or probabilities that don't sum to 1 |
| `service/predictor.py` | `Predictor(rule_caller, action_text_mapper)`: runs each branch's effects on the state to give the outcome distribution; `predict_joint(model, state, joint)` for actions taken at once |
| `builder/predictor_builder.py` | `PredictorBuilder`: wires a predictor with its rule compiler, rule runner and action text mapper |
| `factory/predictor_factory.py` | `create_predictor()` |

## Usage

```python
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.state import State

hit = Branch(0.25, PythonRule("score = score + POINTS"))
miss = Branch(0.75, PythonRule(""))
model = TransitionModelBuilder().with_transition("shoot", (hit, miss)).with_definitions(PythonRule("POINTS = 1")).build()

create_predictor().predict(model, State((("score", 0),)), Action("shoot", ()))
# OutcomeDistribution(outcomes=((State(variables=(('score', 1),)), 0.25), (State(variables=(('score', 0),)), 0.75)))
```

## How transitions apply

- Each branch's effects script runs on its own copy of the state, with the action's parameters by name and the model's
  definitions (see `rule/README.md` for what a script sees). What it leaves in the state's variables is the outcome;
  statements run in order, so each sees what the ones before it changed.
- A state can gain variables, never lose them: an effect on an index the state doesn't have, under a base it has, adds
  the variable after the state's own (see `rule/README.md`). A game whose history has no known length, such as the
  prisoner's dilemma without a known last round, grows its state this way.
- Each branch gives its own entry, even when two branches produce the same state.
- Predicting an action that has no transition raises `KeyError`.
- Branches are discrete; continuous distributions come with the first domain that needs them.

Actions taken at once (`predict_joint`, a `JointAction`):

- Each player's action runs in the joint's order, on every outcome so far, its effects reading its parameters and its
  player as `player`: `hand[player] = shape`. The branches' probabilities multiply, one entry per combination.
- Then the model's resolution runs on each outcome, reading the combined choices: it compares the hands, sets the
  payoffs, and says who acts next by setting the players' flags (see `world/README.md`).
- A joint without an action, or an action with a parameter named `player`, raises `ValueError`.

## Logs

Logger `openmind.predictor.service.predictor`, per branch, one line per variable whose value changed, in the state's
order, then the summary:

- `DEBUG Set cell(1,1) = 'X'`
- `DEBUG place(col=1, row=1) gives 1 outcome(s) with probabilities [1.0]`
- `DEBUG A: throw(shape='rock'), B: throw(shape='paper') gives 1 outcome(s) with probabilities [1.0]`, for actions taken
  at once

Actions are written by `ActionTextMapper`.

## Notes

- Tests: `builder/predictor_builder_tests.py`, `builder/transition_model_builder_tests.py`,
  `factory/predictor_factory_tests.py`, `service/predictor_tests.py`; integration:
  `test/integration/tictactoe_transitions_tests.py`, `test/integration/tictactoe_fourinarow_transitions_tests.py`.
