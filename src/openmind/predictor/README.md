# predictor

## Purpose

What performing an action does to a state. Given a state, an action and the effects rules that say what that action
leads to, the predictor gives the outcome probability distribution: each possible new state with its probability.
Effects also end the game and set payoffs, since a payoff is a predicted outcome of a move to a win or a draw. Games of
chance are actions with several effects rules, each with its chance.

The predictor is the RBS's for its `effects` rules, the way the CSP is its solver for legal moves: the RBS knows which
effects rules belong to which action and hands them over (`rbs/service/rule_based_system.py`), so `rbs.outcomes(state,
action)` is how a game is played forward. The predictor itself knows nothing of a game.

## Content

| File | What it is |
|---|---|
| `model/outcome_distribution.py` | `OutcomeDistribution(outcomes)`: each outcome (new state) with its probability |
| `service/predictor.py` | `Predictor(rule_caller, action_text_mapper)`: `predict(state, action, effects, definitions=None)` runs each effects rule on the state to give the outcome distribution, `effects` being `(chance, rule)` pairs; `predict_joint(state, joint, effects, together=(), definitions=None)` for actions taken at once, `effects` each action's pairs by name and `together` what the players' choices lead to once all are made |
| `builder/predictor_builder.py` | `PredictorBuilder`: wires a predictor with its rule compiler, rule runner and action text mapper |
| `factory/predictor_factory.py` | `create_predictor()` |

## Usage

```python
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rbs.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.state import State

hit, miss = (0.25, PythonRule("score = score + POINTS")), (0.75, PythonRule(""))

create_predictor().predict(State((("score", 0),)), Action("shoot", ()), (hit, miss), PythonRule("POINTS = 1"))
# OutcomeDistribution(outcomes=((State(variables=(('score', 1),)), 0.25), (State(variables=(('score', 0),)), 0.75)))
```

## How effects apply

- Each effects rule runs on its own copy of the state, with the action's parameters by name and the definitions
  given (see `rbs/README.md` for what a script sees). What it leaves in the state's variables is the outcome;
  statements run in order, so each sees what the ones before it changed.
- A state can gain variables, never lose them: an effect on an index the state doesn't have, under a base it has, adds
  the variable after the state's own (see `rbs/README.md`). A game whose history has no known length, such as the
  prisoner's dilemma without a known last round, grows its state this way.
- Each effects rule gives its own entry, even when two give the same state.
- Predicting an action without an effects rule raises `KeyError`.
- Outcomes are discrete; continuous distributions come with the first game that needs them.

Actions taken at once (`predict_joint`, a `JointAction`):

- Each player's action runs in the joint's order, on every outcome so far, its effects reading its parameters and its
  player as `player`: `hand[player] = shape`. The chances multiply, one entry per combination.
- Then the `together` rules run on each outcome, reading the combined choices: it compares the hands, sets the
  payoffs, and says who acts next by setting the players' flags (see `world/README.md`).
- A joint without an action, or an action with a parameter named `player`, raises `ValueError`.

## Logs

Logger `openmind.predictor.service.predictor`, per outcome, one line per variable whose value changed, in the state's
order, then the summary:

- `DEBUG Set cell(1,1) = 'X'`
- `DEBUG place(col=1, row=1) gives 1 outcome(s) with probabilities [1.0]`
- `DEBUG A: throw(shape='rock'), B: throw(shape='paper') gives 1 outcome(s) with probabilities [1.0]`, for actions taken
  at once

Actions are written by `ActionTextMapper`.

## Notes

- Tests: `builder/predictor_builder_tests.py`, `factory/predictor_factory_tests.py`, `service/predictor_tests.py`; integration:
  `test/integration/tictactoe_transitions_tests.py`, `test/integration/tictactoe_fourinarow_transitions_tests.py`.
