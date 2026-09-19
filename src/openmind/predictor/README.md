# predictor

## Purpose

The prediction task: from a state and the actions every agent takes at once, the possible outcomes, each with its
probability. An end state carries each player's payoff. Any model can fill the task (`model/predictor.py`, the port);
the one here runs a ruleset's effects rules. Games of chance are actions with several effects rules, each with its
chance.

## Content

| File | What it is |
|---|---|
| `model/predictor.py` | `Predictor[Model]`, the port: `predict(model, state, joint) -> OutcomeDistribution`, a stateless service given the model it runs |
| `model/outcome_distribution.py` | `OutcomeDistribution(outcomes)`: each outcome (new state) with its probability |
| `service/rule_predictor.py` | `RulePredictor(effects_runner)`: stateless, runs the effects rules of the RBS it is given (`rbs/model/rule_based_system.py`); `predict(rbs, state, joint)`, and `predict_action(rbs, state, action)` for one action run without a player |
| `service/effects_runner.py` | `EffectsRunner(rule_caller, action_text_mapper)`: `run(state, action, effects, definitions=None)` and `run_joint(state, joint, effects, together=(), definitions=None)`, the effects given as `(chance, rule)` pairs |
| `factory/predictor_factory.py` | `create_effects_runner()`, `create_rule_predictor()` |

## How effects apply

- Each effects rule runs on its own copy of the state, with the action's parameters by name and the definitions its
  effects see. What it leaves in the state's models is the outcome; statements run in order.
- Each player's action runs in the joint's order, on every outcome so far, its effects reading its player as `player`.
  The chances multiply, one entry per combination. Then the rules for what the actions lead to together run.
- A joint without an action, or an action with a parameter named `player`, raises `ValueError`; an action without an
  effects rule raises `KeyError`.
- Outcomes are discrete; continuous distributions come with the first game that needs them.

## Logs

Logger `openmind.predictor.service.effects_runner`, per outcome, one line per model whose value changed, then the
summary:

- `DEBUG Set cell = Grid(...)`
- `DEBUG place(col=1, row=1) gives 1 outcome(s) with probabilities [1.0]`
- `DEBUG A: throw(shape='rock'), B: throw(shape='paper') gives 1 outcome(s) with probabilities [1.0]`

## Notes

- Tests: `service/effects_runner_tests.py`, `service/rule_predictor_tests.py`.
