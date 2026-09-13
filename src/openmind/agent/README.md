# agent

## Purpose

The agent and the domains it works on. The agent chooses actions by searching with MCTS. Tic-tac-toe is a domain
within the agent, not a domain in the code: it exists only as this domain's factory recipes and their constants.

## Content

| File | What it is |
|---|---|
| `model/domain.py` | `Domain(name, initial_state, problem, transitions, players)`: a domain within the agent |
| `builder/domain_builder.py` | `DomainBuilder`: collects a domain's parts; rejects missing parts |
| `factory/domain_factory.py` | `create_domain(name)`: creates a domain from its name (`"tictactoe"`) |
| `service/agent.py` | `Agent`: searches a domain's state with MCTS, guided by a rater when built with one; `search` gives the whole result, `choose` the action |
| `model/policy.py` | `Policy`: anything with `choose(domain, state) -> Action`; `Agent` and `RandomPolicy` are policies |
| `service/random_policy.py` | `RandomPolicy`: chooses uniformly among the legal actions; a baseline opponent |
| `builder/agent_builder.py` | `AgentBuilder`: sets iterations, exploration, seed and guidance (`with_guidance(rater)`), and wires the services the agent searches with; rejects missing settings and fewer than 1 iteration |
| `factory/agent_factory.py` | `create_agent(iterations=1000, seed=None)`: an agent searching with the exploration weight √2 |
| `constant/agent_constant.py` | Default iterations (1000), exploration weight (√2), and the guidance's prior weight (1.0) and rollout temperature (0.2) |
| `constant/tictactoe_constant.py` | Domain name, board size, players, empty and unset values, payoff values, variable and action names |
| `factory/tictactoe_factory.py` | `create_tictactoe_domain()`, assembled from `create_tictactoe_initial_state()`, `create_tictactoe_problem()`, `create_tictactoe_transitions()` and `create_tictactoe_players()` |

## Tic-tac-toe

### Players

`Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)"))`: `turn` names the player to act, and `payoff(X)` and
`payoff(O)` hold the players' payoffs.

### State variables

| Variable | Values | Initial |
|---|---|---|
| `cell(row,col)`, row and col in 1..3 | `"X"`, `"O"`, or `None` when empty | `None` |
| `turn` | `"X"` or `"O"` | `"X"` |
| `payoff(X)`, `payoff(O)` | 1 for a win, 0 for a loss, 0.5 each for a draw; `None` until the game ends | `None` |

### Constraints (CSP)

The action `place(row, col)`, with row and col in 1..3, is legal when all of these hold, checked in this order:

1. `payoff(X)` is unset.
2. `payoff(O)` is unset.
3. `cell(row,col)` is empty.

### Transitions (predictor)

`place(row, col)` has one branch with probability 1. Its effects apply in order:

1. `cell(row,col)` = `turn`.
2. For each player P, when one of the 8 lines (3 rows, 3 columns, 2 diagonals) is all P: `payoff(P)` = 1 and the
   other player's payoff = 0.
3. When `payoff(X)` is unset and no cell is empty: `payoff(X)` = 0.5 and `payoff(O)` = 0.5.
4. When `turn` is X, `turn` = O; otherwise `turn` = X.

## Usage

```python
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.domain_factory import create_domain

domain = create_domain("tictactoe")
action = create_agent(iterations=500, seed=1).choose(domain, domain.initial_state)
```

Using the solver and predictor directly:

```python
from openmind.agent.factory.domain_factory import create_domain
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

names = VariableNameMapper()
interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
domain = create_domain("tictactoe")
actions = Solver(interpreter, expression_text, action_text).solve(domain.problem, domain.initial_state)
# 9 actions; actions[0] is Action(name='place', parameters=(('col', 1), ('row', 1)))
predictor = Predictor(interpreter, names, expression_text, action_text)
distribution = predictor.predict(domain.transitions, domain.initial_state, actions[0])
# OutcomeDistribution(outcomes=((State(variables=(('cell(1,1)', 'X'), ('cell(1,2)', None), …, ('turn', 'O'))), 1.0),))
```

## Notes

- Tests: `builder/agent_builder_tests.py`, `builder/domain_builder_tests.py`, `factory/agent_factory_tests.py`,
  `factory/domain_factory_tests.py`, `factory/tictactoe_factory_tests.py`, `service/agent_tests.py`, `service/random_policy_tests.py`; integration:
  `test/integration/tictactoe_actions_tests.py`, `test/integration/tictactoe_search_tests.py`,
  `test/integration/tictactoe_transitions_tests.py`; end-to-end: `test/end_to_end/play_tictactoe_tests.py`.
