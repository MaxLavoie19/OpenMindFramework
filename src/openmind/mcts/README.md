# mcts

## Purpose

Monte-Carlo Tree Search: estimates how good each legal action is by simulating many games from a state, then picks
the most visited action. It finds legal actions with the CSP, outcomes with the predictor, and who acts and what each
player gets through the variables named in `Players`. It knows nothing about any particular domain.

## Content

| File | What it is |
|---|---|
| `model/search_settings.py` | `SearchSettings(iterations, exploration, seed)`: how long and how widely to search; `seed=None` is unseeded |
| `model/action_statistics.py` | `ActionStatistics(action, visits, mean_payoff)`: a root action's visits and mean payoff for the player acting at the root (0.0 when never visited) |
| `model/search_result.py` | `SearchResult(player, statistics, chosen)`: every root action's statistics and the most visited action |
| `model/decision_node.py` | `DecisionNode`: a state in the tree where a player picks an action; mutable |
| `model/chance_node.py` | `ChanceNode`: an action in the tree with its possible outcomes; mutable |
| `service/tree_search.py` | `TreeSearch`: runs the search |

## Usage

```python
import math

from openmind.agent.factory.domain_factory import create_domain
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader

names = VariableNameMapper()
interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
tree_search = TreeSearch(
    Solver(interpreter, expression_text, action_text),
    Predictor(interpreter, names, expression_text, action_text),
    StateReader(),
    action_text,
)
domain = create_domain("tictactoe")
result = tree_search.search(
    domain.problem, domain.transitions, domain.players, domain.initial_state, SearchSettings(500, math.sqrt(2), 1)
)
result.chosen   # Action(name='place', parameters=(('col', 2), ('row', 2)))
# result.statistics holds every root action's visits and mean payoff
```

`agent/builder/agent_builder.py` does this wiring for the agent.

## How a search works

Each iteration:

1. **Selection:** from the root, a decision node with untried legal actions tries one at random. Otherwise it follows
   the child with the highest UCT score: the acting player's mean payoff plus
   exploration × √(ln parent visits / child visits).
2. **Chance:** a chance node draws an outcome by its probability; each distinct outcome has its own decision node.
3. **Rollout:** from the first new decision node, random legal actions are played until none is legal.
4. **Backpropagation:** the final state's payoffs are added to every chance node on the path, and every node on the
   path counts a visit.

After the iterations, the most visited root action is chosen; ties go to the first action in the solver's order.

- A search from a state with no legal action raises `ValueError`.
- A state with no legal action whose payoffs are not all numbers raises `ValueError`.
- The same seed gives the same result.
- The nodes are mutable models, used only inside `TreeSearch`.

## Logs

Logger `openmind.mcts.service.tree_search`:

- `INFO Searching <iterations> iterations for <player>`
- `INFO <action>: <visits> visits, mean payoff <mean> for <player>`, once per root action
- `INFO Most visited: <action>`
- `DEBUG Iteration <n>: <actions from the root>, rollout of <n> actions, payoffs <player>=<payoff> ...`

The usage example above logs these INFO lines, and this DEBUG line for iteration 17:

```
INFO  Searching 500 iterations for X
DEBUG Iteration 17: place(col=2, row=1) > place(col=2, row=2), rollout of 7 actions, payoffs X=0.5 O=0.5
INFO  place(col=1, row=1): 65 visits, mean payoff 0.7 for X
INFO  place(col=2, row=1): 61 visits, mean payoff 0.6885245901639344 for X
INFO  place(col=3, row=1): 41 visits, mean payoff 0.5975609756097561 for X
INFO  place(col=1, row=2): 42 visits, mean payoff 0.5952380952380952 for X
INFO  place(col=2, row=2): 86 visits, mean payoff 0.7732558139534884 for X
INFO  place(col=3, row=2): 40 visits, mean payoff 0.5875 for X
INFO  place(col=1, row=3): 48 visits, mean payoff 0.6354166666666666 for X
INFO  place(col=2, row=3): 43 visits, mean payoff 0.6046511627906976 for X
INFO  place(col=3, row=3): 74 visits, mean payoff 0.7364864864864865 for X
INFO  Most visited: place(col=2, row=2)
```

At DEBUG, every rollout step also logs the solver's candidates and the predictor's effects.

## Notes

- Tests: `service/tree_search_tests.py`; integration: `test/integration/tictactoe_search_tests.py`.
