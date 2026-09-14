# mcts

## Purpose

Monte-Carlo Tree Search: estimates how good each legal action is by simulating many games from a state, then picks
the most visited action. It finds legal actions with the CSP, outcomes with the predictor, and who acts and what each
player gets through the variables named in `Players`. It knows nothing about any particular domain. A model behind the
`ActionRater` interface, such as the RBS, can guide it, and a model behind `PositionValuer`, such as the RBS's value
rules, can value the positions its rollouts reach instead of playing them to the end.

## Content

| File | What it is |
|---|---|
| `model/search_settings.py` | `SearchSettings(iterations, exploration, seed)`: how long and how widely to search; `seed=None` is unseeded |
| `model/action_statistics.py` | `ActionStatistics(action, visits, mean_payoff)`: a root action's visits and mean payoff for the player acting at the root (0.0 when never visited) |
| `model/action_sample.py` | `ActionSample(state, player, action, visits, mean_payoff)`: an action expanded anywhere in the tree, with its visits and mean payoff for the player to act |
| `model/search_result.py` | `SearchResult(player, statistics, chosen, samples)`: every root action's statistics, the most visited action, and a sample for every expanded action |
| `model/action_rater.py` | `ActionRater`: the interface of a model rating actions, `rate(state, actions)` giving each action's expected payoff for the player to act, or `None` |
| `model/guidance.py` | `Guidance(rater, prior_weight, rollout_temperature, guided_rollouts=True)`: how a rater steers the search, and whether rollouts follow its ratings |
| `model/position_valuer.py` | `PositionValuer`: the interface of a model valuing positions, `value(state)` giving each player's expected payoff in the order of the players' names, or `None` |
| `model/leaf_valuation.py` | `LeafValuation(valuer, rollout_actions=0)`: how a valuer ends iterations, valuing the position a rollout reaches after that many actions |
| `model/decision_node.py` | `DecisionNode`: a state in the tree where a player picks an action, with the rater's ratings when guided; mutable |
| `model/chance_node.py` | `ChanceNode`: an action in the tree with its possible outcomes; mutable |
| `service/tree_search.py` | `TreeSearch`: runs the search, guided or not |

## Usage

```python
import math

from openmind.agent.factory.domain_factory import create_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader

tree_search = TreeSearch(create_solver(), create_predictor(), StateReader(), ActionTextMapper())
domain = create_domain("tictactoe")
result = tree_search.search(
    domain.problem, domain.transitions, domain.players, domain.initial_state, SearchSettings(500, math.sqrt(2), 1)
)
result.chosen   # Action(name='place', parameters=(('col', 2), ('row', 2)))
# result.statistics holds every root action's visits and mean payoff; result.samples every expanded action's
```

To guide the search, pass `Guidance(rater, prior_weight, rollout_temperature, guided_rollouts=True)` after the settings;
to value positions, pass `LeafValuation(valuer, rollout_actions=0)` after that. `agent/builder/agent_builder.py` does
this wiring for the agent.

## How a search works

Each iteration:

1. **Selection:** from the root, a decision node with untried legal actions tries one. Otherwise it follows the child
   with the highest UCT score: the acting player's mean payoff plus exploration × √(ln parent visits / child visits).
2. **Chance:** a chance node draws an outcome by its probability; each distinct outcome has its own decision node.
3. **Rollout:** from the first new decision node, legal actions are played until none is legal, or, with a valuation,
   until the valuer values the position.
4. **Backpropagation:** the final state's payoffs are added to every chance node on the path, and every node on the
   path counts a visit.

After the iterations, the most visited root action is chosen; ties go to the first action in the solver's order.

Without guidance, untried actions are tried in random order and rollouts pick actions uniformly. With guidance:

- The rater rates a decision node's legal actions when the node is created. An action rated `None` gets the mean of
  the other ratings; a node where every rating is `None` is searched unguided.
- Untried actions are tried from the highest to the lowest rating; equal ratings keep a random order.
- Selection adds `prior_weight × rating / (child visits + 1)` to the UCT score.
- Rollouts draw each action with probability proportional to `exp((rating − best rating) / rollout_temperature)`.
  With `guided_rollouts` false, rollouts pick uniformly and the rater only rates the tree's decision nodes: rating
  every rollout step is where a costly rater, such as rules reading `wins()`, spends most of a search.

With a valuation:

- After `rollout_actions` rollout actions, 0 valuing the new decision node itself, a position with a legal action gets
  the valuer's payoffs, which are backpropagated like a finished game's. When the valuer gives `None`, the rollout plays
  on to the end.
- A finished game keeps its own payoffs: the valuer only values positions in play. A valuer whose values stay inside the
  payoff range, as value rules' do, never rates a position above a real win.

Also:

- A search from a state with no legal action raises `ValueError`.
- A state with no legal action whose payoffs are not all numbers raises `ValueError`.
- The same seed gives the same result.
- The nodes are mutable models, used only inside `TreeSearch`.

## Logs

Logger `openmind.mcts.service.tree_search`:

- `INFO Searching <iterations> iterations for <player>`
- `INFO <action>: <visits> visits, mean payoff <mean> for <player>`, once per root action
- `INFO Most visited: <action>`
- `DEBUG Iteration <n>: <actions from the root>, rollout of <n> actions, payoffs <player>=<payoff> ...`, with
  `, then valued` after the rollout's length when a valuer gave the payoffs

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
