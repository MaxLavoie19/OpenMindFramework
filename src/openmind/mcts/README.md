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
| `model/search_settings.py` | `SearchSettings(iterations, exploration, seed, rollout_limit=None, unfinished_payoff=None, regret_exploration=0.1, seconds=None, selection='ucb1', puct_exploration=1.5, prior=None)`: how long and how widely to search; `selection` is `ucb1` or `puct`, with PUCT's exploration weight and prior (None: every action alike); the search stops at its iterations or its seconds, whichever comes first, and needs at least one of them; `seed=None` is unseeded; a rollout limit stops rollouts after that many actions; `regret_exploration` is the share of uniform choice mixed into regret matching where players act at once |
| `constant/mcts_constant.py` | `DEFAULT_REGRET_EXPLORATION` (0.1); the selections `UCB1` and `PUCT`, `DEFAULT_PUCT_EXPLORATION` (1.5) and `DEFAULT_PRIOR_TEMPERATURE` (0.1); the priors entry points name, `uniform`, `rater` and `value` |
| `model/move_prior.py` | `MovePrior`: the interface of a prior, `priors(state, actions)` giving one share per action summing to 1, and `name` for the logs |
| `service/uniform_prior.py` | `UniformPrior`: every action alike |
| `service/rater_prior.py` | `RaterPrior(rater, temperature)`: a rater's ratings through a softmax; an action it doesn't rate takes the mean of those it does |
| `service/valuation_prior.py` | `ValuationPrior(valuer, rbs, players, temperature)`: each action's outcomes valued for the player to act, weighed by their probabilities, through a softmax; a finished outcome counts at its payoffs, an action the valuer can't value takes the mean of the others; the RBS's `RuleValuer` works here |
| `service/softmax.py` | `softmax(values, temperature)`: shares from values, None taking the mean of the known ones |
| `factory/move_prior_factory.py` | `create_move_prior(kind, temperature, domain, rater=None, valuer=None)`: the prior a name stands for; a name it doesn't know, or a prior without the model it reads, raises `ValueError` |
| `model/simultaneous_node.py` | `SimultaneousNode`: a state in the tree where players act at once, with each player's legal actions, regrets, summed strategies, visits and payoffs, and the strategies predicted for other players at the root; mutable |
| `model/action_statistics.py` | `ActionStatistics(action, visits, mean_payoff)`: a root action's visits and mean payoff for the player acting at the root (0.0 when never visited) |
| `model/action_sample.py` | `ActionSample(state, player, action, visits, mean_payoff)`: an action expanded anywhere in the tree, with its visits and mean payoff for the player to act |
| `model/search_result.py` | `SearchResult(player, statistics, chosen, samples, strategy=(), iterations=0, seconds=0.0, budget=None, depth=0, option='search')`: every root action's statistics, the action chosen (the most visited), a sample for every expanded action, the iterations completed in the seconds taken, on a clock the step's budget, and the tree's depth, the most actions from the root an iteration went; `option` says how the move was chosen (`fallback and search`, `search` or `random`, see `agent/README.md`); the seconds, the budget and the option don't take part in comparing results |
| `model/action_rater.py` | `ActionRater`: the interface of a model rating actions, `rate(state, actions)` giving each action's expected payoff for the player to act, or `None` |
| `model/guidance.py` | `Guidance(rater, prior_weight, rollout_temperature, guided_rollouts=True)`: how a rater steers the search, and whether rollouts follow its ratings |
| `model/position_valuer.py` | `PositionValuer`: the interface of a model valuing positions, `value(state)` giving each player's expected payoff in the order of the players' names, or `None` |
| `model/leaf_valuation.py` | `LeafValuation(valuer, rollout_actions=0)`: how a valuer ends iterations, valuing the position a rollout reaches after that many actions |
| `model/decision_node.py` | `DecisionNode`: a state in the tree where a player picks an action, with the rater's ratings when guided and its actions' priors under PUCT, worked out once when the node is made; mutable |
| `model/chance_node.py` | `ChanceNode`: an action, or actions taken at once, in the tree with its possible outcomes; mutable |
| `service/tree_search.py` | `TreeSearch(state_reader, action_text_mapper, time_source=None)`: runs the search over a game's RBS, guided or not, drawing each iteration's state from the states a position could be when given them |

## Usage

```python
import math

from openmind.agent.factory.game_factory import create_game
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader

tree_search = TreeSearch(StateReader(), ActionTextMapper())
rbs = create_game("tictactoe", create_knowledge_base("tictactoe"))
result = tree_search.search(rbs, rbs.start(), SearchSettings(500, math.sqrt(2), 1))
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

**Under PUCT** (`selection='puct'`), selection doesn't try every legal action first. A decision node follows, among all
its legal actions, tried or not, the one with the highest Q + c · P · √N / (1 + n): Q the action's mean payoff for the
player to act, or, for an action not visited yet, the node's own mean payoff so far (0 before its first visit); P the
action's prior; N the node's visits and n the action's. Ties go to the higher prior, then to the first action. An
untried action chosen is expanded as under UCB1. A move the prior dislikes can stay unvisited while the iterations go
deep into the ones it likes, and √N still opens it eventually. Under drawn states, a legal action the node didn't have
when made takes the mean of its priors. Under PUCT a rater guides through `RaterPrior`, not through UCB1's rating bonus;
rollouts follow ratings as before. Where players act at once, selection stays regret matching. A selection other than `ucb1` or `puct`, or a negative PUCT exploration, raises
`ValueError`.

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

With a rollout limit:

- A rollout still in play after `rollout_limit` actions stops, and every player gets `unfinished_payoff`; 0 stops at
  the new decision node itself. A valuer due at the same length values the position first.
- A negative limit, or a limit without an unfinished payoff, raises `ValueError`.
- Domains whose random games run long, such as chess, need one unless a valuer values their positions.

Given the states a position could be (`search(..., possible)`), each iteration draws one of them, by its probability,
and plays on it: legal actions and outcomes come from the drawn state. The statistics of every drawn state add up in
the same tree. Legal actions can differ between draws: untried ones are still tried first, and selection chooses among
the tried actions legal in the drawn state.

OMF runs agents, not the game, so nothing hides a state from the searching player: the states it draws are the ones
it is given, never a referee's view of it.

Where players act at once (`search(..., player=..., predicted=...)`, a state whose players to act are flags; see
`world/README.md`), the search is simultaneous-move MCTS with regret matching (Lanctot, Lisý and Winands, 2013):

- Every node is a `SimultaneousNode`. Each player to act picks its action by regret matching: in proportion to its
  positive regrets, uniformly when none is positive, mixed with `regret_exploration` of uniform choice. The joint
  action's outcomes come from `Predictor.predict_joint`.
- After the iteration, each player's picked action adds the payoff to its statistics, and every action's regret grows
  by its outcome-sampling estimate, the picked action's payoff divided by the probability it was picked at, 0 for the
  others, minus the payoff. Each regret-matching strategy adds to the player's summed strategies.
- From the first new node, a rollout draws every player's action uniformly; valuations and the rollout limit apply as
  above. Guidance isn't used.
- `player` names the searching player, one of the players to act at the root, or `ValueError`. Its statistics are its
  root actions' visits and mean payoffs; `SearchResult.strategy` is its average strategy, the summed strategies
  normalized, and the action chosen is drawn from it, so the searching player isn't predictable.
- `predicted` gives other players to act at the root, by name, the strategy they play there instead of regret matching,
  as (action, probability) pairs. The searching player's strategy then heads toward a
  best response to it. A player not to act, the searching player itself, an action it can't take, a negative
  probability, or probabilities not summing to 1 raise `ValueError`.

Also:

- A search from a state with no legal action raises `ValueError`.
- A state with no legal action whose payoffs are not all numbers raises `ValueError`.
- The same seed gives the same result.
- The nodes are mutable models, used only inside `TreeSearch`.
- **Time.** `TreeSearch` reads time from the `TimeSource` it is built with, wall time (`WallTimeSource`) unless given
  another. A search with `seconds` checks the time between iterations and stops once they have passed, or at its
  iterations if those come first; it never stops an iteration halfway and always completes at least one. No
  iterations and no seconds, iterations below 1, or seconds of 0 or less raise `ValueError`.

## Logs

Logger `openmind.mcts.service.tree_search`:

- `INFO Searching <limits> for <player>`, the limits written `<n> iterations`, `for <s> seconds` or
  `up to <n> iterations or <s> seconds`
- `INFO <player> sees <n> states that could be true`, given the states the position could be
- `INFO <action>: <visits> visits, mean payoff <mean> for <player>`, once per root action
- `INFO Most visited: <action>`
- `INFO Searched <n> iterations in <s> seconds for <player>, <ucb1, or puct with the <prior> prior>, tree depth <d>`, after the iterations, before the root actions
- `DEBUG Iteration <n>: <actions from the root>, rollout of <n> actions, payoffs <player>=<payoff> ...`, with
  `, then valued` or `, then stopped at the rollout limit` after the rollout's length when a valuer or the limit gave
  the payoffs

The usage example above logs these INFO lines, and this DEBUG line for iteration 17:

```
INFO  Searching 500 iterations for X
DEBUG Iteration 17: place(col=2, row=1) > place(col=2, row=2), rollout of 7 actions, payoffs X=0.5 O=0.5
INFO  Searched 500 iterations in <seconds, as long as it took> seconds for X, ucb1, tree depth <d>
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

Where players act at once, the same logger writes instead:

- `INFO Searching <limits> for <player>, acting at once with <players>`
- `INFO Searched <n> iterations in <s> seconds for <player>, <selection>, tree depth <d>`
- `INFO <player> is predicted to play <action>=<probability> ...`, per predicted player
- `INFO <action>: <visits> visits, mean payoff <mean>, average strategy <probability> for <player>`, once per root action
- `INFO Sampled from the average strategy: <action>`
- `DEBUG Iteration <n>: <joint action> > ..., rollout of <n> actions, payoffs <player>=<payoff> ...`

## Notes

- Tests: `service/tree_search_tests.py`; integration: `test/integration/tictactoe_search_tests.py`,
  `test/integration/prisoners_dilemma_search_tests.py`.
