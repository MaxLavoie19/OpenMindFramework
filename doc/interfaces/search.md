# Interfaces: search, agent models and hypotheses

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-19)

- **Planning is a task, and a search is one model of it.** OMF enforces no search: minimax suits tic-tac-toe, Monte-
  Carlo tree search suits chess, semi-determinized Monte-Carlo tree search suits poker or Stratego, and a conversation
  is improvised rather than planned. `mcts` goes; the searches become models of the planning task, keeping what still
  fits of it — regret matching, PUCT, the node statistics.
- **An agent model is not another AI.** It is the position value and move value heuristics trained to predict one
  agent's play, through the ports that already exist. So there is no agent model port, and `agent model` stops being a
  named task: an agent's models live in that agent's context.
- **The caller passes the agent models.** Playing Minidrow19 calls for Minidrow19's; playing a stranger calls for a
  generic one, such as a network trained to predict human moves.
- **Hypotheses about hidden information are a probability distribution**: over the possibilities where they can be
  counted, over policies where they can't. How a model arrives at one is its own business.
- **An optimizer reads the knowledge base**, because composing a message needs shared references, what was said
  before, and how much the other knows of each topic. A heuristic still reads only a node.
- **A win always carries the payoff value.** A finished position is worth what the game paid, never what a model
  guesses.

## hypothesis: what can't be seen

```python
@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One guess at what can't be seen, with how likely it is. Either the hidden part filled in — the cards that player
    holds — or the policy that agent is following, where the possibilities can't be counted."""

    likelihood: float
    state: State | None = None            # the state as it would be, hidden parts filled in
    policy: str | None = None             # a policy id: the way that agent is playing


class Hypotheses[Model](Protocol):        # the hypothesis task
    """What the hidden parts might be, as a distribution: every hypothesis with how likely it is, summing to 1. Empty
    where the model knows nothing, and empty for a game that hides nothing."""

    def about(self, model: Model, node: Node, agent: str) -> tuple[Hypothesis, ...]: ...
```

The task is named `hypothesis` in the task constants. A game that hides nothing registers no model for it, and the
search then runs on the state it was given.

## search: planning models

```python
class Planner[Model](Protocol):           # the planning task
    """What to do from this node, and what exploring found: a search plans by exploring, an improvised planner doesn't
    plan at all. None where the model has nothing to say here."""

    def plan(self, model: Model, knowledge_base, node: Node, guidance: Guidance, settings) -> SearchResult | None: ...
```

The planning models this step builds:

| Model | What it does | Suits |
|---|---|---|
| `Minimax` | the tree to its end, each agent taking its best | small games: tic-tac-toe |
| `MonteCarloTreeSearch` | PUCT over the move value prior, leaves valued by the position heuristic, no playouts | chess |
| `SemiDeterminizedMonteCarloTreeSearch` | the same, run over the hypotheses about what can't be seen | poker, Stratego |
| `Improvised` | no tree: the policy picked is optimized once, and that is the move | a conversation, or no time to think |

They share their pieces — expanding, backing up, regret matching — so what they differ in is how much they explore and
over what.

```python
@dataclass(slots=True)
class SearchNode:
    """A node of the tree: the heuristic's node, what exploring it found, and the children it led to."""

    node: Node                            # the state, its game, its features
    visits: int = 0
    total_value: float = 0.0              # summed over the players, in the players' order, for each visit
    children: dict[JointAction, "SearchNode"] = field(default_factory=dict)


class TreeSearch:                         # stateless: built once, given everything it needs
    def search(self, knowledge_base, game, node, settings, guidance) -> SearchResult: ...


@dataclass(frozen=True, slots=True)
class Guidance:
    """The models the search runs with: whose they are, and which the caller chose."""

    player: str                                        # who the search is for
    position_value: tuple[RuleBasedSystem | object, PositionValuer]   # the model and the service running it
    move_value: tuple[object, MoveRater]
    agents: Mapping[str, tuple[object, MoveRater]]     # each other agent's model, by name
    hypotheses: tuple[object, Hypotheses] | None = None
    policies: tuple[Policy, ...] = ()
```

**How a node expands**, as the design says:
1. the CSP gives the legal actions of every player with one;
2. the policies worth expanding are picked, and each gives its candidates by expanding or optimizing;
3. the predictor gives the outcomes of the joint action.

**What a leaf is worth:** the payoffs where the game ended, the position value heuristic otherwise, and nothing where
neither can say. Utility weighs the payoffs over the goals and preferences.

**Where several agents act at once:** regret matching over the joint actions, as today.

**Where another agent acts:** its own move value model rates its actions, so the reply comes from the agent model the
caller passed.

**Hidden information:** the search draws hypotheses from the hypothesis model, runs over each, and weighs what it
found by the hypothesis's likelihood. A policy hypothesis narrows what that agent is taken to be doing; a state
hypothesis replaces the node's state.

## What this step leaves alone

- **How much to search:** the time management policy, at the budget step. Until then the settings carry a node count.
- **Training an agent model** on an opponent's games: the training step.
- **Rhetoric's optimizer** reading the knowledge base: the port carries it, the models come with rhetoric.

## Open

1. **What `search` replaces.** `mcts` holds the old tree search, its nodes, priors and settings, and `agent` holds the
   loop calling it. Deleting `mcts` breaks `agent`, `training`, `dashboard` and the entrypoints until their own steps.
   Delete now and leave them broken, or keep `mcts` until the agent step, with `search` beside it?
2. **What a search gives back.** Today's `SearchResult` carries the chosen action, the statistics per action, the
   samples for training and the budget. Keep that shape, or cut it to what the agent loop needs and let training ask
   for what it wants?

## As coded (2026-09-19)

- `mcts` is deleted. `agent`, `training`, `dashboard` and `openmind-play` import it and stay broken until their own
  steps; `openmind-solve` doesn't and still runs.
- `search/model`: the `Planner` port, `Strategy` (a move distribution per state), `Guidance`, `SearchSettings`,
  `Hypothesis` and the `Hypotheses` port.
- `search/service`: `Improvised`, which picks a policy and takes its optimizer's action, and `Minimax`, which reads a
  small game out to its end, valuing a finished position at what the game paid.
- Task constants: `hypothesis` is named, `agent model` is gone — an agent model is the heuristics trained for that
  agent.
- `RulePolicyValuer` answers "nothing known" where there is no model, so a caller planning without a policy value
  model still gets the default policy.

### Not built here

- Monte-Carlo tree search and its semi-determinized form: later, as models of the same task.
- Utility isn't read by the planners yet: minimax values a finished position at its payoffs, and weighing goals and
  preferences into a leaf's value comes when the agent loop chooses between models.
- Nothing registers a planner as a model yet; the agent step wires the choice.
