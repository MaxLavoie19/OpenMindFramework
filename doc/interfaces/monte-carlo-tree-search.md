# Interfaces: the Monte-Carlo tree search

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

A search is one model of the planning task, and this is the one chess needs: OMF has to generate moves and explore
the lines they lead to. `Minimax` reads a small game out to its end and `Improvised` doesn't explore at all; neither
can play chess.

## One search, not two

The old search was two: PUCT down a tree where players take turns, and regret matching where they act at once. In the
new design there are no turns — every player plays all the time, and the constraints leave a player no action outside
their turn — so the two are one search over the players who can act here:

| Acting here | How the action is chosen |
|---|---|
| one player | PUCT: the move value heuristic's rating as the prior, the visits and values as the rest |
| several players | regret matching, each acting player over its own actions, and the node's average strategy is what it settles on |

Both back up the same thing: what the leaf was worth to every player.

```python
class MonteCarloTreeSearch:                   # a model of the planning task
    """Explores the lines a position leads to, as many nodes as it is given, and gives the strategy it settled on."""

    def __init__(self, utility: Utility, trees: SearchTreeCache) -> None: ...

    def plan(self, model, knowledge_base, node: Node, guidance: Guidance, settings: SearchSettings) -> Strategy | None: ...
```

## An iteration

1. **Select** down the tree, node by node, until a node that isn't expanded or a position the game is over in. Each
   acting player picks as the table above says; the joint action is what they picked together.
2. **Follow the joint action**: the predictor gives its outcomes, and one is drawn by their probabilities. A child is
   kept per outcome, so a chance node's branches are what the game actually offers.
3. **Expand** the node reached: its legal actions per acting player, narrowed by the policy picker where the caller
   gave policies, each with the prior its move value model rates it at — uniform where no model rates it.
4. **Value the leaf**: what the game paid, where it is over — a win always carries the payoff value — or what the
   position value heuristic says. **No playouts**: a game isn't played out to its end to find out what a position is
   worth. Where neither can value it, the iteration adds nothing and the node is left for another model.
5. **Back up** what the leaf is worth to each player, along the path: visits and values everywhere, and regrets at the
   nodes where several acted.

**Another agent's moves are rated by that agent's model**, as `Guidance.agents` holds them, so the reply explored is
the reply expected of that agent rather than of the agent itself.

```python
@dataclass(slots=True)
class SearchNode:
    """A node of the tree: the position, what exploring it found, and where its joint actions led."""

    node: Node
    visits: int = 0
    values: tuple[float, ...] = ()                       # summed over the visits, per player
    acting: tuple[str, ...] = ()                         # the players with an action here
    actions: dict[str, tuple[ActionStatistics, ...]] = field(default_factory=dict)
    children: dict[tuple[JointAction, State], "SearchNode"] = field(default_factory=dict)


@dataclass(slots=True)
class ActionStatistics:
    """One action of one player at one node: what it was rated, how often it was taken, what it brought, and what not
    taking it cost, which is what regret matching reads."""

    action: Action
    prior: float
    visits: int = 0
    value: float = 0.0
    regret: float = 0.0
    strategy: float = 0.0                                # summed, so the average strategy can be read off it
```

## What it gives back

A `Strategy`: a move distribution per state, as the design asks. At every node where the player it plans for acts, the
distribution is what exploring settled on — the visits per action where one player acts, the average strategy where
several did. The actor reads the state it is in and plays what the strategy says there; where the strategy says
nothing, it waits and the planner strategizes from there.

## How much it explores

`SearchSettings.nodes` is how many nodes it may expand, which the time management policy works out from the seconds
and what a node has been costing. `depth` caps how far it looks. Two more settings belong to how it explores:

```python
@dataclass(frozen=True, slots=True)
class SearchSettings:
    nodes: int | None = None
    depth: int | None = None
    seed: int = 0
    exploration: float = 1.5                   # PUCT's weight on the prior against what the visits found
    regret_exploration: float = 0.6            # how much of a regret-matching choice stays uniform
    temperature: float = 1.0                   # 1 plays what it explored, 0 the best of it alone
```

They are settings rather than constants in the code: a caller sets them, and the time management policy sets them
once it is trained on what each setting brought.

## The tree between moves

The design has the planner keeping its tree between moves and pruning what can no longer be reached. The services are
stateless, so the tree lives in an injected cache, as the solver's solutions and chess's boards do:

```python
class SearchTreeCache:                         # injected, this process's own
    """The tree each root was explored from, so the next move carries on from what the last one found."""

    def tree(self, state: State) -> SearchNode | None: ...
    def keep(self, state: State, tree: SearchNode) -> None: ...
```

A move that was explored is already a child of the root, so the next search starts from that subtree and the rest is
dropped, which is the pruning the design asks for. It registers with the memory guard, as every other cache does.

## What this step leaves alone

- **The semi-determinized search**, which runs the same search over the hypotheses about what can't be seen: the same
  port, the hypothesis models, and its own step.
- **Optimizing instead of expanding** within a policy: the budget step decides between them; this search expands.
- **Utility over several goals** at a leaf: minimax doesn't read it yet either, and the two should start together.

## What the search recommends

**Decided (Maxime):** trying every move once is not a requirement — a move the search didn't think worth a single
look isn't one to recommend, so it simply carries none of the strategy. What was explored is what is played, each as
often as it was looked at, so a move tried once among dozens carries almost none of the distribution.

**And exploring a move isn't the same as playing it**, which is the temperature's: at 1 the strategy is what was
explored, which is what bootstrapping and self-play want, since a game only teaches what it was allowed to try; at 0
it is the best of what was explored and nothing else, which is what competitive play wants. In between it leans
toward the best without dropping the rest.

## A search with nothing to value a position with

**Decided (Maxime):** a search without heuristics only works if it finds a win. So it runs either way — in a small
game it stumbles on wins and learns from what they paid — and where it valued nothing at all, it says so in the logs.
What answers this for chess is not refusing to search: it is inferring heuristics from the rules and the relaxed
games, which is the bootstrap's step.

## Open

Nothing is open; the step waits for your OK.
