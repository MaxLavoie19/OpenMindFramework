# Interfaces: the context hierarchy

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-19)

- **A parent delegates a goal and a budget.** The child runs its own agent loop, in its own context, with its own
  models, and reports back what came of it. It is a call, not a task queue.
- **The parent decides, per delegation, whether the child's seconds are carved out of its own or run alongside it.**
  Carving leaves the parent that much less; running alongside costs it nothing in seconds. Noted and not dug further:
  seconds aren't the only thing shared — a child running alongside takes a core the parent could have used — and
  nothing here weighs cores.
- **A level sees an abstraction of the state, made by a model of the `abstraction` task.** Every level abstracts from
  what the world holds, each in its own way: macromanagement sees the economy and the army, micromanagement sees the
  units in this fight. A level with no abstraction model sees the state as it is.
- **What a delegation brought is recorded in the knowledge base**, as the task record it already has, so the
  next-best-task loop can learn what delegating to that level is worth.
- **The child keeps its parent informed after every step**, handing up where its level stands. The parent abstracts
  that into its own state, the way it abstracts anything else, and acts on it while it matters.
- **Delegating is an action of the parent's game.** The parent's rules declare it, the CSP says when it is legal, and
  the parent's actor dispatches it into the hierarchy. A parent could play the child's level out itself, but it
  wouldn't do it well: a level is solved by the models that suit it — minimax for tic-tac-toe, Monte-Carlo tree search
  for chess, its semi-determinized form for poker — and those models live in the child's context.
- **Fetching specifics is an action too.** A level holding a file's path and not its content reads the file as a move
  of its own game, whose effect puts the content in that level's state. The abstractor stays stateless.

## knowledge: nothing new

`Context` already carries `parent`, so the hierarchy is the contexts. A level's models, policies, goals and rulesets
are found by its context id, as they are today.

## budget: carving

```python
@dataclass(frozen=True, slots=True)
class Budget:
    seconds: float
    of_clock: Clock | None = None

    def carve(self, seconds: float) -> tuple["Budget", "Budget"]:
        """The child's budget and what is left of this one. Carving more than there is gives the child what there is
        and leaves nothing."""

    def alongside(self, seconds: float) -> "Budget":
        """A budget of its own, on the same clock, taking nothing from this one."""
```

The parent picks one per delegation, and which it picked is visible in what it has left.

## abstraction: what a level sees

```python
class Abstractor[Model](Protocol):        # the abstraction task
    """The state as one level sees it: the detail that level works on, and none of the rest. None where the model has
    nothing to say about this state."""

    def abstract(self, model: Model, node: Node, context: str) -> State | None: ...
```

It takes a node, so an abstraction can read the features already extracted, and gives a state, which the level's own
world holds. The model this step builds is `RuleAbstractor`: a ruleset of the `abstraction` kind whose rules say which
models of the state the level keeps, the way every other ruleset is declared. Summaries and decoders come with the
codec step, relaxations with the inference step — the same port either way.

## agent: levels and delegation

```python
@dataclass(frozen=True, slots=True)
class Delegation:
    """What a parent hands a child: which level, what it is after, what it may spend, and who it plays as."""

    context: str                          # the child's context id
    goal: str                             # a goal id in that context
    budget: Budget
    player: str = ""                      # who the child plays as, where its level has players


@dataclass(frozen=True, slots=True)
class Report:
    """What a child gives back: where it left its level, what the goal was worth there, and what it spent."""

    delegation: Delegation
    state: State | None                   # the child's level as it left it
    value: float | None                   # what its goal was worth as it left, by that level's utility
    seconds: float                        # what it actually spent
    reached: bool                         # the goal reached, or the budget ran out


class Level:
    """One level of the hierarchy: a context, the world it sees, the models it runs with, and the agent loop running
    in it.

    Its world is an abstraction of the world everything shares; where it has no abstraction model, it is that world."""

    def __init__(self, context: str, agent: Agent, guidance: Guidance,
                 abstractor: tuple[object, Abstractor] | None = None) -> None: ...

    def perceived(self, knowledge_base, node: Node) -> State: ...   # abstracts and holds what it is told
    def run(self, knowledge_base, delegation: Delegation, informing=None) -> Report: ...   # informs after every step


class Hierarchy:                          # stateless over the levels it was built with
    """The levels, by context. A parent delegates by context id and the hierarchy runs that level."""

    def level(self, context: str) -> Level | None: ...
    def children(self, context: str) -> tuple[Level, ...]: ...
    def perceived(self, knowledge_base, state: State) -> None: ...  # pushes it through every level's abstractor
    def delegate(self, knowledge_base, delegation: Delegation) -> Report: ...
    def delegate_alongside(self, knowledge_base, delegation: Delegation) -> Delegated: ...


class Delegated:
    """A child running alongside its parent, in a thread of its own."""

    def running(self) -> bool: ...
    def report(self, seconds: float | None = None) -> Report | None: ...   # waits for it, or None while it runs
    def stop(self) -> None: ...
```

**Each level runs the same machinery**: its own `Agent`, its own time management policy call, and so its own planning
model — `Allocation` already names a model per task, and the models of a level are the models of its context. Nothing
new is needed for "its own planning model per level"; the level passes its context to the registry.

**What a delegation brought** is written as a `Task` record in the child's context: name, goal, `expected_time` and
`value` beliefs, `status` running then done. Drifting those values from reports is the next-best-task loop's step;
this step writes what happened.

## What this step leaves alone

- **Choosing to delegate**: nothing here decides that a delegation is worth making; the parent's own planner does,
  delegating being an action of its game.
- **Drifting a task's value** from the reports: the next-best-task loop step.
- **Summaries and decoders as abstraction models**: the codec step. Relaxations: the inference step.
- **The next-best-task choice across contexts**, which runs at the top of the hierarchy: its own step.

## The parent is kept informed

A child tells its parent where it stands **after every step**, not only when it is done. What the parent makes of what
it is told is its own abstraction's: the child hands up a node of its own level, and the parent's abstraction model
turns it into the state the parent holds.

That is what a coach needs. The coach delegates "move toward interesting positions" to a child that plays the game
mechanically; the child plays a move and tells the coach the position it made, and the coach — whose own loop is
running — comments on what the student chose, or warns them to pay attention. A coach told only at the end would have
nothing to say while it mattered.

Nothing new carries it: the parent's `perceived` is what the child informs, which is the same call the shared world
makes to every level.

## How delegating is declared

A parent's game declares the delegation as it declares any other action: a name, its parameters, the constraints
saying when it is legal, a duration and a cooldown. Its effect is the child's report, so the predictor of the parent's
level predicts what delegating brings, and what it actually brought is evidence like any other.

```python
declarer.rule("take the fight").played_by("me").definitions(...).constraint(...)
declarer.rule("take the fight").leads_to(...)     # what the report does to the parent's state
```

`Hierarchy` is what dispatches it: the parent's `Dispatcher` recognises a delegating action and runs the child level.
It pushes nothing into the parent's world itself — the child keeps the parent informed as it works, and the parent
abstracts what it is told.

## Open

Nothing is open; the step waits for your OK.

## As coded (2026-09-19)

- **`budget`:** `Budget.carve(seconds)` gives the child's budget and what is left of the parent's, and
  `Budget.alongside(seconds)` a budget of its own on the same clock. Both keep the clock they were carved from.
- **`abstraction`**, a package of its own: the `Abstractor` port and `RuleAbstractor`, which runs an abstraction
  ruleset — each rule giving the models the level holds, by name, a later rule replacing a model an earlier one gave.
  A rule that can't be read here leaves the others to say what the level sees; a ruleset whose rules all said nothing
  abstracts nothing, and the level sees the state as it is. `abstraction` is a rule kind now, beside `position`,
  `move` and `optimum`.
- **`agent`:** `Delegation(context, goal, budget, player)`, `Report(delegation, state, value, seconds, reached,
  task)`, `Level`, `Hierarchy` and `Delegated`. `Hierarchy.delegate` hands the child its parent's `perceived`, so the
  parent is told where the child stands after every step and abstracts it into its own state. A level plays its own agent loop over and over on its own world until
  its level is over, the budget runs out or it is stopped, and reports where it left it, what that was worth by its
  own utility, and what it spent.
- **What a delegation brought** is written as a task in the child's context: its expected time is what it was given,
  and its value, once done, what the level it left was worth.
- **A level's own models:** `create_level` finds its context's abstraction ruleset, and the time management policy
  already picks a model per task from the level's own context, so each level plans with what suits it.

### Found while coding

- **A game that doesn't say why it ended.** `ended` reads the game's own ending rule, and tic-tac-toe has none: it is
  over when the constraints leave nobody an action. A level is therefore over when its game says it ended **or** no
  player has a legal action left, which is what minimax reads too.
- **A level waits rather than spins.** Where a step changed nothing — the level is waiting on whatever performs its
  actions — it waits 10 milliseconds before looking again, as the actor does.

### Not built here

- **Nothing declares an abstraction ruleset**: the game declarer declares a game's own rules into its simulation
  ruleset, and an abstraction ruleset goes into the knowledge base the way a heuristic's does. Who declares what a
  level sees is the codec step's, where summaries and decoders fill the same port.
- **Nothing delegates yet**: no game declares a delegating action, and no dispatcher recognises one. A caller
  delegates by calling `Hierarchy.delegate` itself, as the tests do.
- **Drifting a task's value** over the runs: the next-best-task loop step.
