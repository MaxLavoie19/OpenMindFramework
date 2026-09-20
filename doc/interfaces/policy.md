# Interfaces: policy, optimizer and utility

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-19)

- **A policy** is a way of playing tuned to a subset of the actions: the agent picks how it wants to play, then picks
  the action that serves that best. It is a pair of heuristics specialised to it — a position value valuing the states
  it is after, a move value valuing the moves that serve it — under a sub-goal.
- **A policy is manufactured or learned:** an agent is told to learn freeze, fight and flight, or it learns policies
  by policy optimization.
- **A policy performs two tasks**, and one model can perform both: one network with two heads, or two rulesets. A
  model record names its tasks, so one record can perform several (this changes `ModelRecord.task` into `tasks`).
- **The optimizer is an alternative to expanding.** Expanding lists the valid actions and sorts them; optimizing
  produces the one action that serves the sub-goal, as an artillery piece calculates its firing solution rather than
  enumerating every ballistic one.
- **Picking at random is a policy**, the one that saves time when time is short. With time to spare, an agent follows
  several complementary policies and learns when each one pays.
- **Preferences live in the knowledge base**, so the agent can know and explain its own choices.
- **Values are quantitative in some domains and qualitative in others**: centipawns in chess, "sounds selfish" in
  rhetoric.
- **The time management policy keeps its name**, though it is a different thing.
- Rhetoric's 36 tactics are policies too, and a message carries several at once. That is the rhetoric step's; this one
  only has to leave room for it.

## knowledge: the policy record

```python
@dataclass(frozen=True, slots=True)
class Policy:
    name: str                      # "flee", "charge", or "" for the default policy
    context: str                   # a context id
    sub_goal: str                  # what it is after, in words: "stay alive", "take the centre"
    position_value: str = ""       # a model id: the heuristic valuing the states it is after
    move_value: str = ""           # a model id: the heuristic valuing the moves that serve it
    tags: Tags = ()
    id: str = ""                   # policy-<uuid4>
```

`KnowledgeBase` gains `policy(record)`, `policy_by_id`, `policy_named(context_id, name)`, `policies(context_id, tags)`
and `readable_policy`, with a `policies.jsonl` store.

**The default policy** is the one with no name and no heuristics: its candidates are the legal actions, valued by
whatever the game's own heuristics say, and at worst picked at random. A game with no policy of its own gets it.

## policy: choosing one, and optimizing within it

```python
class PolicyValuer[Model](Protocol):          # the policy value task
    """What each policy is worth following in this node, in the policies' order; None where the model knows nothing."""
    def rate(self, model: Model, node: Node, policies: tuple[Policy, ...], player: str) -> tuple[float | None, ...]: ...


class PolicyPicker[Model](Protocol):          # the policy picking task
    """Which policies are worth expanding here, best first, and which aren't worth considering at all: the picked ones,
    in the order they are worth expanding. Picking at random is rarely worth expanding, and when it is, nothing else is
    worth considering."""
    def pick(self, model: Model, node: Node, policies: tuple[Policy, ...], player: str) -> tuple[Policy, ...]: ...


class Optimizer[Model](Protocol):             # the optimizing task
    """The next best step from this node for the policy's sub-goal, without listing the alternatives; None where the
    model has no solution here. The CSP checks what it gives, since a computed solution isn't legal by construction."""
    def optimize(self, model: Model, policy: Policy, node: Node, player: str) -> Action | None: ...
```

Two ways to plan within a policy, and the time management policy picks between them at the budget step:

| Way | What it does | When |
|---|---|---|
| expanding | the CSP lists the valid actions, the policy's move value heuristic rates them, the search explores them | discrete games, where listing is affordable |
| optimizing | a model computes the action serving the sub-goal — a firing solution, an edit a coding agent proposes | continuous or huge action spaces |

The optimizing models this step builds:

| Model | What it does |
|---|---|
| `RandomPicker` | one of the valid values, picked at random: what a policy does when there is no time to think |
| `RuleOptimizer` | runs a ruleset that computes the action, the way a game declares any other rule: equations, a controller, whatever the rules say |
| `SolverOptimizer` | the CSP working the values out one at a time, so what it gives is legal by construction |

A controller such as a PID is a `RuleOptimizer` whose rules hold its state in the game's own models; a model proposing
solutions for an NLP or coding agent is the same port again, at its own step.

**Control systems fit here** without changing the port:
- a system with several inputs and several outputs is a game whose actions carry several parameters, which the CSP and
  the predictor already handle;
- an optimizer gives the next best step from where it is, and **model predictive control's plan** is the series of
  steps the optimizers give as the search explores, so the horizon is the search's;
- **model reference adaptive control** adapts its parameters as it runs, and what it adapted is its own data, kept
  where its model record says, so it outlives the run.

Neither is built in this step: the port is what they need, and the models come when a problem needs them.

**Checking what an optimizer gives.** Equations and controllers know the goal, not the rules, so the CSP checks their
action: `Solver.allows(state, action, values, constraints) -> bool` runs the action's constraints against the values it
carries, rather than searching for values that satisfy them. A `SolverOptimizer` needs no check.

The policy value task rates a policy; the policy picking task says which of them are worth expanding at all. The
picker this step builds reads the ratings and takes those above what the cheapest policy is worth; a better picker is
a model like any other. Whether a picked policy expands or optimizes is the time management policy's at the budget
step.

## utility: what a move is worth

```python
class Utility:                                # stateless
    def of(self, knowledge_base, node, outcomes: OutcomeDistribution, player: str) -> float | None: ...
        # each outcome's value on every goal, weighed by the player's preferences, times its probability, summed


class Binner[Model](Protocol):                # the binning task
    """Continuous outcomes grouped into bins, each with its likelihood."""
    def bins(self, model: Model, outcomes: OutcomeDistribution, player: str) -> tuple[Bin, ...]: ...


@dataclass(frozen=True, slots=True)
class Bin:
    low: float
    high: float
    likelihood: float
```

**Goals and preferences** are records in the knowledge base, so the agent can look back on what it preferred:

```python
@dataclass(frozen=True, slots=True)
class Goal:
    name: str                      # "win", "teach", "stay alive"
    context: str
    tags: Tags = ()
    id: str = ""                   # goal-<uuid4>


@dataclass(frozen=True, slots=True)
class Preference:
    goal: str                      # a goal id
    holder: tuple[str, ...]        # whose preference it is: () for the agent's own, ("black",) for another agent's
    weight: float
    role: str = ""                 # "player", "coach", …: the role it applies in, "" for any
    tags: Tags = ()
    id: str = ""                   # preference-<uuid4>
```

A game's payoff is the goal every game has, so a game with no goal of its own gets one named after its payoff, weighed
1.

## What this step leaves alone

- **Qualitative values**: this step computes on numbers. A domain whose values are judgements, such as rhetoric, gets
  them at the rhetoric step, where they are first needed.
- **Proposing optimizers**: the port, no model.
- **Choosing between expanding and optimizing**: the budget step. This step makes both available.
- **Learning policies** by policy optimization: the training step. This step records what is manufactured.
- **Soft goals** valued by a judge's opinion: rhetoric.
- **Following several policies at once** and weighing what each proposes: the search and budget steps, where there is
  a time budget to spend on it.

## Open

Nothing is open; the step waits for your OK.

## As coded (2026-09-19)

- **Records:** `Policy`, `Goal` and `Preference` in `knowledge/model`, with their mapper and stores, and the knowledge
  base's `policy`, `policies`, `goal`, `goals`, `prefer`, `preference` and `preferences`. A preference is held per
  goal, holder and role, and a role's preference is taken over the one held for any role.
- **Ports** in `policy/model`: `PolicyValuer`, `PolicyPicker`, `Optimizer`.
- **Models** in `policy/service`: `RulePolicyValuer` rates policies by a ruleset whose rules read the policy's name and
  sub-goal; `RatedPolicyPicker` keeps what is rated above picking at random, and picks the default policy alone when it
  is best; `RandomPicker`, `SolverOptimizer` and `RuleOptimizer` are the optimizers.
- **`Solver.allows(state, action, values, constraints, definitions, player)`** checks one action, and
  `Simulation.allows` and the facade pass it through. `RuleOptimizer` checks what its rules computed with it.
- **Utility** in `utility/`: `Utility.of(...)` and `.value(...)`, the `Binner` port, `Bin`, and `EvenBinner`.
- **Tasks named:** `policy value`, `policy picking` and `optimizing` joined the task constants, and `tactic value`
  became `policy value`. The rule kind `optimum` is what an optimizer's ruleset holds.
- **A model record names its tasks** now, not one task, so a network with two heads is one record.

### Not built here

- No caller chooses a policy yet: the search step wires the picker, the optimizers and the utility into how a move is
  chosen, and the budget step decides how many policies are followed and whether each expands or optimizes.
- The default policy is a record with no name; nothing creates one yet, since nothing reads policies yet.
- Values are numbers. Qualitative values arrive with rhetoric.
