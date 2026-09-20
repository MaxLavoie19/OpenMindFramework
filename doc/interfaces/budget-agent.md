# Interfaces: budget, the time management policy and the agent loop

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-19)

- **The time management policy picks both** the set of models a step runs with and how much to explore, from the time
  available, the models available and what each one trades. It starts from a simple bootstrap policy and is trainable
  later.
- **Dispatching actions is a port** an integrator can fill — a robot's control loop, a game client's connection — with
  OMF's own thread as the default. It waits where the strategy has nothing prepared for the state it is in.
- **Observations are pushed:** the integrator hands OMF what it perceived, the decoders turn it into models, and the
  actor and the planner read the same current state.
- **The planner keeps its tree between moves** and prunes what can no longer be reached.
- **Acting and strategizing run at once**: a fast model plays a pre-move from the strategy while the planner keeps
  strategizing.

## budget: what there is to spend

```python
@dataclass(frozen=True, slots=True)
class Budget:
    """What a step may spend: the seconds it has, and what is left of the whole allowance it is carved from."""

    seconds: float
    of_clock: Clock | None = None                 # the clock it was carved from, where the game runs on one


class TimeManager[Model](Protocol):               # the time management task
    """Which models a step runs with, and how much they may explore, given what there is to spend."""

    def manage(self, model: Model, knowledge_base, node: Node, budget: Budget, tasks: tuple[str, ...]) -> Allocation: ...


@dataclass(frozen=True, slots=True)
class Allocation:
    """What the policy decided: the model chosen for each task, and what the planner may explore."""

    models: Mapping[str, ModelRecord]             # by task
    settings: SearchSettings
```

`PlainTimeManager` is the bootstrap model: the best measured model of each task (`ModelRegistry.best`), and a node
count from the seconds available and what a node has been costing, read from the models' measured processing time.
Nothing is learned yet; the training step fits a model of this task from what each allocation brought.

## world: the current state

```python
class Observer(Protocol):
    """What an integrator pushes: what it perceived, for the decoders to turn into models."""

    def observed(self, observation: object) -> None: ...


class World:                                      # stateless service over a state held for the agent
    def current(self) -> State: ...
    def perceived(self, observation: object) -> State: ...    # decodes and replaces the current state
    def happened(self, action: Action, outcome: State) -> State: ...
```

The actor and the planner read `current()`. A state it replaces is what the planner prunes against.

## agent: acting while strategizing

```python
class Dispatcher(Protocol):
    """Performs what a strategy calls for. OMF's own runs a thread; an integrator's may drive motors or a game
    client."""

    def dispatch(self, action: Action) -> None: ...
    def performing(self) -> tuple[Action, ...]: ...           # what is still being performed


class Actor:                                      # OMF's dispatcher: a thread taking actions from the strategy
    def act(self, world: World, strategy: Strategy) -> None: ...
    def stop(self) -> None: ...


class Agent:
    """The loop: perceive, process, plan, communicate, act, under one budget.

    The planner strategizes while the actor acts. Where the strategy has nothing for the current state, the actor
    waits and the planner strategizes from there."""

    def play(self, knowledge_base, world: World, guidance: Guidance, budget: Budget) -> Strategy: ...
```

## What this step leaves alone

- **Training the time management policy**: the training step.
- **The next-best-task loop** (ponder, self-play, review, study, train, play): its own step.
- **Rhetoric and the communicate step of the loop**: the codec and rhetoric steps.
- **Reviving `training` and `dashboard`**: their own steps. This step revives `agent` and `openmind-play`.

## Open

1. **Threads and the debugger.** A pause freezes OMF's clocks and the debugger tracks frames per thread. An actor
   thread pausing at a breakpoint would hold the game up. Should the actor open frames at all, or stay out of the
   debugger until the dashboard can show it?
2. **What `openmind-play` becomes.** It drives the game itself today: prompts a human, applies the action, prints the
   state. With the integrator pushing observations, is it the integrator in that setup — pushing what the human played
   and dispatching what OMF chooses — or does it become a thin example of one?
