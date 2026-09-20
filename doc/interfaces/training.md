# Interfaces: training

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

The aim is plain: **OpenMindChess can train.** Chess plays through OMF now but has no heuristic of its own, so it
plays no better than the moves it is handed.

## What is broken, and what isn't

`training` never collected after the search and agent steps, but the breakage is narrow: four services import
`AgentBuilder` and a few constants of the deleted `mcts`. Everything they build on is alive and tested:

| Alive | What it does |
|---|---|
| `rbs/service/value_generator.py` | generates candidate position rules and fits their weights; already registers the ruleset it declares as a model of the position value task |
| `rbs/service/sparse_fitter.py` | the sparse linear fit choosing which terms earn their place |
| `inference/service/expression_search.py`, `expression_generator.py`, `mechanics.py` | where the candidate terms come from: the readings a state offers |
| `training/mapper/position_row_mapper.py` | self-play games into the rows a fit is made on |

So training is a rewiring, not a rebuild: what played the games is gone, what learns from them is not.

## Self-play on the new loop

```python
class SelfPlay:
    """Plays a context against itself, games in the task runner's workers, and gives back what was played."""

    def play(self, knowledge_base, context: str, games: int, planner, settings: SelfPlaySettings) -> tuple[PlayedGame, ...]: ...
    def play_game(self, knowledge_base, context: str, planner, seeds: tuple[int, int], budget: Budget) -> PlayedGame: ...
```

**A self-play game has an integrator of its own**, as `openmind-play` is one for the terminal: a `Referee` that runs
the game through its own rules, dispatches what each agent chose, draws an outcome and pushes back what came of it.
OMF simulates; something always runs the game, and in self-play that something is OMF's own referee, which the
entrypoint already showed the shape of.

**Every player is an agent of the new loop**: its planner, its time management policy, its budget. The caller passes
the planner, since nothing registers one as a model yet.

```python
@dataclass(frozen=True, slots=True)
class PlayedGame:
    """A game played: every position in order, the actions played, the final payoffs, and why it ended."""

    states: tuple[State, ...]
    actions: tuple[Action, ...]
    payoffs: tuple[float, ...]
    ending: str | None = None
    agent_seed: int | None = None
    outcome_seed: int | None = None
```

What `PlayedGame` loses is what no longer exists: the search samples (a planner gives a strategy, not samples), the
arms, and the per-step clocks. `search_values` goes too — fitting on what a search thought a position was worth waits
until a search is worth listening to.

## Fitting a heuristic from what was played

```python
class ValueDistiller:
    """Self-play, then the position rules fitted on what it played: the training games' positions valued at the
    target, the fits chosen on held-out games, and the chosen rules declared as a ruleset and registered as a model of
    the position value task."""

    def distill(self, knowledge_base, context: str, planner, settings: ValueDistillationSettings) -> ValueDistillationResult: ...
```

Nothing changes in how rules are generated, fitted or chosen. What changes is that the result is a **model**: the
ruleset is registered with its measured accuracy, so the time management policy can pick it over anything else filling
the same task, and the next self-play plays with it.

## What is dropped, and where it goes

| Dropped | Why, and where it returns |
|---|---|
| `continuous_trainer`, `continuous_training_settings`, `continuous_constant` | playing continuously and always doing the next best thing is the next-best-task loop's step |
| `arm_selector`, `arm_library` and its mapper and repository | arms were variants carrying their own rules, picked by UCB1; the model registry is what compares models now |
| `ending_walker`, `game_study`, `game_lesson` | walking a game back to prove its positions is inference; it returns at the inference step |
| `game_replayer` | it replayed a game from its actions and outcome seed for the dashboard; it returns there |

Git keeps them all.

## What this step leaves alone

- **Learned model families** — a network, a decision tree, a lookup table — beyond the rule-based one. The port is a
  model record and a task; the families come when one is needed.
- **Training the time management policy** from what each allocation brought: it needs allocations worth comparing.
- **Agent models**: the same heuristics fitted on someone else's games, which needs games of theirs.

## Open

1. **What plays the training games.** There is no Monte-Carlo tree search yet, so a chess self-play game is played by
   `Improvised` — a policy picked, its optimizer's move taken — which without a policy value model is close to random.
   Random chess games are long and teach little, but they are what there is until a search exists. Train on them now,
   or build the Monte-Carlo tree search first and train on games worth learning from? I lean to training on what
   there is: the pipeline is what is being fixed, and the games it learns from can get better afterwards.
2. **Whether a fitted heuristic plays the next games.** Once a position value model is registered, the time management
   policy picks it, so the next round of self-play plays with what the last one learned. That is the loop the design
   asks for, but it is also what the next-best-task step is meant to run. I lean to leaving the loop to that step and
   giving this one a single pass: play, fit, register.
