# training

## Purpose

Trains models from the agent's own play. For now it generates a rule base once from self-play, validates it on
held-out games and measures it; the expansion-and-distillation loop over rounds comes next.

## Content

| File | What it is |
|---|---|
| `model/distillation_settings.py` | `DistillationSettings(games, held_out_games, iterations, seed, generation)`; `generation` is the rule generator's `GenerationSettings` |
| `model/distillation_result.py` | `DistillationResult(rule_base, training_samples, held_out_samples, rating_error, mean_conditions, patterns, hypotheses, covered)`; `hypotheses` holds every hypothesis's test, `covered` the validated rules a simpler rule covers |
| `constant/training_constant.py` | Default games (20), held-out games (5), iterations (200) and seed (1); the range of per-game seeds |
| `service/self_play.py` | `SelfPlay`: an agent plays a domain against itself, games in the task runner's workers; returns the samples of every search |
| `service/distiller.py` | `Distiller`: self-play, rule generation and validation, and the result's measures |
| `builder/distiller_builder.py` | `DistillerBuilder`: sets how many worker processes self-play runs in (`with_workers`, 1 by default) and wires self-play, the rule generator, and the rule compiler, runner and consequence library its rater checks conditions with |
| `factory/training_factory.py` | `create_distiller(workers=1)` |

## How distillation works

1. `SelfPlay` plays `games` training games, then `held_out_games` more. Every game draws two seeds from `seed` up
   front: its agent, built from the given builder with `iterations`, searches with one, and its outcomes are drawn with
   the other. Games don't depend on each other, so they run in the task runner's workers (see `parallel/README.md`)
   with the same samples whatever the number of workers. The samples of every search are kept, in game order.
2. `RuleGenerator` discovers hypotheses on the training samples and validates them on the held-out samples, with `seed`
   drawing the permutations (see `rbs/README.md`). Validation needs enough held-out states to reach the false discovery
   rate: with few held-out games, nothing is validated and every rule is an action's mean payoff.
3. The result is measured:
   - **accuracy:** `rating_error`, the visit-weighted mean absolute difference between the rules' ratings and the
     searches' mean payoffs on held-out samples with at least `min_visits` visits (`None` without any). The held-out
     samples also validated the rules, so this error isn't independent of their selection;
   - **explainability:** the number of rules and `mean_conditions`, the mean number of conditions per rule;
   - **evidence:** the goal patterns found, every hypothesis's discovery and validation effects, p-value and
     q-value, and each validated rule left out with the simpler rule covering it.

   Guidance, whether the rules steer the search away from low-value actions, and speed show in
   `openmind-evaluate --rules` (see `evaluation/README.md`).

## Usage

```python
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

generation = GenerationSettings(5, 2, 50, 0.05, 0.95, 20, 2, 2, 200, 0.05, 10_000)   # see rbs/README.md
result = create_distiller(workers=8).distill(
    create_domain("tictactoe"),
    AgentBuilder().with_exploration(EXPLORATION),   # distill sets the iterations and each game's seed
    DistillationSettings(games=20, held_out_games=20, iterations=200, seed=1, generation=generation),
)
```

From the terminal: `openmind-distill tictactoe` (see `entrypoint/README.md`).

## Logs

- `openmind.training.service.self_play`: `INFO Self-play game <n>: <samples> samples, payoffs <player>=<payoff> ...`
- `openmind.training.service.distiller`: `INFO Distilled <n> rules, <mean> conditions per rule on average, from <m>
  training samples; <k> of <h> hypotheses validated, <c> covered by a simpler rule, and rating error <error> on <s>
  held-out samples`

Every search also logs its summary (see `mcts/README.md`), and generation logs its patterns, hypotheses and rules (see
`rbs/README.md`).

## Notes

- Tests: `builder/distiller_builder_tests.py`, `factory/training_factory_tests.py`, `service/distiller_tests.py`,
  `service/self_play_tests.py`; integration: `test/integration/tictactoe_distillation_tests.py`; end-to-end:
  `test/end_to_end/distill_tictactoe_tests.py`.
