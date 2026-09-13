# training

## Purpose

Trains models from the agent's own play. For now it distills a rule base once from self-play and measures it; the
expansion-and-distillation loop over rounds comes next.

## Content

| File | What it is |
|---|---|
| `model/distillation_settings.py` | `DistillationSettings(games, held_out_games, iterations, seed, induction)` |
| `model/distillation_result.py` | `DistillationResult(rule_base, training_samples, held_out_samples, rating_error, mean_conditions)` |
| `constant/training_constant.py` | Default games (20), held-out games (5), iterations (200) and seed (1); the range of per-game seeds |
| `service/self_play.py` | `SelfPlay`: an agent plays a domain against itself; returns the samples of every search |
| `service/distiller.py` | `Distiller`: self-play, rule induction, and the result's measures |
| `builder/distiller_builder.py` | `DistillerBuilder`: wires self-play, the rule inducer and the interpreter |
| `factory/training_factory.py` | `create_distiller()` |

## How distillation works

1. `SelfPlay` plays `games` training games, then `held_out_games` more. Each game uses an agent built from the given
   builder with `iterations` and its own seed, drawn from `seed`. The samples of every search are kept.
2. `RuleInducer` induces the rule base from the training samples (see `rbs/README.md`).
3. The result is measured:
   - **accuracy:** `rating_error`, the visit-weighted mean absolute difference between the rules' ratings and the
     searches' mean payoffs on held-out samples with at least `min_visits` visits (`None` without any);
   - **explainability:** the number of rules and `mean_conditions`, the mean number of conditions per rule.

   Speed shows in the `seconds_per_choice` of `openmind-evaluate --rules`.

## Usage

```python
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.domain_factory import create_domain
from openmind.rbs.model.induction_settings import InductionSettings
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

result = create_distiller().distill(
    create_domain("tictactoe"),
    AgentBuilder().with_exploration(EXPLORATION),   # distill sets the iterations and each game's seed
    DistillationSettings(games=20, held_out_games=5, iterations=200, seed=1, induction=InductionSettings(5, 2, 50, 0.05)),
)
```

From the terminal: `openmind-distill tictactoe` (see `entrypoint/README.md`).

## Logs

- `openmind.training.service.self_play`: `INFO Self-play game <n>: <samples> samples, payoffs <player>=<payoff> ...`
- `openmind.training.service.distiller`: `INFO Distilled <n> rules, <mean> conditions per rule on average, from <m>
  training samples; rating error <error> on <h> held-out samples`

Every search also logs its summary (see `mcts/README.md`), and induction logs its rules (see `rbs/README.md`).

## Notes

- Tests: `builder/distiller_builder_tests.py`, `factory/training_factory_tests.py`, `service/distiller_tests.py`,
  `service/self_play_tests.py`; integration: `test/integration/tictactoe_distillation_tests.py`; end-to-end:
  `test/end_to_end/distill_tictactoe_tests.py`.
