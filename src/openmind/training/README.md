# training

## Purpose

Trains models from the agent's own play. It generates a rule base from self-play, validates it on held-out games
and measures it; it selects, from many candidate rules, the smallest set that plays no worse than all of them; and it
fits value rules, which value positions, on the positions of self-play games, once or round after round, each round's
self-play valuing positions with the previous round's rules.

## Content

| File | What it is |
|---|---|
| `model/distillation_settings.py` | `DistillationSettings(games, held_out_games, iterations, seed, generation)`; `generation` is the rule generator's `GenerationSettings` |
| `model/distillation_result.py` | `DistillationResult(rule_base, training_samples, held_out_samples, rating_error, mean_conditions, patterns, hypotheses, covered)`; `hypotheses` holds every hypothesis's test, `covered` the validated rules a simpler rule covers |
| `constant/training_constant.py` | Default games (20), held-out games (5), iterations (200) and seed (1); the range of per-game seeds |
| `service/self_play.py` | `SelfPlay`: an agent plays a domain against itself, games in the task runner's workers; returns every game |
| `model/played_game.py` | `PlayedGame(samples, states, search_values, payoffs)`: a self-play game's search samples, its positions with the search's mean payoff for the player to act in each, and its final payoffs |
| `service/distiller.py` | `Distiller`: self-play, rule generation and validation, and the result's measures |
| `builder/distiller_builder.py` | `DistillerBuilder`: sets how many worker processes self-play games and rule condition checks run in (`with_workers`, 1 by default) and wires self-play, the rule generator, and the rule compiler, runner and consequence library its rater checks conditions with |
| `factory/training_factory.py` | `create_distiller(workers=1)`, `create_rule_selector(workers=1)`, `create_value_distiller(workers=1)` and `create_value_training_loop(workers=1)` |
| `constant/training_constant.py` | Also the value distillation defaults: 100 games and 25 held-out games; the targets, `outcome` and `search` |
| `mapper/position_row_mapper.py` | `PositionRowMapper`: self-play games to the position rows value rules are fitted on, at a target |
| `model/value_distillation_settings.py` | `ValueDistillationSettings(games, held_out_games, iterations, seed, target, values, pondering=None)`; `values` is the value generator's `ValueSettings` |
| `model/value_distillation_result.py` | `ValueDistillationResult(value_base, fits, chosen, candidates, training_rows, held_out_rows, held_out_error, pondering=None)` |
| `service/value_distiller.py` | `ValueDistiller`: self-play, pondering, value rule generation, and the chosen rules' error on held-out rows |
| `builder/value_distiller_builder.py` | `ValueDistillerBuilder`: sets how many worker processes self-play games, deductions and term evaluations run in (`with_workers`, 1 by default) and wires the value distiller and its ponderer |
| `model/pondering_settings.py` | `PonderingSettings(positions, budget)`: how many positions a round ponders and the `DeductionBudget` each gets |
| `model/pondering.py` | `Pondering(rows, deductions, seeds, sources)`: the rows with proven targets, every deduction, and the seeds induced, the most often induced first |
| `model/pondering_summary.py` | `PonderingSummary(positions, proven, seeds, seeds_kept, seeds_in_rules)` |
| `service/position_ponderer.py` | `PositionPonderer.ponder(domain, rows, value_base, settings)`: deduces the positions the rules missed most, values proven rows at the proven payoffs, and gathers their seeds |
| `model/value_training_settings.py` | `ValueTrainingSettings(rounds, distillation, rollout_actions, rollout_limit, unfinished_payoff, evaluation_games, start_file, deduction=None)`; `deduction` is the budget agents fall back on when their rules have no clue |
| `model/training_round.py` | `TrainingRound(number, value_base, fits, chosen, training_rows, held_out_rows, held_out_error, baselines, against_previous, seconds, pondering=None)`: one round's value rules, fits, games and pondering |
| `model/training_report.py` | `TrainingReport(domain, created_at, settings, rounds, complete)` |
| `constant/training_constant.py` | Also the training loop defaults: 3 rounds, 200 games and 50 held out, 100 iterations, 10 rollout actions before valuing, 20 games per opponent; `START_RULES` |
| `service/value_training_loop.py` | `ValueTrainingLoop`: rounds of self-play valuing positions with the previous round's rules, value fitting, and games against the random policy, untrained MCTS and the previous round's agent |
| `builder/value_training_loop_builder.py` | `ValueTrainingLoopBuilder`: sets how many worker processes self-play, term evaluations and games run in (`with_workers`, 1 by default) and wires the loop |
| `mapper/training_report_json_mapper.py` | `TrainingReportJsonMapper`: a training report as JSON text, every round's value rules, fits and games included |
| `mapper/training_report_text_mapper.py` | `TrainingReportTextMapper`: a training report as a table, one line per round |
| `repository/training_report_repository.py` | `TrainingReportRepository`: saves a report as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json`, named after the training's start, overwriting it as rounds end |
| `model/selection_settings.py` | `SelectionSettings(positions, reference_iterations, iterations, guided_rollouts, margin, confidence, resamples, seed, max_hours)` |
| `model/non_inferiority.py` | `NonInferiority(positions, regret_difference, upper_bound, margin, holds)`: a paired comparison of regret against a margin |
| `model/removal_test.py` | `RemovalTest(rule, pass_number, regret_difference, upper_bound, optimal_difference, removed, free, seconds)`: one try at removing a rule |
| `model/selection_report.py` | `SelectionReport(domain, created_at, candidates_file, settings, candidates, selected, full, subset, confirmation, tests, complete)` |
| `constant/training_constant.py` | Also the selection defaults: 10 iterations, margin 0.005, confidence 0.95, 10,000 resamples; `BOOTSTRAP_BATCH` (1,000), `CONFIRMATION_SEED_OFFSET` (1), `SELECTION_KIND` |
| `service/non_inferiority_test.py` | `NonInferiorityTest`: the percentile bootstrap's one-sided upper bound on a mean regret difference, against a margin |
| `service/rule_selector.py` | `RuleSelector`: removes candidate rules one at a time while play stays no worse than with all of them, then confirms the selection |
| `builder/rule_selector_builder.py` | `RuleSelectorBuilder`: sets how many worker processes the searches run in (`with_workers`, 1 by default) and wires the selector |
| `mapper/selection_report_json_mapper.py` | `SelectionReportJsonMapper`: a selection report as JSON text, rules as readable text |
| `mapper/selection_report_text_mapper.py` | `SelectionReportTextMapper`: a selection report as a readable summary |
| `repository/selection_report_repository.py` | `SelectionReportRepository`: saves a report as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json`, named after the selection's start, overwriting it as the selection goes |

## How distillation works

1. `SelfPlay` plays `games` training games, then `held_out_games` more. Every game draws two seeds from `seed` up
   front: its agent, built from the given builder with `iterations`, searches with one, and its outcomes are drawn with
   the other. Games don't depend on each other, so they run in the task runner's workers (see `parallel/README.md`)
   with the same games whatever the number of workers. The samples of every search are kept, in game order.
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

## How rule selection works

Generation keeps every rule validated on its own; selection keeps the rules that matter for play, together. It starts
from a candidate rule base, typically from `openmind-distill --explore`, which generates widely and keeps covered
rules.

1. **Positions and values.** With `reference_iterations` unset, exact search gives every position (`positions` None)
   or a sample of them, and every legal action's value. Otherwise, positions come from random games and values from
   long unguided searches, as in `evaluation/README.md`; they need a number of positions.
2. **All the rules.** An agent guided by every candidate, with `iterations` and `seed`, searches each position; each
   search records whether its choice was optimal and its regret.
3. **Passes.** Each pass counts how many ratings each rule decides over every legal action of every position, then:
   - when the positions are every exact-search position, removes the rules with conditions that decide none, without a
     search: no search reads a rating they give;
   - tries removing each other rule with conditions, least used first, then fewest visits: the positions are searched
     again without it, and its regret differences with all the rules, position by position, go to
     `NonInferiorityTest`. The rule stays out when the one-sided `confidence` bound on the rise in mean regret, from
     `resamples` bootstrap resamples, is below `margin`.

   Every comparison is against all the rules, not against the previous selection, so small losses don't add up.
   Rules without conditions are always kept. Passes stop when one removes nothing, or when `max_hours` is up; the
   report then isn't complete.
4. **Confirmation.** The selected rules against all of them, with the seed plus one, on the same positions when they
   are every position and on a new sample otherwise: one more non-inferiority comparison, since selecting after many
   comparisons on the same positions favours rules that passed by chance.

Searches run in the task runner's workers (see `parallel/README.md`). After every decision the selector hands the
report so far to `on_progress`; `openmind-select` saves it, so a long selection can be read while it runs.

## How value distillation works

1. `SelfPlay` plays `games` training games, then `held_out_games` more, as in distillation. Every game keeps its
   positions, in order, with the search's mean payoff for the player to act in each (the visit-weighted mean of the
   root actions' mean payoffs), and its final payoffs.
2. `PositionRowMapper` turns the games into rows at the `target`:
   - `outcome`: every position, once for each player, valued at that player's final payoff. One game's result is a
     noisy value for its early positions, but it is what the position led to in the agent's own play;
   - `search`: every position, for the player to act, valued at the search's mean payoff there: smoother, but only as
     good as the search, and leaning toward what its rollouts find.
3. **Pondering,** with `pondering` settings. `PositionPonderer` reasons about the training positions the rules know least
   about:
   - every training row's miss is how far its target is from the previous rules' value for its player (the value base
     `distill` is given, the previous round's in a training loop), or from the mean target without rules or where they
     can't value the position;
   - the positions in play with the largest misses, `positions` of them, are deduced within the budget in the task
     runner's workers (see `inference/README.md`). Without rules, the value is constant, so the positions of decisive
     games come first;
   - a proven position's training rows take the proven payoffs as targets, in place of the search's estimate or the
     game's outcome: a proof is exact where both are guesses;
   - every proven deduction induces seeds, and a seed more positions induced comes first, since it's likelier to help.
4. `ValueGenerator` fits value rules on the training rows, trying the seeds first, and chooses a price on the held-out
   rows (see `rbs/README.md`). The result's `pondering` counts the positions pondered and proven, the seeds, the seeds
   the search kept and those among the value rules.
5. `held_out_error`: the mean absolute difference between the chosen rules' value for each held-out row's player and the
   row's target, over the rows the rules can value. The held-out rows also chose the price, so this error isn't
   independent of that choice.

How the value rules play shows in `openmind-evaluate --values` (see `evaluation/README.md`).

## How the value training loop works

`ValueTrainingLoop.train(domain, start, settings, on_round)` runs `rounds` rounds. In round k:

1. **Self-play.** The self-play agents search with the distillation's iterations and the rollout limit. With value rules
   from the previous round, or the `start` rules in round 1, they value the positions their rollouts reach after
   `rollout_actions` rollout actions (see `mcts/README.md`); without, they play rollouts out. A few rollout actions
   before valuing keep each round's positions from being valued only by the previous round's own estimates, and
   finished games always keep their real payoffs. With a `deduction` budget, an agent whose rules have no clue in a
   position deduces it first and plays a proven move without searching (see `agent/README.md`); self-play records the
   proven payoff as that position's value. Untrained MCTS never deduces.
2. **Fitting.** `ValueDistiller` distills new value rules with the distillation's settings, seeded with the seed plus
   k, pondering against the previous round's rules (see "How value distillation works").
3. **Games.** With `evaluation_games`, the new rules' agent, searching as in self-play, plays that many games against
   the random policy, untrained MCTS searching with the same iterations and rollout limit, and the previous round's
   agent (the start rules' in round 1, none without them), switching seats every game. Every agent searches with its
   game's own seed (`create_seeded_agent`), so no two games are searched alike, and the games are seeded from the
   seed plus k.
4. **Report.** The round joins the report, handed to `on_round`: `openmind-train-values` saves the round's value base
   and the report at once, so a training can be stopped between rounds and read. The last report is complete.

The games run in the task runner's workers; the value rules' agents travel there with their `RuleValuer`.

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

```python
from openmind.training.factory.training_factory import create_rule_selector
from openmind.training.model.selection_settings import SelectionSettings

settings = SelectionSettings(None, None, 10, True, 0.005, 0.95, 10_000, 1, None)   # every position, 10 iterations
report = create_rule_selector(workers=8).select(create_domain("tictactoe"), candidates, "candidates.json", settings)
report.selected   # the selected RuleBase
```

From the terminal: `openmind-select tictactoe --rules <candidates>.json`.

```python
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.value_distillation_settings import ValueDistillationSettings

values = ValueSettings(20, 6, 2, (0.1, 0.03, 0.01, 0.003, 0.001), 1000, 1e-6)   # see rbs/README.md
result = create_value_distiller(workers=8).distill(
    create_domain("tictactoe"),
    AgentBuilder().with_exploration(EXPLORATION),
    ValueDistillationSettings(games=100, held_out_games=25, iterations=200, seed=1, target="outcome", values=values),
)
result.value_base   # the ValueBase
```

From the terminal: `openmind-distill-values tictactoe`.

```python
from openmind.training.factory.training_factory import create_value_training_loop
from openmind.training.model.value_training_settings import ValueTrainingSettings

distillation = ValueDistillationSettings(200, 50, 100, 1, "search", values)
settings = ValueTrainingSettings(3, distillation, 10, 100, 0.5, 20, None)   # 3 rounds, rollout limit 100
report = create_value_training_loop(workers=8).train(create_domain("chess"), None, settings, on_round=print)
report.rounds[-1].value_base   # the last round's value rules
```

From the terminal: `openmind-train-values chess --rounds 3 --rollout-limit 100`.

## Logs

- `openmind.training.service.self_play`: `INFO Self-play game <n>: <samples> samples, payoffs <player>=<payoff> ...`
- `openmind.training.service.distiller`: `INFO Distilled <n> rules, <mean> conditions per rule on average, from <m>
  training samples; <k> of <h> hypotheses validated, <c> covered by a simpler rule, and rating error <error> on <s>
  held-out samples`

- `openmind.training.service.rule_selector`:
  - `INFO All <n> rules at <iterations> iterations on <positions> positions: <optimal> optimal choices, mean regret <regret>`
  - `INFO Removed <rule>: it decides no rating`
  - `INFO Removed <rule>: regret <difference> (bound <bound> < margin <margin>), optimal choices <difference>, <seconds> seconds`, or `Kept <rule>: ... (bound <bound> >= margin <margin>) ...`
  - `INFO Pass <n> removed <m> rules: <kept> of <candidates> kept`
  - `INFO Out of time after <hours> hours`
  - `INFO Confirmation with seed <seed> on <positions> positions: regret <difference>, bound <bound> < margin <margin>` (or `>=`)
- `openmind.training.service.position_ponderer`:
  - `INFO Pondered <n> positions the rules missed most, the largest miss <miss>: <p> proven within <plies> plies, <s> seeds`
  - `DEBUG Seed <source>`, one line per seed, the most often induced first
- `openmind.training.service.value_distiller`: `INFO Distilled <n> value rules from <m> training rows valued at the
  <target> target; mean absolute error <error> on <h> held-out rows`, and with pondering `INFO Pondering: <n> positions,
  <p> proven; <s> seeds, <k> kept by the search, <r> in the value rules`
- `openmind.training.service.value_training_loop`:
  - `INFO Round <k> of <n>: self-play without value rules`, or `... self-play valuing positions with the start rules`,
    or `... with round <k-1>'s rules`
  - `INFO Round <k>: <r> value rules, held-out loss <loss>, held-out error <error>`
  - `INFO Round <k> against <opponent>: <games> games, <wins> wins, <draws> draws, <losses> losses`, for `random`,
    `untrained MCTS` and `start rules` or `round <k-1>`
  - `INFO Round <k> took <seconds> seconds`

Every search also logs its summary (see `mcts/README.md`), and generation logs its patterns, hypotheses and rules (see
`rbs/README.md`).

## Notes

- Tests: `builder/distiller_builder_tests.py`, `factory/training_factory_tests.py`, `service/distiller_tests.py`,
  `service/self_play_tests.py`, `service/non_inferiority_test_tests.py`, `service/rule_selector_tests.py`,
  `builder/rule_selector_builder_tests.py`, `mapper/selection_report_json_mapper_tests.py`,
  `mapper/selection_report_text_mapper_tests.py`, `repository/selection_report_repository_tests.py`,
  `mapper/position_row_mapper_tests.py`, `service/value_distiller_tests.py`, `builder/value_distiller_builder_tests.py`,
  `service/position_ponderer_tests.py`,
  `service/value_training_loop_tests.py`, `builder/value_training_loop_builder_tests.py`,
  `mapper/training_report_json_mapper_tests.py`, `mapper/training_report_text_mapper_tests.py`,
  `repository/training_report_repository_tests.py`; integration: `test/integration/tictactoe_distillation_tests.py`;
  end-to-end: `test/end_to_end/distill_tictactoe_tests.py`, `test/end_to_end/select_tictactoe_tests.py`,
  `test/end_to_end/distill_values_tictactoe_tests.py`, `test/end_to_end/train_values_tictactoe_tests.py`.
