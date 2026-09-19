# training

## Purpose

Trains models from the agent's own play. It fits value rules, which value positions, on the positions of self-play
games; and it plays games continuously between arms, remembering every game and every position a decisive game's walk
back proved.

## Content

| File | What it is |
|---|---|
| `constant/training_constant.py` | Default iterations (200); the range of per-game seeds |
| `service/self_play.py` | `SelfPlay`: an agent plays a domain against itself, games in the task runner's workers; returns every game, with its searches' samples unless `keep_samples` is false, and leaves out a game that took its worker over the memory cap in a fresh worker too; `play_arms(domain, builders, games, scores, selector, exploration, rng)` plays games between arms, a game's two arms chosen when a worker starts it. Both take `time_control`: games are then played on a clock, each move timed around the agent's search and charged to its player's clock, given to the agent with the steps its player has played, and a player whose time runs out doesn't play that move, the domain's timeout rule ending the game (the builders need a time budget estimator, see `agent/README.md`) |
| `model/played_game.py` | `PlayedGame(samples, states, search_values, payoffs, arms=(), actions=(), time_control=None, seconds=(), budgets=(), clocks=(), flagged=None)`: a self-play game's search samples, its positions with the search's mean payoff for the player to act in each, its final payoffs, in a game between arms the arm each player followed, and the actions played in order; on a clock, the time control, each step's seconds and budget in order (a step that ran its player's time out included, though its action wasn't played), each player's clock at the end, and the player whose time ran out; then the game's seeds and why it ended |
| `factory/training_factory.py` | `create_value_distiller(workers=1, memory_cap=None, game_memory=None)` and `create_continuous_trainer(knowledge_base, workers=1, memory_cap=None)` |
| `constant/training_constant.py` | Also the value distillation targets, `outcome` and `search` |
| `mapper/position_row_mapper.py` | `PositionRowMapper`: self-play games to the position rows value rules are fitted on, at a target |
| `model/value_distillation_settings.py` | `ValueDistillationSettings(games, held_out_games, iterations, seed, target, values, time_control=None, expected_steps=30, time_reserve=0.05, selection='ucb1', puct_exploration=1.5, prior='uniform', prior_temperature=0.1)`; `values` is the value generator's `ValueSettings`; `time_control` puts self-play on a clock, every agent budgeting its moves with `PlainTimeBudgetEstimator(expected_steps)`; every agent searches by `selection`, following the `value` prior from its own value rules when it has them and the uniform prior otherwise |
| `model/value_distillation_result.py` | `ValueDistillationResult(context, rules, fits, chosen, candidates, training_rows, held_out_rows, held_out_error, records=())`: the context the position rules were declared under and the rules themselves |
| `service/value_distiller.py` | `ValueDistiller`: self-play, value rule generation, and the chosen rules' error on held-out rows; with a game memory (`ValueDistillerBuilder.with_game_memory`), every game, training and held-out, is remembered as it ends with the model each player played (`distill(..., round_number, model_name)` name them) |
| `mapper/played_game_summary_mapper.py` | `PlayedGameSummaryMapper`: a self-play game as a `GameSummary`, with its kind, round, number, models and record |
| `builder/value_distiller_builder.py` | `ValueDistillerBuilder`: sets how many worker processes self-play games and term evaluations run in (`with_workers`, 1 by default) and the memory each holds at most (`with_memory_cap`, no cap by default), and wires the value distiller |
| `service/arm_selector.py` | `ArmSelector.pair(scores, pending, arms, exploration, rng)`: the two arms with the highest UCB1 bounds on their game scores; games under way count without points, an arm never chosen goes first, ties are drawn at random |
| `model/continuous_training_settings.py` | `ContinuousTrainingSettings(games, iterations, seed, arm_exploration, rollout_actions, rollout_limit, unfinished_payoff, deduction=None, ponder_endings=0, time_control=None, expected_steps=30, time_reserve=0.05, selection='ucb1', puct_exploration=1.5, prior='uniform', prior_temperature=0.1)`: how continuous training runs; walking back without a deduction, negative games, arm exploration or walked back positions raise `ValueError` |
| `model/game_lesson.py` | `GameLesson(game, walk=())`: a game a worker played, and the deductions of its positions walked back from its end, the last position first |
| `service/game_study.py` | `GameStudy(self_play, ending_walker).play_and_study(domain, builders, arms, agent_seed, outcome_seed, settings)`: plays a game between two arms and, for a decisive game with `ponder_endings` and a deduction, walks it back, in its worker |
| `service/game_replayer.py` | `GameReplayer.replay(domain, summary)`: a remembered game played again from its actions and outcome seed, its positions exactly as they were |
| `service/continuous_trainer.py` | `ContinuousTrainer.train(domain, library, settings)`: plays games continuously, remembering every game and every proof; `ContinuousTrainerBuilder` wires it with its workers, memory cap and knowledge base |
| `service/ending_walker.py` | `EndingWalker(position_deducer).walk_back(domain, states, budget, limit)`: a game's positions deduced from the last one backward, stopping after the first one not proven or at the limit |
| `model/arm_library.py` | `ArmLibrary(domain, contexts=())`: the context each of a game's arms plays, by arm name; an arm is a variant of the game carrying its own position rules |
| `mapper/arm_library_json_mapper.py` | `ArmLibraryJsonMapper`: an arm library as JSON text and back, each arm with its context; an arm written under the older `signal` key reads the same |
| `repository/arm_library_repository.py` | `ArmLibraryRepository`: writes an arm library at a path and loads it back |
| `constant/continuous_constant.py` | `CONTINUOUS_GAME` (`"arms"`), `PROOF_KEYWORD` (`"proof"`), `NO_VALUE_RULES` (`"no value rules"`), `DEFAULT_ARM_EXPLORATION` (√2) |

## How value distillation works

1. `SelfPlay` plays `games` training games, then `held_out_games` more. Every game draws two seeds from `seed` up front:
   its agent searches with one, and its outcomes are drawn with the other. Games run in the task runner's workers. Every game keeps its
   positions, in order, with the search's mean payoff for the player to act in each (the visit-weighted mean of the
   root actions' mean payoffs, none for a random move played out of time), and its final payoffs. The searches' samples aren't kept: value rules don't read them,
   and a chess game's thousands of samples would fill the main process. A game that took its worker over the memory cap
   in a fresh worker too is left out (see `parallel/README.md`).
2. `PositionRowMapper` turns the games into rows at the `target`:
   - `outcome`: every position, once for each player, valued at that player's final payoff. One game's result is a
     noisy value for its early positions, but it is what the position led to in the agent's own play;
   - `search`: every position, for the player to act, valued at the search's mean payoff there: smoother, but only as
     good as the search, and leaning toward what its rollouts find; a position nothing was searched in gives no row.
3. `ValueGenerator` fits value rules on the training rows and chooses a price on the held-out rows (see
   `rbs/README.md`).
4. `held_out_error`: the mean absolute difference between the chosen rules' value for each held-out row's player and the
   row's target, over the rows the rules can value. The held-out rows also chose the price, so this error isn't
   independent of that choice.

## How continuous training works

`ContinuousTrainer.train(rbs, library, settings)` plays games between arms, the contexts an arm library holds, one
after another. There are no rounds, and nothing is learned from the games yet. While the library holds fewer than two
arms, the seats they leave are taken by an agent without position rules, the arm `no value rules`, whose search values
nothing and falls back on the deduction where there is one.

1. **A game starts** when a worker is free. UCB picks its two arms from the scores the knowledge base counts (see
   `GameMemory`). No search tree outlives its move.
2. **The worker studies the game** before taking the next one (`GameStudy`): a decisive game, one whose players' payoffs
   differ, is walked back from its end (`EndingWalker`), at most `ponder_endings` positions, stopping after the first
   position not proven.
3. **The game is remembered here** as it arrives, with its arms' models, and every position its walk proved is
   believed, its payoffs drawn by deduction, tagged with the keyword `proof`, its game and its ply.

It stops after `games` games, or runs until stopped.

Logs, besides every game's own lines:

- `INFO Playing continuously from game <n> on <games or until stopped>: <k> arms`
- `INFO Walked back <n> positions from the end, <p> proven`, from the worker
- `INFO Remembered arms game <n>: <p> of <w> positions walked back proven`

## Usage

```python
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.game_factory import create_game
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.value_distillation_settings import ValueDistillationSettings

knowledge_base = create_knowledge_base("tictactoe")
rbs = create_game("tictactoe", knowledge_base)
declarer = RuleDeclarer(knowledge_base, "tictactoe distilled")
declarer.inherits(rbs.context)                   # the fitted rules go to a variant of the game
values = ValueSettings((0.1, 0.03, 0.01, 0.003, 0.001), 1000, 1e-6, 3600.0, 8 * 1024**3)   # see rbs/README.md
result = create_value_distiller(knowledge_base, workers=8).distill(
    rbs,
    AgentBuilder().with_exploration(EXPLORATION),
    ValueDistillationSettings(games=100, held_out_games=25, iterations=200, seed=1, target="outcome", values=values),
    declarer,
)
result.rules   # the position rules declared under result.context
```

## Logs

- `openmind.training.service.self_play`: `INFO Self-play game with seeds <agent seed> and <outcome seed> finished in
  <plies> plies: <samples> samples, payoffs <player>=<payoff> ...`, logged by the worker as soon as the game ends, with
  ` by <ending>` after the plies when the domain says why games end (see `agent/README.md`), ` by <player>'s flag` when a
  player's time ran out, and on a clock ` on <time control>, clocks <player>=<seconds left> ...`; and `INFO Self-play game
  with seeds <agent seed> and <outcome seed> record: <record>` when the domain records games, such as a chess game's PGN;
  on a clock, `DEBUG Step <n>: <player> took <seconds> seconds of a <budget> second budget, <seconds> left` for every
  move
- `openmind.training.service.self_play`, with arms:
  - `INFO Self-play game with seeds <agent seed> and <outcome seed>, <arm> against <arm>, finished in <plies> plies: <samples> samples, payoffs <player>=<payoff> ...`, from the worker
  - `INFO Arms game <n>: <arm> against <arm>, payoffs <player>=<payoff> ...; <arm> scores <score> over <games> games, ...`, as each game's result comes back
- `openmind.training.service.value_distiller`: `INFO Distilled <n> value rules from <m> training rows valued at the
  <target> target; mean absolute error <error> on <h> held-out rows`

Every search also logs its summary (see `mcts/README.md`), and value generation logs its search and fits (see
`rbs/README.md`).

## Notes

- Tests: `factory/training_factory_tests.py`, `service/self_play_tests.py`, `mapper/position_row_mapper_tests.py`,
  `service/value_distiller_tests.py`, `builder/value_distiller_builder_tests.py`, `service/arm_selector_tests.py`,
  `service/ending_walker_tests.py`, `service/game_replayer_tests.py`, `mapper/arm_library_json_mapper_tests.py`.
