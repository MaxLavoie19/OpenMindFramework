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
| `service/self_play.py` | `SelfPlay`: an agent plays a domain against itself, games in the task runner's workers; returns every game, with its searches' samples unless `keep_samples` is false, and leaves out a game that took its worker over the memory cap in a fresh worker too; `play_arms(domain, builders, games, scores, selector, exploration, rng)` plays games between arms, a game's two arms chosen when a worker starts it. Both take `time_control`: games are then played on a clock, each move timed around the agent's search and charged to its player's clock, given to the agent with the steps its player has played, and a player whose time runs out doesn't play that move, the domain's timeout rule ending the game (the builders need a time budget estimator, see `agent/README.md`) |
| `model/played_game.py` | `PlayedGame(samples, states, search_values, payoffs, arms=(), actions=(), time_control=None, seconds=(), budgets=(), clocks=(), flagged=None)`: a self-play game's search samples, its positions with the search's mean payoff for the player to act in each, its final payoffs, in a game between arms the arm each player followed, and the actions played in order; on a clock, the time control, each step's seconds and budget in order (a step that ran its player's time out included, though its action wasn't played), each player's clock at the end, and the player whose time ran out; then the game's seeds and why it ended |
| `service/distiller.py` | `Distiller`: self-play, rule generation and validation, and the result's measures |
| `builder/distiller_builder.py` | `DistillerBuilder`: sets how many worker processes self-play games and rule condition checks run in (`with_workers`, 1 by default) and wires self-play, the rule generator, and the rule compiler, runner and consequence library its rater checks conditions with |
| `factory/training_factory.py` | `create_distiller(workers=1)`, `create_rule_selector(workers=1)`, `create_value_distiller(workers=1, memory_cap=None)` and `create_value_training_loop(workers=1, memory_cap=None)` |
| `constant/training_constant.py` | Also the value distillation defaults: 100 games and 25 held-out games; the targets, `outcome` and `search` |
| `mapper/position_row_mapper.py` | `PositionRowMapper`: self-play games to the position rows value rules are fitted on, at a target |
| `model/value_distillation_settings.py` | `ValueDistillationSettings(games, held_out_games, iterations, seed, target, values, pondering=None, signals=None, time_control=None, expected_steps=30, selection='ucb1', puct_exploration=1.5, prior='uniform', prior_temperature=0.1)`; `values` is the value generator's `ValueSettings`; `time_control` puts self-play, arms games and the loop's games on a clock, every agent budgeting its moves with `PlainTimeBudgetEstimator(expected_steps)`; every agent searches by `selection`, following the `value` prior from its own value rules when it has them and the uniform prior otherwise |
| `model/value_distillation_result.py` | `ValueDistillationResult(value_base, fits, chosen, candidates, training_rows, held_out_rows, held_out_error, pondering=None)` |
| `service/value_distiller.py` | `ValueDistiller`: self-play, pondering, value rule generation, and the chosen rules' error on held-out rows; with a game memory (`ValueDistillerBuilder.with_game_memory`), every game, self-play and arms, training and held-out, is remembered as it ends with the model each player played (`distill(..., round_number, model_name)` name them), and arms' scores for UCB are counted from what it remembers instead of the signal library |
| `mapper/played_game_summary_mapper.py` | `PlayedGameSummaryMapper`: a self-play game as a `GameSummary`, with its kind, round, number, models and record |
| `builder/value_distiller_builder.py` | `ValueDistillerBuilder`: sets how many worker processes self-play games, deductions and term evaluations run in (`with_workers`, 1 by default) and the memory each holds at most (`with_memory_cap`, no cap by default), and wires the value distiller and its ponderer |
| `model/pondering_settings.py` | `PonderingSettings(positions, budget)`: how many positions a round ponders and the `DeductionBudget` each gets |
| `model/pondering.py` | `Pondering(rows, deductions, seeds, sources)`: the rows with proven targets, every deduction, and the seeds induced, the most often induced first |
| `model/pondering_summary.py` | `PonderingSummary(positions, proven, seeds, seeds_kept, seeds_in_rules)` |
| `service/position_ponderer.py` | `PositionPonderer.ponder(domain, rows, value_base, settings)`: deduces the positions the rules missed most, values proven rows at the proven payoffs, and gathers their seeds |
| `constant/signal_constant.py` | `WIN` (`"win"`), the signal of winning itself; `UNIFORM` (`"uniform"`), the aggregation where every signal votes the same; `WEIGHTED` (`"weighted"`), the aggregation where every signal votes by its reliability; `DEFAULT_ARMS` (8); `DEFAULT_SIGNAL_HORIZON` (0); `DEFAULT_ARM_EXPLORATION` (√2); the deduced signals' names, `OPTIONS`, `MY_OPTIONS`, `THEIR_OPTIONS`, `OWNED`, `TAKING`, `LOSING`, `TAKING_MOVES`, `LOSING_MOVES`, `FORK`, `MATERIAL` and `HANGING` |
| `model/signal_settings.py` | `SignalSettings(arms=8, horizon=0, exploration=√2, goal_limit=2)`: how many signals a round follows besides winning and the aggregations, how many plies later signals are read for targets, UCB's exploration between arms, and how many moves ahead the deduced goal distance looks for a win; negative numbers, or a goal limit below 1, raise `ValueError` |
| `model/signal_readings.py` | `SignalReadings(anchors, proven, games, decisive, signs)`: what signals read on a round's anchors, +1, -1 or 0 per anchor by signal name |
| `service/arm_selector.py` | `ArmSelector.pair(scores, pending, arms, exploration, rng)`: the two arms, signals an agent follows, with the highest UCB1 bounds on their game scores; games under way count without points, an arm never chosen goes first, ties are drawn at random |
| `service/signal_ranker.py` | `SignalRanker`: `best(library, count)`, the most reliable signals read from positions, ties going to the most read; `reliabilities(library, signals)`, a signal never read counting as fully reliable |
| `service/signal_targeter.py` | `SignalTargeter(term_evaluator)`: `rows(domain, games)`, every position for each player; `targets(domain, games, signals, reliabilities, horizon, deductions=())`, each followed signal's targets on those rows |
| `model/signal.py` | `Signal(name, source=None, parts=(), premises=())`: something that reads a position for a player: winning itself (no source, no parts), an expression's value for the player (a source), or an aggregation voting with the signals its parts name; a signal deduced from the rules names its premises |
| `service/heuristic_deducer.py` | `HeuristicDeducer(expression_generator, mechanics)`: `deduce(domain, goal_limit=2)`, the signals deduced from a two-player domain's rules alone as general principles to start from, each naming its premises |
| `model/signal_record.py` | `SignalRecord(signal, agreements=0, disagreements=0, games=0, wins=0, draws=0, losses=0)`: how often the signal pointed to the coming winner over every round, and the games an agent following it played; `accuracy` (0.5 before any reading) and `reliability` (2 × accuracy − 1, at least 0) |
| `model/rule_support.py` | `RuleSupport(term, strengths)`: a value rule's term and, by signal name, its standardized weight in that signal's fit |
| `model/signal_library.py` | `SignalLibrary(domain, records=(), supports=(), value_bases=())`: every signal's record, the rules the signals support, and the value base last fitted to each followed signal, kept across rounds and runs |
| `service/signal_recorder.py` | `SignalRecorder(term_evaluator)`: `read(domain, signals, games, deductions=())` reads every signal at every anchor for the winner and the loser; `aggregate(readings, aggregation, weights=None)` adds an aggregation's votes; `add(library, signals, readings)` adds the agreements and disagreements to the records; `record` reads then adds |
| `service/signal_library_updater.py` | `SignalLibraryUpdater.update(library, fits)`: replaces each fitted signal's strengths, weighs rules by standing, drops rules no reliable signal supports, and keeps this round's fits as the value bases; `standing(library, support)`; `score(library, games)`, each game between arms a win, a draw or a loss on its arms' records |
| `mapper/signal_library_json_mapper.py` | `SignalLibraryJsonMapper`: a signal library as JSON text and back, every record with its accuracy and reliability, every support with its strengths |
| `repository/signal_library_repository.py` | `SignalLibraryRepository`: writes a signal library at a path and loads it back |
| `model/value_training_settings.py` | `ValueTrainingSettings(rounds, distillation, rollout_actions, rollout_limit, unfinished_payoff, evaluation_games, start_file, deduction=None)`; `deduction` is the budget agents fall back on when their rules have no clue |
| `model/training_round.py` | `TrainingRound(number, value_base, fits, chosen, training_rows, held_out_rows, held_out_error, baselines, against_previous, seconds, pondering=None)`: one round's value rules, fits, games and pondering |
| `model/training_report.py` | `TrainingReport(domain, created_at, settings, rounds, complete)` |
| `constant/training_constant.py` | Also the training loop defaults: 3 rounds, 200 games and 50 held out, 100 iterations, 10 rollout actions before valuing, 20 games per opponent; `START_RULES` |
| `service/value_training_loop.py` | `ValueTrainingLoop`: rounds of self-play valuing positions with the previous round's rules, value fitting, and games against the random policy, untrained MCTS and the previous round's agent; with the signals target and no value bases yet, it prepares the deduced signals first; with a game memory (`ValueTrainingLoopBuilder.with_game_memory`), every game, evaluation series included, is remembered as it ends |
| `service/signal_preparer.py` | `SignalPreparer(heuristic_deducer)`: `prepare(domain)`, a library with a record for every deduced signal and the value bases round 1's arms follow: `deduced`, every weight 1 over the number of signals, and `deduced, <signal> doubled` for each signal |
| `builder/value_training_loop_builder.py` | `ValueTrainingLoopBuilder`: sets how many worker processes self-play, term evaluations and games run in (`with_workers`, 1 by default) and the memory each holds at most (`with_memory_cap`, no cap by default), and wires the loop |
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
   root actions' mean payoffs), and its final payoffs. The searches' samples aren't kept: value rules don't read them,
   and a chess game's thousands of samples would fill the main process. A game that took its worker over the memory cap
   in a fresh worker too is left out, and so is a pondered position (see `parallel/README.md`).
2. `PositionRowMapper` turns the games into rows at the `target`:
   - `outcome`: every position, once for each player, valued at that player's final payoff. One game's result is a
     noisy value for its early positions, but it is what the position led to in the agent's own play;
   - `search`: every position, for the player to act, valued at the search's mean payoff there: smoother, but only as
     good as the search, and leaning toward what its rollouts find.
3. **Pondering,** with `pondering` settings. `PositionPonderer` reasons about the training positions the rules know least
   about:
   - first, with `endings`, it walks back from the end of every decisive training game, one whose players' payoffs
     differ, in the task runner's workers: each game deduces its positions from the last one backward and stops at the
     first position not proven, or after its share of `endings`, the number divided evenly between the decisive games
     (at least 1 each). A won game's last position is a win in one, the one before a win in a few: its proofs are exact
     targets where the search's estimates sit near a draw, and they induce seeds like any proof;
   - every training row's miss is how far its target is from the previous rules' value for its player (the value base
     `distill` is given, the previous round's in a training loop), or from the mean target without rules or where they
     can't value the position;
   - the positions in play with the largest misses, `positions` of them, leaving out those the walks proved, are deduced within the budget in the task
     runner's workers (see `inference/README.md`). Without rules, the value is constant, so the positions of decisive
     games come first;
   - a proven position's training rows take the proven payoffs as targets, in place of the search's estimate or the
     game's outcome: a proof is exact where both are guesses;
   - every proven deduction induces seeds, and a seed more positions induced comes first, since it's likelier to help.
4. `ValueGenerator` fits value rules on the training rows, trying the seeds first, and chooses a price on the held-out
   rows (see `rbs/README.md`). The result's `pondering` counts the positions pondered and proven, the seeds, the seeds
   the search kept and those among the value rules, and the positions the walks deduced and proved.
5. `held_out_error`: the mean absolute difference between the chosen rules' value for each held-out row's player and the
   row's target, over the rows the rules can value. The held-out rows also chose the price, so this error isn't
   independent of that choice.

How the value rules play shows in `openmind-evaluate --values` (see `evaluation/README.md`).

## Signals

Everything that reads a position for a player is a signal: winning itself, a piece count, a move count, any expression
the engine builds. Winning is exact but sparse, one result per game and nothing from draws; smoother signals are read at
every position, and each is weighed by how well it has pointed to the coming winner. Signals are discovered, never
written by hand. A round learns from them with the `signals` target, in a two-player domain:

0. **Arms.** Once the library holds two value bases or more, those of the signals the previous round followed, every
   self-play game, training and held out, is between two agents each following one of them. Games wait in a queue; when
   a worker takes one, `ArmSelector` chooses its two arms by UCB1 on their scores so far, points per game with a win 1
   and a draw 0.5, games under way counting without points and arms never chosen first, and the game's first arm plays
   first in even games. Every game's result counts for the next choice as soon as it ends (`TaskRunner.stream`, see
   `parallel/README.md`), and adds a win, a draw or a loss to its arms' records. Games between arms depend on the order
   games finish, so on the number of workers. Round 1 without such a library plays plain self-play.
1. **Pondering,** as with the other targets, on rows valued at the search target: its deductions, the walks' included,
   and its seeds serve the signals.
2. **Candidates.** Every signal read from positions this round, named by its source, once: the seeds, the rules the
   library's signals support, the signals already recorded, and the leaves of the training games' positions (see
   `inference/README.md`); winning too.
3. **Records** of every candidate on the training games' anchors, added to the library.
4. **Following:** winning, the `arms` signals with the best records (`SignalRanker.best`), and two aggregations over
   those, `uniform` and `weighted`, recorded too. Until arms play each other, nothing else chooses them.
5. **Targets** (`SignalTargeter`) of every followed signal on every training and held-out position, for each player: a
   signal read `horizon` plies later as the player's reading minus the other's, standardized over the round, through
   expit; winning, the player's final payoff; `uniform`, expit of the mean of every followed signal's standardized
   difference; `weighted`, the same weighed by reliability. A proven position takes the proven payoffs in every target.
   A blank reading (the signal giving `None`) is a player lacking what the signal reads and counts as 0 in the
   difference; blank for both players is no reading: that signal's target there is 0.5, its spread is taken over the
   rows with a reading, and the aggregations average only the signals with a reading at each row.
6. **Fitting:** one expression search against every target, then a fit per target
   (`ValueGenerator.generate_for_targets`), and the library's supports updated.
7. **The round's rules,** those its games against the baselines use and the next round's plain agent follows, are the
   value base of the fitted signal with the best game score, ties going to the one that played most; the `weighted`
   signal's while no fitted signal has played. Their held-out error is measured against that signal's targets. Every
   fitted signal's value base becomes one of the library's, the arms of the next round. The result holds the library
   and the followed signals' records.

`openmind-train-values --target signals` saves the library after every round and can start from an earlier run's
(`openmind-distill-values` refuses the target: one distillation has no next round to hand the library to).

**Signals deduced from the rules.** Before any game there is no data, so `HeuristicDeducer` tests nothing: it infers
general principles from the rules for a clueless agent to fall back on, and games weigh them afterwards, until trades
worked out in play replace them. A base is a player's things when its values in the initial position name players,
such as `color` in chess. Every signal is normalized by the position's own totals, so it reads from -1 to 1, and 0 when
its divisor is 0; most read 0 or blank in the initial position. Each signal names its premises:

| Signal | Principle | Reads, for the player | Premises |
|---|---|---|---|
| `options` | more options is better | (its legal moves − the other player's) ÷ both | none |
| `my options` | more options for me | its legal moves ÷ both players' | `options` |
| `their options` | an opponent with fewer moves is easier | − the other player's legal moves ÷ both players' | `options` |
| `goal distance` | being closer to winning is better | (the other player's distance to a win − its own) ÷ both, a distance being the fewest moves to a win if the other player did nothing (`solo_distance`), up to the goal limit; blank when neither can win within it | none: it reads the rules' win condition |
| `owned <base>` | moves come from things, so more things is better | (its things − the other player's) ÷ both | `options` |
| `taking <base>` | taking their things is good | the most of the other player's things it can take with one move ÷ their things; blank when none | `owned <base>` |
| `losing <base>` | losing my things is bad | − the most of its things the other player can take with one move ÷ its things; blank when none | `owned <base>` |
| `taking moves <base>` | taking things with many moves is better | the most legal moves the other player loses from one of its taking moves ÷ their moves; blank when none | `taking <base>`, `their options` |
| `losing moves <base>` | losing things with many moves is worse | − the most legal moves it loses from one of the other player's taking moves ÷ its moves; blank when none | `losing <base>`, `my options` |
| `fork <base>` | a double threat is good | how many of its moves leave at least two moves each taking one of the other's things ÷ its moves; blank when none | `taking <base>` |
| `material <base>` (a grid) | things with many moves are worth more | (the worth of its things − the other's) ÷ both; a thing is worth its kind's average legal moves alone on every cell, its kind being what the other grids with the same cells hold there (a piece in chess) | `owned <base>`, `taking moves <base>` |
| `hanging <base>` (a grid) | a thing they can take and I can't take back is bad | − how many of its things the other player can take where, were the thing the other's, it couldn't take it, ÷ its things; blank when none | `losing <base>` |

**Preparation.** With the signals target and a library without value bases (a new run, or a library saved before
preparation existed), `ValueTrainingLoop` has `SignalPreparer` deduce the signals before round 1. Their records join the
library, never read yet, and the library's value bases become `deduced`, every signal weighing 1 over their number, bias
0, payoffs from 0 to 1, and `deduced, <signal> doubled` for each signal. Round 1's self-play games are between those
bases, as arms; every round records the deduced signals with the others, under their own names, so games give them their
worth. A library loaded with value bases is used as it is. Logged as `Prepared <n> signals deduced from the rules: value
base deduced, every weight <w>, and <n> variations doubling one weight each`.

Each is logged as `Derived <signal> from <premises>: <source>`, and a grid's worth table as `Worth of a thing of <base>
by its kind (<grids>), its average legal moves alone on every cell: <table>`; the table is written into the `material`
signal's source. Tic-tac-toe starts with an empty board, so only the
options are derived.

- **Anchors.** Every position of a round's games whose coming winner is known: a position a deduction proved with
  payoffs that differ, or any other position of a decisive game. A proven draw is no anchor. A position met in several
  games is an anchor in each.
- **Records.** `SignalRecorder` reads each signal with a source at every anchor for the winner and for the loser, in the
  term evaluator's workers. The winner reading higher agrees, the loser reading higher disagrees, a tie counts neither.
  A blank reading counts as 0, the player lacking what the signal reads; blank for both is a tie.
  Winning always agrees; an aggregation votes with its parts, each part pointing to the player it reads higher. A signal
  that can't be read on every anchor adds nothing that round. Records keep counting across rounds.
- **Reliability.** 2 × accuracy − 1, at least 0: 1 when a signal always pointed to the winner, 0 when it did no better
  than a coin flip, 0.5 accuracy before any reading.
- **Support.** One expression search runs against every followed signal's targets and keeps an expression any of them
  supports (see `inference/README.md`); each signal then gets its own fit (`ValueGenerator.generate_for_targets`, see
  `rbs/README.md`). A rule's strength from a signal is its standardized weight in that fit.
- **Standing.** `SignalLibraryUpdater` replaces the strengths of every signal fitted this round, keeps those of signals
  not fitted, and weighs each rule by its standing, Σ reliability × |strength| over its signals; a signal never read yet
  counts as fully reliable. A rule whose standing is 0 leaves the library; there is no limit on how many stay.

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
4. **Report.** With `evaluation_games`, the report is first handed to `on_round` as soon as the round's rules are
   fitted, the round without games yet, and again once its games are done; without, once. `openmind-train-values`
   saves a round's value base the first time it sees the round, and the report every time, so a training stopped or
   killed during a round's games keeps that round's rules. The last report is complete.

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
- `openmind.training.service.value_distiller`, with the signals target:
  - `INFO The round's rules are the <signal> signal's: score <score> over <games> games, the best of the signals followed`, or `...: no signal followed has played a game yet`
  - `INFO <n> candidate signals: <s> seeds, <r> supported rules, <k> recorded signals, and the leaves of <g> games' positions`
  - `INFO Following <signal>: reliability <r> from <a> agreements and <d> disagreements`, per followed signal
  - `INFO Distilled <n> value rules from <m> training rows valued at the signals target, the weighted signal's; mean absolute error <e> on <h> held-out rows`
- `openmind.training.service.signal_targeter`:
  - `INFO Targets for <n> signals on <m> rows, read <h> plies later; <p> rows proven`
  - `INFO Signal <name> couldn't be read on every row: no target this round`
- `openmind.training.service.signal_recorder`:
  - `INFO Recorded <n> signals on <a> anchors, <p> of them proven, from <d> decisive games of <g>; <u> signals couldn't be read`
  - `DEBUG Signal <name>: <a> agreements and <d> disagreements this round, reliability <r>`
- `openmind.training.service.signal_library_updater`:
  - `INFO Signals support <n> rules after fitting <k> signals; dropped <m> rules no reliable signal supports`
  - `INFO Arm <name>: <w> wins, <d> draws, <l> losses of <g> games, score <s>`
  - `DEBUG Rule <term>: standing <s>; <signal>=<strength> ...` and `DEBUG Dropped rule <term>: no reliable signal supports it`
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
  - `INFO Round <k> fitted: handing it over before its games`, with evaluation games and an `on_round`
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
