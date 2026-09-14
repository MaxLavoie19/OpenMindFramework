# rbs

## Purpose

The rule-based system: an explainable model learned from search. Its rules rate actions by expected payoff for the
player to act, and those ratings guide MCTS. The RBS is one kind of model behind the generic `ActionRater` interface in
`mcts`; a decision tree, a DNN or another model could implement the same interface later, chosen for its speed,
accuracy and explainability.

Its second kind of rules, value rules, turn a position into each player's expected payoff, so that a search can value
the positions it reaches without playing them to the end, which random play does badly in deep games. They sit behind
`PositionValuer` in `mcts`, the way rules about actions sit behind `ActionRater`: weighted Python terms, fitted from
self-play with a price on every weight, so the terms that don't pay for themselves drop out.

Rules are generated for any domain from its rules and its search samples: nothing here knows a game. A rule generator
works as an inference engine: it proposes hypotheses about which actions deserve more or less exploration, discovers
them on some games and tests them on others with a permutation test and a false discovery rate, and only the
hypotheses that hold become rules.

## Content

| File | What it is |
|---|---|
| `model/rule.py` | `Rule(action, conditions, expected_value, visits, priority=False)`: for an action with that name, when every condition (a Python rule) holds, none meaning any state, its expected payoff for the player to act over that many search visits; a priority rule rates before the others |
| `model/rule_base.py` | `RuleBase(domain, rules)` |
| `model/generation_settings.py` | `GenerationSettings(min_visits, max_conditions, min_rule_visits, min_gain, confidence, beam_width, max_offset, solo_limit, patterns, false_discovery_rate, permutations, coverage=True)` |
| `model/action_row.py` | `ActionRow(state, action, visits, mean_payoff, advantage)`: an action's merged samples in a state |
| `model/row_arrays.py` | `RowArrays(states, visits, advantages, payoffs)`: action rows as numpy arrays, states numbered |
| `model/goal_pattern.py` | `GoalPattern(action, conditions, moves)`: what a winning move needs, seen in that many winning moves |
| `model/hypothesis.py` | `Hypothesis(action, conditions, parent, direction, effect, states, visits, expected_value, priority, score)`: a discovered claim that the actions matching the conditions, within the parent's scope, have a higher (direction 1) or lower (-1) advantage |
| `model/hypothesis_test.py` | `HypothesisTest(action, conditions, direction, discovery_effect, discovery_states, validation_effect, validation_states, p_value, q_value, validated)` |
| `model/generation_result.py` | `GenerationResult(rule_base, patterns, hypotheses, covered)`: the rule base, the goal patterns, every hypothesis's test and the validated rules a simpler rule covers |
| `model/coverage.py` | `Coverage(rule, covering)`: a validated rule left out of the rule base, and the simpler kept rule matching every row it matches, whose rows' mean advantage is within `min_gain` of its rows' |
| `constant/generation_constant.py` | Default generation settings; the exploration presets (`EXPLORE_BEAM_WIDTH` 60, `EXPLORE_MAX_CONDITIONS` 3, `EXPLORE_MIN_GAIN` 0.02, `EXPLORE_FALSE_DISCOVERY_RATE` 0.2); `MIN_STATES` (5), `PRIORITY_MARGIN` (0.05), `QUANTITY_CUTS` (6), `ANCHOR_PROBES` (50), `PERMUTATION_BATCH` (1,000) |
| `constant/consequence_constant.py` | The names generated rules read (`me`, `other`, `action`, `win_chance`, `wins`, `solo_distance`, `near`, `OUTSIDE`) and the consequence cache size (200,000) |
| `service/consequence_library.py` | `ConsequenceLibrary`: what generated rules read besides state variables, worked out with the domain's own solver and predictor |
| `service/goal_pattern_miner.py` | `GoalPatternMiner`: the variables winning moves need |
| `service/condition_evaluator.py` | `ConditionEvaluator`: a rule's values, or where it is true, on action rows; several rules at once in the task runner's workers, the rows split in slices |
| `service/primitive_generator.py` | `PrimitiveGenerator`: the single conditions hypotheses are built from |
| `service/advantage_contrast.py` | `AdvantageContrast`: per state, the mean advantage of the actions a condition matches minus that of the others |
| `service/hypothesis_discoverer.py` | `HypothesisDiscoverer`: hypotheses from discovery rows, by beam search over conjunctions |
| `service/hypothesis_validator.py` | `HypothesisValidator`: sign-flip permutation tests on validation rows, Benjamini-Hochberg q-values |
| `service/coverage_filter.py` | `CoverageFilter`: leaves out the validated rules a simpler rule already covers |
| `service/rule_generator.py` | `RuleGenerator`: mines, generates, discovers and validates, and returns the rule base |
| `service/rule_rater.py` | `RuleRater`: an `ActionRater`; rates each action with its first matching rule and explains the rating |
| `mapper/action_row_mapper.py` | `ActionRowMapper`: search samples to action rows, action rows to arrays |
| `mapper/hypothesis_text_mapper.py` | `HypothesisTextMapper`: a tested hypothesis as readable text |
| `mapper/rule_text_mapper.py` | `RuleTextMapper`: a rule as readable text |
| `mapper/rule_base_json_mapper.py` | `RuleBaseJsonMapper`: a rule base as JSON text and back, each condition as its Python source; `priority` defaults to false on load |
| `repository/rule_base_repository.py` | `RuleBaseRepository`: saves a rule base as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and loads it |
| `builder/consequence_library_builder.py` | `ConsequenceLibraryBuilder`: wires the library's solver, predictor, state reader and variable name mapper |
| `builder/rule_generator_builder.py` | `RuleGeneratorBuilder`: sets how many worker processes conditions are checked in (`with_workers`, 1 by default) and wires the generator's services around one consequence library |
| `builder/rule_rater_builder.py` | `RuleRaterBuilder`: sets the rule base and the domain and wires the rater; rejects a missing rule base or domain |
| `factory/rbs_factory.py` | `create_rule_generator(workers=1)`, `create_rule_rater(rule_base, domain)`, `create_value_generator(workers=1)` and `create_rule_valuer(value_base, domain)` |
| `model/value_rule.py` | `ValueRule(term, weight)`: a term of a position's value, a Python value rule giving a number for the player valued, and its weight |
| `model/value_base.py` | `ValueBase(domain, bias, low, high, rules)`: a player's value is `low + (high - low) × logistic(bias + Σ weight × term)` |
| `model/position_row.py` | `PositionRow(state, player, target)`: a position valued for a player, and the payoff its value is fitted to |
| `model/value_settings.py` | `ValueSettings(pair_pool, cuts, solo_limit, prices, max_steps, tolerance)` |
| `model/sparse_fit.py` | `SparseFit(weights, bias, steps, settled)`: weights fitted at a price; a weight of 0 drops its term |
| `model/value_fit.py` | `ValueFit(price, terms_kept, steps, settled, training_loss, held_out_loss)`: one price of a sweep |
| `model/value_generation_result.py` | `ValueGenerationResult(value_base, fits, chosen, candidates)` |
| `constant/value_constant.py` | Default value settings: pair pool 20, 6 cuts, prices 0.1, 0.03, 0.01, 0.003 and 0.001, 1,000 steps, tolerance 1e-6; `COUNT_VARIABLE` (`value`) |
| `service/term_evaluator.py` | `TermEvaluator`: a term's values on position rows as numbers; several terms at once in the task runner's workers, the rows split in slices |
| `service/term_generator.py` | `TermGenerator`: the single terms value rules are fitted from |
| `service/sparse_fitter.py` | `SparseFitter`: a logistic fit with an L1 price on its weights, by accelerated proximal gradient |
| `service/value_generator.py` | `ValueGenerator`: generates terms and their pairs, fits them at every price, and returns the value base of the fit best on held-out rows |
| `service/rule_valuer.py` | `RuleValuer`: a `PositionValuer`; values each player's position with a value base and explains what each rule adds |
| `mapper/value_rule_text_mapper.py` | `ValueRuleTextMapper`: a value rule as readable text, `+0.42 × wins(me)` |
| `mapper/value_base_json_mapper.py` | `ValueBaseJsonMapper`: a value base as JSON text and back, each term as its Python source |
| `repository/value_base_repository.py` | `ValueBaseRepository`: saves a value base as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and loads it |
| `builder/value_generator_builder.py` | `ValueGeneratorBuilder`: sets how many worker processes terms are evaluated in (`with_workers`, 1 by default) and wires the generator around one term evaluator |
| `builder/rule_valuer_builder.py` | `RuleValuerBuilder`: sets the value base and the domain and wires the valuer; rejects a missing value base or domain |

## What generated rules read

A condition is a Python value rule (see `rule/README.md`). Besides the state's variables and the action's parameters, it
reads:

- `action`: the action being rated; an action parameter named `action` is rejected;
- `me` and `other`: the player to act and the next player in the domain's order; in a value rule, the player the
  position is valued for and the next player;
- `win_chance(action)`: the probability that the action ends the game in a win for the player taking it;
- `wins(player, action=None)`: the summed win chances of the actions `player` could take if it were their turn, now or,
  expected over its outcomes, after `action`: `wins(me, action) >= 2` is a fork;
- `solo_distance(player, action=None, limit=2)`: the fewest of `player`'s own actions after which a win is possible, if
  nobody else moved, now or after `action`; `limit + 1` when none is found;
- `near(action, *offset)`: the value, now, of the variable at that index offset from the indexed variable the action
  sets (its *anchor*), or `OUTSIDE`: `near(action, 0, -1) == me` is "my mark just left of where I play".

A win is an outcome with no legal action left in which the player's payoff is higher than every other player's. The
consequence library works these out with the domain's solver and predictor and caches them, cleared when 200,000
entries are reached.

## How rules are generated

`RuleGenerator.generate(domain, discovery, validation, settings, seed)` takes the search samples of discovery games
and, separately, of validation games.

1. **Action rows.** Samples of the same action in the same state are merged. Rows with fewer than `min_visits` visits
   are dropped. A row's *advantage* is its mean payoff minus the best mean payoff in its state, so it only compares
   actions of the same state; a state left with a single row is dropped.
2. **Goal patterns.** Up to `patterns` distinct discovery moves that win with certainty are probed: every variable
   holding a player's name is changed, one at a time, to a value that isn't a player's name. A variable whose change
   makes the move illegal or no longer a certain win is needed. The needed variables, written relative to the mover
   (`me`, `other`) and, when they share the anchor's base and index count, as `near(action, ...)`, form the move's goal
   pattern; patterns are counted over the moves and sorted by count.
3. **Primitives,** for each action name, from the rows and the domain's players only:
   1. every state variable at every value seen, a player's name written as `me` or `other`: `cell[2, 2] == me`; the
      variable naming the player to act stays absolute: `turn == 'X'`;
   2. every indexed variable read at the action's parameters, in every order: `cell[row, col] == None`;
   3. every parameter at every value seen: `col == 3`;
   4. `near(action, ...)` at every offset up to `max_offset`, at `me`, `other`, `OUTSIDE` and the other values the
      anchor's base takes;
   5. the conditions of the action's goal patterns;
   6. thresholds on quantities, at up to `QUANTITY_CUTS` values seen (quantiles when there are more):
      `win_chance(action)`, `wins(me)`, `wins(other)`, `wins(me, action)`, `wins(other, action)`, the changes
      `wins(me, action) - wins(me)` and `wins(other, action) - wins(other)`, and, unless `solo_limit` is 0,
      `solo_distance(me, action, solo_limit)` and `solo_distance(other, action, solo_limit)`.
4. **Discovery.** A primitive that raises `KeyError`, `NameError` or `TypeError` on a row, is true everywhere or nowhere,
   or matches the same rows as an earlier one (or their complement) is dropped.
   - A primitive true for some actions of a state and false for others *splits* exploration. Within a scope, the
     contrast is taken state by state: the visit-weighted mean advantage of the matching actions minus that of the
     others, in every state where both occur. It becomes a hypothesis when the matching actions have
     `min_rule_visits` visits, the contrast covers `MIN_STATES` states and its mean, the *effect*, is at least
     `min_gain` away from 0. Its score is the effect over its standard error.
   - A primitive with the same value for every action of a state can't split exploration, but can *gate*: it becomes a
     scope for the others, such as `wins(other) >= 1` for "when the opponent threatens".
   - The `beam_width` best-scored hypotheses of each size, and the gates, are extended with one more splitting
     primitive, up to `max_conditions` conditions. Each goal pattern is also tried whole, as one hypothesis.
   - No hypothesis has a condition that changes nothing on the discovery rows. A combination is skipped when one of
     its conditions can be dropped without changing the rows it matches, such as `wins(other) >= 1 and
     wins(other, action) >= 1` when every action leaving the opponent a win is in a position where the opponent could
     already win. A goal pattern drops its conditions, first to last, while the rows it matches stay the same.
     Hypotheses matching the same rows are merged into the one with the fewest conditions, then the highest score.
     Fewer hypotheses also make the false discovery rate correction milder.
   - A hypothesis's expected value is the visit-weighted mean payoff of its matching rows. It is a *priority*
     hypothesis when that mean, `confidence` bound included (its standard error shrunk toward the payoffs' spread),
     lies within `PRIORITY_MARGIN` of the spread from the highest or lowest payoff seen: a sure win or a sure loss.
5. **Validation,** on the validation rows only. For each hypothesis, the same contrast is taken within its parent's
   scope. Under the null hypothesis, the condition doesn't tell the actions apart, so each state's difference is as
   likely to have either sign: a one-sided sign-flip permutation test in the discovered direction, with `permutations`
   permutations drawn from `seed`, gives the p-value `(1 + permutations at least as extreme) / (1 + permutations)`. A
   hypothesis with fewer than 2 validation states, or one that can't be evaluated there, gets a p-value of 1. The
   p-values of every hypothesis, of every action, are adjusted together with Benjamini-Hochberg. A hypothesis is
   *validated* when its q-value is within `false_discovery_rate` and its validation effect points the discovered way.
   The smallest p-value a test can give is `1 / (1 + permutations)`, so with many hypotheses, few validation states or
   few permutations, nothing can be validated.
6. **Rule base.** For each action name, in the order first seen: a rule without conditions, its visit-weighted mean
   payoff, then one rule per validated hypothesis, with its conditions, expected value, visits and priority.
7. **Coverage**, unless `coverage` is off, as with `openmind-distill --explore`, whose candidates `openmind-select`
   narrows down by play instead. A validated rule is left out, as *covered*, when a kept rule already covers it: one with no more
   conditions, a priority rule whenever it is one, matching every discovery row it matches, whose rows' visit-weighted
   mean advantage is within `min_gain` of its own rows'. Advantage compares moves within their own positions, so how
   good those positions are cancels out. Rules are checked fewest conditions first, then most rows matched, so
   `win_chance(action) >= 1` covers `cell[1, 7] == None and win_chance(action) >= 1`. With first-match rating, the rule
   that rates a covered rule's actions may be another than the one covering it.

   Judging coverage on expected values was tried first, and measured on tic-tac-toe's 4,520 positions against the same
   generation without coverage (46 rules): it left 15 rules whose ratings alone picked an optimal move in 4,105
   positions instead of 4,247. Judged on advantage, 14 rules picked one in 4,375, guided search chose an optimal move
   more often at 10 iterations (154 positions against 93, McNemar p 0.0001) and no less often at 20, 50 or 100, and
   each choice took about a quarter less time.

## How rules rate

For each action, the rules for its name are tried priority rules first, then those with the most conditions, then the
most visited; the first whose conditions all give `True` rates the action. An action name without rules gets `None`. A
condition raising `KeyError`, `NameError` or `TypeError`, such as one reading a variable the state doesn't have, doesn't
hold. `explain(state, action)` returns the rule behind the rating.

## How value rules are generated

`ValueGenerator.generate(domain, training, held_out, settings)` takes position rows, each a position, the player it is
valued for and the payoff to fit, from training games and, separately, from held-out games; `training/README.md` says
where rows come from.

1. **Payoff range.** `low` and `high` are the lowest and highest training payoffs, and payoffs are scaled from 0 to 1
   between them. When every training payoff is the same, nothing is fitted: the value base has no rules and values every
   position at that payoff.
2. **Single terms,** from the rows and the domain's players only, a player's name written as `me` or `other`:
   1. every state variable at every value seen: `cell[2, 2] == me`; the variable naming the player to act is also
      written absolutely: `turn == 'X'`;
   2. for every indexed variable, how many of its variables hold each value seen:
      `sum(value == me for value in cell.values())`;
   3. `wins(me)`, `wins(other)` and, unless `solo_limit` is 0, `solo_distance(me, None, solo_limit)` and
      `solo_distance(other, None, solo_limit)`, as they are and at up to `cuts` thresholds seen: `wins(other) >= 1`.
3. **Usable terms.** A term is dropped when it raises `KeyError`, `NameError` or `TypeError`, or gives something other
   than a finite number, on a training or a held-out row; when it has the same value on every training row; or when its
   training values repeat an earlier term's.
4. **Pairs.** The `pair_pool` single terms most correlated with the scaled payoffs are multiplied two by two:
   `(wins(me) >= 1) * (turn == me)`. A product is the smooth form of a conjunction, so a pair can say what neither of its
   terms says alone. Pairs go through the same filter.
5. **Fits.** Columns are standardized on the training rows. For each price, from the highest down, starting from the
   previous price's weights, `SparseFitter` minimizes the mean logistic loss of the scaled payoffs plus price × the sum
   of the weights' absolute values, the bias unpriced. It takes accelerated proximal gradient steps (FISTA) of
   `4 × rows / ‖columns with a bias column‖²`, a step the logistic loss's curvature can't make overshoot; each step
   shrinks every weight toward 0 by step × price and lands it on 0 when it would cross, so the terms that don't pay
   their price drop out. A fit stops after `max_steps`, or once no weight moves by more than `tolerance` × the largest of
   1 and the largest weight.
6. **Choice.** The fit with the lowest mean logistic loss on the held-out rows is kept, or on the training rows without
   held-out rows; ties go to the fewest terms. Its nonzero weights, converted back to their terms' own units, become the
   value rules, the largest standardized weight first.

## How value rules value

`RuleValuer.value(state)` gives each player, in the order of the players' names,
`low + (high - low) × logistic(bias + Σ weight × term)`, every term read with `me` being that player. The value stays
strictly between the lowest and highest payoffs, so a finished game's own payoffs always rank beyond it. A term raising
`KeyError`, `NameError` or `TypeError`, or giving something other than a finite number, leaves the position unvalued
(`None`), and a search plays that rollout to the end. `explain(state, player)` gives each rule with what it adds to the
player's score: its weight times its term.

## Usage

```python
from openmind.agent.factory.domain_factory import create_domain
from openmind.rbs.factory.rbs_factory import create_rule_generator, create_rule_rater
from openmind.rbs.model.generation_settings import GenerationSettings

domain = create_domain("tictactoe")
settings = GenerationSettings(
    min_visits=5, max_conditions=2, min_rule_visits=50, min_gain=0.05, confidence=0.95, beam_width=20, max_offset=2,
    solo_limit=2, patterns=200, false_discovery_rate=0.05, permutations=10_000,
)   # the defaults, in constant/generation_constant.py
result = create_rule_generator().generate(domain, discovery_samples, validation_samples, settings, seed=1)  # SearchResult.samples
rater = create_rule_rater(result.rule_base, domain)
rater.rate(state, actions)          # one rating per action
rater.explain(state, actions[0])    # the rule behind the first rating
```

Training does this from self-play: see `training/README.md` and `openmind-distill`.

Value rules:

```python
from openmind.rbs.factory.rbs_factory import create_rule_valuer, create_value_generator
from openmind.rbs.model.value_settings import ValueSettings

settings = ValueSettings(
    pair_pool=20, cuts=6, solo_limit=2, prices=(0.1, 0.03, 0.01, 0.003, 0.001), max_steps=1000, tolerance=1e-6
)   # the defaults, in constant/value_constant.py and constant/generation_constant.py
result = create_value_generator().generate(domain, training_rows, held_out_rows, settings)   # PositionRow
valuer = create_rule_valuer(result.value_base, domain)
valuer.value(state)               # each player's value, in the order of the players' names
valuer.explain(state, "X")        # each rule with what it adds to X's score
```

Training does this from self-play too: see `openmind-distill-values`.

## Logs

- `openmind.rbs.service.goal_pattern_miner`:
  - `INFO Found <n> goal patterns in <m> winning moves`
  - `DEBUG Goal pattern of <action> in <moves> moves: <condition> and ...`
- `openmind.rbs.service.hypothesis_discoverer`:
  - `INFO <action>: skipped <n> combinations a condition adds nothing to, shortened <p> goal patterns, merged <d> hypotheses matching the same rows`
- `openmind.rbs.service.rule_generator`:
  - `INFO <action>: <n> primitives, <m> hypotheses discovered from <rows> rows in <states> states`
  - `INFO Validated <k> of <m> hypotheses on <rows> rows in <states> states at a false discovery rate of <q>`
  - `INFO <action>: kept <r> of <v> validated rules; <c> covered by a simpler rule within min_gain`
  - `DEBUG Covered: <rule as text> by <rule as text>`
  - `DEBUG <tested hypothesis as text>`, one line per hypothesis, such as `place when win_chance(action) >= 1: raises advantage, discovery 0.61 over 40 states, validation 0.58 over 12 states, p 0.0001, q 0.002, validated`
  - `DEBUG <rule as text>`, one line per rule, such as `place when wins(me, action) >= 2: EV 0.9 over 120 visits, priority`

- `openmind.rbs.service.value_generator`:
  - `INFO <n> single terms generated, <u> usable on <t> training and <h> held-out rows`
  - `INFO Every training payoff is <payoff>: nothing to fit`
  - `INFO <c> candidate terms: <s> single, <p> products of pairs among the <k> single terms most correlated with the payoffs`
  - `INFO Price <price>: <k> of <c> terms kept in <steps> steps, settled|not settled; training loss <loss>, held-out loss <loss>`
  - `INFO Chose price <price>: <r> value rules, bias <bias>, payoffs from <low> to <high>`
  - `DEBUG <value rule as text>`, one line per rule, such as `+0.42 × wins(me)`

The rater and the valuer don't log: they run inside searches.

## Notes

- Tests: `builder/rule_generator_builder_tests.py`, `builder/rule_rater_builder_tests.py`, `factory/rbs_factory_tests.py`,
  `mapper/action_row_mapper_tests.py`, `mapper/hypothesis_text_mapper_tests.py`, `mapper/rule_base_json_mapper_tests.py`,
  `mapper/rule_text_mapper_tests.py`, `repository/rule_base_repository_tests.py`,
  `service/advantage_contrast_tests.py`, `service/condition_evaluator_tests.py`, `service/coverage_filter_tests.py`,
  `service/consequence_library_tests.py`, `service/goal_pattern_miner_tests.py`,
  `service/hypothesis_discoverer_tests.py`, `service/hypothesis_validator_tests.py`,
  `service/primitive_generator_tests.py`, `service/rule_generator_tests.py`, `service/rule_rater_tests.py`,
  `builder/rule_valuer_builder_tests.py`, `builder/value_generator_builder_tests.py`,
  `mapper/value_base_json_mapper_tests.py`, `mapper/value_rule_text_mapper_tests.py`,
  `repository/value_base_repository_tests.py`, `service/rule_valuer_tests.py`, `service/sparse_fitter_tests.py`,
  `service/term_evaluator_tests.py`, `service/term_generator_tests.py`, `service/value_generator_tests.py`;
  integration: `test/integration/tictactoe_distillation_tests.py`.
