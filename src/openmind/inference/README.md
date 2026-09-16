# inference

## Purpose

The inference engine: what a domain's own rules imply about a position, and a search for expressions of it that value
positions. Nothing here knows a game. An expression reads a position through a view that looks ahead with the domain's
actions, so a tactic such as "a piece attacking two pieces wins one, since the opponent has one move" is deduced by
playing the opponent's replies, not written in.

Expressions are Python, like every rule (see `rule/README.md`): a generated value rule is the source of an expression
reading `here`.

## Content

| File | What it is |
|---|---|
| `constant/inference_constant.py` | `ME`, `OTHER`, `OUTSIDE`, `HERE`, the template's `VIEW` placeholder, look-ahead kinds, pattern relations, combinations, thresholds, `MEMORY_CHECK_INTERVAL` (1,000), `SCREENING_SHARE` (0.1), `MIN_SCREENING_ROWS` (500), `CANDIDATE_BATCH` (200), and the defaults: `DEFAULT_SEARCH_SECONDS` (3,600), `DEFAULT_SEARCH_MEMORY` (half the machine's memory), `DEFAULT_PROCESS_MEMORY` (that half shared between the logical CPUs; both from `parallel/constant/parallel_constant.py`), `DEFAULT_DEDUCTION_SECONDS` (10) and `DEFAULT_HIGHEST_PAYOFF` (1.0) |
| `model/deduction_budget.py` | `DeductionBudget(plies, seconds, highest=1.0)`: how far and how long one position is reasoned about, and the highest payoff a player can get |
| `model/deduction.py` | `Deduction(state, player, action, payoffs, line, plies)`: what reasoning about a position proved for the player to act, and `proven` |
| `service/position_deducer.py` | `PositionDeducer.deduce(domain, state, budget)`: proves a position's best action from the domain's own rules alone, one ply deeper at a time |
| `service/deduction_inducer.py` | `DeductionInducer.seeds(domain, deduction, vocabulary, highest)`: candidate expressions scoped to what a proven deduction touched |
| `model/expression.py` | `Expression(template, clauses, plies, pattern=None)`: Python source reading its position where `{view}` stands, what the fit prices, how many actions it looks ahead, and its pattern when it counts one |
| `model/aggregate.py` | `Aggregate(base, pair, kind, body, body_clauses, body_plies=0, readings=(), operations=())`; `readings` and `operations` are the body's own parts, in the order they were added, so `body` is their fold and a candidate can be read from shared readings; a body built another way, such as one comparing a body with itself at `j`, records none: a body read at every index `i`, or every pair of different indices `i` and `j`, of a base, then counted where it holds, summed, or taken at its lowest or highest; `body_plies`, how many actions the body looks ahead |
| `model/pattern.py` | `Pattern(anchor, conditions)`: conditions around an index, counted over every index of the anchor base |
| `model/pattern_condition.py` | `PatternCondition(base, steps, relation, value=None, other_condition=None)`: a base read at the index shifted by steps, `==` or `!=` a value or another condition's variable |
| `model/vocabulary.py` | `Vocabulary(players, to_act, values_by_variable, values_by_base, indices_by_base, offsets_by_arity)` |
| `model/search_budget.py` | `SearchBudget(seconds, memory_bytes, candidates=None)`: how long a search runs, how many bytes its process holds, and how many candidates it tries, `None` for no limit |
| `model/expression_search_result.py` | `ExpressionSearchResult(expressions, training, held_out, generations, stopped, tried)` |
| `constant/sentence_constant.py` | The words and templates of literal readings: players, places and entries, comparisons, arithmetic, look-aheads, aggregates, patterns |
| `mapper/expression_sentence_mapper.py` | `ExpressionSentenceMapper.to_sentence(source)`: an expression's Python source read literally in English, construct by construct; a construct without a template is quoted as its source |
| `service/mechanics.py` | `Mechanics`: views of positions and the outcomes of any player's actions, with the domain's solver and predictor; `limit_memory(bytes)` and `clear()` |
| `service/position_view.py` | `PositionView`: a position as expressions read it |
| `model/relaxation.py` | `Relaxation(name, dropped=(), widened=(), passing=False)`: fewer rules, to see what a game teaches without them: constraints dropped by action and position, parameters given wider values by a rule, and whether every player also has a `pass` action |
| `service/domain_relaxer.py` | `DomainRelaxer(rule_caller)`: `relaxations(domain)`, the relaxations the rules allow (each constraint of each action dropped, and passing where one player acts at a time); `relax(domain, relaxation)`, a relaxed copy, the domain given left as it is. `pass` takes the constraints that read no parameter, so a player passes while the game goes on and not once it is over, and its effects hand the turn to the next player |
| `service/expression_generator.py` | `ExpressionGenerator`: the vocabulary, leaves, pattern children, thresholds, combinations and look-aheads; `pattern_expression(pattern, vocabulary)`, the expression counting a pattern |
| `service/expression_search.py` | `ExpressionSearch`: grows expressions generation by generation within a budget, trying given seeds before the leaves |

`rbs` wires these in: `ConsequenceLibrary.names` gives rules `here`, and `ValueGenerator` fits the expressions the
search keeps (see `rbs/README.md`).

## What a view reads

- Its variables, as attributes gathered the way rules see them: `here.turn`, `here.cell[2, 3]`, `here.payoff[me]`.
- `offset(base, at, *steps)`: the variable of `base` at index `at` shifted by the steps, or `OUTSIDE`.
- `moves(player)`: for each action `player` could take if it were their turn, its outcomes as `(view, probability)`.
- `mobility(player)`: how many such actions there are.
- `changed(player, base, at)`: how many of those actions change the variable of `base` at index `at`, each outcome
  weighted by its probability. The mechanics work out every variable's changes once per position and player, from the
  moves, so reading every cell of a grid costs one pass over the moves (`Mechanics.changes`).
- What if, each the view of an edited copy, the player to act and every other variable left as they are:
  `with_value(base, at, value)`, one variable set (a variable the position doesn't have raises `KeyError`);
  `cleared(at)`, every grid's cell at `at` set to the grid's empty value, the value most of its cells hold in the
  domain's initial position (`Mechanics.empties`); `copied(source, target)`, every grid's value at `source` also placed
  at `target`; `alone(at)`, every grid emptied except at `at`. In chess, `here.with_value('color', at, other).changed(me,
  'color', at)` counts the defenders of my piece on `at`, `here.alone(at).mobility(me)` its moves on an empty board, and
  `here.copied(j, at).changed(me, 'color', at)` whether I attack an empty square `at`, pawns' diagonals included.
- `best(player, reading)` and `worst(player, reading)`: the highest and lowest reading expected after one of those
  actions; the reading here when the player has none.
- `count(player, reading)`: how many of those actions the reading is expected to hold after.

A reading is a function of the view after the action, usually a lambda: `here.worst(other, lambda v2: v2.best(me,
lambda v1: v1.payoff[me] == 1.0))` is "whatever the other player does, I can win next". A look-ahead's result is kept on
its view by the reading's code, the views it closes over, and the `me`, `other` and `here` it reads. Views and moves
are kept per process. Every 1,000 entries remembered the process's memory is read, and they are all cleared when it
holds more than its share: `DEFAULT_PROCESS_MEMORY` until a search sets it, then the search's memory budget shared
evenly between the evaluator's workers, a share copies sent to workers carry. The process's memory guard clears them
too (see `parallel/README.md`); the memory is read with `parallel`'s `MemoryMeter`.

A search whose term evaluations take a worker over its memory cap in a fresh worker too stops with "the memory budget
ran out", keeping what it found.

## How expressions are generated

From the rows' positions and the domain's players only. A player's name, as a value or as an index, is written relative
to the player valued: `me` or `other`. Nothing assumes a board: a value is either a name (a mark, a piece, a player,
`None`) or a number (a position, a clock, a count), and a variable or base is *numeric* when every value seen is a
number. A number is never compared for equality with a value seen; it is read as it is, and thresholds, arithmetic
and aggregates build on it.

1. **Leaves:**
   - a numeric variable as it is: `{view}.x[me]`, `{view}.halfmove`;
   - every other variable at every value seen: `{view}.cell[2, 2] == me`;
   - the player to act, also absolutely: `{view}.turn == 'X'`;
   - for every indexed base of names and every value, how many indices hold it, a pattern of one condition:
     `sum(1 for at in {view}.cell if {view}.cell[at] == me)`;
   - for every numeric indexed base, the sum, the lowest and the highest of its values, aggregates of one reading:
     `min((({view}.x[i]) for i in {view}.x), default=0)`;
   - for every grid of names (a base whose indices are whole numbers, a `Grid`) and every value, how many cells hold it
     and are of odd parity, and how many hold it and some action of `me`, or of `other`, changes:
     `sum(1 for i in {view}.cell if (({view}.cell[i] == None) and ({view}.changed(me, 'cell', i))))`,
     one action ahead;
   - `{view}.mobility(me)` and `{view}.mobility(other)`.
2. **Pattern children,** for grids, where indices are whole numbers: one more condition, on any base read at the
   pattern's index shifted by any offset seen between two indices of one base (the same index included): `==` or `!=`
   any name that base takes, `OUTSIDE`, or the variable another condition reads. A base with other indices than the
   anchor's is read with `offset`. Patterns grow without a size limit: `sum(1 for at in {view}.cell if {view}.cell[at] ==
   me and {view}.offset('cell', at, 0, 1) == me and {view}.offset('cell', at, 0, 2) == me)`.
3. **Aggregate children,** for any indices, whole numbers or not: the body grows by an operation (`+ - * / abs >= <= ==
   and or`) with a reading of any base sharing the aggregate's indices, at `i`, or also at `j` once it reads pairs
   (a numeric base as it is, a base of names equal to each name, or equal at `i` and `j`); a body read at `i` also turns
   into its difference, distance, order and equality at `i` and at `j`, over every pair of different indices. Every new
   body is counted, summed, and taken at its lowest and highest: `min(((abs(({view}.x[i]) - ({view}.x[j]))) for i in
   {view}.x for j in {view}.x if i != j), default=0)` is the smallest gap between two entities. Over a grid, the readings
   also ask the grid by coordinates: at `i` (and `j`), the cell's parity, how many of `me`'s or `other`'s actions change
   it (one action ahead), how many neighbours hold each name, and whether a ray in each direction (every coordinate
   stepping by -1, 0 or 1) meets each name; over pairs, the distance and alignment of `i` and `j`, and whether each name
   lies between them. What if, one action ahead: the moves `me` or `other` would have alone at `i`; for a grid whose
   values name players, how many of `me`'s moves would change `i` were it `other`'s, and the other way round; and over
   pairs, how many of `me`'s or `other`'s moves would change `i` with `j`'s values copied there.
4. **Thresholds:** at least each value an expression takes above its lowest, and at most each below its highest; and
   the expression's absolute value.
5. **Combinations** of two expressions: `+`, `-`, `*`, `/ max(1, ·)`, `max`, `min`, `>=`, `==`; their clauses add up.
6. **Look-aheads,** for `me` and for `other`: `best` and `worst` of the expression and of its change
   (`(E(v)) - (E(here))`), `count` of the actions after which it holds, raises it or lowers it. The lambda variable is
   numbered by the plies of its body (`v1`, `v2`, …), so nesting never shadows a variable an inner body reads.

A child has one clause more than its parent, a combination the clauses of both.

## How the search runs

`ExpressionSearch.search(domain, training, held_out, targets, price, max_steps, tolerance, budget)`, targets being the
training payoffs scaled from 0 to 1. The screening rows are every row up to 500 rows, otherwise a tenth of them, evenly
spread, 500 at least. Each generation:

1. The kept expressions are fitted at the price, each weight priced by its clauses times the share of rows where its
   expression isn't blank (`SparseFitter` with costs), and the residual, prediction minus payoff, is taken on the
   training rows.
2. From the second generation, every kept expression is expanded, the weighted first, by weight then gradient: its
   look-aheads, absolute value, thresholds, pattern children and aggregate children once, and its combinations with each
   kept expression it hasn't met. Then the expressions tried since the last expansion that varied without being kept
   are expanded too, the steepest first, into their look-aheads, pattern children and aggregate children: a relation
   can matter when its parts don't, such as the gap between two positions when neither position does. Parents take
   turns, one candidate each, and a parent makes its next candidate only when it is asked for one, so a generation's
   candidates are never all in memory, however many there are.
3. Candidates not tried before are tried in batches of 200 or the candidates left, whichever is fewer; the time, the
   candidates and the memory left are checked before each batch. A candidate is
   evaluated on the screening rows; it goes on when its gradient, the mean of its standardized values times the
   residual, is above price × clauses × the share of rows where it isn't blank, that is when the fit would give it a
   weight. It is kept when, on every training row, it still is, it varies, its values are finite where not blank, and no
   kept expression has the same values. Thresholds and combinations are computed from their parents' kept values, a row
   blank in a parent staying blank; an aggregate whose body recorded its parts is folded from its readings, each read
   once per position and reused by every candidate that reads it (`ReadingCache`, see `rbs/README.md`), while an
   aggregate over pairs of indices and a body built another way keep their source; the others are evaluated as rules by `TermEvaluator`, in its workers. A candidate
   raising an error or giving something other than a finite number or `None` on a row is dropped.
4. When the kept values and the fit's two copies of them would pass the memory budget (8 bytes per row per expression,
   three times), the expressions without weight and with the smallest gradients are evicted first.

**Blanks.** A term giving `None` on a row is blank there: what it reads isn't there at that moment, as a fork detector
without a fork. `TermEvaluator` gives NaN on that row. `ExpressionSearch.scaling(column)` gives the center and scale the
fit reads a column with: its mean and standard deviation without blanks; with blanks, 0 and the root mean square of its
values, uncentered, so a blank reads as 0 and adds nothing, and a detector always worth the same when it fires still
tells its rows apart (centering it would make it a constant). `standard(column, scaling)` applies it, and
`share(column)` is the share of rows that aren't blank: a rare term is priced for the rows where it speaks.

Given several targets by name (`targets` a mapping instead of one array), such as the targets of several signals, each
generation fits every target, a candidate goes on and is kept when its gradient against any target's residual passes
its price, and expansion and eviction go by an expression's largest weight and steepest gradient over the targets:
an expression is kept when any target supports it. One target given as an array is named `SINGLE_TARGET` (`"target"`)
and searches as before; no target raises `ValueError`. The generation log then gives each target's training loss,
`<name>=<loss> ...`.

Memory is measured, not estimated. Before each batch, when this process holds more than the memory budget, the search
clears the views, forgets the expressions passed over and evicts every unweighted column; if the process still holds
more, it stops. Tried candidates are remembered as 8-byte digests of their sources.

The search stops when the time, memory or candidate budget runs out, or when a generation has nothing left to try. A
generation that keeps nothing doesn't stop it: the next generation expands the expressions it passed over, so a
relation, pattern or look-ahead can be found when no leaf pays its price alone. The budget is the only limit on what it
looks at: how many leaves, conditions, clauses or actions ahead. A search limited by time depends on the machine's
speed; one limited by candidates doesn't.

## Deduction and induction on one position

The search above looks at every position at once; it is fast when rules exist, and blind when none do. Deduction works
on one position instead, where the question is concrete: why does a move win here?

**Deduction.** `PositionDeducer.deduce(domain, state, budget)` reasons with nothing but the domain's own rules: legal
actions, outcomes and the payoffs of finished games. It deepens one ply at a time, up to `budget.plies`, within
`budget.seconds`:

- a finished game is proven: its payoffs;
- an action is proven when every outcome with a chance above 0 is, and is worth their payoffs weighted by their chances;
- a position in play is proven when every legal action is, the player to act taking the best for them (ties go to the
  solver's first), or as soon as one proven action gives that player `budget.highest`, which no other action can beat.

Nothing is estimated: a position whose lines don't end within the plies stays unproven. The first depth that proves the
position gives the `Deduction`: the best action, each player's payoffs, and the line, each action with the state it led
to (the likeliest outcome). A budget of fewer than 1 ply or no seconds, or a position without a legal action, raises
`ValueError`. In chess, every ply multiplies the work by about 30.

**Induction.** `DeductionInducer.seeds(domain, deduction, vocabulary, highest)` turns a proven deduction into candidate
expressions scoped to what it touched, a player written as `me` for the deduction's player or `other` for the next one:

1. **The proof as a look-ahead:** whether the player the proof pays `highest` gets it, read along the line, each ply the
   best over that player's actions and the worst over the other's. A win within 3 plies of mine is `{view}.best(me,
   lambda v3: v3.worst(other, lambda v2: v2.best(me, lambda v1: v1.payoff[me] == 1.0)))`; a loss within 2 is
   `{view}.worst(me, lambda v2: v2.best(other, lambda v1: v1.payoff[other] == 1.0))`. It is built with the generator's own
   look-aheads, so the search reads it like any other.
2. **What the first action changed, as a pattern:** every variable with whole-number indices it changed, as it was
   before, at its offset from the first one: a rook of mine and an empty square seven ranks up, counted wherever they
   hold together.

A look-ahead or pattern that can't be written this way, such as a proof paying a third player, gives no seed. The
search tries seeds before the leaves (`search(..., seeds)`), and training's `PositionPonderer` feeds it the seeds of the
positions the rules missed most (see `training/README.md`); an agent can fall back on deduction when its rules have no
clue (see `agent/README.md`).

## How an expression is read

`ExpressionSentenceMapper` parses the source and reads each construct with its template, variables keeping their
own names, so it works for any domain:

| Expression | Literal reading |
|---|---|
| `here.best(other, lambda v2: v2.worst(me, lambda v1: (sum(1 for at in v1.color if v1.color[at] == other)) - (sum(1 for at in v2.color if v2.color[at] == other))))` | the highest, over the opponent's moves, of the lowest, over my moves, of the change in the number of places where the color there is the opponent's |
| `sum(1 for at in here.piece if here.piece[at] == 'knight' and here.color[at] == me)` | the number of places where the piece there is knight and the color there is mine |
| `min(((abs((here.x[i]) - (here.x[j]))) for i in here.x for j in here.x if i != j), default=0)` | the lowest, over every pair of different entries, of the distance between the x of one entry and the x of the other entry |
| `here.count(me, lambda v1: (v1.mobility(other)) < (here.mobility(other)))` | the number of my moves after which the number of moves the opponent could make is less than the number of moves the opponent could make now |

A reading on the position the innermost look-ahead reaches needs no words; one on an earlier position says which
(`now`, `after the opponent's move`). A subtraction of the same reading on two positions is a change. `rbs`'s rule
explainer gives these readings to a language model, which turns them into plain sentences (see `rbs/README.md`).

## Usage

```python
from openmind.inference.model.search_budget import SearchBudget
from openmind.rbs.factory.rbs_factory import create_value_generator
from openmind.rbs.model.value_settings import ValueSettings

settings = ValueSettings(
    prices=(0.1, 0.01, 0.001), max_steps=1000, tolerance=1e-6, seconds=600.0, memory_bytes=8 * 1024**3, candidates=None
)
result = create_value_generator(workers=8).generate(domain, training_rows, held_out_rows, settings)
for rule in result.value_base.rules:
    print(rule.weight, rule.term.source)   # e.g. here.worst(other, lambda v2: v2.best(me, lambda v1: v1.payoff[me] == 1.0))
```

## Logs

- `openmind.inference.service.expression_search`:
  - `INFO Searching expressions for <seconds> seconds within <bytes> bytes, trying any number of|at most <n> candidates: <k> seeds, <l> leaves, <t> training rows, <s> screened`
  - `INFO This process held <bytes> bytes, over the memory budget of <budget>: cleared the views, forgot <p> expressions passed over and evicted <e> unweighted columns; it now holds <bytes> bytes`
  - `INFO Generation <g>: <tried> candidates tried, <kept> kept, <evicted> evicted; <n> expressions kept, <w> weighted at price <price>, looking up to <plies> actions ahead; training loss <loss> before the generation; <total> candidates tried in all; <seconds> seconds left; <bytes> bytes held`
  - `DEBUG Kept <source>`, one line per expression kept
  - `INFO Search stopped after <g> generations and <total> candidates: <reason>; <n> expressions kept`

- `openmind.inference.service.position_deducer`:
  - `INFO Deduced <action> for <player> within <plies> plies: payoffs <player>=<payoff> ..., along <action> > <action> ...`
  - `DEBUG Nothing proven for <player> within <plies> plies`, with `: the <seconds> seconds ran out` when they did

Mechanics, views and the inducer don't log.

## Notes

- `inference` and `rbs` depend on each other by design: the search evaluates with `rbs`'s term evaluator and fits with
  its sparse fitter, and `rbs`'s consequence library gives rules the mechanics' view.
- A look-ahead evaluates every action of a player, and every action after each of those: in chess, about 30 actions a
  ply. Deep expressions are slow to evaluate, and the time budget decides how many get tried.
- Tests: `mapper/expression_sentence_mapper_tests.py`, `service/mechanics_tests.py`, `service/position_view_tests.py`,
  `service/expression_generator_tests.py`,
  `service/expression_search_tests.py`, `service/position_deducer_tests.py`, `service/deduction_inducer_tests.py`;
  integration: `test/integration/tictactoe_inference_tests.py`; in OpenMindChess, `test/integration/chess_inference_tests.py`
  and `test/integration/chess_deduction_tests.py`.
