# Open questions

These are gaps, contradictions and unknowns in `doc/architecture.md`. Each one waits for Maxime's decision. None is
settled until it is struck from this file and the design says so.

**A question is struck once it is decided, and its number is never reused.** What was decided lives in the code it
shaped — the docstrings of this project carry their reasons — and in this file's git history, where each entry sits
in the commit that struck it. The index at the foot keeps the numbering legible, since docstrings and plans cite
questions by number.

## Still open

**A standing ruling, because three separate services broke it and each read as reasonable while doing so.**
*A thing is never valued by what its rules admit.* Counting how much a piece can do and calling the number its
worth is the answer being supplied and then found. OMF is told to look for heuristics; that a piece has a value
is a heuristic it finds in what games paid, or does not have. What was removed under this ruling, and what each
said of itself:

- `WorthReasoner` — mobility as worth, wired into pondering and seeding the search with weighted terms. Its own
  premise check printed *"resting on nothing, since no position bore out that doing less is worse"* while it
  produced `rook at 4.29412`.
- `HeuristicDeriver` — every public method of it: `derive` gave each thing *"the value its own rules admit"*,
  `afforded` and `allowed()` counted mobility against the constraints and against the game, `holdings` and
  `seeds` carried the result into the search as starting weights.
- `Inferrer` and `RuleReasoner.reaching` — *"a rook's rules reach fourteen squares, so a rook is worth at least
  fourteen"*, said outright and emitted as a fact's value. Never wired to anything, which is why it survived.

`RuleReasoner.entails` stays: whether one rule allows everything another allows is logic, and true. What does
not follow is that allowing more is being worth more, and that step is where each of these went wrong.


26. **How a rule comes to see through what is in the way.** (found while splitting legality into generators and
    constraints)
    What a generator proposes and a constraint then refuses *because something stands in the way* is worth keeping:
    that one thing bears on a square through another is a battery where the two are aligned and friendly, a pin
    where what stands between cannot move without exposing what is behind it, and a discovered attack where it
    moves anyway. None of that is chess — it is what any game of lines and blocking has — and it only exists to be
    had if some rule proposes the blocked move in the first place.
    The wrong answer, tried and reverted: name the readings a generator may consult, so that it cannot see what
    occupies the way. Every name in such a list is a grid concept, so a game without a grid gets a generator that
    may read nothing; and telling the engine which readings describe how a thing moves is doing its finding for it,
    which is the same mistake as valuing a thing by what its rules admit and calling the answer a discovery.
    What makes it hard is that neither objective asks for it. A generator that reads what is in the way misses
    nothing — the blocked move really is illegal — so "never miss" does not force x-raying. Nor does cost: a rule
    that consults occupancy is not longer than one that does not. The reason to x-ray is that the blocked
    proposals are knowledge worth having, which is a wish about what OMF can state, not a fact about legality.
    - **Options:**
      - Ask the question of the rules afterwards instead of building it into them: put the learned rules to a
        position with nothing in the way and see what they propose. That is relaxing the game, which `GameRelaxer`
        already makes a context, and it needs no vocabulary partition and no declaration at all.
      - Learn two generators against two targets — what is legal, and what would be legal were the board empty —
        and let the difference be the x-ray facts. Honest, and it doubles the learning.
      - Leave it. Legality converges, and batteries, pins and discovered attacks are never available to reason
        with.
    - **Undecided.**


28. **A consequence says when it happens and nothing asks.** (found joining the predictor to the constraints so
    that "the king must not be left attacked" could be written)
    `Consequence.when` is learned — the conditions under which a change happens, by the machinery that learns
    what a game refuses — and `ConsequenceDrawer.drawn` never reads it. It draws a change wherever the change's
    places resolve, so a capture is drawn for every move whose landing square is on the board, a castling's rook
    would move on every move, and a clock is reset whether or not the condition holds. Nothing has noticed
    because nothing drew consequences until now: the drawer's only caller was its own tests.
    It is not a small fix. Answering `when` means putting a clause to a case, which is `RefusalLearner.covers` —
    and the refusal learner holds the hypothetical, which holds the drawer, so the dependency would close a ring.
    The pattern for breaking it is already in `_could_take`, which hands the hypothetical a way to ask rather
    than a thing to hold.
    It is also not urgent in the way it looks. For the board a move leads to, a removal drawn at the landing
    square removes a piece where there is one and nothing where there is not — so with the changes made in the
    order the game made them, an unconditional capture draws the right board anyway. What it gets wrong is
    everything else: rights cleared that were not, clocks counted that should have reset, a castling's rook
    moving on a move that is not a castling.
    - **Options:**
      - Give the drawer a `holds(clause, case)` callable, as `_could_take` gives the hypothetical a `refuses`.
        Nothing is held, the ring does not close, and a caller with no learner draws unconditionally as now.
      - Draw every consequence and let the caller filter, which puts the same question one level up and needs
        the same callable.
      - Leave it, and say in `Consequence.when`'s docstring that it is learned for reading and not for drawing.
        Honest, but it makes a learned thing decorative.
    - **Undecided.** Blocking nothing today; king safety works without it, for the reason above.


31. **A player's side is deduced, and the learner that most needs it cannot see it.** (found striking question 24,
    which turned out to have been answered in code a fortnight before it was read)
    `SideDeducer` works out which way each player faces from which way their one-way pieces go, and `Sides.toward`
    gives nought where a game has no side to tell — so a game whose pieces all move both ways has nothing deduced
    and nothing offered. That is question 24 answered, and answered the way Maxime would answer it.
    **But two vocabularies grew apart and only one of them got it.** `ActionReadings` takes `sides` and reads how
    far a move goes forward and which rank a square stands on counted from the player's own end; it feeds
    `RuleDeducer` and the extraction scripts (`WorthReasoner` is gone, under the ruling above). `CandidateReadings` takes no `sides` at all — the
    word does not appear in the file — and it is what `RefusalLearner` and `learn_constraints.py` use. So the
    constraint learner, which is the live loop and the thing whose pawn rules are split by colour, is the one
    learner that cannot say "forward". The deduction is available to deductions and not to constraints, which is
    half of what it was for.
    **And a side is a row step, not a vector.** `Sides.facing` is `tuple[tuple[Value, int], ...]` — one integer
    per player, the step along the row. It says everything chess needs, because a pawn walks up a file. It cannot
    say a forward that runs along a column, or diagonally, or on a board that is not squares — and a vector is
    what the fact actually is.
    - **Options:**
      - Give `CandidateReadings` the `sides` that `ActionReadings` already takes, and offer the same readings
        from both. The deduction exists, the readings exist, and this is plumbing — but the two files have
        diverged far enough that "the same readings" is a claim to check rather than assume.
      - Fold the two into one vocabulary. Bigger, and the right shape if they are meant to be one thing, which is
        a question nobody has asked out loud.
      - Widen `facing` from a row step to a vector, separately from the wiring. Nothing in chess needs it, so it
        would be built against a game we do not have — which this project's own practice says not to do.
      - Leave it. The constraint learner keeps learning pawn rules twice, and what question 24 complained about
        stays true where it was complained about.
    - **Decided (Maxime): fold them into a single vocabulary.** What that leaves open is stated below, because
      mapping the two turned up more than plumbing.
    - **What the fold has to settle.** They are two complete parallel stacks, both live:
      `ActionReadings` → `ReadingLiterals` → `ClauseLearner`, used by `dig_and_learn` and `explore_and_learn`;
      and `CandidateReadings` → `RefusalLearner`, used by `learn_constraints`, the scout and the worker. About
      five ideas are said twice — rows and columns apart against `places apart`, `{model} at {parameter}`
      against `holds`, `things between` against `on line`, share-a-row-or-column and on-a-diagonal against what
      `_along` derives, row and column against the place decomposition.
      - **Literal-native, one service. Chosen, and its first half is done** — see below. `ActionReadings` stops
        filling templates into strings and emits literals; `ReadingLiterals` goes, having existed only to undo
        that flattening; the six candidate predicates move in. The rework doc already calls the flattening the
        fault — it *"uses the filled string as a key, which throws away the very thing that makes it
        general"* — so not flattening is that same decision one step earlier.
      - **Keep the bridge, merge behind it.** `CandidateReadings` becomes more `ActionReadings` templates and
        `ReadingLiterals` stays the only road to literals. Smaller, and it preserves a fault the rework already
        logged: three templates are the same shape once filled, so their readings land under one name and their
        evidence is pooled.
    - **Two things the fold can silently lose, and both are load-bearing.**
      - **The whole-structure read.** `CandidateReadings.of_state` reads every cell of every grid;
        `ActionReadings` reads grids only where the parameters point. Reading only the named cells *"cannot
        work"* in its own words — nothing in two cells tells a blocked way from a clear one, so blocking is not
        hard to find there, it is absent.
      - **`tied`.** It is what bounds the search: a case carries about a hundred and fifty readings and `tied`
        offers about two dozen, which is C(23,4) against C(151,4). The distinction is between what a rule may be
        *built* from and what it may *consult*, and `ActionReadings` has no notion of it. A merged vocabulary
        that drops it makes the constraint learner intractable rather than merely slower.
    - **Step one, done: the readings are said outright and the bridge is gone.** `ActionReadings.read` is now
      the one computation and returns `Reading`s that hold their slots apart from their template; `literals`
      and `of` are two presentations of it, and `of` is marked as going when the stand-in engine does.
      `RuleDeducer` gained the same split. `ReadingLiterals` is deleted, with its regexes, its type-guessing
      and its order-dependent template tuple.
      **What the bridge was costing, counted rather than asserted.** Over three fixtures, of 693 literals
      **225 were unchanged and 468 changed — and every change is a reach.** Four faults, one per template:

      ```
      the reading                                     the bridge made of it                                        said outright
      black can reach color 'black'                   can reach owned('black','color','black')                     can reach a thing('black','color','black')
      black can reach color 'black' and piece 'pawn'  can reach owned('black','color',"black' and piece 'pawn")    can reach holding('black',"color 'black' and piece 'pawn'")
      black can reach another player's piece 'pawn'   can reach owned('black','another',"player's piece 'pawn'")   can reach owned('black',"another player's","piece 'pawn'")
      black can reach source                          can reach holding('black','source')                          can reach('black','source')
      ```

      The second splits a quoted string mid-quote. The third splits the role `another player's` at its space,
      so `another` becomes the whose. The fourth is the king-safety condition, named as though the square the
      action points at were a thing standing on one — a clause tying that argument to another reading's holding
      would have tied a square to a piece. Nothing outside the reach family moved.
      A value also keeps the type it was read as, where the bridge parsed it back out of a name and made every
      character that looked like a number into one.
      `test_readings_named_the_same_shape_still_come_apart_into_their_own_terms` was deleted rather than
      carried: its own docstring said which template a name came from *"cannot be recovered, and it is not
      guessed at"*, so it pinned the stand-in. What replaces it asserts four predicates and four arities.
    - **Step two, not started:** merge the two vocabularies, keeping `tied` and the whole-structure read.
    - **What it buys, which is why it was asked for.** `row of {parameter}, from the player's own side` and
      `rows from {first} to {second}, forward` arrive in the constraint learner's vocabulary, and the pawn rules
      stop being learned twice.
    - **And it buys being attacked, which is the larger half and was found later.** `ActionReadings` already
      carries `{player} can reach {parameter}`, whose own comment says what it is: *"A square holding a piece is
      reached by a move that takes it, so this is what being attacked is: the others have a move that would
      capture it."* With `sides` it reads by role — `another player can reach the one acting's king` names no
      colour and no square. That is the king-safety condition, induced from the game's own allowed actions.
      **The constraint learner could not see it, so king safety was hand-composed instead** — `Hypothetical.taken`
      asks the same question by holding the predictor and one ply of lookahead. One relation, arrived at twice,
      because the two vocabularies were apart. This is the clearest case the fold has.

32. **Induction should coin shorthands, and nothing asks for one.** (raised by Maxime while reading question 27)
    A relation worth a name — being attacked, a fork, a pin — is a body of conditions that recurs. Induction
    ought to coin it, and whatever needs it ought to read it: the constraints should discover that they need
    "the king is attacked" because they cannot otherwise account for what the game refuses; the heuristics
    should later discover that a thing reaching two of another's at once is worth a name of its own.
    **Three quarters of the machinery is built and none of it is joined up.**
    - *Coining a name for what a family shares* is `CoveringLearner.principles` — "the rules restated as
      principles, with what specializes each one under it", with tests for a family within a family. It names
      what a rule set shares, not a predicate other services may ask for.
    - *Storing a body once and pointing at it* is `RuleLibrary`: `put`, `told`, `shared`, and a `cost` of a
      charge per rule plus a charge per pointer — which is exactly the price of a shorthand against saying it
      out each time. Built, tested, **wired to nothing**.
    - *Being attacked* is already a reading. `ActionReadings` carries `{player} can reach {parameter}` and
      `{player} can reach {whose} {holding}`, induced from the allowed actions the way Maxime describes it: a
      move that lands where a thing stands takes it. Question 31 is why the constraint learner cannot see it.
    - *Whoever needs it reads it* is the architecture already agreed — the engine puts rules in the knowledge
      base and a rule-based system retrieves those its context calls for. Nothing about this idea cuts across
      that.
    **What is genuinely absent is two things, and the first is the hard one.**
    - **Asking for a concept one does not have.** Every reading today is offered unconditionally to everybody.
      Nothing anywhere notices that it cannot account for something and asks for a new predicate as a result.
      That is predicate invention, which is the open problem of its field rather than a gap in this codebase —
      and the residue this project now computes is exactly the signal that would drive it.
    - **Counting over a relation, without which a fork cannot be said.** `{player} can reach {holding}` is a
      yes or a no. A fork is a *two* — this thing reaches two of another's at once — and no reading counts how
      many of a kind a thing reaches. A shorthand language that cannot count cannot reach the second of
      Maxime's two examples, and the first needs no new language at all.
    - **Options:** undrafted. The fold of question 31 comes first either way, since it decides the vocabulary a
      shorthand would be coined in, and `RuleLibrary` is where one would be kept.
    - **Undecided.**


33. **Most of what the side deduction concludes is wrong, and every conclusion is stated the same way.** (raised
    by Maxime on reading the sides/no-sides A/B: *"It's likely that most of these are poor hypothesis. A few
    might be good and it might be hard to tell which is best"*)
    `SideDeducer` concludes about fifty ownership claims per chess position. Named in the log rather than
    counted, they fall into four kinds, and nothing in the output distinguishes them:
    - **Right.** `black owns, of what 'holds x self column grid' reads: piece(black, pawn) seen 12,
      piece(black, bishop) 2, knight 2, rook 2, king 1, queen 1.` Every value is one of that player's pieces.
    - **Backwards.** `white owns, of what 'holds y self column grid' reads: piece(black, pawn) seen 1,
      piece(black, queen) seen 1.` White is given black's pieces, on one sighting each: some square elsewhere on
      the board happened to hold a black queen while only white was on the move.
    - **Nonsense.** `black owns, of what 'holds x self column square colour' reads: dark seen 10, light seen 10.`
      Both players play over the same light and dark squares; neither owns either, and here one player owns both.
    - **Circular.** `black owns, of what 'belongs self column x square colour' reads: another player seen 20`,
      six such lines. `belongs` is the reading `CandidateReadings` builds *from the previous deduction*, so the
      deducer is concluding ownership of its own prior output. That loop is a fault on its own, separate from
      whether any individual claim is sound.
    The criterion — a value only one player was ever seen moving — is satisfied identically by all four. It is
    also the only thing being asked. Sighting counts now travel with each claim in the log, which separates
    *thin* from *firm* but not *right* from *wrong*: the backwards claims are thin, but so is a correct
    `piece(black, queen) seen 1`, and the nonsense claims are the thickest of the lot at 10 and 20.
    **The knowledge base cannot tell them apart either, which is the sharper half.** `Fact.from_rules` and
    `Fact.from_facts` are the framework's own answer to "how sure is this and why" — the docstring says so —
    and `SideDeducer.facts` leaves both empty, so all fifty enter resting on "the rules themselves". That is
    not a display problem.
    **One judge has already ruled, on one path.** Over 28 positions, the sides arm of the A/B priced out 1164
    candidate readings against the baseline's 262 and kept **none** carrying `belongs` or `toward`. So for
    clause learning, the price has answered: none of these were worth having. Both arms meanwhile found
    ownership without an ownership reading at all, by sharing a variable —
    `refused :- turn(X2), lands on(self, x, y, grid, piece(X2, X3))`. The seed path of the heuristics plan is
    where the question is still live, because there the terms are Python over grids and unification is not
    available to say "mine".
    - **Options:** undrafted, and the shape of the answer is the question. Candidates visible in the evidence,
      none reasoned through: whether a *reading* rather than a value is the thing being judged, since right and
      wrong divide almost perfectly by reading; whether a reading whose claims partition the players cleanly is
      a better reading than one giving the same value to several, which `Information.told` is already the
      measure for; whether ownership should be concluded at all before something needs it, rather than deduced
      eagerly and offered to everybody; and what an `owns` fact ought to rest on, given that today it rests on
      nothing.
    - **Undecided.**


34. **The dashboard's front page watches a training that no longer exists.** (found restarting the runs on
    cinamon after the induced game went in; postponed by Maxime)
    It watches twelve things and five of them were deleted in the 2026-09-18 rework, including the three it
    depends on most: `entrypoint/train_values`, which is how it recognises a training process at all;
    `training/service/value_training_loop`, which is where it reads the round from; and
    `evaluation/service/match_runner`. With three OMF processes running on the machine the page reports
    **"Training: not running, Workers: 0, No training process"**, and the newest thing on it is an earlyoom kill
    from four days earlier.
    The deeper reason is not the deleted modules. The whole progress section is built on **rounds** — *round N
    of M*, games this round, matches this round — and rounds were abandoned: *"no more rounds: games run
    continuously"*. There is no round line in any current log, so the section never appears even where the
    modules are found.
    The pages that read the knowledge base rather than the logs are unaffected: `/games` and `/game/<id>` fill
    as games are played, and now show each side's heuristic with its rules.
    **This is the same shape as `training/README.md`**, which the heuristics plan already flags as documenting
    about twelve files that are not on disk. The rework moved the code and left behind the things that watch it.
    - **Options:** undrafted, and the question is what the front page is *for* now rather than what to rename.
      With no loop to show progress through, "where is this training" has to become something else — what each
      run is learning and how fast, which is what the constraint arms already print per position and what the
      finder prints per game.
    - **Undecided**, postponed.


35. **A term is priced for what it costs and nothing says what its accuracy is worth.** (found on checking the
    literature after pricing terms by what they take to read)
    `ExpressionSearch.cost` now multiplies a term's clause count by `TermEvaluator.dearness`, so a term that
    reads a hundred times slower than the cheapest here is penalised a hundredfold in the L1 price. That
    mechanism has a name — **cost-sensitive feature selection**, in its *embedded* form, a per-feature cost
    folded into the penalty — and `SparseFitter` already took a `costs` array per column, so what changed is
    only what goes into it. Two things the literature says I did not account for:
    - **Cost there is given, not measured.** An acquisition cost is money, minutes, a medical procedure,
      supplied by whoever knows the domain. Timing the reading is the right move for a learner that knows
      nothing about its game, and it is why `dearness` is relative rather than in seconds — but it is noisy and
      machine-dependent in a way a given cost is not.
    - **Misclassification cost is the other half, and there is none.** The framing is cost *against what a
      mistake costs*. Here a mistake is a worse heuristic, priced only through held-out loss, on a scale that
      has no common unit with dearness. So the trade is one-sided: terms were made dearer without anything
      saying what accuracy is worth.
    **Game playing has the same trade under another name — knowledge against speed.** Stockfish runs thin
    nodes, little evaluation and enormous node counts; Leela runs thick ones, far better evaluation at about a
    thousandth the nodes; they reach comparable strength. **Thick nodes are a real and winning design**, so the
    four-tenths-of-a-second heuristic that provoked this was not wrong for being expensive. What was wrong is
    that nothing chose: the fit optimised accuracy alone and the planner was then left four nodes, which is
    neither thin nor thick. The narrower justification for the change as it stands is therefore not *dear terms
    are bad* but **the fit and the planner must agree about what a node costs**, and nothing yet makes them.
    - **Options:** undrafted. What is visible without reasoning it through: whether the planner's budget is the
      missing unit, since nodes per position is a number both sides could be told; whether accuracy's worth is
      measurable only by playing, which would make it games rather than held-out loss that settles it; and
      whether a measured reading time belongs on the model record as a belief, which `ModelRecord`'s own
      docstring already says it should be — *"its measured accuracy and processing time are beliefs about it"* —
      and which nothing writes today.
    - **Undecided.**


36. **What a move does is scored on boards, and another rule has to ask it what happened.** (found putting the
    sniper to king safety, after the question it needs was made available to the search)
    `TAKEN_AFTER` — "taken from that player, once this is done, by something they could then do" — is the rule
    a king may not be left where it can be taken, and the only one chess has that nothing could express. It is
    answered by `Hypothetical._takes`, which draws what a candidate does and looks for a `Removed` change.
    **On a mature predictor there are none.** A run that had learned `Removed | grid` at one in the morning had
    learned only `Moved`, `Placed` and `Told` by eight, and the question is therefore false everywhere.
    **The predictor is not wrong.** `Placed` is documented as "something put where there was nothing, *or over
    what was there*" and `Moved` as "what stood at one place now at another". So a capture may be said as
    `Removed(destination)` then `Moved(origin, destination)`, or as `Moved(origin, destination)` alone, and the
    two make the same board out of every position. Nothing in fitting a predictor against boards prefers
    either, and the shorter one won.
    **`change.py` states the problem in its own first note** — *"An effect that hands back a position says what
    the position became and never what the action did. Whatever differs between the two has to be worked out,
    and which difference was the point cannot be... so what it did is there to be read rather than inferred."*
    The changes exist so that what an action did is readable. But which changes are learned is settled by which
    boards they reproduce, and that is blind to the distinction the changes were introduced to carry.
    **This is the first place the two halves have had to agree about anything.** Until now the predictor learned
    what a move does and the constraint learner learned what is refused, and neither read the other. The first
    rule needing both at once is the one that cannot be asked, and it fails silently: a question that cannot be
    answered comes back no, and a king who is never in danger looks like a rule that is simply not there.
    - **Options:** undrafted. What is visible: whether `_takes` should ask *whether the thing is still there
      afterwards* rather than whether a removal was drawn, which is general and correct and costs a board
      comparison per candidate; whether a change onto a place already holding something counts as taking it,
      which is reading what `Placed` and `Moved` already say they mean rather than adding anything; and whether
      a predictor should be scored on what it says an action *did* as well as on the board it gives, which is
      the only one of the three that stops the next such disagreement rather than this one.
    - **Undecided.**


37. **A question about what the other side could do is answered by rules that are not yet good enough to
    answer it.** (found running the sniper at king safety once the vocabulary could say it)
    The layering says the question goes to the rules that ask no such question themselves, and that is right.
    What it does not say is that those rules have to be *complete enough* for the answer to mean anything.
    Measured, with eighty-eight learned constraints as the layer below:

    ```
    mover: black
    a move that leaves the king attacked  [bishop, king, knight, pawn, queen, rook]
    a legal move (a8a6)                   [bishop, king, knight, pawn, queen, rook]
    a legal move (a8a7)                   [bishop, king, knight, pawn, queen, rook]
    a legal move (b6b5)                   [bishop, king, knight, pawn, queen, rook]
    ```

    **Every kind can be taken after every move**, so the reading discriminates nothing and no body built from
    it can refuse a check violation without refusing every legal move. The constraints leave roughly a hundred
    of fourteen thousand candidates standing; among a hundred surviving replies there is always one landing on
    the king's square, so the answer is yes everywhere.
    **The circularity is the point.** The rule that would tighten the constraints cannot be learned until the
    constraints are tight. Everything else in the chain works: the reading is produced, offered, and
    combinable, and a search finds it where it discriminates — pinned on a case with nothing of chess in it.
    What fails is the evidence the question rests on.
    - **Options:** undrafted. What is visible: whether the question should be put to the *game* rather than to
      the rules while the rules are young, which is truthful but is asking an oracle and teaches nothing;
      whether a reading should say how much it is trusted, so a question resting on loose rules is a weak
      reading rather than a false one; and whether this is simply an ordering — king safety is the last rule
      to learn, not the first, and the honest move is to say so and come back.
    - **Undecided.**

    **Since measured again with tight rules below, and the circularity is not the only thing in the way.**
    Run with the forty written constraints as the layer below — every one of them but king safety, so the
    rules below are tight and silent about kings — the same reading says `king` *never*, on any move:

    ```
    board  the game says      move  what could be taken next
    1      refused for check  e8f7  nothing
    1      allowed            a8a6  nothing
    2      refused for check  e2e3  bishop, pawn, rook
    2      allowed            a1b1  bishop, pawn, rook
    3      refused for check  e7d6  bishop, queen
    3      allowed            a7a5  bishop, queen
    ```

    Not yes everywhere but no everywhere, and the reason turned out to be stale evidence rather than a limit:
    the knowledge base read from had been filled before a removal's condition generalised. Learned again from
    two thousand moves of legal play on current code, the condition is `lands on(self, x, y, grid,
    piece(X1, X2))` — both colour and kind are variables, and a king is a piece like any other. Those two
    thousand moves took 186 things, ten distinct kinds, both colours, every kind but a king, which is never
    taken in legal play. **The general condition is reached from evidence that never contains the case it has
    to cover.**

    **Run again with tight rules below and consequences learned here, the reading separates exactly:**

    ```
    board  the game says      move  what could be taken next
    1      refused for check  e8f7  king
    1      allowed            a8a6  nothing
    1      allowed            b6b5  pawn
    2      refused for check  e2e3  bishop, king, pawn, rook
    2      allowed            a1b1  bishop, pawn
    3      refused for check  e7d6  bishop, king, queen
    3      allowed            a7a5  bishop, pawn, queen
    4      refused for check  g2f1  king, knight, pawn, rook
    4      allowed            a1b1  bishop, knight, pawn, rook
    ```

    `king` after six of six moves refused for check and after none of sixteen legal moves. The rule is a body
    of one condition. **So only the loose half of this question was ever real**, and what it costs is settled:
    the hard rule is cheap exactly to the degree the easy ones are right.
    - **Decided (Maxime): iteratively, and there is nothing to detect.** Improve the ruleset until it is
      stable but for the harder rules still missing, then dig deeper, each pass uncovering more hypotheses
      until the set is coherent. **Stability is not a thing to measure and wait for — it happens**, as more
      data arrives and the rules are reconciled against it. And depth is not raised on a signal either: the
      maximum is raised when the search stops finding better solutions.
      That removes two of the three things listed below as open. There is no definition of stable to settle,
      because nothing waits on stability; and there is no stopping rule to find, because deeper is tried
      exactly when shallower has stopped paying. What is left is the middle one.
    - **What that leaves open, and none of it is started:**
        - **What deeper opens.** A reading that rests on the rules below is offered unconditionally today.
          Under a loop it would be earned — withheld while the layer below is loose, since offering it early
          is what produced the yes-everywhere answer above. This is the one sub-question the decision above
          does not dissolve, because raising the depth says nothing about which readings become askable at
          which depth.
        - **Consequences do not outlive their run.** `plain2` and `sides2` persist none; `windowed` cannot be
          loaded at all, holding the pre-rewrite spelling. A loop that re-learns what a move does every pass
          is paying for it every pass. Plumbing rather than a question, and unstarted.


40. **Nothing measures whether advice helped.** (found deciding what a coaching comparison should be, and
    being told rightly that OMF should decide it)
    Advice is the difference between two models: what a player would do, and what would be better. What
    *better* is has two candidates and they are not the same claim — the optimal model, which says what is
    theoretically better, and a stronger player's predictive model, which says what somebody rated well above
    them actually does differently. Both are buildable from what exists and the database carries the ratings
    the second needs.
    **Which is better advice is not for anybody here to pick**, and that is settled: it is the same shape as
    which rater goes first, and `CascadeOrder` already refuses to privilege a candidate for being what it is.
    The currency is the caller's and the measurement decides.
    **The gap is that there is no currency to give it.** Every other choice in this design is settled by
    evidence the system can gather: a heuristic is judged by games it plays, a predictive model by how little
    it was surprised, a fit by held-out rows. Advice is judged by whether the person who took it got better,
    and that is months away, outside the system, and confounded by everything else they did in the meantime.
    A cascade ordered on a currency nobody can measure is ordered on nothing.
    - **What is visible, and none of it is chosen:** whether the proxy should be agreement — does following
      the advice move the player's choices toward the stronger model's, measured on their later games, which
      is gatherable but measures compliance rather than improvement; whether it should be the player's own
      result over time, which is the real thing and is slow and noisy enough that a season of it may not
      settle one comparison; whether advice should be scored the way a rule is, by how much of the gap it
      accounts for, which measures the explanation rather than its effect; and whether the honest answer is
      that this one is not the system's to settle and the two comparisons are both offered, named, with what
      each rests on, for a person to judge.
    - **Undecided.** Nothing is built on either comparison meanwhile.


41. **Every game leaves a candidate behind, and nothing takes one away.** (found wiring the drawer, working out
    what bounds the pool once each worker ponders after each game)
    The specification ponders at the end of every game and accumulates what it settles. A game takes a
    fraction of a second and a ponder takes half a minute, so each worker leaves behind roughly one new
    candidate heuristic a minute, and every one of them is registered as a model of what a position is worth.
    **Everything downstream is priced per candidate.** Judging rates all of them against every game that
    ended; the players are offered all of them, and each one crosses to a worker as its rules and weights.
    Neither is expensive at three candidates and both are linear in a pool that grows all night.
    **The drawer does not settle it.** It prefers what is unproven, so a growing pool is explored rather than
    ignored — but exploring more things is the cost, not the cure.
    - **What is visible:** retiring a candidate that has answered enough decisions and did no better than
      knowing nothing, which is the score against its own ignorance baseline and needs no new measurement;
      retiring the weaker of two that agree with each other, which is the other half of specification step 6
      — *"which correlate with each other"* — and needs the per-decision chances that `AgreementScorer`
      currently aggregates away; keeping one registered heuristic per worker, re-fitted each game, which
      bounds the pool at the cost of throwing away a good one for being old; and pondering less often than
      every game, which is the departure already refused.
    - **Decided (Maxime): not yet, and not on this argument.** Nothing is retired until candidates have been
      properly tested, shown not to work, and shown to actually clutter. Disk and memory are not short. What
      this question is waiting for is a measurement from a night's run — how many candidates there are by
      morning, and what judging and offering cost by then — not a cap chosen in advance.


42. **Listing a rule in a ruleset costs the whole ruleset, so a ruleset costs the square of what it lists.**
    (found when an overnight run filled a filesystem)
    `KnowledgeBase.link` sets the links and calls `ruleset`, which appends the record to the store. So linking
    the *k*th rule appends a record carrying *k* links, and listing *N* rules writes *N(N+1)/2* of them.
    Measured, one ruleset in an empty store:

    ```
    links   rulesets.jsonl
       50           79,816 bytes
      100          284,316
      200        1,068,316
      400        4,136,316
    ```

    Doubling the links quadruples the file, which is the square plainly.
    **It is not only a worker's problem.** A night's run left a worker's `rulesets.jsonl` at 63 GB against 43
    MB of rules, six of them filled a 457 GB filesystem to nothing, and the run spun for hours failing to
    write. But the run's own store shows the same shape at the same moment — 12.5 MB of rulesets against 558
    KB of rules, after three positions — so this is about how a ruleset is kept, not about who keeps one.
    Everything that lists many rules over time meets it: the induced ruleset gains a constraint a position,
    and a fit lists every term it kept.
    **The store is append-only on purpose**, and that is worth keeping: it is a record of what was believed
    and when, read back with the last write winning. What is quadratic is not the appending but appending a
    *whole* record to say one small thing about it.
    - **Options, none chosen:**
      - **A link is its own record.** Links become their own append-only stream, so listing a rule costs one
        small append and *N* links cost *N*. The honest fix, and it changes the store's shape and everything
        that reads a ruleset back.
      - **Link in one go.** `link_all(ruleset, pairs)` appends once for a whole fit, turning a hundred and
        fifteen growing records into one. Small, compatible, and still quadratic across fits — it buys a
        constant, not the exponent.
      - **Compact the store.** Rewrite it when it is loaded or when it passes a size, keeping the last record
        per id. Keeps the format and the reading, and gives up the order of belief that the append-only shape
        exists to carry.
      - **Bound what writes.** A short-lived store is thrown away before it grows, which is what a player's
        own store now does. It does nothing for a store meant to outlive the run, which is the one that
        matters.
    - **Undecided.** The player's store is bounded meanwhile, so nothing is filling a disk today; the run's
      own store still grows this way and nothing has measured how fast over a full night.

43. **The budget thins the rules in a heuristic and does nothing about how many heuristics there are.**
    - **Measured.** On a real chess ponder the gate took a fit of five terms down to three at ten bits a
      round; on tictactoe against a fit keeping thirty-nine, thirty bits wrote seven and a hundred wrote
      sixteen. So each candidate is proportionally cheaper to judge and to link, which is real relief for
      question 42, whose cost scales with how many rules a ruleset lists.
    - **What it does not touch.** Three prices are swept with `keep_every_price`, so every ponder mints three
      rulesets, `_adopted` names each one afresh so that nothing overwrites anything, and nothing ever
      removes one. Judging costs candidates × decisions × moves, and the candidate count is untouched. At six
      workers on a five-minute cycle that is on the order of two hundred heuristics an hour.
    - **The three things that would bite**, in the order they look worth trying: retiring whole heuristics
      on the `mass - offered` the judging already computes every round and throws away (question 41, and the
      smallest change); `RuleTenure`, which thins rules further but no rulesets; and not keeping every price,
      which is three times fewer rulesets at the cost of the candidate pool the sweep exists to create.
    - **Undecided**, and waiting on the same night's measurement question 41 is waiting on.

44. **A binary constraint over large domains is unaffordable in the CSP even where its relation is sparse, and
    three smaller things around it.** Found while stating a positioning problem — which stretch of a wall a few
    colored markers identify — as a constraint problem
    ([marker-wall.md](interfaces/marker-wall.md)). The trail and `circuit` work
    ([search-trail.md](interfaces/search-trail.md), [circuit-constraint.md](interfaces/circuit-constraint.md))
    fixes none of these, because the encoding it settled on has no binary constraints at all. So they are
    recorded rather than met.
    - **A support table is materialised over the whole cross product.** `solver.py:171-181` builds a
      two-parameter constraint's allowed pairs by iterating `domains[first] × domains[second]` and checking
      every pair. Where both domains are large, that is paid up front whatever the relation turns out to hold.
      The encoding this was found with — a chain of variables each ranging over every window pattern, each
      neighbouring pair constrained to overlap — costs `N` tables of `N²` checks:

      ```
          positions N   checks to build the tables   verdict
                  343                    4.0 x 10^7   minutes
                2 401                    1.4 x 10^10   out
      ```

      The relation itself holds only `N × Q` pairs, so what is wasted is the whole difference. A lazily
      checked binary constraint, or one built from the relation where a caller can offer it, would not pay
      it — and would also not have arc consistency's support counts to work from, which is the real tension.
    - **Identical constraints each get their own table.** In that encoding all `N-1` overlap constraints are
      *the same relation*, and each is materialised separately. Keying the table by the prepared source and
      the two domains would build one. Small, and it buys a factor of `N`.
    - **An all-different group is written as one Python source string.** `Solver._group` reads its operands
      out of a `PythonRule`, so a group of `N` parameters is a source string with `N` operands: over a
      megabyte before anything is solved once `N` reaches six figures, compiled and cached as one rule. The
      values-as-values path already exists for parameter domains — `DomainRule` was added for exactly this
      reason — and there is no counterpart for a constraint's operands.
    - **`AllDifferentPropagator` recurses.** `augment` and `_components` are both recursive
      (`all_different_propagator.py:49` and `:82`), so stack depth is bounded by the group size. No
      `RecursionError` fired on the groups measured, but that is those instances' luck rather than a
      guarantee, and nothing about the group size is under OMF's control.
    - **Undecided**, and nothing is filling a disk or crashing a run today: the problem that found these
      states `circuit` alone and forms no group and no table, so it meets none of them. They wait for the
      next problem that does.

45. **"Hardcoded" and "held as axiomatic" are being treated as one thing, and one of them is a model we wrote.**
    (found when a hand-written rule turned away a legal move and the mending went to rewrite it)
    A rule record is frozen unless `open`, which is the right shape: a belief held as basic is one nothing inside
    may revise, and `open` is how a caller says *except this one*. Given constraints are now written frozen and
    the mending honours it.
    What that flattens is a real difference between two things both given by hand. The constraints are the rules
    of chess — the game is what they say it is, so calling them basic is calling the game the game. The
    predictions are a *model* of what a move does, written by somebody who could be wrong, and one of them is
    known to be incomplete today: the square a pawn may be taken in passing on is recorded only where a pawn
    stands to take there, and whether that pawn is pinned wants the hypothetical, which would have the predictor
    waiting on the constraints that wait on it.
    So a constraint held as basic can turn away a legal move for a reason that is entirely the predictor's, and
    both are frozen. Nothing inside can settle which of two things held as basic is the wrong one — the warning
    says so and stops there, which is right and is also an admission that the two were declared alike when they
    are not. What the epistemic domain would add is somewhere for the conflict to go: confidence in what a move
    is predicted to do is a thing that can move, where the rules of the game are not.
    **Undecided.** Writing the predictions `open` would let the mending rewrite them, which is worse — a model
    corrected to fit one position is the fault this project has a standing rule against. What is wanted is not a
    different flag but a layer that holds degrees of belief, and that is the epistemic domain's job rather than
    this file's.

46. **The teller only measures the heuristics it is asked to narrow, so where there is nothing to cut it
    measures nobody.** (found after the rule-level teller signal was removed, when the heuristic-level line had
    never appeared once in a running build)
    `_told` returns early on `len(models) <= keeping`, which is correct for what it was written to do: narrow a
    pool before the dear payoff question, and a pool already smaller than the cut needs no narrowing. The live
    run is `--teller-keeping 24` over 20 heuristics, so it has taken that branch every time and the teller has
    not been asked anything.
    That was invisible while a rule signal was printing teller numbers every ponder. With the signal gone this
    is the only place outside knowledge enters, and the early return now skips two things that are not
    narrowing: `JudgingRecord().told(...)`, which is what the Heuristics page reads for its `tracks the teller`
    column, and the measurement itself. Asked for where the teller's opinion is visible, the honest answer today
    is nowhere whenever the pool is at or below `keeping` — and the column reads `not asked` for every
    heuristic, which is true and looks like a wiring fault.
    Two purposes looked like they were wearing one function. They are not: narrowing is the whole job, and the
    measurement is a by-product of it rather than a thing owed to anybody.
    **Answered: the teller stays silent where there is nothing to narrow.** A teller that measures a pool it is
    not cutting is spending engine time to fill a column, and the column is not what it is for. What was
    actually wrong is the other end — a pool of twenty against a cut of twenty-four is a pool too small to need
    cutting, and the fix belongs where heuristics are generated rather than where they are measured. So the
    early return stays, and `tracks the teller` reading `not asked` across a small pool is the truth.

## Struck, by number

Decided, and removed from the body. Read the reasons in `git log -p doc/open-questions.md`.

- **38.** A term flat along a walk is not a term that carries nothing — and neither is a term flat over the
  positions being fitted. Order by steadiness, drop nothing.
- **39.** The price sweep could not tell its models apart. Near enough now counts as equal, by the standard
  error of the loss.

- **A1.** Default tactic
- **A2.** Accuracy and precision
- **A3.** Certainty
- **A4.** Goal weights are beliefs
- **A5.** Time management levels
- **A6.** The bootstrap time management policy
- **A7.** CSP and continuous actions
- **A8.** Hierarchy
- **A9.** State abstraction
- **A10.** Planning in a coding agent
- **A11.** Soft goals
- **A12.** Epistemology
- **A13.** Search and hidden information
- **A14.** The package map
- **B1.** Expected utility vs fuzzy distance
- **B2.** Anchors that can be wrong
- **B3.** The zero-sum assumption vs agent models
- **B4.** Payoff vs utility vs rhetorical gain
- **B5.** Words in the search
- **C1.** Where the hierarchy lives
- **C2.** Opinions held by the agent itself
- **C3.** How a tactic is defined
- **C4.** Rhetorical tactics
- **C5.** What "literature" is
- **C6.** The budget's unit when levels overlap
- **C7.** Task value
- **C8.** Debug vs dashboard vs logs
- **C9.** The debugger's stack and the real-time clock
- **C10.** Measurements taken across a pause
- **C11.** "Tactic" means two things
- **C12.** Detectors: where they come from and where they live
- **C13.** Interconnecting modules
- **C14.** Documentation
- **C15.** What "no learning" freezes
- **C16.** Simulation and the predictor
- **C17.** The heuristic and model step
- **C18.** The policy and optimizer step
- **C19.** The agent model and search step
- **C20.** The budget and agent loop step
- **C21.** The context hierarchy step
- **C22.** A game whose starting state uses a grid's aliases can't declare where it starts
- **C23.** What a grid answers where nothing stands
- **C24.** Where a player's own direction comes from — deduced, not declared, and absent where a game has none
- **C25.** What a finished game has to pay before a value can be selected
- **C27.** What a shape settles, nothing says
- **C29.** Constraints are not carried forward, they are regrown — and mending is dead code
- **C30.** A symbol that says its part perfectly is thrown away for being rare
