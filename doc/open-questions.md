# Open questions

These are gaps, contradictions and unknowns in `doc/architecture.md`. Each one waits for Maxime's decision. None is
settled until it is struck from this file and the design says so.

**A question is struck once it is decided, and its number is never reused.** What was decided lives in the code it
shaped — the docstrings of this project carry their reasons — and in this file's git history, where each entry sits
in the commit that struck it. The index at the foot keeps the numbering legible, since docstrings and plans cite
questions by number.

## Still open

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
    `RuleDeducer`, `WorthReasoner` and the extraction scripts. `CandidateReadings` takes no `sides` at all — the
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


## Struck, by number

Decided, and removed from the body. Read the reasons in `git log -p doc/open-questions.md`.

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
