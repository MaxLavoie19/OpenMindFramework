# Putting the findings into the learner

What to change, in the order I would change it, from [learner-practices.md](learner-practices.md). Each step is
one component and goes through the usual two stops: interfaces approved, then code approved.

The measure of the whole thing is already in place and does not change: the hand-written constraints in
`openmind_chess.learning.service.chess_constraints`, the four counts on `/constraints`, and `matching / 72`. Every
step below says what it should do to those numbers, so a step that does nothing is visible rather than assumed.

## 1. Strip by what it costs, not by what it gets away with

**The fault.** `ConstraintDistiller._stripped` removes a condition whenever removing it does not happen to refuse
a legal move. Nothing measures what the condition was worth, so `refused :- origin(3, X1)` survives: in a position
whose rank 6 is empty it refuses no legal move, and it refuses every move in the next position.

**The change.** Grow against one part of the legal moves and strip against the other, which is IREP's split.

There is no second half to this, and the reason is worth writing down because I proposed one and it was wrong. A
condition's removal cannot *lose* refused candidates: a shorter body is implied by a longer one, so what the
constraint refuses only grows. So there is nothing to measure about the removal itself, and the usual heuristics
have nothing to weigh — with refusing a legal move forbidden rather than priced, every removal the guard permits
scores alike. Nothing in one position's twenty legal moves distinguishes `origin(3, X1)` from a good constraint,
because both are consistent with all twenty. Only evidence the constraint was not grown against can tell them
apart. That is the whole of the fix.

**Where.** The caller holds each position's legal moves back in two parts: one goes into the cases
`RefusalLearner.learn` grows against, the other never does and is what `ConstraintDistiller` strips against.
Both accumulate across positions, so the pruning side is thin at position one and honest thereafter.

**Verified by.** `origin(3, X1)` and its kind stop appearing; `wrongly refused` stays at zero without repair
having to drop anything; `refused accounted for` stops falling as it does now.

## 2. Honour the kinds the game declared

**The fault.** `_best_pairing` will pair a row place with a column place, and `_distances` emits the distance
between a row and a column as a reading. Neither can mean anything, and both cost time on every case.

**The change.** Carry each parameter place's kind, from the schema the game already declares, and refuse pairings
and distances across kinds. This is Progol's and Aleph's mode-and-type bias, and it is not knowledge smuggled in:
the game said a cell has a row and a column and that they are different kinds.

**Where.** `CandidateReadings` and `RefusalLearner` take the `ActionKind` at construction, as `Hypothetical`
already does. `_distances` pairs only places of the same kind; `_best_pairing` skips pairings whose places
disagree.

**Verified by.** Readings per case fall (six distance readings become two in chess); learning a position gets
faster with the same constraints found; nothing in the hand-written set stops being expressible.

## 3. Let a failure constrain the space, not just the clause

**The fault.** When a constraint refuses a legal move, `repaired` tries to add a condition its kept cases all
share, finds none, and the caller drops it — losing everything it rightly refused. Worse, nothing stops the same
constraint being learned again in the next position, since nothing recorded that it was refuted.

**The change.** Keep what was refuted. A candidate constraint is rejected out of hand if it is more general than
something already shown to refuse a legal move — which is a subsumption test, and `Subsumer` exists for it. This
is Popper's generate–test–constrain: the failure teaches the search rather than only the clause.

Repair keeps its place for the case where a condition does separate the cases. What changes is that dropping is no
longer the end of it.

**Where.** A `Refuted` record in `inference/model`, held by the caller and passed to `learn` and `distilled`
alongside `starting`, since the services stay stateless. Checked wherever a candidate is accepted: `_widened`,
`_widest`, `_folds`, `_stripped`.

**Verified by.** The same wrong constraint stops reappearing across positions; the count of dropped constraints
falls while `refused accounted for` holds.

## 4. Regrow each constraint at the end

**The fault.** A constraint learned at position one was grown when nothing was covered and nothing since has
reconsidered it, only stripped it.

**The change.** RIPPER's optimisation pass: for each constraint build two rivals — one grown afresh from the cases
it covers, one grown by extending it — and keep whichever gives the shortest description while refusing the same
candidates. The rules this rescues are exactly the early ones.

**Where.** A pass in `ConstraintDistiller` after folding and stripping, under its own share of the budget.

**Verified by.** Total readings fall on a run of ten positions without `refused accounted for` falling.

## 5. Stop by description length

**The fault.** `learn` stops when every refused candidate is covered, so the last few stubborn ones get a
constraint of their own that explains nothing and costs a lot.

**The change.** RIPPER's stopping rule: stop adding constraints once the description length of the constraints
plus what they leave unexplained exceeds the best seen. What is left uncovered is then a residue to be reported
rather than a thing to be papered over.

**Where.** `RefusalLearner.learn`, beside the existing `least` and `positions` guards.

**Verified by.** Constraint count falls; `refused accounted for` falls slightly and deliberately; the residue is
logged as what to go and find evidence about.

## 6. Ask about directions

Not a change — a question, because it is a new reading. Every pawn rule in the hand-written set is written twice,
once per colour, because we can say how far apart two numbers are and not which way. The chess ILP work had
directions in its background knowledge for this reason. One generic reading — which way one number lies from
another — would halve them.

If the answer is yes, it goes in `CandidateReadings._distances` beside the distance, and `chess_constraints`
should then be rewritten to use it, to confirm the rules get shorter rather than merely different.

## 7. Choose the next position instead of taking the one that arrives

**The fault.** Positions come one move apart from a single random walk, so consecutive positions teach almost
nothing new, and castling, en passant and promotion effectively never appear.

**The change.** Pick the next position by where the constraints and the engine disagree, or where a constraint has
never been tested. We score every candidate in every position already, so the signal is in hand.

**Where.** `learn_constraints.py`, which currently plays a uniformly random legal move.

**Verified by.** `matching / 72` rises faster per position than under the random walk, on the same budget.

## 8. Walk several games at once

Twenty-four workers, each its own game and seed, learning from their own positions and handing back constraints;
the parent merges them by union and distils against the union of everyone's legal moves. `TaskRunner` does the
process handling already, and only the clauses cross between processes.

This belongs after 1 and 3, not before: merging twenty-four sets of constraints that each over-strip and cannot
record a refutation would multiply the mess rather than the evidence.

**Verified by.** Positions per hour; and that `wrongly refused` falls rather than rises as the guard grows.

## What this does not touch

King safety, and with it castling through check, needs the predictor — a position that does not exist yet, read
and asked about. That is the other half of the design and none of the above brings it closer.
