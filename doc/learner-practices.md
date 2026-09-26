# How rule learners are built, and what ours does differently

What the literature on inductive logic programming settled on, set against the learner in `inference/service`, so
the places we departed from it are deliberate rather than accidental. Sources at the end.

## The shape everybody uses

**Sequential covering.** Learn one rule, set aside the examples it accounts for, learn another for the rest. Our
`RefusalLearner.learn` is this already, and it is the uncontroversial part.

**Scored selection, not vetoed selection.** FOIL grows a rule top-down and picks the next condition by *adjusted
information gain* — how many positive examples a condition keeps against how many negatives it lets in. Every
condition in the final rule is there because it earned its place by separating the classes.

Ours does not do this, and it is the single biggest gap. `ConstraintDistiller._stripped` removes a condition
whenever removing it does not happen to refuse a legal move. That is a veto, not a measure: nothing asks how much
a condition contributes. It is how we ended up holding `refused :- origin(3, X1)` — "refused where the move starts
on row 3" — which is true of a position whose rank 6 is empty, classifies no better than a coin, and refused every
move in the next position.

**Top-down versus bottom-up.** FOIL searches general-to-specific. We grow bottom-up by anti-unification, and that
is worth keeping: it is where the variables come from. Nothing top-down can propose
`places apart(origin, 1, destination, 1, N)` without first inventing the variable structure to score. The right
combination is ours to generate candidates and theirs to choose between them.

## What to do when a rule turns out too general

**Learning from failures (Popper).** Generate, test, constrain. When a hypothesis entails a negative example, do
not merely patch it — record a constraint that **prunes every generalisation of it** from the space. When it fails
to entail a positive, prune its specialisations. The failure teaches the search, not just the clause.

Ours patches. `RefusalLearner.repaired` adds conditions to the offending clause and the caller drops it when
nothing separates the cases — and in practice it always drops, because after stripping, the candidates a
constraint must keep refusing share nothing but the board. So each correction costs everything that constraint
rightly refused, and the same wrong constraint is free to be learned again next position. A constraint recorded
against the space could not be.

**Theory revision** is the older name for this, and its literature warns that "the most general correct
specialisation" is the wrong thing to ask for — minimality has to be defined carefully or revision oscillates.

## Choosing between rule sets

**Minimum description length.** The model that compresses the data best is the one to keep.

~~Our cost — `per_clause + len(body) ** 2`, minimised subject to refusing the same candidates — is an MDL
criterion in all but name, and this is the part of our design the literature most straightforwardly endorses.~~

**That was wrong on both halves, and the search that found it out is worth keeping.** A criterion in all but
name it is not: it is a made-up penalty rather than a code, which
[Galbrun's survey](https://arxiv.org/pdf/2007.14009) §8 names as "short-circuiting the principle" — using unit
costs "avoids making decisions". And it is not minimised subject to refusing the same candidates *as a
trade-off*; that is a hard invariant, so nothing prices what the rules leave unexplained, and a set's size is
therefore settled by a repair heuristic rather than by any pressure we have claimed.

Worse, the sign runs against us. Under a real code a body of k conditions from a pool of C costs
`L_N(k) + log₂C(C,k)`, whose marginal cost per further condition **falls** with k; under `1 + k²` it **rises**.
So the quadratic over-prefers short constraints — and in refusal semantics a short constraint is an over-general
one, which is the error this project calls unrecoverable.

What the literature does endorse, and what our setting turns out to be, is
[URPILS](https://eda.rg.cispa.io/pubs/2024/urpils-wiegand,klakow,vreeken.pdf): constraints, survivors, no
negative examples, and a data term of `log|F_M|` — each observed legal move costs the bits to name it among what
survives. The asymmetry we care about then falls out of the encoding at about 450:1 rather than being weighted
in by hand.

## What game-rule induction in particular has learned

**Structured induction beats learning everything at once.** The Progol work on chess decomposed the move
generator into some forty sub-problems, fifteen levels deep, each level's learned clauses becoming background
knowledge for the next. They did not try to learn `legal move` in one go, and neither should we.

Our `Hypothetical` and `allowed by the rules below` is a two-level version of this, arrived at independently and
for the same reason. The gap is that our levels are fixed by whether a constraint asks a hypothetical, while
theirs were a designed hierarchy — and the general form of getting there by yourself is predicate invention,
which we do not do at all.

**The same three rules defeated them.** Progol's chess learner excluded castling, en passant and promotion.
Castling and en passant because they depend on history the board does not show; promotion because the move needs
a parameter it has not got. We set aside exactly those three, for exactly those reasons, before reading any of
this. That is not a failure to fix — it is the known shape of the problem.

**Check needs more than one position.** They report that check and double-check "required multiple board
positions to learn effectively", which is our finding that king safety is a fact about the board a move leads to
and needs the predictor.

**Their vocabulary was ours plus one thing.** They restricted background knowledge to "immediately perceivable
patterns — pieces, squares, colours, distances and directions", chosen so a non-expert could grasp them. That is
our reading vocabulary almost exactly, and we arrived at it by the same argument. The exception is **directions**.
We added distances (`apart`) and have nothing for direction, which is why every pawn rule we wrote by hand had to
say `less` or `at_least` on a row and be written twice, once per colour. A pawn moving *forward* is one rule; a
white pawn moving to a smaller row and a black pawn moving to a larger one is two rules and a coincidence.

**Expect it to be hard.** On the Inductive General Game Playing benchmark — inducing rules from traces of fifty
games — the best existing ILP system learned 40% of the tasks perfectly. Inducing game rules is not a solved
problem with an implementation detail in the way.

## What I would change, in order

1. **Score conditions instead of vetoing them.** Replace `_stripped`'s criterion with information gain over
   refused and allowed candidates. This directly kills `origin(3, X1)` and it is the change with the most behind
   it in the literature.
2. **Constrain the space on failure, not just the clause.** When a constraint is caught refusing a legal move,
   record that its generalisations are out, so the next position cannot learn it again.
3. **Ask about directions.** One generic reading — which way one number lies from another — halves every rule
   that is currently written once per player, and it is the one primitive the chess ILP work had that we lack.
   It is a new reading, so it is a question rather than a change.
4. **Learn intermediate predicates.** Structured induction is how chess was actually managed. We have the
   layering; what we lack is any way to invent the middle concepts that the layers would be built from.

## The practical toolbox

Heuristics and techniques the field converged on, with what each would mean here.

### Scoring a condition

The measures form a spectrum from consistency to coverage, and the choice is the task's:

| Measure | What it is | What it favours |
|---|---|---|
| Precision | `p / (p + n)` | consistency; overfits, since one example is perfectly consistent |
| Laplace | `(p + 1) / (p + n + 2)` | precision with a prior, so a rule covering one case is not worth as much as one covering a hundred |
| m-estimate | precision pulled toward the prior by `m` | whichever you want; `m` is the dial between the two |
| Information gain (FOIL) | `p · (log(p/t) − log(P/T))` | conditions that improve purity *and* keep positives |
| Weighted relative accuracy | coverage × (precision − prior) | coverage; used where a broad rule is worth more than a pure one |

**What this means for us.** Our classes are four thousand refused against twenty allowed, and refusing an allowed
one is not a cost to be traded off — it is forbidden. So the measure collapses: with `n = 0` required, precision
is 1 for every acceptable constraint and says nothing, and what is left to maximise is `p`, the refused candidates
still covered. That is the number `_stripped` never looks at. It is a smaller change than adopting FOIL wholesale:
keep the veto, and among the conditions the veto permits removing, remove the one that loses the fewest refused
candidates — and stop when removing any of them would lose some.

### Not overfitting the evidence in front of you

**Grow and prune on different examples.** RIPPER (and IREP before it) splits the examples: grow the rule on one
part, then simplify it while it still does well on the *other* part. That is a direct answer to our
`origin(3, X1)`: stripped against the same twenty moves it was grown from, any condition looks removable; stripped
against twenty it has not seen, it does not. We have been doing the equivalent of pruning on the training set,
which is the textbook way to get exactly the rule we got.

**Stop by description length.** RIPPER stops adding rules once the description length of the rules plus the data
exceeds the best seen by more than a threshold. Ours has no stopping rule of its own — it stops when every refused
candidate is covered — and MDL would let it stop earlier rather than reaching for the last few stubborn cases with
a constraint that explains nothing.

**Optimise afterwards, not only during.** RIPPER's last pass takes each rule and builds two rivals — one regrown
from nothing, one grown from the rule itself — and keeps whichever gives the shortest total description. Rules
learned early, when little was covered, are the ones this rescues, and our constraints from position one are
exactly that.

### Bounding the search before you search it

**Types and modes.** Progol and Aleph make the game declare each argument's type and whether it is input or
output, and refuse to build clauses that violate them. This is the cheapest large win available to us: our schema
already declares that a cell has a `row` and a `column` as separate kinds, and `_best_pairing` currently considers
pairing a row place with a column place, while `_distances` emits the distance between a row and a column as a
reading. Neither can mean anything. Honouring the kinds the game already declared would cut both, and it is not
knowledge we are smuggling in — the game said it.

**Bottom clause and inverse entailment.** Progol builds the most specific clause entailing an example and searches
only within it. Our `_outright` plus generalising is that method, arrived at independently, which is reassuring
about the shape even where the selection is wrong.

### Making it fast

**Coverage testing is the known bottleneck.** Theta-subsumption — asking whether a clause covers a case — "is
often the bottleneck of an ILP system", which is what our profiling found before we knew to expect it. The
literature's answers are caching coverage, testing subsumption as a constraint satisfaction problem, and
QuickFOIL's: map the operations to relational algebra and let query processing share operators, cache and group
the joins, which bought orders of magnitude over FOIL and Aleph.

We have done the cheap parts — cached hashes, indexed cases by reading, ordered conditions by selectivity. The
relational framing is the next order of magnitude and a real rewrite.

**Parallelism is standard here.** Data and task parallelism over ILP is well trodden; our plan to walk several
games at once and merge is the data-parallel form of it.

### Choosing what to learn from

Nothing above helps if the evidence is poor. Our positions come one move apart from a single random walk, and a
constraint that is wrong in a way no position in that walk exposes will never be caught. Active learning — going
and finding the position that would settle a question, rather than taking what arrives — is the standard answer,
and here it is unusually easy: a position where the constraints and the engine disagree is exactly what we score
already.

## Sources

- FOIL, sequential covering and information gain — [Mooney, *Rule Learning*, CS391L](https://www.cs.utexas.edu/~mooney/cs391L/slides/rules.pdf); [FOIL overview](https://www.geeksforgeeks.org/first-order-inductive-learner-foil-algorithm/)
- Learning from failures, generate–test–constrain — [Cropper and Morel, *Learning programs by learning from failures*](http://andrewcropper.com/pubs/popper.pdf)
- The state of the field — [*Inductive logic programming at 30: a new introduction*](https://arxiv.org/pdf/2008.07912)
- Theory revision and the minimality of specialisation — [*On the proper definition of minimality in specialization and theory revision*](https://link.springer.com/chapter/10.1007/3-540-56602-3_128)
- Inducing game rules from traces, and how well it goes — [Cropper et al., *Inductive general game playing*](https://arxiv.org/abs/1906.09627)
- Chess move legality by structured induction — [Muggleton et al., *Inductive Learning of Chess Rules Using Progol*](https://www.doc.ic.ac.uk/~shm/GAchess.html)
- RIPPER: growing and pruning sets, the Laplace pruning metric, MDL stopping and the optimisation pass — [Cohen, *Fast Effective Rule Induction*](https://static.aminer.org/pdf/PDF/000/334/623/fast_effective_rule_induction.pdf); [JRip's description of the algorithm](https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Classification/JRip)
- Consistency against coverage in rule heuristics — [*On the Trade-off Between Consistency and Coverage in Multi-label Rule Learning Heuristics*](https://arxiv.org/pdf/1908.03032)
- Scaling: relational algebra, caching and pruning — [Zeng et al., *QuickFOIL: Scalable Inductive Logic Programming*](https://pages.cs.wisc.edu/~jignesh/publ/QuickFoil.pdf)
- Coverage testing as the bottleneck — [*Fast Theta-Subsumption with Constraint Satisfaction Algorithms*](https://link.springer.com/article/10.1023/B:MACH.0000023150.80092.40)
- Chess rules as theory revision — [*Chess Revision: Acquiring the Rules of Chess Variants through FOL Theory Revision from Examples*](https://link.springer.com/chapter/10.1007/978-3-642-13840-9_12)
