# Interfaces: one vocabulary

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
is a question, not a decision.

Nothing about what is learned changes. What changes is that the parts of this project stop saying the same
thing in different languages, so that a thing learned once is available everywhere rather than learned again at each
level.

## What is being asked

> *"I'd rather reconcile all vocabularies so that they all are compatible by design and it doesn't have to learn
> the same thing at multiple levels."* — Maxime

## The vocabularies there are

| where | what it says things in | example |
|---|---|---|
| `world/model/change.py` | four kinds of change | `Moved("grid", (2,5), (4,5))` |
| `rule/model/drawn.py` | nine ways of drawing a place or value from an action | `Standing(grid, Stepped(self, row, x))` |
| `inference/service/candidate_readings.py` | seven reading predicates over named places | `lands on(self, x, y, grid, piece(…))` |
| `inference/service/action_readings.py` | a second set of reading predicates | question 31: "two parallel stacks, both live" |
| `inference/service/evaluable_predicates.py` | computed relations | `apart(A, B, 2)`, `same(A, B)` |
| `inference/service/hypothetical.py` | three questions about what could happen | `taken after(Mover, king)` |
| `inference/model/aggregate.py`, expressions | Python over a position view | `sum(here.cell[i] == me for i in …)` |
| `language/model/happening.py` | what a move is called | roles over parts of a happening |

Eight, and at least five of them can say *the square a move lands on*.

## The finding

**They are one language spelled several ways, and the code says so about itself.** `Stepped`'s docstring:
*"The same gap the readings had, where `lands on` fills it."* `Place`'s: the general form of `Row` and `Column`.
`Functor`'s, in the term language clauses are already written in: *"A function of terms standing for a term: the
square one step on from another, the player whose turn follows."*

Said side by side, the duplication is exact:

```
the predictor            Standing("grid", Stepped("self", "row", "x"))
the constraint learner   lands on(self, x, y, grid, V)
the heuristics           here.cell[i]   (with i ranging, and the stepping written into the expression)
```

**And it has already cost a day.** A capture can be said as `Removed(destination)` then `Moved(origin,
destination)`, or as `Moved(origin, destination)` alone — the same board either way, so nothing in fitting a
predictor prefers either. The predictor settled on the second overnight; `Hypothetical._takes` was reading the
first; and king safety became a rule that could be written by hand and never found, silently. That is question
36, and it is not a typo. It is two halves of one system agreeing in prose and nowhere else.

## What the literature says

**This vocabulary exists, it is called the Game Description Language, and it has been proven universal.** It
distinguishes eight relations — `role`, `init`, `true`, `does`, `next`, `legal`, `goal`, `terminal` — with
`true` allowed only in the bodies of rules and `next` only in their heads. Thielscher proved the language
universal for finite deterministic games of complete information, and its stochastic extension universal for
any finite extensive-form game. So the moments below are not a new idea: `true` is now, `next` is after, and
keeping them apart is the standard discipline.

**The project is already doing the task the literature calls learning it.** Inductive general game playing —
Cropper, Evans and Law, 2019, which the heuristics plan already cites for "the best performing system solves
only 40% of tasks perfectly" — is *given game traces, learn the rules that could produce them*, where the rules
are written in that language. That is what `learn_constraints.py` does. Adopting the vocabulary therefore makes
this project's results comparable with a published benchmark of fifty games rather than only with themselves.

**But one part of it should be refused, and this project has already refused it for a good reason.** `next` is
a *complete* specification: everything true in the next position must be derived, or it is false. So a game
written in that language writes out what does not change, by hand — *"the cell stays marked unless somebody
marks it"* — one statement per fact per action that leaves it alone. That is why such descriptions run long,
and a large part of why learning them is hard.

`change.py` has already taken the other side, and says why: *"An effect that hands back a position says what
the position became and never what the action did. Whatever differs between the two has to be worked out, and
which difference was the point cannot be."* Saying the *difference* rather than the whole is the position the
planning languages take — a list of what an action adds and a list of what it takes away — and `Placed` and
`Removed` are exactly those two lists.

**What bridges them is the known answer to the frame problem.** Reiter's successor state axioms compile
statements about effects into one statement per fact, quantified over every action there is: the fact holds
after an action if that action made it hold, or it held already and that action did not unmake it. That takes
the number of statements from the order of facts times actions down to the order of facts plus actions.
Thielscher's fluent calculus takes the position itself as a term and gives update axioms for the same problem,
paying in inference where the other pays in writing.

**Somebody has already learned rules in the event calculus, and their concessions are the price.** ILED —
Katzouris, Artikis and Paliouras, *Machine Learning* 2015 — learns event definitions incrementally by
inductive logic programming. It is the closest published thing to what this project is building, and it buys
its result with two assumptions this project does not have: the dialect is **fixed and cut down**, not the
full calculus (*"We call the Event Calculus dialect used in this work Simplified Discrete Event Calculus"*),
and supervision is assumed **noise-free and complete**. It was shown on nine interrelated events over a
thousand time points.

**The predicates themselves can be learned.** SIFT — Gösgens, Jansen and Geffner, *Learning Lifted STRIPS
Models from Action Traces Alone* — is given no predicates: *"it involves learning the domain predicates as
well."* So a designed vocabulary is a choice here, not a necessity, and this plan has to say why it is the
right one rather than assume it.

**Learning from what is refused as well as what is allowed is standard.** AMLSI takes *"two training datasets,
I+ and I−"*, the second being infeasible sequences, and outputs a planning domain by grammar induction;
TempAMLSI extends it to temporal models. Two learners, one over what a game allows and one over what it
refuses, is the ordinary shape rather than an eccentric one.

So the position this plan takes: **adopt the separation of now from after and the discipline of one term
language; keep this project's differences rather than a complete `next`; treat what does not change as
something compiled rather than written; and restrict the dialect deliberately rather than discover the
restriction later**, which is what the literature has done since 1991.

The evidence behind these paragraphs, with what survived adversarial verification and what did not, is in
[one-vocabulary-evidence.md](one-vocabulary-evidence.md).

Sources: [the Game Description Language specification](http://ggp.stanford.edu/readings/gdl_spec.pdf) ·
[The General Game Playing Description Language Is Universal](https://www.ijcai.org/Proceedings/11/Papers/189.pdf) ·
[Inductive general game playing](https://arxiv.org/pdf/1906.09627) ·
[Situation calculus and successor state axioms](https://en.wikipedia.org/wiki/Situation_calculus) ·
[From situation calculus to fluent calculus](https://www.sciencedirect.com/science/article/pii/S0004370299000338) ·
[The frame problem](https://en.wikipedia.org/wiki/Frame_problem)

## What real time changes, and it changes the moment

Maxime, this session: *"we'll eventually test OMF on real time video games"*. That is not a later problem. It
decides what a moment is indexed by, and choosing the other way is cheap now and expensive afterwards.

**The situation calculus says outright that it cannot do it.** Its situations are snapshots at instants,
actions change one into another, and those actions "are instantaneous, have no duration, and have immediate
and permanent effects". All three are false of a held key, a projectile in flight, or a cooldown.

**The Game Description Language's `next` is the same assumption wearing a predicate.** `next` means *the next
turn*, and a real-time game has no next turn.

**The event calculus was built for exactly this** and was developed as the alternative to the situation
calculus. It assumes "an explicit linear time-structure, independent of any events", it covers "both action
events performed by agents and external events outside any agent's control", its time is continuous with a
discrete variant limiting it to integer points, and it carries a **default persistence**: facts hold until
something ends them.

Two of its pieces are already here under other names:

```
Initiates(event, fact, time)   ↔   Placed  — something now the case that was not
Terminates(event, fact, time)  ↔   Removed — something no longer the case
default persistence            ↔   saying the difference and not the whole, which `change.py` already argues for
```

So the correction: **a moment is indexed by time, not by an action.** A turn-based game is the discrete case
where the clock advances once per joint action, and `after(A)` is `at(T + 1)`. That costs nothing to choose
today — it is the same work — and it is what makes a game where things happen that nobody did sayable at all.

The one thing it adds now, and it is worth having for chess too: **an event need not be anybody's action.** A
clock running out, a piece promoting, a pawn taken in passing — these are things that happen, and saying them
as events rather than as parts of somebody's move is what `Consequence.order` is already working around.

**And this is a frontier rather than a thing to adopt.** Laird — sole author, Soar's own creator — reports
that ACT-R and Soar have at most an optional module estimating short durations, that **neither can judge
longer time scales**, and that this is an open research problem. Two architectures and four decades between
them, and the thing going at the centre of this vocabulary is the thing they name as unsolved. That is a
reason to keep the time structure as thin as the work actually needs, not a reason to avoid it.

Sources: [the event calculus](https://en.wikipedia.org/wiki/Event_calculus) ·
[Event calculus, handbook chapter](https://dai.fmph.uniba.sk/~sefranek/kri/handbook/chapter17.pdf)

## The one vocabulary

Three things, of which two already exist.

### Terms — what something is, and they are `Term`

`Constant`, `Number`, `Variable`, `Functor` — unchanged. What changes is that the drawings become functors
rather than a parallel set of classes:

```
Place("self", "row")              →  place(self, row)
Stepped("self", "row", "x")       →  stepped(place(self, row), x)
Standing("grid", Place("self"))   →  holding(grid, place(self, row), place(self, column))
Other()                           →  other(mover)
More("halfmove clock")            →  more(halfmove clock)
Row / Column                      →  place(…, row) / place(…, column), which `Place` already wants
```

Nothing is invented: `Functor` is documented for exactly this, and unification over them already works.

### Facts — what is the case, and they are `Literal`

A reading is a fact. What changes is that a reading's arguments are terms rather than strings that name terms:

```
holds(self row, self column, grid, V)   →  reads(holding(grid, place(self, row), place(self, column)), V)
lands on(self, x, y, grid, V)           →  reads(holding(grid, stepped(place(self, row), x),
                                                                stepped(place(self, column), y)), V)
```

`lands on` stops being a separate predicate and becomes `reads` of a stepped place. Every other reading is a
relation over terms in the same way: `apart(A, B, N)`, `same(A, B)`, `on line(A, B, V)`.

### Moments — when it is the case, which is the new part

A fact holds *at* a time. Not after an action: at a time, with an action being one of the things that can
happen at one.

```
at(now): reads(T, V)
at(W): reads(T, V)          where W is a time, and a turn-based game's clock ticks once per joint action
happens(E, W)               where E is an action somebody took, or something that simply happened
```

This is not new — it is `true` and `next` from the Game Description Language, and that separation is what
makes the language universal. It is what the rest of this has been working around:

- **A consequence becomes an equation between moments.** *"This move carries what is at the origin to the
  destination"* is `after(move): reads(holding(grid, stepped(…)), V) :- now: reads(holding(grid, place(self,
  …)), V)`. The predictor is then learning clauses, in the language the constraint learner learns clauses in,
  and `Consequence`'s `where`/`onto`/`value` become the terms of one.
- **What is taken stops being machinery.** `taken after` is `now: reads(T, mine) ∧ after(A): ¬reads(T, mine)`,
  said rather than computed. Question 36 cannot recur, because there is no second reading of what a capture is
  to disagree with the first.
- **A change becomes a difference between moments**, which is what `change.py`'s own note says it exists to
  carry: *"An effect that hands back a position says what the position became and never what the action did."*
  The four kinds stay as the *shorthand* a predictor draws in; what they mean is the difference.

## The readings are learned, not written

**Decided (Maxime).** What a candidate is read as is discovered, not designed.

Today seven predicates are hand-written. Somebody judged that *what stands between two places* was worth
having and wrote `_on_line`; that *how far apart two numbers are, and which way* was worth having and wrote
`_distances` and `_from_nothing`. The learner searches combinations of those seven and never invents an
eighth. When a pursuit finishes having found nothing, what it has proved is that no combination of *these*
tells a candidate apart — which is a fact about somebody's judgement, reported as a fact about the game.

**The term language is what makes this possible, and it was already in the plan.** Once a reading is
`reads(T, V)` with T a term, a reading *is* a term, and the space of readings is the space of compositions.
`lands on` stops being a predicate somebody wrote and becomes one composition among many:

```
lands on(self, x, y, grid, V)   is   reads(holding(grid, stepped(place(self, row), x),
                                                         stepped(place(self, column), y)), V)
```

Nothing about that composition is privileged. It is reachable by building from the primitives, and so are the
ones nobody thought of.

**Half of it is already built, on the other side of the project.** `ExpressionGenerator` is documented as
generating expressions *"for any domain from the positions of its rows and its players only; nothing here
knows a game"* — it reads a vocabulary off the positions, makes leaves, expands them by pattern, aggregate,
unary, threshold and combination, and the search keeps what pays. `TermEvaluator` prices each by what it costs
to read. `SparseFitter` and `ConstraintSelector` decide what earns its place. The heuristic half invents its
vocabulary and prices it; the constraint half is handed seven predicates. **The asymmetry is inside the
project, not between it and the literature.**

**Nothing has to be chosen as primitive, because the integrator already hands over the structures.** Maxime,
this session: *"the integrator uses datastructures from OMF so we should have all we need to read it"*. That
is right, and it is stronger than choosing a small set well. A game does not hand over pixels; it hands over
OMF's own types, and those types declare what can be read:

| the game declares | so a reading can say |
|---|---|
| `Grid.shape`, `Grid.cells` | what stands at a place, and how many dimensions a place has |
| `Grid.directions` | which steps mean something on this grid, rather than every offset |
| `Grid.aliases` | what the game calls a place |
| `Map.items`, `List.items`, `Scalar.value` | what is held at a key, in order, or on its own |
| `Record.parts`, `Kind.parts` | that a thing has parts, and what they are called |
| `Kind.values` | what a thing may be — the whole domain, not what has been seen |
| `Kind.nothing` | whether emptiness is one of the things it may be |
| `ActionKind.parameters` with their `Kind` | what an action names, and of what sort |

**So the reading vocabulary is generated from the schema, not written against a game.** A place is a
coordinate of a grid the game declared; a step is that coordinate moved by a number, and `directions` says
which moves are worth considering; a part is what `Kind.parts` says a value has; two things are comparable
when the game gave them the same `Kind`. Everything the seven hand-written predicates say is a composition of
what is in that table.

**Which means the hand-written readings are re-deriving what the schema already states.** `_alike_named`
compares the *names* of places to decide whether two readings are of the same sort — "self row" against "self
column" — when `Kind` declares it outright. `_distances` computes how far apart every pair of numbers is,
when `Grid.directions` says which steps the game considers meaningful. That is not a small saving: it is the
difference between a vocabulary that happens to suit chess and one a game generates about itself.

**And it answers the bootstrap.** A position can be read before anything has been learned about reading it,
because the schema says how. What is learned is not *whether* a place can be read but *which* compositions
are worth keeping — which is a price question, and the project already prices terms.

**What is hard, and none of it is solved here.**

- **Cost, and it is the same cost Cyc paid.** The flat readings were flat so the search could combine them
  cheaply, and 152 a case is already the expensive part. If a reading is a term to be discovered, the space
  is terms times candidates times the guard. The flattening does not disappear — it becomes a *cache of what
  was found worth reading*, which is a different and harder thing than a fixed list.
- ~~**The bootstrap.**~~ Answered above: the schema says how to read a position before anything is learned
  about reading it. What remains is narrower — the first compositions have to be cheap enough that a position
  can be read at all before any of them has earned its place.
- **What makes a reading worth having.** Today none of them has to earn anything. Learned, each needs a price
  and a payoff, and the project already has both shapes — description length on one side, dearness on the
  other — built for terms rather than for readings.

**The expressions stay too**, for the same reason: a heuristic is read millions of times and must compile to
arithmetic. But an `Aggregate` already records "its base, its kind, whether it reads pairs of indices, its
readings and the operations joining them" — it is a structured form waiting for its readings to be terms.

## Where it lives: a domain of its own, called `statement`

**Decided (Maxime).** The vocabulary is its own bounded context, and it is called `statement`.

**Its own, because today it is owned by nobody.** Terms, facts and clauses sit in `rule`; changes sit in
`world`; values sit in `structure`; and six domains — inference, predictor, heuristic, language, rbs, csp —
all speak it. A thing every context depends on and none owns is the definition of a missing one. Its
invariants are about *form*: a term ground or not, a clause's variables, one moment before another. Those
change for a different reason than "how is a rule kept and run" or "what is the case", which is the test.

**`statement` and not `claim`, because this domain has no epistemics and must not sound as though it does.**
Whether anybody is committed to what is said belongs to `knowledge`, which has beliefs, opinions, direct
experiences and evidence, and to `epistemology`, which has certainty and justification. A clause about to be
refuted is the same statement before and after; what changes is the claim on it. The project also keeps a
careful line between an opinion and a belief, and a third word of that family attached to a layer with none
of that in it would blur a distinction it has been precise about. `claim` is better left free for a domain
that does track who said what and whether they stand by it.

```
openmind/statement/model/term.py          what a statement is made of
openmind/statement/model/literal.py       one thing said
openmind/statement/model/clause.py        several, standing or falling together
openmind/statement/model/moment.py        when it is said to be the case
openmind/statement/model/change.py        a difference between two moments
openmind/statement/model/drawn.py         how a part is named from an action
openmind/statement/model/consequence.py   what an action makes the case
```

What the others keep, and both are more coherent for it: `rule` keeps the rule kinds and the caller, compiler
and runner — how a rule is kept and run. `world` keeps positions, actions, joint actions and players — what
is the case. `structure` keeps values, grids, records and coordinates — what things *are*, which is what
terms denote.

Afterwards `statement` depends on `structure` alone and everything else depends on `statement`.

**Corrected after doing it.** This section first claimed the move would fix a backwards arrow — `rule`
importing `world` six times because `Consequence` and `Drawn` lived in one and `Change` in the other. That was
wrong. All six are `State` and `Action`, and they stay: a rule is *run against a position*, so a rule caller
needs one. The dependency was proper all along. What the move actually buys is the first line of that
paragraph and nothing more, which is still the point — a shared thing owned by somebody.

**The cost is import churn and it is large**: counting both projects, `clause` is imported by 76 files,
`literal` by 65, `term` by 52, `consequence` by 17, `drawn` by 13, `change` by 13. Mechanical, and the sort of
move that happens in one commit or not at all.

## Where the vocabulary stops, for now

Chess hands over a position: a grid named `grid` holding `Piece(white, pawn)`. A video game hands over pixels,
and something has to produce the thing a term refers to. For a video game that something is a network, and
then the vocabulary has a seam in it.

**Nobody has shown one vocabulary spanning that seam.** OpenCog Hyperon is the most ambitious current claim,
and its own account is wrapping rather than unifying: the Atomspace is *"the primary meta-representational
hub"*, but there is *"a broader Space API that permits the creation of multiple specialized types of Spaces
within it"*, and neural integration is *"you could wrap a large language model or other deep neural networks
in an Atomspace API, and perform pattern matching against"* them.

**The ports are already the wrap, and that is the right call for now.** A heuristic reads a `Node` and returns
numbers; a network can fill that without speaking terms at all, and `ModelRecord` already has a family for
one. The seam only becomes unavoidable when perception is *learned* rather than handed over. So this plan
stops at the edge of it deliberately, and the boundary is written here rather than discovered later.

## Stages

**Stage zero — the domain exists. Done.** The moves above, no behaviour, one commit: six modules and two test
files moved, 109 files rewritten across both projects, both suites unchanged at 1464 and 63. `statement`
depends on `structure` alone; `world` depends on it once, where `Changer` applies a change; `rule` depends on
it six times.


Each is a component and goes through the usual two stops.

**One — `Drawn` becomes `Term`. Done.** The nine drawing classes are functor shapes, with builders and a
reader beside them. They unify; the consequence mapper drops its own name-to-class map, its own reading of a
dataclass's fields, and its own third route through the value mapper. Both suites unchanged at 1464 and 63.

Two things it found. **`Row` and `Column` do not fold into `Place`**, although both docstrings implied it:
`Place` takes the named part of whatever a parameter holds, while `Row` resolves a name the game gave a square
through the grid's own aliases. A game whose parameters hold records needs one and a game whose parameters
hold square names needs the other, so both stay. And **the drawings had three ways of being written down** —
this mapper's own, the term mapper's, and the value mapper's for a record inside one — which is the same
shape as question 36 one level down.

**Two — readings take terms.** `CandidateReadings` emits `reads(T, V)` where T is a term, and the flat
predicates become the projection. `lands on` and `holds` merge. Question 31's fold of the two reading stacks
falls out of this rather than being separate work.

**Three — moments.** A clause may be asserted at `now` or at `after(A)`. `Hypothetical`'s three questions are
rewritten as clauses over moments and its special-case machinery goes.

**Four — consequences become clauses.** `Consequence` is a clause at `after`, learned by the same learner. The
predictor and the constraint learner become one learner given two kinds of target.

**Five — aggregates over terms.** A heuristic term is an aggregate over a term rather than over Python source,
so the seeds the engine reasons out and the terms the search generates are the same objects.

## The risk this runs, named from the one system that ran it longest

**Cyc kept its universal representation and lost the ability to reason over it.** Lenat and Marcus, 2023: four
decades and 2000 person-years, and *"Cycorp's experiments with larger-sized teams generally showed a net
decrease in total productivity"* — it does not parallelise. Then footnote 9, in Lenat's own hand: *"We noticed
empirically that the general theorem-proving reasoner actually took so long that over a million queries in a
row that called on it, as a last resort, just timed out. Going back farther, we saw that that had happened for
decades…"*

**The usual moral drawn from this is the wrong one.** The claim that Lenat repudiated one maximally expressive
universal representation was put to adversarial verification and **refuted**: the same paper defends it —
*"only higher order logic can represent the same breadth of thought as a natural language"* — and presents
Cyc's speed as a success. What was abandoned in practice was the *general reasoner*, replaced by a thousand
specialised ones.

So the specific risk here is not that one vocabulary is too expressive to be true. It is that **one vocabulary
can be too expensive to reason over, quietly, for years**, while everything that works is a special case
beside it. This plan's answer is the flattening: the searchable readings stay, derived from the terms rather
than written beside them, and the inner loop never resolves a term. That answer is now load-bearing rather
than a convenience, and it is the thing to measure first.

## What it costs

- **Unification gets dearer.** A variable may stand for a composed place, so the search space grows. The
  flattening keeps it out of the inner loop, but the boundary has to be watched.
- **`Row` and `Column` disappear** into `Place`, which its own docstring already asks for.
- **Everything touches it.** Eight files speak one of these today and all of them change.
- **It can be slower and still be right**, which is the dangerous shape. The measurement has to be the same
  one used all along: positions learned per hour, and what the constraints leave standing.

## Verification

| Stage | Green when |
|---|---|
| one | Every drawing round-trips as a term and back; the predictor learns what it learned before |
| two | A reading and the term it flattens agree on every candidate of a position, mechanically |
| three | `taken after` is deleted and king safety is still expressible, as a clause |
| four | One learner, given the same sightings, produces the consequences the predictor produced |
| five | A seeded heuristic term and a generated one are the same object |

**The claim, measured:** a thing learned by one part is available to another without being learned again. The
concrete case to hold it to is the one that cost a day — a predictor that says a capture one way and a
constraint learner that reads it another cannot disagree, because there is no second reading.

## Open questions

- **Whether a moment is a modality or an argument.** `at(W): reads(T, V)` as a prefix on a clause, or
  `reads_at(W, T, V)` as an argument. The literature has taken both: the situation calculus and the event
  calculus both make it an argument — `holds(f, do(a, s))`, `HoldsAt(f, t)` — while the Game Description
  Language makes it a predicate, `true` against `next`, and forbids each from appearing where the other
  belongs. The predicate form stays within Datalog and therefore stays decidable; the argument form is the
  one that survives a game with no turns. **Since real time is coming, the argument form is what this plan
  takes**, and staying decidable becomes a restriction on the *clauses* rather than on the language.
- **How the frame is kept, now that there is a frame to keep.** Saying differences rather than whole states is
  what avoids writing frame axioms, and it is why `change.py` exists. But once a consequence is a clause at a
  moment, something has to say what did *not* change. Reiter's answer — one statement per fact, quantified
  over every action — is a thing to compile rather than a thing to learn. **Open: whether it is compiled, or
  differences stay primitive and what does not change is never formed at all.** The second is cheaper and is
  what is done today; the first is what makes the rules provable rather than only runnable.
- **Whether the four kinds of change survive at all.** They could be pure shorthand for differences between
  moments, or they could stay as what a predictor draws in. Keeping them keeps question 36's shape alive in
  miniature. The event calculus suggests the answer: `Placed` and `Removed` are `Initiates` and `Terminates`
  and earn their place; `Moved` is the two of them together and is the one that caused question 36; `Told` is
  a scalar and may be either.
- **What happens to `ActionReadings`.** Question 31 decided the two reading stacks fold into one. This makes
  that fold a consequence rather than a task, but the two have diverged far enough that "the same readings" is
  still a claim to check.
- **Whether this is affordable.** Nothing here has been measured. The only honest answer is stage one, then the
  same numbers the arms already print. Cyc says what failure looks like: not a wrong representation, a
  representation nothing can afford to reason over.
- **How far the dialect is cut down, and said out loud.** The one published system that learns in the event
  calculus fixed a simplified discrete dialect to make learning tractable. This plan should name its
  restriction deliberately — which terms may nest, whether a moment may be a variable, whether an event may
  be quantified over — rather than arrive at one by finding out what is too slow.
- **~~Whether the readings stay designed or become learned.~~ Decided: learned.** What remains open under it
  is which four or five things stay primitive, how a position is read before anything has been learned about
  reading it, and what a reading has to earn to be kept.
- **What is assumed about supervision.** The published result assumes it noise-free and complete. A learner
  watching a game gets neither: it sees the moves a game listed and must take everything else as refused,
  which is exactly what `Evidence` already says. So one of the two assumptions that bought that correctness
  guarantee is already given up here, and nothing yet says what is lost with it.
