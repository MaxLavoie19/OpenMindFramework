# Interfaces: the inference engine

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

The engine replaces the stand-in in `inference`: `Inferrer`, `RuleReasoner`, `RuleDeducer`, `CoveringLearner` and
`HeuristicProposer`. It must support every feature the stand-in stood in for — the inventory at the end is the
acceptance checklist.

**What it is for.** Deduction, chaining and learning are the machinery; **generating heuristics is the point**. An
agent handed a game and no advice has to work out for itself what a good position looks like, and the only thing it
has to go on at the start is the rules. So the engine reads the rules and reasons to what they imply about value —
what a thing is worth from what its rules afford, how far a win is once a constraint is lifted, which principles
hold across games — and hands those over as the agent's first heuristics. Everything else here exists to make that
possible.

**Where the engine stops and the search starts.** The engine generates *the logic that applies*: which rules bear
on a position and what shape the heuristic has. Where the rules imply a number it supplies that too — a piece's
value from the squares its rules cover is reasoned, not guessed, and it is a far better place to begin than zero.
But it is a beginning. The search does the expanding and the distilling, and it is what tunes the trainable
parameters: the weight each rule carries in a context, the specific values. The engine says a rook is worth about
fourteen because fourteen squares is what a rook's rules reach; the games say what a rook is really worth.

## The engine across OMF's life

The interfaces below are answerable to this. Each phase is the same machinery — induce clauses, chain them, weigh
them by how often they held — pointed at different examples.

1. **An engine is connected.** An integrator plugs in a game engine and says nothing about the rules. OMF plays at
   random and asks it what happened.
2. **The rules are induced from that play.** Two kinds, and both are clauses:
   - **legality** — what makes a move one that may be played;
   - **prediction** — what a move leads to from a state, and **with what probability**. This is why clauses carry
     probabilities. `ConsequenceLearner` today learns what an action does and its conditions, and has no way to say
     that an outcome happens four times in five; a game with chance in it cannot be modelled at all.
3. **Bootstrap heuristics are inferred** for move value and position value, with starting values where the rules
   imply them — and for **two aims at once**: *optimal* play, which tries to win, and *predictive* play, which says
   what a player will actually do. They are not the same question and they part company immediately: the move that
   wins and the move a club player finds are different moves.
4. **The search strategizes with them**, and at the end of **every** game the tunable parameters are updated. Not
   in rounds — games run continuously and each one's end informs the next.
5. **A player's style is induced** when one is worth modelling: their games are the examples, and what comes out is
   the set of clauses that best explains how they play — predictive heuristics for that player, in that player's
   context. An agent model is a ruleset like any other; `task_constant.py` already says so, and this is what fills
   it.
6. **Play keeps correcting the model.** Three things can be discovered, and each is a different repair:
   - a move we did not know we could play — the legality clauses are too narrow;
   - a move we did not know we could *not* play — a legality clause is too broad, and how long it had stood says
     whether it was a guess or a rare rule of the game finally met;
   - an outcome we did not predict, or one whose probability was wrong — the prediction clauses are incomplete, or
     their counts are.

   Each re-opens the game model, and the same three re-open a player model when it is a player being predicted.
7. **What was learned here is offered elsewhere.** A clause stated wholly in generic terms is a principle and can
   be tried in another game — a fork is possible in tic-tac-toe and in chess, and OMF should learn it once. It
   arrives as a candidate with a weight per context, never as something already true.
8. **A black box is explained.** A network may predict a player better than any ruleset OMF can train. The engine
   then learns an explainable model *from the network*: the network is asked, its answers are the examples, and
   clauses are induced from them. Nothing about the learner changes — only where the examples come from.
9. **Rules are shared in words.** A clause is structured, so it can be encoded to natural language by a language
   model, and eventually decoded back. The same holds for what OMF infers rather than what it rules: the RBS
   offers structured information, and the codec takes it out to words and brings words back in. The engine's part
   is to keep everything it concludes in a form that *has* a structure — which is what a clause is and what a
   fitted Python expression is not.

Phases 1 to 7 are this rework. Phases 8 and 9 are not built here; what is built here is the interface that makes
them possible — one learner whose examples come from a port, and one conclusion form that is structured.

## Decided so far (2026-09-21)

- **Generating heuristics is the main purpose.** The engine derives them by reasoning from the game's rules.
- **Two aims, and the same tasks.** A heuristic aims at *optimal* play — maximise the win — or at *predictive*
  play — what this player will do. No new task is added: `task_constant.py` already settles that an agent model is
  the position-value and move-value heuristics trained to predict that agent's play, in that agent's context. What
  the engine adds is saying which aim a heuristic was derived for, since the two are fitted against different
  targets and must never be fitted against each other's.
- **The engine generates the logic that applies, and bootstraps the values it can reason out.** Which rules bear on
  a position, what form the heuristic takes — and, where the rules imply a number, that number: a piece's value
  from the squares it covers, before a single game is played.
- **The search expands and distils: it tunes the trainable parameters** — the weight per rule, the specific values.
  `ExpressionSearch`, `ValueGenerator` and `ValueDistiller` are not stand-ins and stay. What changes is where they
  start: from what the rules imply rather than from zero. `HeuristicProposer`'s random sampling goes.
- **Build fresh.** The z3 layer deleted by commit `724860b` is not restored.
- **One mechanism: resolution.** Resolving a rule against facts gives a new fact; two rules give a new rule; a
  denied goal gives a proof; the refinement inverted gives learning; a counter-example is a contradiction.
- **General clauses** — negation normal form, Skolemization, conjunctive normal form — with a definite-clause fast
  path. Learning works in the definite fragment.
- **Z3 is a backend**, for goals with arithmetic and for mathematical induction. It cannot chain rules into rules
  or learn, so it is never the engine.
- **Learning is bottom-up generalisation** (changed 2026-09-21, after top-down was built and measured). A case
  said outright is a clause covering it; two cases are generalised by anti-unification, which keeps the terms they
  agree on and puts a variable where they differ, the same variable for the same differing pair throughout. It is
  generalised against further cases while nothing that should be refused slips through.

  I had advised top-down on the grounds that bottom-up needs mode declarations. **That was wrong here**: modes bound
  a bottom clause built by inverting entailment against a background theory, and there is no such clause to bound
  when every case already *is* a finite set of ground literals. Nothing is declared.

  Top-down was built, and on a real game it could not run: thousands of candidate literals scored against tens of
  thousands of cases, a hundred million matches for one refinement step. Six passes at the cost bought a factor of
  1.4 against the hundredfold needed. Bottom-up removes the search rather than making it cheaper — the cases say
  what the clause is — and the clause tying a thing's side to the acting player's, which top-down needed a special
  pairing rule to reach at all, falls out of one anti-unification.

  The point worth keeping: **anti-unification is only possible because a reading has arguments.** A flattened
  reading name is atomic and there is nothing in it to generalise. The change that made the engine slow is the same
  change that made the fast method available, and top-down was the old representation's algorithm kept on past its
  representation.
- **Learning carries across positions.** `learn(..., starting=)` takes what is already believed: a case already
  accounted for is passed over, and a case that is not is first tried as a *widening* of something already held
  before anything new is begun. So an agent learns by playing — each position asks only what it did not know — and
  one rule stays one rule instead of being rediscovered per position.
- **A clause carries a probability.** A conclusion's chance is computed over *every* proof of it, accounting for
  the premises those proofs share.
- **The engine computes the chance; epistemology judges it.** What crosses is one piece of evidence.
- **Statistics are estimation and measurement**, never hypothesis tests: what keeps a rule is utility (2026-09-17).
- **Clause models live in `rule/model/`**, beside `PythonRule`, so a clause can be a rule the knowledge base holds
  without `rule` having to depend on `inference`.
- **Nothing the stand-in guessed at is carried over as a number.** `Surprise.rare`'s 100, `grow`, `positions`,
  `depth`: each is learned, or is a budget the caller sets. None is a constant in the engine.
- **Sorts are optional.** `Variable.sort` is a string a game need not fill, and the engine does not insist: where a
  game says nothing, the readings keep a player and a square apart by being different predicates.
- **Four services move in from `epistemology`**: `Justifier`, `CoherenceChecker`, `AccuracyScorer`, and the
  certainty models. Epistemology keeps the doctrine.
- **Layering**: `rule` → `knowledge` → `inference` → `epistemology`.

## Change to `rule`

A clause is a kind of rule body, so it lives beside `PythonRule` and joins the `Rule` union.

```python
# rule/model/term.py
@dataclass(frozen=True, slots=True)
class Variable:
    """A place in a clause that stands for anything: what lets one rule say what many rules say now."""
    name: str
    sort: str = ""              # "" where the game says nothing about what kind of thing it is


@dataclass(frozen=True, slots=True)
class Constant:
    """A named thing: a player, a piece kind, a square."""
    name: Value


@dataclass(frozen=True, slots=True)
class Number:
    value: int | float


@dataclass(frozen=True, slots=True)
class Functor:
    """A function applied to terms, such as the cell one step on from another."""
    name: str
    arguments: tuple["Term", ...]


#: What a literal is said of.
type Term = Variable | Constant | Number | Functor
```

```python
# rule/model/literal.py
@dataclass(frozen=True, slots=True)
class Literal:
    """One thing said of some terms, or its denial.

    The predicate is a reading's name with its slots opened up: what `ActionReadings` writes as the single key
    "rows from source to target" is `rows(source, target, N)` here, so the source, the target and the number can
    each be a variable. That is the whole difference between this and the conditions it replaces."""

    predicate: str
    arguments: tuple[Term, ...]
    negated: bool = False

    @property
    def ground(self) -> bool:
        """Whether nothing in it is a variable."""
```

```python
# rule/model/clause.py
@dataclass(frozen=True, slots=True)
class Clause:
    """A disjunction of literals: everything the engine reasons with is one of these.

    A fact is a clause of one positive literal with no variables. A rule is a clause with one positive literal and
    some negative ones, read as "the positive holds where all the negatives do". A goal is a clause with no
    positive literal. A contradiction is the clause with no literals at all, and deriving it is what a proof is.

    `probability` is how often the rule holds where its body does, 1.0 for a rule that simply holds. A rule with a
    probability below 1 is read as the rule plus one independent chance of its own firing, so there is one
    semantics and not two.

    `name` is what it is called where it has been named, for logs and for the knowledge base."""

    literals: tuple[Literal, ...]
    probability: float = 1.0
    name: str = ""

    @property
    def head(self) -> Literal | None:
        """The one positive literal where the clause is definite; None for a goal, and where there are several."""

    @property
    def body(self) -> tuple[Literal, ...]:
        """The negative literals, as the conditions they read as."""

    @property
    def definite(self) -> bool:
        """Whether exactly one literal is positive: the fragment the fast path and the learner work in."""

    @property
    def empty(self) -> bool:
        """The contradiction."""

    @property
    def readable(self) -> str:
        """`worth_at_least(Thing, N) :- reaches(Thing, N)`, or with `0.97::` in front where it has a probability."""
```

```python
# rule/model/clause_rule.py
@dataclass(frozen=True, slots=True)
class ClauseRule:
    """A rule the engine can reason with, rather than only run.

    A rule kept as Python is opaque the moment it is stored: it can be called and nothing more. Kept as a clause it
    can be resolved against other rules, subsumed, challenged and revised, and `ClauseCompiler` gives an RBS the
    Python it runs when it needs it."""

    clause: Clause


# rule/model/rule.py — the union gains it
type Rule = PythonRule | ClauseRule | Callable[..., object]
```

`knowledge/mapper/rule_record_json_mapper.py` maps a `ClauseRule` to and from JSON. `RuleRecord.probability`
already exists for an effects rule's outcomes; a clause's probability is the same notion and is written there, not
in a second field beside it.

## Change to `predictor`

Prediction is half of what is induced from random play, so this is not a rewiring note.

`Consequence.when` is `tuple[Covering, ...]` today and becomes `tuple[Clause, ...]`. `ConsequenceLearner` keeps
what it does — asking where each part of each change could have come from, so what an action does is said in terms
of the action and carries to the next position — and delegates the conditions to `ClauseLearner` as it already
delegates them to `CoveringLearner`.

What it gains is the probability. A consequence today happens under its conditions or does not, which is enough
for chess and cannot state a game with chance in it at all: an outcome that comes about four times in five has
nowhere to be written down. With a probability on the clause it has, `ChanceFitter` fits it from the counts, and
`ClauseChallenger.mispredicted` can then tell a missing rule from a wrong rate — which are different problems with
different repairs, and are indistinguishable while everything is certain.

## `inference`

It depends on `rule`, `knowledge`, `world` and `structure`.

### Models

```python
# model/formula.py — for stating things; clauses are what they become
type Formula = Atom | Not | And | Or | Implies | Iff | ForAll | Exists | Truth
```

```python
# model/substitution.py
@dataclass(frozen=True, slots=True)
class Substitution:
    """What variables were bound to, on the way to a conclusion."""
    bindings: tuple[tuple[Variable, Term], ...] = ()

    def of(self, variable: Variable) -> Term | None: ...
    def applied(self, node: Term | Literal | Clause) -> Term | Literal | Clause: ...
    def then(self, other: "Substitution") -> "Substitution": ...
```

```python
# model/signature.py
@dataclass(frozen=True, slots=True)
class Signature:
    """What can be said in a context: each predicate by name and arity, which of them are computed rather than
    derived, and what each argument may hold.

    The learner's candidate literals come from here and from nowhere else, so what OMF can learn to say is what the
    game's own readings let it say."""

    arities: tuple[tuple[str, int], ...]
    evaluable: tuple[str, ...] = ()
    values: tuple[tuple[str, int, tuple[Value, ...]], ...] = ()    # predicate, position, the values seen there
    numeric: tuple[tuple[str, int], ...] = ()
```

```python
# model/derivation_step.py
@dataclass(frozen=True, slots=True)
class DerivationStep:
    """One step: its number, the rule applied as the engine names it (`given`, `resolved`, `factored`, `evaluated`,
    `induction`), the steps it uses, what it concludes, and what was bound to get there."""
    number: int
    rule: str
    premises: tuple[int, ...]
    clause: Clause
    substitution: Substitution = Substitution()


# model/derivation.py
@dataclass(frozen=True, slots=True)
class Derivation:
    """One way a conclusion was reached, readable back to the clauses it started from.

    `chances` are the probabilistic clauses it used, which is what a chance is computed over: two derivations
    naming the same one are not independent, and that is the thing a stand-in gets wrong."""

    conclusion: Clause
    steps: tuple[DerivationStep, ...]
    chances: tuple[str, ...] = ()

    @property
    def rests_on(self) -> tuple[str, ...]:
        """The names of the clauses it was given, in the order they were used."""
```

```python
# model/answer.py
#: What the engine can say. It never guesses: where the clauses don't decide, it says so and why.
PROVED, DISPROVED, UNKNOWN = "proved", "disproved", "unknown"


@dataclass(frozen=True, slots=True)
class Answer:
    """What asking a goal came to: every derivation found within the budget, not the first.

    A chance needs all of them, and so does knowing whether a conclusion has one reason or several. `reason` says
    why an unknown is unknown — the budget ran out, or the readings cannot tell these apart, which is the case to
    go and look into rather than a failure."""

    goal: Clause
    status: str
    derivations: tuple[Derivation, ...] = ()
    chance: "Chance | None" = None
    seconds: float = 0.0
    reason: str | None = None
```

```python
# model/chance.py
@dataclass(frozen=True, slots=True)
class Chance:
    """How likely a conclusion is, and how it was worked out.

    `spread` is how much the count behind it supports it, so a rule fitted on six examples is not mistaken for one
    fitted on six thousand. `exact` says whether the proofs were counted exactly or sampled past the budget."""

    value: float
    spread: float = 0.0
    derivations: int = 0
    exact: bool = True
```

```python
# model/inference_budget.py
@dataclass(frozen=True, slots=True)
class InferenceBudget:
    """What one question may spend. Nothing here is a cap the engine set itself: every one is the caller's.

    `derivations` is how many proofs to collect before stopping, `nodes` how large the decision diagram may grow
    before the chance is sampled instead of counted."""

    seconds: float
    steps: int | None = None
    depth: int | None = None
    derivations: int | None = None
    nodes: int = 100_000
```

```python
# model/example.py
@dataclass(frozen=True, slots=True)
class Example:
    """One thing that was so, or wasn't: what the learner learns from."""
    literal: Literal
    holds: bool
    where: object | None = None         # the position it was read in, for support across positions
```

`model/surprise.py` stays as it is: a rule that held until it didn't, how long it stood, and the position kept to
test later rules against. **Open:** `Surprise.rare`'s threshold of 100 is hardcoded; it should be a caller's
budget or learned, per "constants are learned or inferred, not hardcoded".

### Ports

```python
# model/prover.py
class Prover(Protocol):
    """Answers a goal from a set of clauses, within a budget. The native resolver fills it; so does the Z3 backend,
    for goals it decides and resolution doesn't."""

    def ask(self, clauses: Sequence[Clause], goal: Clause, budget: InferenceBudget) -> Answer: ...


# model/anchoring.py
class Anchoring(Protocol):
    """What counts as something support can rest on. Epistemology's doctrine, injected, so the engine traces
    support without holding a position on what knowledge is founded on."""

    def anchor(self, knowledge: KnowledgeBase, entry_id: str) -> bool: ...


# model/examples_source.py
class ExamplesSource(Protocol):
    """Where the things to learn from come from.

    One port, because the learning never changes and only the evidence does. Playing at random and asking the
    engine what happened gives the rules of the game. A named player's games give how that player plays. A trained
    network, asked and asked again, gives what it would say — and clauses induced from its answers are an
    explainable model of a thing that cannot explain itself.

    A source that can be asked about a case of OMF's choosing, rather than only handing over what it has, answers
    `probe`: that is what turns a challenge into an experiment instead of a wait."""

    def examples(self, about: str, most: int | None = None) -> tuple[Example, ...]: ...

    def probe(self, goal: Clause) -> tuple[Example, ...]:
        """Cases meeting that goal, where the source can be asked for them; empty where it cannot."""
```

### Services — the language

```python
class Unifier:
    """What two literals have to agree on to be the same literal, or None where they can't be.

    The occurs check is done: a variable is never bound to a term containing it, which is the difference between a
    prover and a prover that loops."""

    def unify(self, one: Literal, other: Literal) -> Substitution | None: ...
    def unify_terms(self, one: Term, other: Term, so_far: Substitution = Substitution()) -> Substitution | None: ...


class Clausifier:
    """A formula as the clauses that say the same thing: negation pushed in, existentials Skolemized, the rest
    distributed. This is how anything anyone writes gets into the one form the engine reasons with."""

    def clauses(self, formula: Formula, name: str = "") -> tuple[Clause, ...]: ...


class Subsumer:
    """Whether one clause says everything another says, so the other adds nothing.

    This is the first-order lift of the stand-in's `entails`, and it does one thing more. Plain subsumption cannot
    see that "worth at least 8" makes "worth at least 5" pointless, because neither is an instance of the other. So
    it consults the evaluable orderings, and dominance is caught — which is what `Inferrer._held` was doing by hand."""

    def __init__(self, unifier: Unifier, evaluable: EvaluablePredicates) -> None: ...

    def subsumes(self, one: Clause, other: Clause) -> bool: ...
    def within(self, clauses: Sequence[Clause], others: Sequence[Clause]) -> bool: ...
    def ordered(self, by: Mapping[Value, Sequence[Clause]]) -> tuple[tuple[Value, Value], ...]:
        """Which of those things affords at least what another does, concluded from their clauses alone: the queen
        and the rook, before a position is looked at."""
    def compatible(self, one: Clause, other: Clause) -> bool:
        """Whether the two bodies can hold of the same thing."""
```

### Services — deduction and chaining

```python
class EvaluablePredicates:
    """The literals the engine answers by computing rather than by resolving.

    Arithmetic and the orderings are here because every game has numbers in it and deriving `3 < 4` a step at a
    time would be absurd. They hold for any game whatever, which is what lets them be built in.

    Anything narrower is **registered, not built in**. `reaching` — how far a set of rules reaches over a grid of
    a given shape — is carried over from the stand-in unchanged in what it computes, and it is registered by
    whatever knows the game has a grid. A game with no grid never sees the predicate, and nothing in the engine
    asks whether there is one. This is the rule for everything of the sort: the engine holds what is true of all
    games, and a game's own shape arrives as a registration."""

    def evaluable(self, predicate: str) -> bool: ...
    def holds(self, literal: Literal) -> bool | None:
        """Whether it holds, or None where it isn't ground enough to say."""
    def dominates(self, one: Literal, other: Literal) -> bool:
        """Whether holding the first makes the second add nothing."""
    def register(self, predicate: str, arity: int, answering: Callable[..., object],
                 ordering: str = "") -> "EvaluablePredicates":
        """The same predicates with one more, for a game that has something worth computing.

        It gives back a new one rather than changing this one: the services are stateless and are handed what they
        run, so a game's own predicates travel with the game and never leak into another's reasoning."""


class Resolver:
    """One step: two clauses sharing a literal one affirms and the other denies give a third, with what had to be
    bound. Factoring collapses a clause's own duplicate literals, which a general clause set needs and a definite
    one never does.

    Every conclusion the engine reaches is some number of these."""

    def resolve(self, one: Clause, other: Clause) -> tuple[tuple[Clause, Substitution], ...]: ...
    def factors(self, clause: Clause) -> tuple[Clause, ...]: ...


class ForwardChainer:
    """Everything that follows from those clauses, until nothing further does.

    This is what replaces `Inferrer.chain`, and the difference is where the reasoning lives. The stand-in's
    `_further` was eight kinds of conclusion written as Python branches; here the kinds of conclusion are clauses
    like any other, so OMF concludes something new by being given or by learning a rule, not by being edited.

    A clause another already subsumes is not kept, which is `_held`'s dominance test done properly."""

    def chain(self, clauses: Sequence[Clause], budget: InferenceBudget) -> tuple[Derivation, ...]: ...


class GoalProver:
    """What the clauses say about a goal, by denying it and looking for a contradiction.

    It collects **every** derivation within the budget rather than stopping at the first. A chance cannot be worked
    out from one proof, and neither can whether a conclusion has more than one reason behind it — which is what
    makes a conclusion worth more than its best single support.

    Goal-driven, since there are infinitely many things that follow from a set of rules and only some of them are
    wanted: the set of support keeps the search on the question asked. Deepening a step at a time, so a shallow
    answer is never missed behind a deep one."""

    def __init__(self, resolver: Resolver, subsumer: Subsumer, evaluable: EvaluablePredicates,
                 proof_weigher: ProofWeigher) -> None: ...

    def ask(self, clauses: Sequence[Clause], goal: Clause, budget: InferenceBudget) -> Answer: ...
```

### Services — probability

```python
class ProofWeigher:
    """How likely a conclusion is, from every proof of it.

    Two proofs of the same thing are not two independent reasons when they lean on the same probabilistic clause,
    and multiplying them as though they were is the mistake this exists to avoid. So the proofs are turned into a
    formula over the clauses they used, that formula is compiled to a decision diagram — which is what makes the
    shared parts shared — and the diagram is weighted from the bottom up.

    Exact while the diagram stays within the budget's nodes. Past it the choices are sampled instead, and the
    answer says which it was rather than pretending."""

    def __init__(self, justifier: Justifier) -> None: ...

    def chance(self, derivations: Sequence[Derivation], budget: InferenceBudget) -> Chance: ...


class ChanceFitter:
    """A rule's probability from how often it held, as a Beta–Bernoulli.

    The mean is the probability and the spread says how much the counting supports it, so six examples and six
    thousand do not come out looking alike. Nothing is kept or dropped on a threshold: what a rule is worth is
    decided by utility, not by whether it clears a bar."""

    def fit(self, clause: Clause, examples: Sequence[Example]) -> Chance: ...
```

### Services — learning and the discovery loop

```python
class ReadingLiterals:
    """The game's readings as literals, with their name templates opened up.

    `ActionReadings` names a reading by filling a template — "rows from {first} to {second}" — and then uses the
    filled string as a key, which throws away the very thing that makes it general. Here the slots become the
    literal's arguments, so a rule can quantify over them. This is why one clause can say what a pawn does for
    both players, where the stand-in needed one rule per colour."""

    def literals(self, readings: Mapping[str, Value]) -> tuple[Literal, ...]: ...
    def signature(self, readings: Sequence[Mapping[str, Value]]) -> Signature: ...


class ClauseLearner:
    """The rules that account for what was seen, learned one at a time.

    Being legal is not one thing — a knight's move is legal for reasons a pawn's is not — so a clause is grown to
    cover some of the examples rather than all, and another is grown for what is left. Each is grown from the top:
    start from the clause that says nothing and add the literal that best tells the positive examples from the
    negative, taking its variables from those already bound.

    Everything the stand-in learned to do carries over, and each of these is a thing it took work to get right:
    a clause covering examples another already covers is still worth keeping; the conditions a family of clauses
    shares are that family's principle, said once with the specializations under it; what forbids a move belongs to
    the clause it was learned under and never reaches a sibling; and a literal that turns away examples the game
    allows is an artefact of the evidence, not a reason, and comes back out.

    **It learns four things and is one service.** What makes a move legal; what a move leads to and how often,
    which is the predictor's business and needs the probability; how a particular player plays, from their games,
    which is predictive rather than optimal play; and what a black box would say, from asking it. The clauses
    differ in what their heads are about and in nothing else, so the examples come from a port and the learner
    never knows which of the four it is doing."""

    def learn(self, examples: Sequence[Example], signature: Signature, budget: InferenceBudget,
              least: int = 1, loosely: float = 0.0) -> tuple[Clause, ...]: ...
    def distilled(self, clauses: Sequence[Clause], examples: Sequence[Example] = (),
                  least: int = 2) -> tuple[Clause, ...]:
        """The clauses restated as principles, with what specializes each one under it."""
    def excluding(self, clauses: Sequence[Clause], examples: Sequence[Example]) -> tuple[Clause, ...]:
        """Each clause again, carrying what rules out the negative examples it covers."""
    def relaxed(self, clauses: Sequence[Clause], examples: Sequence[Example],
                loosely: float = 0.0) -> tuple[Clause, ...]:
        """The clauses with every literal dropped that turns away examples that hold."""
    def uncovered(self, clauses: Sequence[Clause], examples: Sequence[Example],
                  by: str) -> dict[Value, int]:
        """What no clause covers, counted by what that predicate says of it: where to look next."""
    def conjunction(self, clauses: Sequence[Clause], examples: Sequence[Example],
                    most: int | None = None, cost: float = 0.0) -> tuple[Clause, ...]:
        """The few clauses that together tell the positive from the negative."""


class ClauseChallenger:
    """What would break these clauses, and what happened when something did.

    A learner that only ever sees what a game happens to offer learns what the game happens to offer. So each
    clause's own literals are turned into goals — find the thing that meets every other literal and fails this one
    — and each goal is a position to go and build rather than one to wait for.

    When one is found, how long the clause had stood decides what it means. Broken by the fifth example it met, it
    was a guess. Broken by the three-thousandth it is a rule of the game that almost never comes up, and meeting it
    is the most informative thing that has happened — so the position is kept, and every clause learned afterwards
    is made to face it."""

    def challenges(self, clauses: Sequence[Clause]) -> tuple[tuple[Clause, Literal], ...]:
        """Each clause with each of its own literals: what to look for an exception to."""
    def challenged(self, clauses: Sequence[Clause],
                   examples: Sequence[Example]) -> dict[tuple[Literal, bool], int]:
        """What each literal costs: what it wrongly turns away, and what it fails to turn away."""
    def refuted(self, clauses: Sequence[Clause], examples: Sequence[Example],
                stood: Mapping[str, int] = ()) -> tuple[tuple[Clause, ...], tuple[Surprise, ...]]:
        """The clauses that survived, and what broke the rest."""
    def unexplained(self, clauses: Sequence[Clause],
                    examples: Sequence[Example]) -> tuple[Example, ...]:
        """The negative examples no clause rules out. Either a reading is missing, or what rules them out isn't
        about the thing being read at all — and saying so is the point. Logs a warning."""
    def unforeseen(self, considered: Sequence[Literal], held: Sequence[Literal]) -> tuple[Literal, ...]:
        """What turned out to be so and was never among the candidates: what OMF did not know could be done."""

    def mispredicted(self, clauses: Sequence[Clause], seen: Sequence[Example],
                     budget: InferenceBudget) -> tuple["Misprediction", ...]:
        """Where what happened and what was expected to happen part company.

        The third way the model can be wrong, and the one only probabilities let OMF notice. A move that turns out
        illegal refutes a clause outright and a move nobody knew about is plainly missing — but an outcome is not
        like that. It can be one no clause predicted at all, which is a prediction clause missing; or it can be one
        that was predicted at the wrong rate, which is nothing broken and a count to correct. Told apart, the first
        is a discovery and the second is arithmetic; run together, a game with chance in it looks like a game whose
        rules keep breaking."""
```

```python
# model/misprediction.py
@dataclass(frozen=True, slots=True)
class Misprediction:
    """An outcome that didn't match what was expected, and which kind of not-matching it was.

    `expected` is the chance the clauses gave it and `held` how often it actually came about. `unforeseen` marks
    the outcome nothing predicted at all — where no clause is wrong, one is missing."""

    outcome: Literal
    expected: Chance | None
    held: Chance
    unforeseen: bool = False
    where: object | None = None
```

`LegalityTree` stays, ported to clauses. Its three-valued leaf — allowed, refused, **unknown** — is kept: a tree
that cannot tell two cases apart says so, and that is the case to go and look into. It is a different cost profile
from refinement, not a worse one: one pass per level rather than a search over hypotheses.

### Services — heuristics, which is what the rest is for

```python
# model/parameter.py
@dataclass(frozen=True, slots=True)
class Parameter:
    """A number in a heuristic that the engine reasoned a starting value for, and that the search may move.

    A rook's rules reach fourteen squares. That is not what a rook is worth — it is where to start looking for what
    a rook is worth, and it beats starting at zero or at a number somebody typed in. The engine says fourteen
    because the rules say fourteen; the games say what it should be.

    `holds` names the ground clause carrying the value, so tuning the parameter is revising that clause and the
    revision stays readable as logic rather than disappearing into a weight vector."""

    name: str
    initial: float
    holds: str = ""


# model/heuristic.py
#: What a heuristic is trying to do. Optimal play maximises the win; predictive play says what a player will
#: actually do. Both are valuing a position or a move, and they are not the same valuation.
OPTIMAL, PREDICTIVE = "optimal", "predictive"


@dataclass(frozen=True, slots=True)
class Heuristic:
    """A way of valuing a position or a move, reasoned out of the rules rather than guessed at.

    `clause` is the logic: a head putting a number on a state for a player, and a body of what has to hold. That is
    the engine's real contribution — *which rules apply*, in a form that can be read, resolved against and argued
    with. `kind` is the rule kind it becomes in the knowledge base: a position's value, a move's, a policy's optimum.

    `aim` is what it is for, and it decides what the heuristic is fitted against — which is the whole of the
    difference between the two. Optimal play is fitted on what the game paid: a position is worth what the game
    finally gave, a move is worth the share of the search's visits it took. Predictive play is fitted on what was
    actually played: a move is worth how often this player chooses it, a position what this player appears to make
    of it. Fitting one against the other's target gives a model that is neither. `holder` is whose play it
    predicts, empty for optimal play, which is nobody's in particular.

    `parameters` are the numbers the engine could reason out, each with where it came from. They are starting
    points handed to the search, never answers: what a rule is finally worth in a context is fitted there.

    `derivation` is why it was proposed. A heuristic that plays badly can then be read back to the reasoning that
    suggested it, and the reasoning argued with — which is the thing a tuned number can never be."""

    clause: Clause
    kind: str
    derivation: Derivation
    aim: str = OPTIMAL
    holder: Value | None = None
    parameters: tuple[Parameter, ...] = ()
    rests_on: tuple[str, ...] = ()


class HeuristicDeriver:
    """The agent's first heuristics, reasoned from the game's own rules.

    This is what the engine is for. A new game arrives with its rules and nothing else — no games played, no
    positions valued, nothing to fit against — and the fitting that OMF does well needs targets it does not yet
    have. What it does have is the rules, and the rules say what things are for.

    It derives for both aims. **Optimal** play comes out of the three ways below, which read the rules and say what
    they imply about winning. **Predictive** play — what a given player will actually do — comes out of their games
    where there are any, and before that out of optimal play held separately and marked as the projection it is.

    Three ways to the first, and none of them is a guess:

    - **What a thing affords.** A thing is worth how much of what its owner can do goes away without it. That is
      said without a board, without turns and without an opponent, so it means something in every game: in sudoku
      it is how many placements a filled cell rules out, in the prisoner's dilemma it is nothing, because nothing
      on the table is owned. Where a game *does* lay things out on a grid there is a cheaper way to the same
      answer — count what the rules admit from a square without playing at all, which is what `reaching` does —
      and it is an optimisation available to games that have a grid, never a step the derivation depends on.
      Chained, the afforded amounts order themselves: one thing whose rules take in another's affords at least
      what it does.
    - **What a relaxed game says.** Lift a constraint and the game gets easier; how far a win is in the easier game
      is an estimate of how far it is in the real one, and it can be computed where the real distance cannot.
      `GameRelaxer` already makes a relaxation a context of its own, so a relaxed game is a game like any other.
    - **What holds across games.** A principle — prefer the position with more options, prefer the one where what
      the opponent must answer is more than they can answer in a turn — is a clause like any other, carried between
      contexts with a weight per context, and it earns its place in a game by being fitted there.

    What comes back are candidates, not answers. Whether a heuristic is any good is settled by playing, and the
    weights are the fitter's to find."""

    def __init__(self, goal_prover: GoalProver, forward_chainer: ForwardChainer,
                 clause_compiler: ClauseCompiler) -> None: ...

    def derive(self, clauses: Sequence[Clause], signature: Signature,
               budget: InferenceBudget) -> tuple[Heuristic, ...]:
        """Every heuristic the rules suggest, each with the reasoning that suggested it."""

    def afforded(self, by: Mapping[Value, Sequence[Clause]],
                 budget: InferenceBudget) -> tuple[Heuristic, ...]:
        """What each thing is worth from what its rules afford, and the ordering between them.

        This is where the bootstrapped values come from: a rook's rules cover fourteen squares from a square, a
        queen's take in a rook's, so a queen is worth at least what a rook is. Each number comes back as a
        `Parameter` on the heuristic that uses it, with the ground clause holding it, so it can be tuned later
        without ceasing to be something the agent can say out loud."""

    def relaxed(self, knowledge_base: KnowledgeBase, context: str,
                budget: InferenceBudget) -> tuple[Heuristic, ...]:
        """How far a win is in each of the game's relaxations, as an estimate for the game itself."""

    def principles(self, knowledge_base: KnowledgeBase, context: str) -> tuple[Heuristic, ...]:
        """The clauses that held in other contexts and may hold here, to be fitted and found out."""

    def predictive(self, knowledge_base: KnowledgeBase, context: str, holder: Value,
                   examples: Sequence[Example] = ()) -> tuple[Heuristic, ...]:
        """What this player is likely to do, rather than what would be best.

        With their games, it is induction like any other: their moves are the positive examples, the moves they
        had and passed over are the negative ones, and what comes out is the clauses that account for their
        choices. A probability belongs here more plainly than anywhere else — a player who plays a move three
        times in ten is not a player whose rule is broken.

        Without their games, the only honest starting point is that they will play what looks good, so optimal
        play is where predicting them begins. **That is projection, and it is named as projection**, because it
        is exactly the assumption that fits an engine and does not fit a club player: OMF would be predicting
        itself and calling it them. It is a prior to be moved off as soon as a game arrives, kept as its own
        clauses from the start so that moving off it costs nothing.

        Nothing here decides that the opponent is predicted rather than assumed optimal. Both are models, and
        which one is used is settled where models are settled — on accuracy, on cost, and on what wins games."""

    def seeds(self, heuristics: Sequence[Heuristic]) -> tuple[tuple[Expression, float], ...]:
        """The heuristics as expressions the search can start from, each with the weight to start it at.

        `ExpressionSearch.search` and `ValueGenerator.generate` already take `seeds` and nothing has ever supplied
        them: the search begins at its leaves and builds up, which is how it finds a heuristic nothing in the rules
        suggested, and also how it spends a generation rediscovering that a queen is worth more than a pawn.

        The weight matters as much as the shape. A linear heuristic values a position as the sum of its rules'
        weighted readings, so the weight on "how many rooks I have" *is* what a rook is worth — the engine's
        fourteen is not a separate number to be carried somewhere, it is that term's starting weight."""
```

**One change outside the engine.** `SparseFitter.fit` already takes `start: SparseFit | None` and *"starts from
start's weights and bias when given, from 0 otherwise"*, so warm-starting a fit is supported at the bottom. What is
missing is the plumbing: `ValueGenerator.generate` and `ExpressionSearch.search` take `seeds` but pass no starting
weights into the first fit. Seeds become `tuple[Expression, float]`, and the first fit starts from those weights
with 0 for every leaf. Nothing else about the search changes.

`HeuristicPonderer` stays and becomes this service's caller: it gathers positions, gets targets from payoffs or
from proofs, asks `HeuristicDeriver` for candidates, seeds the search with them and reports each `Labelling`.
`HeuristicProposer` goes — sampling random combinations of vocabulary leaves is the stand-in that this replaces.

### Services — the Z3 backend

```python
class SmtProver:
    """Goals resolution doesn't settle, handed to Z3: the ones that turn on arithmetic and ordering.

    Z3 decides such goals and cannot do the rest — it never gives back a new rule, and it never learns one — so it
    is asked a question and is never the engine."""

    def ask(self, clauses: Sequence[Clause], goal: Clause, budget: InferenceBudget) -> Answer: ...
    def by_induction(self, clauses: Sequence[Clause], goal: Clause, over: Variable,
                     budget: InferenceBudget) -> Answer:
        """A statement about every whole number from a certain point, proved at the base and at the step."""


class TheoryLibrary:
    """What OMF knows before any game does: ordering, arithmetic, sets and membership, the best and worst of a
    value over a set. Generic definitions only — that a queen is worth nine is a game's business, not OMF's."""

    def theory(self, name: str) -> tuple[Clause, ...]: ...
```

### Services — into the knowledge base

```python
class ClauseCompiler:
    """A clause as the Python an RBS runs. The clause stays what is kept; this is what is handed over to be run."""

    def compile(self, clause: Clause) -> PythonRule: ...


class ClauseRecorder:
    """What was concluded, written down so it outlives the run that concluded it.

    A conclusion goes down twice, and each time for a reason. As a **ground clause**, so the engine can read its own
    conclusions back and resolve against them next time, instead of every run re-deriving what the last one knew.
    And as a **belief**, so the certainty, the spread and what it rests on have somewhere to live — which is what
    lets it be argued with.

    Nothing here decides how sure to be. A conclusion drawn from rules a game declared is as good as those rules;
    one drawn from rules OMF induced is as good as the induction, which is a question about the learning and not
    about the reasoning."""

    def record(self, knowledge_base: KnowledgeBase, context: str, derivations: Sequence[Derivation],
               chance: Chance | None = None, at: datetime | None = None) -> tuple[Belief, ...]: ...
    def declare(self, knowledge_base: KnowledgeBase, context: str, ruleset: str,
                clauses: Sequence[Clause]) -> tuple[RuleRecord, ...]: ...
```

### Services moved in from `epistemology`

They keep their names and their behaviour, and each gains what being inside the engine allows.

- **`Justifier`** — unchanged in what it does: traces support through `rests_on` to anchors, records the ids met
  again on the way, counts how many distinct sets of anchors the evidence reaches. It gains
  `justify_derivation(derivation)`, since a derivation and a belief's evidence are the same shape, and
  `ProofWeigher` uses its independent-supports analysis rather than growing a second copy of it. What counts as an
  anchor is now injected as `Anchoring`, so foundherentism stays epistemology's position.
- **`CoherenceChecker`** — keeps both conflicts it finds today (justified evidence naming different values, and a
  belief its anchor contradicts) and gains a third: two clauses that resolve to the empty clause. That is a
  contradiction the value comparison cannot see — "a rook reaches 14" and "nothing reaches more than 10" are about
  different variables and cannot both be true.
- **`AccuracyScorer`** — unchanged. It is the measurement half of the statistics, and it works the same for every
  mechanism, the engine's own included.
- **`CertaintyAssessor`** with `BayesianCertainty`, `GaussianCertainty`, `FuzzyCertainty`, `EvenErrorModel` and the
  `CertaintyModel`/`ErrorModel` ports — unchanged, including "the first model that fits" and `unassessed`.

`epistemology_constant.py` splits with them: the conflict kinds follow `CoherenceChecker`, `ANYTHING_ELSE` follows
`BayesianCertainty`, and the accuracy and spread belief names follow `AccuracyScorer`.

## `epistemology` after the move

It keeps the doctrine and depends on `inference`.

```python
class Epistemology:
    """Makes a context's beliefs rigorous, by foundherentism.

    What is kept here is the position about knowledge, not the machinery: that anchors are direct experiences and
    the rules an application declared, that no claim supports itself, that a belief no model can assess is lived
    with while a task is raised once per kind, and that a conflict is a warning and a task because the model may
    have solved a problem incorrectly. The tracing, the assessing and the finding are the engine's."""

    def review(self, knowledge, belief, need_value=(), need_time=None) -> Belief: ...
    def audit(self, knowledge, context, value, expected_time) -> tuple[Task, ...]: ...
    def immediate(self, context: str) -> bool: ...


class Foundherentism:
    """Fills `Anchoring`: a direct experience, or a rule an application declared and did not declare open."""

    def anchor(self, knowledge: KnowledgeBase, entry_id: str) -> bool: ...
```

## Logs

- `openmind.inference.service.forward_chainer`: `INFO Chained <n> further clauses from <m>; <total> in all` and
  `INFO Nothing further follows from <m> clauses after <steps> steps`.
- `openmind.inference.service.goal_prover`: `INFO <goal>: <status> by <n> derivations in <seconds> seconds`,
  `INFO <goal>: unknown after <seconds> seconds: <reason>`, `DEBUG Step <n>, <rule> from <premises>: <clause>`.
- `openmind.inference.service.proof_weigher`: `INFO <goal> at <chance>, from <n> derivations over <m> chances,
  counted exactly|sampled over <k> draws`.
- `openmind.inference.service.clause_learner`: `INFO Learned <n> clauses covering <c> of <t>, <w> wrongly` and
  `DEBUG Kept <clause>` per clause.
- `openmind.inference.service.clause_challenger`: `INFO <n> challenges from <m> clauses`,
  `INFO <clause> broke after standing <stood>: <what broke it>`,
  `INFO <n> outcomes came out otherwise than expected: <u> nothing predicted, <r> at the wrong rate`, and
  `WARNING <n> of <m> negative examples no clause rules out: either a reading is missing or what rules them out
  isn't about what is being read`.
- `openmind.inference.service.heuristic_deriver`: `INFO Derived <n> heuristics for <context> from its rules alone:
  <a> from what things afford, <r> from relaxations, <p> from principles`, `INFO Derived <n> heuristics predicting
  <holder>'s play from <g> of their games` or `INFO Nothing of <holder>'s to go on: predicting them from optimal
  play, which is projection`, and `DEBUG <clause>, because <derivation>` per heuristic.
- `openmind.inference.service.clause_recorder`: `INFO Wrote down <n> conclusions about <context>`.
- The moved services keep their log lines as they are.

Unifier, clausifier, subsumer and resolver don't log: they are steps, not decisions.

## The inventory this must satisfy

Nothing below may be dropped. Each row needs a test in the new engine, and the stand-in's tests are the
specification — any that cannot be re-expressed is a finding to bring back, not a test to bend.

| Feature of the stand-in | Where it lands |
|---|---|
| `CoveringLearner.learn` — ways of being legal, grow/prune, support across positions, overlapping, `loosely` | `ClauseLearner.learn` |
| `CoveringLearner.distilled` — principles with specializations under them | `ClauseLearner.distilled` |
| `CoveringLearner.excluding` — exclusions that never leak to a sibling | `ClauseLearner.excluding` |
| `CoveringLearner.relaxed` — dropping conditions that are artefacts of the evidence | `ClauseLearner.relaxed` |
| `CoveringLearner.uncovered` — where to look next | `ClauseLearner.uncovered` |
| `CoveringLearner.challenging` / `challenged` | `ClauseChallenger.challenges` / `challenged` |
| `RuleDeducer.conjunction` / `learn` / `refined` | `ClauseLearner.conjunction`, under the same loop |
| `RuleDeducer.refuted` → `Surprise`, and `rare` | `ClauseChallenger.refuted`, `Surprise` kept |
| `RuleDeducer.unexplained`, with its warning | `ClauseChallenger.unexplained` |
| `RuleDeducer.unforeseen` | `ClauseChallenger.unforeseen` |
| `LegalityTree.fit` and its `UNKNOWN` leaf | `LegalityTree`, ported; `Answer` is three-valued throughout |
| `ConditionMasks` — bitset coverage | The same trick, over substitutions |
| `RuleReasoner.entails` / `implies` / `within` | `Subsumer` |
| `RuleReasoner.ordered` | `Subsumer.ordered` |
| `RuleReasoner.together` | `Subsumer.compatible` |
| `RuleReasoner.reaching` | `EvaluablePredicates` |
| `Inferrer.chain` and its eight fact kinds | `ForwardChainer`, the kinds now clauses |
| `Inferrer._held` — dominance | `Subsumer` consulting the evaluable orderings |
| `Fact.rests_on` — provenance | `Derivation.rests_on`, and `Justifier` |
| `FactRecorder.record` | `ClauseRecorder.record` — a clause **and** a belief |
| `SideDeducer`, `WorthReasoner` (with its `settled` grounding check) | Stay; they emit ground clauses instead of `Fact`s |
| `ActionReadings` | Stays; `ReadingLiterals` stops flattening its templates |
| `Justifier`, `CoherenceChecker`, `AccuracyScorer`, the certainty models, best effort | Moved in, behaviour kept |
| `ConsequenceLearner`, `Consequence.when` — what an action does, and when | Kept; `when` becomes clauses, and gains the probability it has no way to say today |
| `HeuristicProposer.readings` / `propose` — random weighted combinations, fitting nothing | Replaced by `HeuristicDeriver`, which reasons instead of sampling |
| `HeuristicPonderer.ponder`, `Labelling.paid` | Stays, as `HeuristicDeriver`'s caller |
| `ExpressionSearch`, `ExpressionGenerator`, `Mechanics`, `PositionView` | Untouched. Not stand-ins; they now start from seeds the engine derived |
| `PositionGatherer` | Untouched |

`PositionDeducer` is untouched too: it reasons over a game tree, not over clauses, and stands in for nothing.

## Transferable knowledge

**A fork is possible in tic-tac-toe and in chess.** In one it is making two winning lines where the opponent can
block one; in the other it is attacking two pieces where they can save one. Nothing about knights or noughts is
doing the work. What is doing the work is: *one action puts more at stake than can be answered before the next*.
Said that way it is neither game's, and carrying it between them is knowledge OMF should not have to learn twice.

Managing that is a job of the engine's, and it has three parts.

**Saying it generically.** A clause transfers when every predicate in it belongs to the shared vocabulary rather
than to one game's own values. `puts_at_stake(Thing, N), N > 1, acts_once` transfers; the same rule written with a
particular piece kind in it does not. So the test for transferability is not a judgement about meaning, it is a
check on the predicates — which makes it something the engine can do rather than something a person has to decide.

**Lifting a clause until it does.** A clause learned in one game usually says more than it needs to, because it was
grown against that game's examples and nothing ever took the surplus out. Generalising it is machinery that already
exists here: `relaxed` drops the literals that are artefacts of the evidence, `distilled` factors a family down to
what they share, and a constant that varies across the examples becomes a variable. What is left, if it is stated
wholly in the shared vocabulary, is a principle.

**Trying it elsewhere, with a weight per context.** A principle arriving in a new game is a candidate and nothing
more. It is fitted there like any other, and it carries a weight *per context* and never one weight everywhere —
a thing that pays in chess may be worthless in the prisoner's dilemma, and the principle is not wrong for that.
What is kept is where it paid and where it did not, so a principle tested in several games is worth more than one
tested in one, and the architecture's rule holds: knowledge crosses contexts on purpose, never by leaking.

```python
# model/principle.py
@dataclass(frozen=True, slots=True)
class Principle:
    """A clause stated in terms no single game owns, and how it has fared where it was tried.

    `paid` is what happened in each context it went to: the weight it was fitted at there. A principle is not true
    or false, it is worth something or nothing in a given game, and the record of that is the principle."""

    clause: Clause
    from_context: str
    paid: tuple[tuple[str, float], ...] = ()


class ClauseGeneraliser:
    """Lifts a clause out of one game's own terms, or says it cannot be lifted.

    It drops what the evidence made necessary rather than what the game does, turns a constant that varied into a
    variable, and then asks the one question that decides it: is every predicate left in the shared vocabulary? A
    clause that still names one game's values is that game's, and saying so plainly is better than carrying
    something that cannot mean anything elsewhere."""

    def generalise(self, clause: Clause, examples: Sequence[Example],
                   shared: Signature) -> Principle | None: ...

    def transferable(self, clause: Clause, shared: Signature) -> bool:
        """Whether every predicate in it belongs to the shared vocabulary."""


class PrincipleLibrary:
    """What has held elsewhere, offered to a game that has not tried it yet.

    It holds no opinion about whether a principle applies here. It offers, the fitting answers, and what the
    fitting answered goes back in — so the next game is offered the principles that have been paying, and OMF gets
    better at starting games rather than only at playing the one it is in."""

    def offer(self, knowledge_base: KnowledgeBase, context: str,
              shared: Signature) -> tuple[Principle, ...]: ...
    def keep(self, knowledge_base: KnowledgeBase, principle: Principle,
             context: str, weight: float) -> Principle: ...
```

`HeuristicDeriver.principles` is the caller: the principles a library offers are one of the three ways it reaches
for candidates, alongside what things afford and what relaxations say.

**The shared vocabulary** is the generic terms OMF implements — a player, a side, an owner, what a thing affords,
what is at stake, that a player acts once before the others do. It is `TheoryLibrary`'s business and it is
deliberately small: every word in it has to mean something in a game with no board, one player, or no conflict.
Nothing in it is a game's. There is no king in it and no checkmate, and there is no fork either — there is only
what a fork is an instance of.

## A principle goes above its specialisations, never instead of them

Found by running it (2026-09-21). Generalising two clauses against each other looks like the obvious way to keep
one rule where there is one rule — and applied as a *replacement* it destroys what it touches.

The case that settles it is a pawn. Played at, OMF learns three clauses without being told anything:

| | what it learned |
|---|---|
| step | `pawn`, `rows = 1`, `columns = 0`, target empty |
| double step | `pawn`, `rows = 2`, `columns = 0`, **`row(source, 2)`** |
| capture | `pawn`, `rows = 1`, `columns = -1`, on a diagonal, target held by the other side |

It found the condition on the double step by itself: `row(source, 2)` is in that clause and in neither of the
others, and nobody said where pawns start.

"A pawn moves one or two" is **correctly two clauses**, because they are not one rule with a varying distance —
the double step carries a condition the single step does not. Anti-unified, `rows` becomes a variable and
`row(source, 2)` is dropped as unshared, giving "a pawn moves some distance up its own column": strictly more
general, strictly wrong, and it permits a pawn to move three squares from anywhere. Two correct clauses are spent
to buy one incorrect one.

Measured, a loop that merged this way never accumulated anything: eighteen moves, every legal move unforeseen
every time, clause count oscillating between one and six instead of growing, where the same loop without merging
grew to twenty-three clauses and sixty-three per cent coverage.

So the rule for the whole engine: **what a family of clauses shares is said above them, with the specialisations
kept under it** — which is what the stand-in's `distilled` did and what `merged` did not. `ClauseLearner.merged`
stays in the package and out of the learning loop until it layers rather than replaces.

## Nothing here knows a game

OMF's convention is that nothing in it is specific to one game, and an engine is where that is easiest to lose:
the examples are always a particular game, and a board is a tempting thing to assume. So, explicitly —

- **No board.** A grid is a thing some games have. `reaching` is registered by whatever knows there is one, never
  built in, and every derivation has a path that does not go through it.
- **No turns.** All players act at once and the constraints say who may; a game played in turns is one where the
  others have no action. Nothing here reads a player to act.
- **No opponent, and no two players.** Sudoku has one player, the public goods game has many, and the prisoner's
  dilemma has two who are not enemies. A player is predicted or assumed optimal by a model chosen on its accuracy,
  never by a rule saying opponents minimise OMF's payoff.
- **No zero sum.** Payoffs come from the game. Nothing infers one player's from another's.
- **No assumption that anything is owned.** Ownership is deduced where a game has it and absent where it does not,
  and a heuristic that values what a player holds simply finds nothing to value.
- **No numbers of OMF's own.** Every constant is learned or is a budget the caller set.

The built-in games are the test of this, and they are chosen to be awkward together: tic-tac-toe has a grid and
turns, sudoku has a grid and one player, rock paper scissors has no grid and simultaneous play, the prisoner's
dilemma has no grid and no conflict. Anything that works for all four is not leaning on a board.

## Open

- **"Distil" is about to mean three things in one package.** `ValueDistiller.distill` learns a heuristic from
  self-play — generate candidates, fit, keep what holds on held-out games — the tuning sense. `CoveringLearner.distilled`
  means factoring the conditions a family of rules shares into a principle. And learning an explainable model out
  of a network by asking it (phase 7) is *model distillation*, the third sense and the most standard of the three.
  All three will appear in the same logs. Rename the principle sense — `generalised`, `principled`? The other two
  are the established words and are at least close kin.
- **The codec boundary.** Phase 8 wants the engine's conclusions encoded to words and decoded back. A clause is
  already structured enough to encode, and `ClauseTextMapper` already renders one readably. What is not settled is
  whether decoding lands on clauses directly, or on `Formula` and through `Clausifier` — the second keeps natural
  language from having to think in normal forms. That is `codec`'s stop, not this one, and nothing here decides it.
- **Three of `ActionReadings`' templates are the same shape once filled**, found while building `ReadingLiterals`:
  `"{player} can reach {parameter}"`, `"{player} can reach {holding}"` and `"{player} can reach {whose} {holding}"`.
  A filled name like `"first can reach target"` matches all three, so which reading it came from cannot be
  recovered and is not guessed at — the slots still come out as terms, which is what learning needs, but the three
  readings land under one name and their evidence is pooled. Nothing else is affected: every other template is
  distinguishable. The fix, if it is wanted, is in `ActionReadings` naming them apart, not in inverting harder.
- **How the learned constants are learned.** Each is settled as learned rather than fixed, which says what they are
  not. What remains is where each one's value comes from before there is anything to learn it from: how long a
  clause must stand before breaking it counts as meeting a rare rule of the game, how much of a clause's examples
  to grow on, how deep a tree may go. A budget the caller sets is the answer for the first run; what replaces it,
  and what it is learned from, is worked out at the code stop for Stage 5.
- **`doc/open-questions.md` §24** — where a player's own direction comes from — looks already settled in code and
  not on paper. `SideDeducer` deduces `facing` from which way a player's one-way pieces go, and `ActionReadings`
  reads `row of {parameter}, from the player's own side` and `rows from {first} to {second}, forward` off it. That
  is the question's third option, built in `1e4daea` on 2026-09-20, after the question was written. It is Maxime's
  to strike, not the engine's, and nothing here touches that file.
