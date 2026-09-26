# Interfaces: constraint learning

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-22)

- **An action's legal parameter values are a CSP.** Not the game — choosing which of the legal moves is best is
  search, heuristics and policy, and none of that is a CSP. What is a CSP is move generation: the parameters of an
  action are the variables, and the legal actions are the assignments no constraint refuses.
- **There are no generators.** Legality is not produced, it is what survives.
- **Two rule types.** Constraints and predictors. This step is constraints only.
- **Domains are the whole structure.** Both parameters of `move` range over every cell — 4096 candidates in chess,
  every one of them allowed before anything is learned. Learning only ever adds or tightens.
- **One grid, whose cells hold whatever the game wants to store.** A cell holds a `Record` with the parts the game
  declared, read as one term however deep it goes: `grid[1][1] = square(white, piece(white, rook))` reads as
  `grid(1, 1, square(white, piece(white, rook)))`. Not parts kept in separate grids side by side: those can be
  moved out of step, and then a square holds one piece's colour and another's kind. A piece's colour follows the
  piece because they are one value, and the square's own colour stays where the square is.
- **A thing does not carry where it is.** A square has a colour; its coordinates are its position in the grid. So
  a reading names the cell in its leading places and says the thing in the last one, and never says where twice.
  Nothing is lost by the single term, since a clause puts a variable wherever it does not care and unification
  goes inside a term.
- **Values are opaque, structure induced.** A cell alias is a coordinate into a grid the state holds. The learner
  gets coordinates; it never gets what a relation over them means. "Same row", "diagonal", "adjacent" are clauses
  it builds, not readings it is handed.
- **The readings cover the whole structure**, not a chosen subset of it. Reading only the cells a candidate names
  cannot express a blocked way; picking out "the cells between" would hand over a chess-shaped notion. So
  everything is read and the learner sorts out what matters. The cost is compute, and it is paid rather than
  designed around.
- **The evidence is what the game shows**: a position with its legal moves, and later what playing a move does to
  the board. The rules are deduced from that. The learner is not driven by error reports.
- **Why deduce rules the engine already enforces.** Not to reproduce the move list — the engine does that, faster
  and correctly. It is to hold them as clauses that can be reasoned with: chained into further rules, asked what
  follows from a position, and used to derive heuristics. Reproducing the list is only how we know the clauses are
  right.
- **Simplicity is the selection pressure.** A rule and a lookup table both account for the positions they were fit
  to; the rule stays the same size as evidence accumulates and the table grows with it.
- **Constraints live in the knowledge base**, as everything the agent knows does, and are fetched from it rather
  than carried along.
- **A constraint that refuses a legal move is repaired where it can be, and dropped where it cannot.** Repairing
  is adding conditions until it stops covering that move while still covering everything it rightly refused.
  Where no reading tells the two apart, the constraint was wrong rather than merely too broad, and it goes.
- **Deferred:** promotion (a third parameter existing only under a condition), and en passant and castling, whose
  legality depends on history the board does not show.

## Where this sits

In `inference`, over `rule`, `structure` and `world`, under nothing. It produces `Clause`s; the existing
[`csp`](../../src/openmind/csp/) package consumes them. It is not a new solver and not a new clause language.

## What is reused rather than rebuilt

`Clause`, `Literal`, `Term` (`Constant`, `Number`, `Variable`); `Example`, `CaseIndex`, `Substitution`,
`InferenceBudget`; `Unifier`, `Subsumer`, `EvaluablePredicates`; `Action`, `State`, `Grid`; the whole `csp`
package. These are code-approved and the previous failure was above them.

## The vocabulary

One case is one candidate assignment in one position, said as ground literals.

| Reading | Shape | From |
|---|---|---|
| What stands in a grid cell | `<model>(Row, Column, Value)`, one per cell of every grid the state holds | `Grid` |
| A value that has named parts | a term in the value's place: `piece(white, knight)`, named after the record and holding its parts in order | `Record` |
| What a scalar says | `<model>(Value)` | `Scalar` |
| What a map says | `<model>(Key, Value)` | `Map` |
| What a list says | `<model>(Index, Value)` | `List` |
| Where a parameter points | `<parameter>(Row, Column)` where its value is a cell of a grid, else `<parameter>(Value)` | `Action` + `GridAliases` |
| Arithmetic and comparison | `less`, `at_most`, `more`, `at_least`, `same`, `other_than` | `EvaluablePredicates`, built in |

Nothing else. In particular **not** `Grid.diagonals`, `.lines`, `.rays`, `.neighbours` or `.distance`, though the
grid offers them: those are the relations the learner is supposed to discover.

This is enough to reach a sliding rule without being told one. Two cases of the same piece moving along a row
anti-unify: `origin(4,1)` with `origin(2,1)` gives `origin(X1,1)`, and `destination(4,5)` with `destination(2,7)`
gives `destination(X1,X2)` — the *same* `X1`, because a differing pair of terms always maps to the same variable
across the whole clause. The equality between the two rows comes out by construction, never by search.

## Models

```python
@dataclass(frozen=True, slots=True)
class Evidence:
    """One position and what the game showed of it: which actions it allows there.

    This is what the learner is given. It is not told which of its rules were wrong, and nothing here is a
    correction — a position and its legal actions is the whole of the input, and everything the learner concludes
    it concludes from that."""

    where: State
    action: str
    legal: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class Disagreement:
    """Where what a set of constraints refuses and what the game refuses differ.

    This scores constraints against evidence; it is not how evidence arrives. `allowed` are candidates no
    constraint refused and the game does not list: the constraints say too little. `forbade` are candidates some
    constraint refused and the game lists: a constraint says too much."""

    where: State
    allowed: tuple[Action, ...]
    forbade: tuple[Action, ...]

    @property
    def settled(self) -> bool: ...          # neither kind occurred
```

## Services

All stateless, built once and injected, given everything they work on per call.

```python
class CandidateReadings:
    """Candidates in a position, said as ground literals — the vocabulary above and nothing else.

    The single place a new reading could ever leak in, so it is small and read as such."""

    def cases(self, evidence: Evidence, domains: Mapping[str, Sequence[Value]]) -> tuple[Example, ...]: ...
    """Every assignment the domains allow, read, and marked. A case **holds** where the game does not list the
    action: covering a case is refusing it, and legality is the absence of a cover. Nothing is sampled and nothing
    is left out — the cases are the whole space the solver will later search."""

    def read(self, state: State, action: Action) -> tuple[Literal, ...]: ...

    def of_state(self, state: State) -> tuple[Literal, ...]: ...
    """The readings that do not depend on the candidate, worked out once for a position and shared by its
    thousands of candidates rather than rebuilt for each."""


class RefusalLearner:
    """The constraints that account for what the game refuses.

    Each clause covers some of the refused candidates and none of the listed ones. Grown from a case said
    outright and widened case by case: where two cases agree a term stays, where they differ a variable goes in,
    and the same differing pair always gives the same variable. A widening that would take in a listed action is
    refused and another case tried.

    Refusing is not one thing — a way may be blocked, a thing may not be the mover's, a king may be left exposed —
    so one clause is grown, what it covers is set aside, and another is grown for the rest."""

    def learn(
        self,
        examples: Sequence[Example],
        budget: InferenceBudget,
        starting: Sequence[Clause] = (),
    ) -> tuple[Clause, ...]: ...
    """`starting` is what is already believed: a case it accounts for is left alone, and a case it does not is
    first tried as a widening of something already held. That is what lets an agent learn position by position
    rather than from a corpus."""

    def refuses(self, clauses: Sequence[Clause], example: Example) -> bool: ...
    """Whether any of them refuses that candidate. There is no second list to consult and no exception to check:
    this is the whole of legality."""

    def scored(
        self, clauses: Sequence[Clause], evidence: Evidence, domains: Mapping[str, Sequence[Value]]
    ) -> Disagreement: ...
    """How those constraints stand against a position: what they let through that the game does not list, and
    what they turn away that it does."""

    def repaired(
        self, clause: Clause, keeping: Sequence[Example], without: Sequence[Example]
    ) -> Clause | None: ...
    """That constraint with conditions added until it stops refusing what the game allows, still refusing what it
    rightly refused — or None where no reading tells the two apart.

    A constraint that turns away a legal move is wrong as it stands, which is not the same as being worth
    nothing: everything it rightly refused would come back if it were simply dropped, and the next position would
    learn it all again. So it is given more to say, since a constraint that says more covers less, and the cases
    it must go on covering are what it may say it from. A condition is taken from the readings they all share, so
    adding it cannot lose any of them.

    None is the honest answer, not a failure: where the moves it must keep refusing and the move it must release
    carry the same readings, the constraint was wrong rather than too broad, and the caller drops it."""


class ConstraintDistiller:
    """The simplest set of constraints refusing exactly what these refuse.

    Cost grows faster than length, so two short rules beat one long one and nothing is gained by shattering a rule
    into pieces. Nothing is perturbed to see whether it survives: what may change is how the constraints are
    said, never what they say, and every candidate is answered after exactly as before."""

    def distilled(
        self,
        clauses: Sequence[Clause],
        examples: Sequence[Example],
        budget: InferenceBudget | None = None,
    ) -> tuple[Clause, ...]: ...


class ClauseConstraint:
    """A learned clause run as one of the solver's constraints.

    `ClauseRule` is declared as a kind of rule and stored as one, but nothing in OMF runs it — `RuleCaller` has no
    branch for it — so nothing the engine learns can currently reach the solver. This is that bridge, and it needs
    no compiler: refusing a candidate is asking whether a clause covers its readings, which is the same operation
    learning already does. One mechanism serves learning and playing."""

    def constraint(self, clauses: Sequence[Clause]) -> ConstraintRule: ...
    """Those constraints as the callable the `Solver` takes: true where no clause refuses."""
```

The `Solver` also wants a `ValuesRule` per parameter. Under this design it is the domain and nothing more — every
cell of the grid — and says nothing about which moves are possible.

## Where constraints are kept

In the knowledge base, through what it already offers — no new store and no new record type.

A learned constraint is a `RuleRecord` of kind `CONSTRAINT`, its `rule` a `ClauseRule` wrapping the clause, its
`action` the action it constrains, its `source` the inference that produced it, and `open` true so OMF may revise
what OMF wrote. It is kept with `declare`, its repair written back with `revise`, and a constraint that could not
be repaired removed with `undeclare`. `frozen` is what keeps the learner's hands off anything a game declared
itself.

What the knowledge base holds is what `learn` is given as `starting`, so each position asks only what it did not
already know and what it knew is widened rather than rediscovered. The learner does not reach into the knowledge
base itself: it is a stateless service and is handed what it works on, as every other one is.

Worth saying because it is a trap: `LISTING` is the same legal-move list, declared as a rule. Where a game
declares it, it is used and the constraints are not. The learner's constraints and a game's listing must agree,
and nothing can tell which was meant when they don't.

## Open

Nothing.

## Verification

- Unit tests beside each service as `*_tests.py`; the end-to-end one under `test/`.
- End to end: over a set of positions, learn constraints, run the `Solver`, compare its solutions to the engine's
  legal moves position by position, and report the two counts with a header — never a claim about quality.
- What is learned and what it cost logged to `data/log/` as it happens.
