# Interfaces: theories — what OMF knows before any game does

Supersedes `domains.md`, which used a word already doing two jobs: a **domain** is a game (`domain: str` is
"chess", in twelve files) and a domain is a set of values (`Domain.WHOLE`, in three). A third meaning would make
every docstring ambiguous, and the docstrings are how this codebase explains itself.

**Theory** is the standard word — a theory of sets, of order, of motion — and it is the project's own earlier
answer: the deleted `TheoryLibrary` had `theory(name) -> tuple[Clause, ...]`.

## The test for whether something is a theory

**The same idea keeps arriving in pieces, each cut to fit whichever caller needed it.** That is how geometry was
found: `places apart`, `from nothing`, `lands on`, `pointed at` and `on line` were each added to
`CandidateReadings` when some rule could not be said without one, each with its own docstring making its own
case. It is also how arithmetic was found, and information.

## The list

| theory | state | the work |
|---|---|---|
| **information** | in use, duplicated | consolidate — Shannon's formula is written *verbatim twice*, `syntax_learner.py:120` and `coupling_learner.py:243`, and `DescriptionLength` shares no notion of bits with either |
| **statistics** | built, unreachable | `ChanceFitter`, `AccuracyScorer`, the certainty layer — all tested, and the live loop reaches **none** of it |
| **sets** | absent | wanted today by `among`, the vocabulary count, and what the constraints leave standing |
| **arithmetic** | scattered in three files | `abs(held - wanted)` twice in `evaluable_predicates`, `held + drawn.by` and `held + by` in `consequence_drawer`, `one + step` in `consequence_learner` |
| **order** | `EvaluablePredicates` already is one | name it: `less`, `at_most`, `more`, `at_least`, and `UPWARD`/`DOWNWARD`, which `Subsumer` already consults |
| **geometry** | invented five times | replace the five readings or only extend them — **and replacing rewrites all 41 hand-written chess constraints**, losing the measurement that says nothing gets through |
| **mechanics** | implicit throughout | identity through motion (`Moved`), exclusion (`Changer`'s order), solidity (`it_passes_over_something`), displacement (`Stepped`) |
| **agency** | across six mechanisms | `Other`, the `acting` callable, `turn(Mover)`, `other_than(Owner, Mover)`, `PLAYERS`, and ten services that mention players |

**Doing now: information, statistics, sets.** The rest as they are wanted, which is how readings have always
been added here.

## The rule that governs all of them

**A theory supplies vocabulary, never assumptions.** Whether two things may share a square is a fact about the
game — chess says no, a game with stacking says yes — so OMF must go on learning it. Mechanics may offer the
words `occupied`, `blocked`, `the same thing as`; it may not assert that things are solid. Get this wrong and
OMF stops being game-agnostic, which is the one thing it is for.

This bites hardest on mechanics and is why that one is not first.

## What a theory is

```python
@dataclass(frozen=True, slots=True)
class Theory:
    name: str
    terms: tuple[Term, ...]
    claims: Callable[[str, object], bool]   # where it is *likely* to pay — a prior, not a gate


@dataclass(frozen=True, slots=True)
class Term:
    predicate: str
    places: tuple[str, ...]      # what each argument is, in the game's own declared kinds
    answered: Callable | None    # computed where it can be; None means derived from clauses
    tied: bool                   # whether the search may build bodies from it
```

**`tied` is what keeps this affordable**, and it is not new: a chess case already carries about 151 readings
while `tied()` hands the search about 33. The board readings are carried — so a constraint can be *checked*
against them — and withheld from what a body is *built* from.

**A claim is a prior, not a fence.** Euclidean distance applies to any two tuples of numbers whether or not
anybody called them a board. What decides whether a term may be offered is that its places typecheck against
what is here; the claim only says where it is likely to earn its keep.

## Two costs, and one of them only became principled today

- **Search is combinatorial and none of this helps.** ΣC(n,k) over the *tied* vocabulary: 33 terms gives about
  46,900 bodies at length four, a hundred gives about four million. Only `tied` controls it.
- **Breadth is now priced.** Under `DescriptionLength` a body of k conditions costs `L_N(k) + log₂C(C, k)`, so
  **C is the vocabulary size and every constraint pays for it**. A theory offering terms nothing uses taxes
  every rule in the set. That is the answer to "how much vocabulary is too much", and there was none yesterday.

## Open

- Whether `tied` belongs to a term or to a term *in a game* — "same row" is about the board in chess and about
  the candidate in a game whose action names a row.
- Whether geometry replaces the five readings. **This gates the geometry work and nothing else.**
- Whether sets are wanted by rules at all. Chess's collections are its piece types, its players and the domains
  of `x` and `y`; a rule saying "the mover is one of the sliders" needs "the sliders" to be declared, and it is
  not. **Sets may be a theory for the machinery before they are one for the rules**, and it is worth knowing
  which is being built.
- `Mechanics` is already a class name — a per-process cache of views and moves for expression generation. The
  theory of motion cannot have it without a rename.

## Verification

- Both suites green (OMF 1084, chess 60).
- **Consolidation changes no answer.** Information and arithmetic are the same formulas in one place; every
  test that passed must pass unchanged, and the syntax learner must still settle on the same sorts.
- **The search does not grow.** `tied` per case stays where it is, or the theory is offering the search what it
  should only be offering the checker.
- **A game with no grid gets no geometry**, when that is built — a theory claiming everything has said nothing.
