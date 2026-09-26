# Interfaces: a notation's syntax, learned before its meanings

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

Sits under [happenings.md](happenings.md), and answers the question left open at the end of it — whether the
decoder's reading of a notation is given or learned. It is learned, and from the notations alone.

## Why

**What we have is a lexicon, and a notation has a grammar.** `CouplingLearner` measures every substring of every
notation against every part of what happened, and `Decoder.read` asks only whether a symbol is *present*. So
`Nf3`, `exd5` and `R4xd4` are three bags of characters, and `Decoder.write` concatenates symbols in order of how
much they tell us while admitting in its own docstring that it knows nothing about order.

**Absolute position was the stand-in and it cannot work.** `Coupling.where` counts from the start of the string,
but a role's index moves with the notation's length: `e` sits first in both `e4` and `exd5` while saying where
the move lands in one and where it started in the other. Measured against 34914 chess moves, that collapses into
one saying right 95.0% of the time, where the truth is two sayings each right always. No threshold separates
them, because the measurement never had the distinction in it.

**And the same four sayings show the cost.** The file letters are the only sayings that narrow *where* a move
lands. Filtering them out for being 95% right leaves 361 of 8064 candidates standing and nothing read as one;
keeping them refuses the move actually played seven times in three hundred. Both are the same defect seen from
its two ends.

## What is learned, in four stages

**Sorts, from the company a character keeps.** For every character, what precedes and follows it across the
corpus. `a`–`h` share a profile, `1`–`8` share another, `x` keeps its own, and `B` separates from `b` because one
leads a notation and the other stands where files stand. That last is capitalisation being significant *as a
result* — nobody says capitals name pieces; those characters simply do not keep the same company. Sorts are
merged agglomeratively, taking at each step the merge that shortens the total description most, and stopping when
no merge shortens it. No chosen number of sorts, and the same selection pressure the constraints already use.

**Shapes, as sequences of sorts.** Every notation rewritten over its sorts: `e4` and `d4` become one shape,
`Nf3` and `Bc4` another, `exd5` a third. Tens of thousands of distinct strings collapse to a handful of shapes,
and those shapes are the syntax.

**Couplings per slot.** Within a shape the layout is fixed, so position finally means something. A coupling stops
attaching to a symbol at an index and attaches to a slot of a shape. `e` in the opening slot of the taking shape
says where the move started; `e` in the square slot says where it lands. Two sayings, each right every time.

**Roles, by merging slots across shapes.** The trailing square is the destination in every shape that has one.
Finding that is the same anti-unification `RefusalLearner` does on cases and `said_as_one` does on symbol
families, and it is what stops the number of sayings growing with the number of shapes — the difference between
a grammar and a table.

What survives all four is genuinely open to syntax. `Nbd2` says the `b` is a disambiguator and not which knight;
that part goes to the constraints. **Syntax assigns the roles, the rules choose among what the roles leave
open.**

## Models

```python
@dataclass(frozen=True, slots=True)
class Sort:
    """One sort of character, found by the company it keeps.

    Not a category anyone named. Characters that appear after the same things and before the same things are
    doing the same job, whatever that job turns out to be — and two characters that look alike to us but keep
    different company are two sorts, which is how a notation's capitals separate from its lower case without
    anyone saying that capitals mean anything."""

    name: str
    characters: frozenset[str]


@dataclass(frozen=True, slots=True)
class Shape:
    """One layout a notation comes in: which sort stands at each place.

    A notation belongs to exactly one shape, and a shape fixes its length. That is what makes a place in it worth
    measuring, where a place in the raw string was not."""

    sorts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Role:
    """One job a notation gives a place, across every shape that has it.

    Held as the places themselves rather than a name, because the name would be ours. A role that covers the last
    square of four shapes is one saying; four roles covering one each is a table of four."""

    places: tuple[tuple[Shape, int], ...]


@dataclass(frozen=True, slots=True)
class Grammar:
    """A game's notation as sorts and the shapes they fall into."""

    sorts: tuple[Sort, ...]
    shapes: tuple[Shape, ...]

    def shaped(self, said: str) -> Shape | None: ...   # which shape that notation is, or None if it is new
    def sort_of(self, character: str) -> Sort | None: ...
```

`Coupling.where` changes from `int | None` to `Role | None`, and `Coupling.said_by` asks the grammar for the
notation's shape before looking at a place in it.

## Services

```python
class SyntaxLearner:
    """The shape of a game's notation, from the notations and nothing else.

    **It never sees what happened.** Sorts and shapes are a fact about the strings, and keeping them independent
    of the happenings is what lets the two measurements check each other rather than agree by construction. It
    also means a game's records can be read for their structure before a single rule is known.

    It keeps nothing: built once, it is given the notations on every call."""

    def learn(self, said: Sequence[str]) -> Grammar: ...


class CouplingLearner:
    """(reworked) What each slot of each shape says about what happened.

    The measurement is unchanged — bits for what a symbol discriminates, precision for how often acting on it is
    right. What changes is where a coupling attaches, and that substrings stop being guessed: a shape says where
    one piece of a notation ends and the next begins, so `longest` goes."""

    def learn(self, sightings, grammar: Grammar, ...) -> tuple[Coupling, ...]: ...
    def said_alike(self, couplings: Sequence[Coupling]) -> tuple[Coupling, ...]: ...   # slots merged into roles
```

## What goes

`Coupling.where` as an absolute index, `longest` and the substring enumeration under it, and
`CouplingLearner._places`. The threshold `surely` stays, but should stop doing the work it is doing now: sayings
that are right in their role do not need filtering out.

## Open

- **Multi-character sorts.** Every token in chess notation is one character, so a shape can be read off the
  characters one at a time. A rank of `10` on a larger board would not be. Leaving it until a notation needs it,
  rather than designing for it blind.
- **A notation whose shape is new.** News rather than failure — it means the shapes learned so far do not cover
  what the game just played — but what the decoder should do with it in the meantime is not decided.
- **Whether sorts may also use what the characters say.** The proposal above uses the company a character keeps
  and nothing else. The happenings are there and would separate sorts that company alone leaves merged. My
  position is to start without them, because independence is worth more than the separation, and to fall back
  only if sorts come out visibly wrong. Not decided.
