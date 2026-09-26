# Interfaces: happenings, and reading a game's notation

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

Replaces the declared action shape. A game stops saying "a move has an origin and a destination", and OMF works
with what can *happen* — which it already has a language for.

## Why

**The notation is the rules' own compression.** An engine offers `Nf3`, not `move(g1, f3)`. It leaves out the
origin precisely because the rules make it recoverable, and puts one in only when they do not. So reading the
notation and knowing the rules are the same problem seen twice: a notation that decodes to exactly one happening
means the rules agreed with it, to several means they are too loose, to none means too tight.

That was staring at us this morning and I argued the wrong way from it — that resolving `Nf3` needs the very
constraint being learned, so it must happen in the adapter where OMF cannot see. The right conclusion is that it
should *be* the problem.

**And it changes where evidence comes from.** Today an engine enumerates legal moves, a scaffold nothing was
supposed to depend on. Reading notation needs only game records: millions of them, real play, no oracle, and the
same two error kinds fall out of decoding rather than out of being told.

## What a candidate is now

**The candidate space comes from the change language rather than from a declared action.** `Moved(grid, source,
target)` has two places, each a cell, which on an eight by eight board is the same four thousand and ninety-six
candidates as today — but the shape is OMF's own structure and no game declares it.

A **happening** is one or more changes. One change is the whole of a plain move; castling is two `Moved`s;
promotion is a `Moved` and a `Placed`. Neither needs a conditional parameter nor a phase, which is two things
parked twice now that stop needing to be solved.

The readings are unchanged in kind. A change's places are named by the change — `source`, `target`, `at` — where
they were named by the action's parameters, so `places apart`, `holds` and the rest follow as they are.

## Models

```python
@dataclass(frozen=True, slots=True)
class Happening:
    """What can come about: one or more changes, together.

    The thing constraints are now over. A plain move is one change; a castling is two; a promotion is a move and
    a placement. Saying it this way is what lets those stop being special cases — they are happenings with more
    changes in them, not actions with stranger shapes."""

    changes: tuple[Change, ...]

    @property
    def places(self) -> tuple[tuple[str, Coordinates], ...]: ...   # each change's places, by the name it gives them
```

## Services

```python
class Happenings:
    """Every happening a position could have, before anything is known to refuse one.

    The domains come from the structures a state holds — a grid's extent, a scalar's kind — rather than from an
    action a game declared. A game says what it is made of and OMF works out what could happen to it."""

    def of(self, state: State, changes: int = 1) -> tuple[Happening, ...]: ...


class Decoder:
    """What a game's notation says, read as a happening — and written back, which is the same rules run the other
    way.

    **One rule-based system does both.** Rules relate a notation to a happening, and a rule read forwards decodes
    while the same rule read backwards encodes. Writing `Nf3` requires knowing that the origin is recoverable,
    which is knowing the rules; so a system that can write the notation has demonstrated the same knowledge as
    one that can read it, and neither has to be given separately.

    **Decoding is scored without an oracle.** Exactly one happening means the rules agreed with the notation;
    several means they let through what the notation assumed they would not; none means they refuse what the
    game played. Those are the two error kinds we have, arriving from a game record instead of from an engine."""

    def read(self, state: State, said: str, refusing: Sequence[Clause]) -> tuple[Happening, ...]: ...
    def write(self, state: State, happening: Happening, refusing: Sequence[Clause]) -> str: ...
```

## What goes

`ActionKind` stops being how candidates are laid out, and the chess adapter stops resolving notation. `Action`
stays where an agent has to name what it is doing, but nothing learns from its shape.

## Open

- ~~**What bounds a happening of several changes.**~~ **Decided (Maxime): the predictor bounds it, and the
  question was badly put.** It read as combinatorial — one change is four thousand candidates, two is that
  squared — and it is not, because the further changes are not free. A happening's backbone is one `Moved` whose
  two places *are* the parameters: what stood at the origin's row and column now stands at the destination's. A
  capture removes at the destination, a piece taken in passing at the row of the origin and the column of the
  destination, a promotion puts its piece down on the destination. Every one is drawn from those same two
  places, so a second change adds no freedom and there is no square to take. The vocabulary for saying this is
  already `predictor.model.drawn`.
  Nor is it circular, which was the other half of the mistake. What ties a castling's rook to its king does not
  have to be searched for, because the game hands back the whole outcome: the predictor watches what happened
  and learns the drawing. "In some occasions it also does something else" is a `Consequence` with a `when`, and
  the `when` is the part not yet learned — it is hardcoded empty, so every consequence is predicted to happen
  every time. That is the real gap, and it is the predictor's.
  Castling is the one case whose extra change is not drawable from the two places: its rook's squares need
  `Always` constants and a `when` telling king-side from queen-side apart. Expressible, and the case where the
  `when` does real work rather than none.
- ~~**Whether the decoder's rules are learned or given at first.**~~ **Decided (Maxime): learned.** A game hands
  over its structures and its notation, and nothing else. See [notation-syntax.md](notation-syntax.md): the sorts
  of character and the shapes they fall into are found from the strings alone, and what each place of each shape
  says is then measured against what happened. Nothing is told that `N` names a kind or that `f3` is a square.
