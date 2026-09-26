# Interfaces: moves as offsets, movers with their own rules

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

Supersedes [shorter-constraints.md](shorter-constraints.md). Sits alongside [happenings.md](happenings.md),
which already takes the declared action shape away; this says what replaces it.

## Why, measured

One flat set of constraints has to separate a bishop's moves from a knight's **without any condition naming
which is which**. There is nothing in the vocabulary that says whose rule this is, so the learner reaches for
whatever else tells the two apart, and only two things do: the whole board, or the coordinates. Both are in the
store today.

```
body length  count          of 140 constraints kept after 12 positions
          1      2
        …
          9      1          fifty of nine conditions or fewer — among them
                            refused :- turn(X4), holds(destination row, destination column,
                                       grid, square(X6, piece(X4, X7)))
         80      7
        …
         84      3          ninety of eighty or more, which is a board
        157      3
        …
        159      7          twenty-eight of about a hundred and fifty-eight, which is two boards
```

Neither half of the day's work moved this. Generalising cannot shorten a clause — offered four thousand
widenings, a hundred-and-sixty-one condition constraint came out at a hundred and fifty-seven, and none of the
four thousand failed for want of a widening. Growing the clause from the case instead shortens it to one
condition and makes every one a coordinate: `refused :- destination(1, 1)`, short and generalising to nothing.
The length was never the disease.

## What a game declares

**The type and the domain, and nothing else.** For chess, a move takes two signed whole numbers.

```python
Action("move", (("x", Domain.WHOLE), ("y", Domain.WHOLE)))
```

`Domain` is OMF's, not the game's: whole numbers, whole and positive, whole and negative, real, real and
positive, and so on. The identical declaration describes checkers, shogi or amazons — the difference between
those games is entirely what gets learned. That is the genericity test, and it is now one anybody can apply by
reading the declaration.

**Note what is not declared.** Not −7 to +7: the whole numbers. So *you may not move off the grid* is a rule
OMF has to learn, where every candidate enumeration we have written so far had it baked in. It is a real rule of
chess and it should be found like the others — and being true of every mover, it lives once.

**And that the rules of a move are the rules of the thing that moves.** `grid[1, 1].move(2, 2)` is the thing
standing at (1,1) being asked about an offset. The game declares the dispatch; OMF learns what fills it.
Declaring that a bishop has a move is not declaring how it moves.

**The mover is a parameter like any other, constrained to one value.** `bishop.move(2, 2)` is `move(self, 2, 2)`
with `self` narrowed to the thing at (1,1) — so dispatch never leaves the CSP and no generator appears at the
edge of it. A rule set "for bishops" is the constraints whose first parameter is constrained to a bishop, which
is also what gives the library a key that was found rather than attached.

**That parameter carries its place, or the offset and the ray have nothing to start from.** It is a reference
into the grid rather than a copy of what stands there, so binding it reads both what the thing is and where it
is. A piece is position-aware without a piece storing coordinates, which would be saying twice what the square's
own index already says.

## What changes in the action

**The parameters are the offset, not the destination.** `move(2, 2)` is two right and two down from wherever the
mover stands; the destination is derived and never named.

That is what makes the postcode **unstatable** rather than merely penalised. Everything tried today was an
attempt to recognise position-specific rules and make them lose. A rule written over offsets has no square in
it, so there is nothing position-specific left to write down.

And the rules collapse to what they should always have been: the two offsets equal for a bishop, exactly one of
them zero for a rook, one and two in some order for a knight, neither above one for a king. One or two
conditions, and true everywhere rather than true where they were read.

## Rays, which are the same idea asked differently

A move by offset (x, y) *is* a ray: direction (sign x, sign y), length max(|x|, |y|). The offset says where the
mover lands; the ray says what it passes through. So nothing in the way is one condition over what the move
already describes — the first occupant along that ray is no nearer than the move's own length.

**This reverses the plan's decision to withhold `Grid.rays`,** and the reversal is the point rather than an
exception. That decision held while moves were absolute destinations, when a ray was an extra relation invented
to connect two otherwise unrelated squares. With moves as offsets the ray is the path the offset traces, and
withholding it is refusing to let the learner see the move it is reasoning about.

It also reaches [open question 26](../open-questions.md), seeing through what is in the way. If the reading says
what the *first* occupant along a ray is, then asking past that occupant is the same reading again, and
batteries, pins and discovered attacks become statable — without relaxing the board or learning two generators
against two targets, which were the options on the table.

## The rule library

**Every rule exists once; a rule set is pointers into it.** When nothing points at a rule it is deleted, which
is what the store already does whenever the log says `Forgot rule-…` — now for a reason rather than as
tidying-up.

**Where the boundary sits between common and per-mover stops being a decision.** A rule pointed at by every kind
is a rule of the game; one pointed at by a single kind is that mover's own. Nobody declares which; it is read
off the pointer count.

**And it is the selection pressure we have been failing to hand-build.** A rule costs once in the library and a
set pays only for its pointers, so a rule serving six movers is a sixth as dear per use as six copies of it. A
postcode can never gain a second pointer — it names a square, so it serves one mover in one position — and goes
on costing full price forever. Generality gets cheap and specificity stays dear, with nothing having to detect
which is which.

**A rule carries no dispatch; its set supplies it.** A rule says "not unequal offsets", never "a bishop may not
move unequal offsets". Otherwise the white bishop's rule and the black bishop's are different rules however
identical their content, the library can share nothing, and the pointer count stops meaning anything. Written
without it, both bishops point at one rule, pawns have their own because their sets differ rather than because
their rules say so, and "you may not land on your own thing" is a single rule every set points at.

## Models and services

```python
@dataclass(frozen=True, slots=True)
class RuleSet:
    """The rules deciding one kind of mover's moves, held as pointers rather than as rules.

    Pointers so that what several movers share is one rule and not six, and so that what they share can be seen
    rather than declared."""

    of: Value                        # what these are the rules of, as the game names it
    rules: tuple[str, ...]           # into the library


class RuleLibrary:
    """Every rule once, however many sets point at it.

    It keeps nothing about who points where beyond the counting, and a rule nothing points at is forgotten. That
    is not housekeeping: it is what makes a rule's cost fall the more it is used, which is the whole of the
    pressure toward rules rather than tables."""

    def put(self, clause: Clause) -> str: ...
    def at(self, pointer: str) -> Clause: ...
    def pointed_at(self, pointer: str) -> int: ...
    def forgotten(self) -> tuple[str, ...]: ...    # those nothing points at any more
```

Readings gained, and they are the only ones:

- `moves by(x, y)` — the offset itself, which replaces `origin(…)` and `destination(…)` as what a rule may be
  about.
- `first along(way, distance, what)` — casting from where the mover stands, how far to the first occupant and
  what it is.

## What goes

`origin(row, column)` and `destination(row, column)` as things a rule may be *about*. The grid still needs
absolute coordinates to say what stands where, so they do not vanish from the readings — they vanish from the
action's parameters, which is what a rule can name.

## How it is judged

Rules per mover and their lengths, as counts with a header. A bishop's set should be one or two conditions, and
if it is not, this has not worked. The two error counts must not get worse — nothing wrongly refused is the
column that has held all day. And the chess declaration read end to end should contain no chess.

## Settled, having first been written down as open

Four questions were listed here. Writing them out plainly answered all four, which is worth recording as much
as the answers are.

- **Where the window for negative evidence comes from.** It is derivable and it is not a cap. On a grid of R by
  C, no offset outside −(R−1)…(R−1) by −(C−1)…(C−1) can land on the board from *any* square, so nothing outside
  it is refused for a reason the inside does not already show. And within it, whether a move leaves the board
  depends on where the mover stands — from a corner every negative offset goes off the edge — so *you may not
  move off the grid* stays learnable, with abundant evidence. The window comes from the structure the game
  already declared: no number from the integrator, no cut chosen by OMF.
- **What identifies the mover.** A set per colour and kind, because a rule carries no dispatch. The two
  bishops' sets both point at one rule, pawns differ because their sets differ, and nothing is learned twice
  that the library would not fold into one. Had rules carried their own dispatch this would have been a real
  choice with no good answer.
- **Castling and promotion.** Not open: [happenings.md](happenings.md) settled it. The rook moving is a
  consequence the predictor learns, not a second action, so castling is the king moving two sideways and lives
  in the king's set; promotion is the pawn's move with the becoming a queen as a consequence. What remains is
  that promotion needs a parameter saying what to become, which the plan already defers.
- **En passant's removal.** Not open either: `predictor.model.drawn` already says it in its own docstring — the
  piece taken in passing stands at the row of the source and the column of the target. A consequence like any
  other.

## Open

Nothing. If something here is wrong it is wrong in the body, not in a question left unanswered.
