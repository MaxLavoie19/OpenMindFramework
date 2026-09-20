# Interfaces: chess movement declared as rules

Waiting for Maxime's OK. Nothing is coded until then.

OMF must enumerate chess's moves itself, from rules it solves, rather than being handed a list. Today chess declares
`move(uci)` and python-chess generates the legal moves, so OMF sees opaque strings: it cannot tell that a move has a
source and a target, that the source holds a queen, or how far a bishop reaches. Nothing it infers can be about
movement, and no relaxation of movement is possible.

## The action

```python
declarer.values(MOVE, SOURCE, my_pieces)        # cells holding one of the player's pieces
declarer.values(MOVE, TARGET, every_cell)       # every cell of the board
declarer.values(MOVE, PROMOTION, promotions)    # a piece kind, or nothing
```

Source, target and promotion are the move. The CSP enumerates the candidates and the constraints say which are legal,
so a reading can speak of what the source holds and where the target is.

## The constraints, one concern each

Each is a rule of its own, so each can be dropped on its own and the relaxation means something:

| Rule | Says |
|---|---|
| it is the player's turn | as now |
| the game goes on | as now |
| the source is the player's own piece | you move your own |
| the target is not the player's own piece | you don't take your own |
| the step suits the piece | the offset between source and target is one the piece makes |
| the path is clear | for a piece that slides, nothing stands between source and target |
| the pawn's step | forward only, one square, two from its start, diagonally only to take, en passant |
| promotion | a pawn reaching the last rank promotes, and nothing else promotes |
| the king is not left attacked | the move doesn't leave the player's own king where the other side could take it |

Castling is its own action, `castle(side)`, with its own constraints: the rights stand, the squares between are empty,
and the king isn't attacked where it stands, passes or lands.

**The steps a piece makes** come from the grid's own vocabulary: a knight's offsets, a bishop's diagonals, a rook's
orthogonals, a queen's both, a king's one step. `Grid.ray(start, direction, blocked)` already walks a line until
something stands in the way, and `Grid.neighbours` gives one step in any direction.

## The one recursion to avoid

"The king is not left attacked" asks whether the other side could take the king — which is another move's legality,
which asks the same question again. It is cut the usual way: an attack is read with the other side's moves as they
would be **without** that rule, so the question is asked once and stops. That rule is therefore the one relaxation
that gives `ignoring check`, and it is the only one that reads the other player's moves.

## What ends a game

Checkmate and stalemate stop being python-chess's: the player to move has no legal action, and the game is lost where
its king is attacked and drawn where it isn't. The fifty-move rule, repetition and insufficient material stay as
they are, read from the state.

## What python-chess is still for

The PGN of a game and the picture of a board — and as an oracle: perft tells us whether the declared rules are right,
which is the only way to know.

## The cost

The CSP will solve perhaps 16 × 64 candidates per position against those constraints, where python-chess generated
about 30 moves directly. Perft to depth 3 takes 0.25 seconds today; declared this way it will take minutes. Accepted:
what OMF gains is a game it can reason about instead of one it can only ask.

## What this opens

- **Reach is enumerable**: one piece on an empty board, its legal targets from every square, closed over until
  nothing new appears — colour-boundedness, most and least reach, moves to cross the board, all by enumeration.
- **Relaxations that shrink the problem**: drop the path rule and pieces jump; drop the step rule and they teleport;
  drop the king rule and check is ignored — each a game of its own to reason in.
- **Readings about movement**: what a move's source holds, what its target holds, how far it goes.

## Open

1. **Where the steps live.** Either each piece kind's offsets are declared as data chess owns, and one rule reads
   them; or one rule per piece kind. The first is shorter and keeps the pattern in one readable place; the second
   makes each piece's movement its own relaxable rule.
2. **Whether the source domain is narrowed.** Restricting source to the player's own pieces makes the enumeration 16 ×
   64 rather than 64 × 64, but it hides from OMF that moving an empty square is illegal — a rule it could otherwise
   learn. Narrow it for speed, or leave it wide and let the constraint say so?
