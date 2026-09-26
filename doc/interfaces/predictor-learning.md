# Interfaces: learning what an action does

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

The second of the two rule types. Constraints say which actions a game allows; predictors say what an action
leads to. Between them they are the whole of a game's rules, and the constraint half has now run into the wall
that only this half can get past: the king must not be left attacked is a fact about a position that does not
exist yet, and so is a king castling through an attack.

## What is already there and is kept

The old design left models for this that are right and were never the problem — the problem was the learners
above them.

- **`world.model.change`** — `Placed`, `Removed`, `Moved`, `Told`, applied in order. Its docstring makes the
  argument this whole half rests on: *an effect that hands back a position says what the position became and
  never what the action did*. A move that takes a piece and a move onto an empty square differ in the same two
  squares, and a pawn taken in passing stands on neither square the move names. Said as changes, what an action
  did is there to be read rather than inferred.
- **`predictor.model.consequence.Consequence`** — one change, said in terms of the action rather than of the
  position it left: *this action removes something, and the square it removes from is the row of where it started
  and the column of where it lands*. That is what holds in every position rather than in the one it was read
  from.
- **`predictor.model.drawn`** — `Row`, `Column`, `Standing`, `Asked`: the ways a change's places and values are
  drawn from the action and the position before it.
- **`predictor.model.outcome_distribution.OutcomeDistribution`** — outcomes with their probabilities.

## What changes

**`Consequence.when` becomes our clauses.** It currently holds `Covering` from `covering_learner`, one of the
four abandoned learners. The conditions under which a change happens are the same kind of thing as the conditions
under which an action is refused — readings of the position and the action — so they are `Clause`s, learned by
the machinery that now learns constraints.

**`Drawn` takes the schema's names.** `Row` and `Column` were written when a cell was two coordinates by
assumption. A game declares its parameters' places and what they are called, so a drawn place is *that place of
that parameter*, and a game whose action has one place or five is said the same way.

## Models

```python
@dataclass(frozen=True, slots=True)
class Outcome:
    """One thing an action may lead to: the changes it makes, and how often it makes them.

    Changes rather than a position, for the reason `world.model.change` gives. A game whose action changes
    everything — a hand redealt, dice rolled — says so as the changes that redeal it, and nothing here needs a
    second shape for that case.

    `chance` is 1.0 for a game that does the same thing every time, which is every outcome in chess."""

    changes: tuple[Change, ...]
    chance: float = 1.0


@dataclass(frozen=True, slots=True)
class Watched:
    """One action played and what was seen of it: the position before, the action, and the changes the game
    reported.

    **The game reports the changes; OMF does not work them out by comparing positions.** Two positions differ in
    some squares and never say which difference was the point, so a learner given only the before and after is
    being asked to guess the thing it is supposed to learn the conditions of.

    `whole` says whether what was seen is all there was. Chess shows the whole outcome; liar's dice shows a
    player their own dice and nothing else. A prediction about what was not seen cannot be scored when it is
    made, and saying so here is what keeps a learner from treating silence as agreement."""

    where: State
    action: Action
    changes: tuple[Change, ...]
    whole: bool = True
```

## Services

```python
class ConsequenceLearner:
    """What an action does, learned from having watched it done.

    Each change seen is a case, as each candidate was a case for the constraints, and the same growing and
    widening applies: a change said outright, widened against further sightings until what is left is the change
    said in terms of the action. Where two sightings differ in which square was emptied and agree that it was the
    square the move started from, what comes out is "the square it started from" — by construction, not by
    search.

    The conditions are learned the same way and for the same reason. Most of a move has none: it happens every
    time. What has conditions is the part a game finds hard to state — taking in passing, castling, promotion —
    and those are the consequences whose `when` is not empty."""

    def learn(
        self, watched: Sequence[Watched], budget: InferenceBudget, starting: Sequence[Consequence] = ()
    ) -> tuple[Consequence, ...]: ...

    def outcomes(self, consequences: Sequence[Consequence], state: State, action: Action) -> OutcomeDistribution: ...
    """What those consequences say that action leads to there: the changes whose conditions hold, in order, with
    the chance of each outcome."""


class PredictedPosition:
    """The position an action leads to, to be read and asked about like any other.

    This is what the constraint half is waiting for. A constraint may not say "the king must not be left
    attacked" over the board as it stands, because that is a fact about a board that does not exist; given the
    position the move leads to, it is a fact about a board like any other, and the readings and the constraints
    work on it unchanged.

    It is also what the hypothetical needs. Asking whether an enemy piece could reach a square means asking of a
    position where it is their turn, and whose turn it is changes by a `Told` — so the counterfactual is a
    prediction, not a special case."""

    def after(self, state: State, outcome: Outcome) -> State: ...
```

## What this unblocks, and what it does not

**Unblocks.** King safety, and castling through an attack, by reading the position an action leads to. And the
counterfactual behind "could an enemy piece reach this square", since whose turn it is is a change like any
other.

**Does not unblock.** Promotion still needs a third parameter that exists only under a condition — a conditional
variable in the constraint problem, which is neither half of this.

## Open

- **Whether a game must report its changes, or may hand back the position it led to.** Reporting is what makes
  the learning possible, and it is also one more thing every game must say about itself. A game that hands back a
  position could have its changes worked out badly, which is worse than not at all.
- **How a prediction about what was not seen is scored.** In chess it never arises. In liar's dice every
  prediction about another player's hand is of this kind, and scoring it means waiting for a reveal that may
  never come.
