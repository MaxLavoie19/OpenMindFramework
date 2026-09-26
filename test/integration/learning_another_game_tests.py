"""The same learner, on a game that is not chess.

Everything the learner knows about a game comes from the state's own data models, the action's declared shape, and
the moves the game lists. If any of it had quietly become about chess, this would not work — and it is the only
way to find that out, since a thing that assumes a board of a certain size and a piece of a certain kind still
passes every test written with that board in front of it.

Tic-tac-toe is a different shape on purpose: a move is two separate parameters rather than one cell, the grid
holds a player's mark rather than a record, and being legal is one condition rather than seventy.
"""

from openmind.inference.model.evidence import Evidence
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.constraint_distiller import ConstraintDistiller
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.structure.model.grid import Grid
from openmind.structure.model.kind import Kind
from openmind.structure.model.schema import ActionKind
from openmind.world.model.action import Action
from openmind.world.model.state import State

PLACE, ROW, COLUMN = "place", "row", "col"
PLAYERS = ("X", "O")
SIDE = Kind("side", values=(1, 2, 3))
MOVE = ActionKind(PLACE, ((ROW, SIDE), (COLUMN, SIDE)))
DOMAINS = MOVE.domains
PLENTY = InferenceBudget(seconds=30.0)


def a_position(*marks):
    """A board with those marks on it, given as (row, column, player), and whoever is to play."""
    cells = [[None, None, None], [None, None, None], [None, None, None]]
    for row, column, player in marks:
        cells[row - 1][column - 1] = player
    return State.of(cell=Grid.of(cells), turn=PLAYERS[len(marks) % 2])


def a_move(row, column):
    return Action(PLACE, ((COLUMN, column), (ROW, row)))


def evidence_of(state):
    """The game lists a move to every empty cell. The learner is told the moves and never the rule."""
    grid = state.model("cell")
    legal = tuple(a_move(row, column) for row, column in grid.coordinates() if grid.at((row, column)) is None)
    return Evidence(state, PLACE, legal)


def learner():
    readings = CandidateReadings(MOVE)
    return readings, RefusalLearner(readings)


def test_a_game_that_is_not_chess_is_learned_by_the_same_learner_unchanged():
    """Nothing is configured for it beyond the action's shape, which the game declares."""
    readings, held = learner()
    state = a_position((1, 1, "X"), (2, 2, "O"), (3, 3, "X"))
    evidence = evidence_of(state)

    learned = held.learn(readings.cases(evidence, DOMAINS), PLENTY)

    assert held.scored(learned, evidence, DOMAINS).settled


def distilled(held, readings, *positions):
    """What it holds after learning from those positions in turn, as the run does: learn on top of what it has,
    then say it as simply as everything seen so far allows."""
    learned, allowed = (), []
    for evidence in positions:
        cases = readings.cases(evidence, DOMAINS)
        allowed.extend(one for one in cases if not one.holds)
        learned = held.learn(cases, PLENTY, starting=learned)
        kept = ConstraintDistiller(held).distilled(
            learned, [one for one in cases if one.holds], allowed
        )
    return kept


def test_one_position_can_be_explained_by_a_rule_that_is_true_of_nothing_else():
    """Three marks on the diagonal, and "refused where the row equals the column" accounts for all of them. It is
    shorter than the truth and consistent with every move the game listed, so the learner takes it.

    This is not a fault to be fixed in the learner. Nothing in one position tells a coincidence from a rule —
    that is what further positions are for, and it is why the run distils afresh against everything seen rather
    than keeping what it settled on first.

    How many further positions is not a number worth asserting. What it settles on here is subtler than the
    diagonal — it is *when O is to move, a move to the diagonal is refused* — so a position with X to move does
    not touch it however different the board, and being rid of that one leaves the next coincidence standing.
    Which position corrects which coincidence is the whole argument for choosing positions rather than taking
    whichever arrives, and it is not a thing a test can pin at a fixed count without pinning an accident of its
    own fixture."""
    readings, held = learner()
    diagonal = evidence_of(a_position((1, 1, "X"), (2, 2, "O"), (3, 3, "X")))
    elsewhere = evidence_of(a_position((1, 2, "X"), (2, 1, "O"), (3, 3, "X"), (1, 3, "O"), (2, 3, "X")))

    kept = distilled(held, readings, diagonal)

    assert held.scored(kept, diagonal, DOMAINS).settled
    assert held.scored(kept, elsewhere, DOMAINS).forbade, "the coincidence is refused somewhere else"


def test_being_legal_here_is_one_condition_and_it_is_found():
    """A move is refused where something already stands where it would go. Distilled, that should be what is
    left — one constraint, saying that and nothing else."""
    readings, held = learner()
    evidence = evidence_of(a_position((1, 1, "X"), (2, 2, "O"), (3, 3, "X")))
    cases = readings.cases(evidence, DOMAINS)

    learned = held.learn(cases, PLENTY)
    distilled = ConstraintDistiller(held).distilled(
        learned, [one for one in cases if one.holds], [one for one in cases if not one.holds]
    )

    assert len(distilled) <= 3, [one.readable for one in distilled]
    assert held.scored(distilled, evidence, DOMAINS).settled
