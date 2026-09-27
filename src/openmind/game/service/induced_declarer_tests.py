from dataclasses import dataclass

from openmind.game.service.induced_declarer import InducedDeclarer
from openmind.inference.service.clause_constraint import ClauseConstraint
from openmind.inference.service.refusal_learner import REFUSED
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, EFFECTS, INITIAL, LISTING, PLAYERS, VALUES
from openmind.knowledge.constant.task_constant import SIMULATION
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.predictor.service.drawn_effects import DrawnEffects
from openmind.csp.factory.csp_factory import create_solver
from openmind.predictor.factory.predictor_factory import create_rule_predictor
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.service.simulation import Simulation
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.statement.model.clause import Clause
from openmind.statement.model.consequence import Consequence
from openmind.statement.model.drawn import Always, Other, Place
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number
from openmind.rule.service.rule_caller import RuleCaller
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_runner import RuleRunner
from openmind.structure.model.grid import Grid
from openmind.structure.model.kind import Kind
from openmind.structure.model.map import Map
from openmind.structure.model.record import Record
from openmind.structure.model.schema import ActionKind, GridKind, Schema
from openmind.world.model.players import Players
from openmind.world.model.state import State

SIDES = ("first", "second")
MARK = Kind("mark", values=("a mark",))


@dataclass(frozen=True, slots=True)
class Cell(Record):
    """Where a mark goes, as a game of parts says it.

    A `Record` and not any pair of numbers: what OMF reads a parameter's named places out of is a record, so a
    game whose parameter is a plain tuple has a parameter nothing can decompose — no reading of its row, and no
    drawing of a square from it."""

    row: int
    column: int


CELL = Kind(
    "cell",
    parts=(("row", Kind("row", values=(1, 2))), ("column", Kind("column", values=(1, 2)))),
    builds=Cell,
)


def a_start() -> State:
    """A two by two board, nobody having played, nobody paid."""
    return State.of(
        grid=Grid.of([[None, None], [None, None]]),
        turn="first",
        payoff=Map.of({"first": None, "second": None}),
    )


def a_schema() -> Schema:
    return Schema(
        models=(("grid", GridKind(MARK.or_nothing(), (2, 2))),),
        actions=(ActionKind("place", (("cell", CELL),)),),
    )


def refusing_a_taken_square() -> Clause:
    """What a learner works out from watching: a square already holding something takes nothing more."""
    return Clause(
        (
            Literal(REFUSED, ()),
            Literal("grid", (Number(1), Number(1), Constant("a mark"))).denied,
            Literal("cell", (Number(1), Number(1))).denied,
        )
    )


def marking() -> tuple[Consequence, ...]:
    """What a learner works out an action does: it puts a mark down, and the turn passes."""
    return (
        Consequence("Placed", "place", "grid", (Place("cell", "row"), Place("cell", "column")), value=Always("a mark")),
        Consequence("Told", "place", "turn", value=Other(), order=1),
    )


def running() -> RuleCaller:
    """A caller that can run what was learned, including whose turn it becomes.

    Both halves are said outright and neither is defaulted: a rule caller travels to worker processes and the
    clause caller cannot, so a game that plays what it learned is built with one rather than every caller in
    the system carrying it. Which of a position's models says who is acting is the game's to tell too."""
    return create_rule_caller(
        clause_caller=ClauseConstraint(),
        consequence_caller=DrawnEffects(acting=lambda state: state.value("turn"), players=SIDES),
    )


def playing(knowledge: KnowledgeBase, context: str):
    """The induced game, given a simulation that can read whose turn it is — the same caller throughout, since
    the effects need it as much as anything else."""
    caller = running()
    return create_rule_based_game(knowledge, context, simulation=Simulation(create_solver(caller), create_rule_predictor(caller), caller))


def induced(knowledge: KnowledgeBase, **held) -> str:
    fields = {
        "context": "an induced game",
        "action": "place",
        "starts_at": a_start(),
        "played_by": Players(SIDES, "payoff"),
        "schema": a_schema(),
        "refused": (refusing_a_taken_square(),),
        "does": marking(),
    }
    return InducedDeclarer(rule_caller=running()).declare(knowledge, **{**fields, **held})


def test_a_game_is_declared_out_of_what_was_learned_of_one(knowledge: KnowledgeBase) -> None:
    """Every kind a game is assembled from is now something OMF can work out for itself, and declared together
    they are a game."""
    context = induced(knowledge)

    ruleset = knowledge.ruleset_named(knowledge.context_named(context).id, SIMULATION)
    kinds = {rule.kind for rule, _ in knowledge.ruleset_rules(ruleset.id)}
    assert {INITIAL, PLAYERS, VALUES, CONSTRAINT, EFFECTS} <= kinds


def test_it_declares_no_listing_and_no_ending(knowledge: KnowledgeBase) -> None:
    """A listing would mean playing the declared game while believing it was playing the learned one. An ending
    would be borrowing a judgement it has not earned: a position its own rules leave no move in is over as far
    as it knows."""
    context = induced(knowledge)

    ruleset = knowledge.ruleset_named(knowledge.context_named(context).id, SIMULATION)
    kinds = {rule.kind for rule, _ in knowledge.ruleset_rules(ruleset.id)}
    assert LISTING not in kinds
    assert "ending" not in kinds


def test_the_induced_game_lists_the_moves_its_own_learned_rules_leave(knowledge: KnowledgeBase) -> None:
    """The whole point, and the thing nothing in the literature has done: a player built out of induced rules.

    Four squares, and the one already holding a mark is refused by what was learned — so three remain."""
    context = induced(knowledge, starts_at=a_start().with_model("grid", Grid.of([["a mark", None], [None, None]])))
    game = playing(knowledge, context)

    actions = game.actions(game.start(), player="first")

    assert len(actions) == 3
    assert all(action.parameters[0][1] != Cell(1, 1) for action in actions)


def test_playing_a_move_of_the_induced_game_leaves_what_was_learned_it_leaves(knowledge: KnowledgeBase) -> None:
    """Legality alone is half a game. `EffectsRunner` raises without an effects rule, so a game that knows what
    is legal and not what a move does cannot take a single step."""
    game = playing(knowledge, induced(knowledge))
    start = game.start()

    after = game.outcomes(start, game.actions(start, player="first")[0]).outcomes[0][0]

    assert sum(1 for at in after.model("grid").coordinates() if after.model("grid").at(at) == "a mark") == 1
    assert after.value("turn") == "second"
