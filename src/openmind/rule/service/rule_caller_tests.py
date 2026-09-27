import pickle

import pytest

from openmind.inference.service.clause_constraint import ClauseConstraint
from openmind.inference.service.refusal_learner import REFUSED
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.statement.model.clause import Clause
from openmind.predictor.service.drawn_effects import DrawnEffects
from openmind.rule.model.clause_rule import ClauseRule
from openmind.statement.model.consequence import Consequence
from openmind.rule.model.consequence_rule import ConsequenceRule
from openmind.statement.model.drawn import column, row
from openmind.statement.model.literal import Literal
from openmind.rule.model.python_rule import PythonRule
from openmind.statement.model.term import Constant, Number
from openmind.rule.service.rule_caller import RuleCaller
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.model.state import State

NAMES = CellNames(("a", "b"), ("2", "1"))


def a_position() -> State:
    return State.of(grid=Grid.of([["a thing", None], [None, None]], NAMES), turn="white")


def refused_where(*literals: Literal) -> ClauseRule:
    """One learned constraint, in the shape the constraint learner declares them in."""
    return ClauseRule(Clause((Literal(REFUSED, ()), *(one.denied for one in literals))))


def running_clauses() -> RuleCaller:
    """A caller built to run learned clauses, as one assembling an induced game would build it."""
    return RuleCaller(RuleCompiler(), RuleRunner(StateNamespaceMapper()), ClauseConstraint())


def test_a_learned_clause_is_answered_rather_than_called():
    """The whole point of stage one: a rule OMF induced can now decide whether a move is legal.

    Before this there was no branch for a clause at all, so a constraint the engine had worked out for itself
    could be declared, stored and reasoned over, and never once consulted."""
    staying = refused_where(Literal("origin", (Number(1), Number(1))))
    caller = running_clauses()

    assert caller.value(staying, a_position(), {"origin": "a2", "destination": "b2"}) is False
    assert caller.value(staying, a_position(), {"origin": "b2", "destination": "a2"}) is True


def test_a_clause_with_nothing_to_run_it_says_what_is_missing():
    """It used to raise `TypeError: 'ClauseRule' object is not callable` from inside the solver, where nothing
    said what was really wrong. A caller that cannot run clauses should say that it cannot run clauses."""
    caller = create_rule_caller()

    with pytest.raises(ValueError, match="run clauses"):
        caller.value(refused_where(Literal("turn", (Constant("white"),))), a_position(), {"origin": "a2"})


def test_a_caller_given_nothing_to_run_learned_rules_with_still_travels_to_a_worker():
    """A rule caller crosses to worker processes, so what it holds has to cross too. Wiring the learned callers
    in by default made every caller in the system unpicklable at once, because one of them holds lambdas
    written inside a constructor."""
    assert pickle.loads(pickle.dumps(create_rule_caller())) is not None


def test_a_clause_passes_the_worker_check_and_survives_being_sent_to_one():
    """A closure over the learned clauses is what this check refuses, and refusing it is right — a worker is
    started fresh and cannot find a function defined inside another. A clause is data, so it travels."""
    learned = refused_where(Literal("turn", (Constant("white"),)))
    caller = running_clauses()

    caller.check(learned)

    assert pickle.loads(pickle.dumps(learned)) == learned


def test_a_clause_reads_in_a_log_as_the_logic_it_is():
    caller = running_clauses()

    said = caller.source(refused_where(Literal("turn", (Constant("white"),))))

    assert "refused" in said and "turn" in said


def test_what_a_clause_reads_is_unaffected_by_how_python_rules_are_read():
    """The two kinds share a caller and must not share a code path: source says which parameters it reads, and a
    clause reads the candidate whole."""
    caller = running_clauses()

    assert caller.value(PythonRule("turn == 'white'"), a_position(), {"origin": "a2"}) is True
    assert caller.prepare(refused_where(Literal("turn", (Constant("black"),))), ("origin", "destination")).arguments == (
        "origin",
        "destination",
    )


def emptying() -> ConsequenceRule:
    """What an action does, as OMF learns it: the square the mover came from empties."""
    return ConsequenceRule((Consequence("Removed", "move", "grid", (row("origin"), column("origin"))),))


def making_them_happen() -> RuleCaller:
    """A caller built to run what was learned an action does, as one assembling an induced game would."""
    return RuleCaller(
        RuleCompiler(),
        RuleRunner(StateNamespaceMapper()),
        consequence_caller=DrawnEffects(acting=lambda state: state.value("turn"), players=("white", "black")),
    )


def test_what_was_learned_an_action_does_is_made_to_happen():
    """The other half of a learned game. A rule OMF induced can decide whether a move is legal; without this
    nothing can play the move it decided about, because `EffectsRunner` raises with no effects rule."""
    after = making_them_happen().apply(emptying(), a_position(), {"origin": "a2", "destination": "b2"})

    assert after.model("grid").at((1, 1)) is None


def test_consequences_with_nothing_to_make_them_happen_say_what_is_missing():
    with pytest.raises(ValueError, match="make consequences happen"):
        create_rule_caller().apply(emptying(), a_position(), {"origin": "a2"})


def test_what_an_action_does_passes_the_worker_check_and_survives_being_sent_to_one():
    """Consequences are data, like a clause and unlike a closure, so a worker rebuilds them rather than having
    to find them."""
    learned = emptying()
    making_them_happen().check(learned)

    assert pickle.loads(pickle.dumps(learned)) == learned


def test_what_an_action_does_reads_in_a_log_as_what_it_does():
    assert "removed grid" in making_them_happen().source(emptying())
