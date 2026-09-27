from openmind.inference.model.formula import And, Atom, Exists, ForAll, Iff, Implies, Not, Or
from openmind.inference.service.clausifier import Clausifier
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Variable


def said(predicate: str, *arguments: object) -> Atom:
    return Atom(Literal(predicate, tuple(arguments)))  # type: ignore[arg-type]


def test_a_thing_said_outright_is_one_clause_of_one_literal() -> None:
    clauses = Clausifier().clauses(said("acts_once"))

    assert clauses == (Clause((Literal("acts_once", ()),)),)


def test_an_implication_becomes_the_conclusion_or_the_premise_denied() -> None:
    formula = Implies(said("owns", Variable("Player")), said("holds", Variable("Player")))

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 1 and len(clauses[0].literals) == 2 and clauses[0].definite


def test_an_implication_s_premise_becomes_the_body_and_its_conclusion_the_head() -> None:
    formula = Implies(said("owns", Variable("Player")), said("holds", Variable("Player")))

    clause = Clausifier().clauses(formula)[0]

    assert clause.head is not None and clause.head.predicate == "holds" and clause.body[0].predicate == "owns"


def test_a_conjunction_gives_one_clause_for_each_of_its_parts() -> None:
    formula = And((said("acts_once"), said("owns", Constant("first"))))

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 2


def test_denying_a_conjunction_gives_one_clause_of_both_denials() -> None:
    formula = Not(And((said("acts_once"), said("owns", Constant("first")))))

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 1 and all(one.negated for one in clauses[0].literals)


def test_denying_for_every_thing_makes_it_there_is_some_thing() -> None:
    formula = Not(ForAll((Variable("Thing"),), said("owned", Variable("Thing"))))

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 1 and clauses[0].ground


def test_there_is_some_thing_becomes_a_thing_invented_for_it() -> None:
    formula = Exists((Variable("Thing"),), said("owned", Variable("Thing")))

    clauses = Clausifier().clauses(formula)

    assert clauses[0].ground and isinstance(clauses[0].literals[0].arguments[0], Functor)


def test_a_thing_invented_under_for_every_depends_on_what_it_was_under() -> None:
    formula = ForAll((Variable("Player"),), Exists((Variable("Thing"),), said("owns", Variable("Player"), Variable("Thing"))))

    clause = Clausifier().clauses(formula)[0]
    invented = clause.literals[0].arguments[1]

    assert isinstance(invented, Functor) and len(invented.arguments) == 1


def test_a_thing_invented_outside_for_every_depends_on_nothing() -> None:
    formula = Exists((Variable("Thing"),), ForAll((Variable("Player"),), said("owns", Variable("Player"), Variable("Thing"))))

    clause = Clausifier().clauses(formula)[0]
    invented = clause.literals[0].arguments[1]

    assert isinstance(invented, Functor) and invented.arguments == ()


def test_an_or_over_an_and_is_distributed_into_a_clause_for_each_way() -> None:
    formula = Or((said("first"), And((said("second"), said("third")))))

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 2


def test_a_clause_holding_something_and_its_own_denial_is_dropped_as_saying_nothing() -> None:
    formula = Or((said("holds"), Not(said("holds"))))

    clauses = Clausifier().clauses(formula)

    assert clauses == ()


def test_an_equivalence_becomes_both_directions() -> None:
    formula = Iff(said("owns", Variable("Player")), said("holds", Variable("Player")))

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 2 and all(clause.definite for clause in clauses)


def test_the_same_letter_used_by_two_quantifiers_does_not_come_out_as_one_thing() -> None:
    formula = And((
        ForAll((Variable("Thing"),), said("owned", Variable("Thing"))),
        ForAll((Variable("Thing"),), said("moved", Variable("Thing"))),
    ))

    clauses = Clausifier().clauses(formula)

    assert clauses[0].variables != clauses[1].variables


def test_a_clause_carries_the_name_and_probability_the_formula_was_stated_with() -> None:
    clauses = Clausifier().clauses(said("acts_once"), "a player acts once", 0.8)

    assert clauses[0].name == "a player acts once" and clauses[0].probability == 0.8


def test_nested_quantifiers_come_out_as_one_clause_saying_the_same_thing() -> None:
    formula = ForAll(
        (Variable("Player"),),
        Implies(said("acts", Variable("Player")), Exists((Variable("Thing"),), said("owns", Variable("Player"), Variable("Thing")))),
    )

    clauses = Clausifier().clauses(formula)

    assert len(clauses) == 1 and clauses[0].definite and len(clauses[0].variables) == 1
