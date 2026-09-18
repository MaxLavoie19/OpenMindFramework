from openmind.inference.constant.logic_constant import INT
from openmind.inference.mapper.formula_text_mapper import FormulaTextMapper
from openmind.inference.model.formula import And, Atom, Compare, Exists, ForAll, Iff, Implies, Not, Or, Truth
from openmind.inference.model.sort import Sort
from openmind.inference.model.symbol import FunctionSymbol, PredicateSymbol
from openmind.inference.model.term import Application, Constant, Number, Variable
from openmind.inference.service.theory_library import TheoryLibrary

LIBRARY = TheoryLibrary()
THING = Sort("Thing")


def test_set_relations_and_operations_are_written_between_their_arguments() -> None:
    x, a, b = Variable("x", THING), Constant("A", LIBRARY.set_sort(THING)), Constant("B", LIBRARY.set_sort(THING))
    union = Application(LIBRARY.operation("union", THING), (a, b))

    text = FormulaTextMapper().to_text(ForAll((x,), Implies(Atom(LIBRARY.member(THING), (x, a)), Atom(LIBRARY.member(THING), (x, union)))))

    assert text == "∀x ((x ∈ A) → (x ∈ (A ∪ B)))"
    assert FormulaTextMapper().to_text(Atom(LIBRARY.subset(THING), (a, b))) == "A ⊆ B"


def test_connectives_quantifiers_comparisons_and_calls_read_as_logic() -> None:
    n = Variable("n", INT)
    f = Application(FunctionSymbol("f", (INT,), INT), (n,))
    even = Atom(PredicateSymbol("even", (INT,)), (n,))
    formula = Exists((n,), And((Compare(">=", n, Number(0)), Or((Not(even), Iff(even, Truth(True)))), Compare("<", f, Number(3)))))

    assert FormulaTextMapper().to_text(formula) == "∃n ((n ≥ 0) ∧ (¬even(n) ∨ (even(n) ↔ ⊤)) ∧ (f(n) < 3))"
