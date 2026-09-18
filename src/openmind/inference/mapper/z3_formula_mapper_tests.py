import z3

from openmind.inference.constant.logic_constant import INT
from openmind.inference.mapper.z3_formula_mapper import Z3FormulaMapper
from openmind.inference.model.formula import Atom, Compare, Equal, Exists, ForAll, Iff, Implies, Not, Or
from openmind.inference.model.sort import Sort
from openmind.inference.model.symbol import FunctionSymbol, PredicateSymbol
from openmind.inference.model.term import Application, Constant, Number, Variable
from openmind.inference.service.theory_library import TheoryLibrary

LIBRARY = TheoryLibrary()
THING = Sort("Thing")


def test_a_formula_goes_to_z3_and_comes_back_the_same() -> None:
    # Written as Z3 keeps it: inside a quantifier, Z3 writes `f(n) >= 2` as `2 <= f(n)`.
    mapper, context = Z3FormulaMapper(), z3.Context()
    x, n = Variable("x", THING), Variable("n", INT)
    sets = LIBRARY.set_sort(THING)
    a, b = Constant("A", sets), Constant("B", sets)
    f = FunctionSymbol("f", (INT,), INT)
    formula = ForAll(
        (x, n),
        Iff(
            Implies(Atom(LIBRARY.member(THING), (x, a)), Not(Atom(LIBRARY.member(THING), (x, b)))),
            Or((Compare("<=", Number(2), Application(f, (n,))), Equal(Application(FunctionSymbol("+", (INT, INT), INT), (n, Number(1))), n))),
        ),
    )

    assert mapper.from_z3(mapper.to_z3(formula, context), (THING, sets)) == formula


def test_sorts_not_given_back_are_named_after_z3_s() -> None:
    mapper, context = Z3FormulaMapper(), z3.Context()
    y = Variable("y", Sort("Other"))
    formula = Exists((y,), Atom(PredicateSymbol("p", (Sort("Other"),)), (y,)))

    assert mapper.from_z3(mapper.to_z3(formula, context)) == formula
