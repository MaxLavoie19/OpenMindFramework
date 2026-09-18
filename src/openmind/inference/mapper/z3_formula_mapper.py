from collections.abc import Iterable

import z3

from openmind.inference.constant.logic_constant import ARITHMETIC, BOOL, INT, REAL
from openmind.inference.model.formula import (
    And,
    Atom,
    Compare,
    Equal,
    Equisatisfiable,
    Exists,
    ForAll,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Truth,
)
from openmind.inference.model.sort import Sort
from openmind.inference.model.symbol import FunctionSymbol, PredicateSymbol
from openmind.inference.model.term import Application, Constant, Number, Term, Variable

_COMPARE = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}
_ARITHMETIC = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b}
_BACK_COMPARE = {z3.Z3_OP_LE: "<=", z3.Z3_OP_LT: "<", z3.Z3_OP_GE: ">=", z3.Z3_OP_GT: ">"}
_BACK_ARITHMETIC = {z3.Z3_OP_ADD: "+", z3.Z3_OP_SUB: "-", z3.Z3_OP_MUL: "*", z3.Z3_OP_DIV: "/", z3.Z3_OP_IDIV: "/"}


class Z3FormulaMapper:
    """Maps formulas to Z3 expressions in a context, and Z3's expressions, the formulas of its proofs included, back to
    formulas. Truth values, whole numbers and real numbers are Z3's own sorts; every other sort is an uninterpreted sort of
    the same name. Arithmetic and comparisons between numbers are Z3's own; every other symbol is an uninterpreted function
    of the same name. Back, a sort is the one of that name among those given, or a new sort of that name; Z3 may have
    written a formula in a form of its own, such as `f(n) >= 2` as `2 <= f(n)` inside a quantifier."""

    def to_z3(self, formula: Formula, context: z3.Context) -> z3.BoolRef:
        return self._formula(formula, context, {})

    def from_z3(self, expression: z3.ExprRef, sorts: Iterable[Sort] = ()) -> Formula:
        known = {sort.name: sort for sort in sorts}
        return self._back_formula(expression, known, [])

    def sort(self, sort: Sort, context: z3.Context) -> z3.SortRef:
        if sort == BOOL:
            return z3.BoolSort(context)
        if sort == INT:
            return z3.IntSort(context)
        if sort == REAL:
            return z3.RealSort(context)
        return z3.DeclareSort(sort.name, context)

    def _formula(self, formula: Formula, context: z3.Context, bound: dict[Variable, z3.ExprRef]) -> z3.BoolRef:
        match formula:
            case Truth(value):
                return z3.BoolVal(value, context)
            case Atom(symbol, arguments):
                function = z3.Function(symbol.name, *(self.sort(sort, context) for sort in symbol.arguments), z3.BoolSort(context))
                return function(*(self._argument(argument, context, bound) for argument in arguments))
            case Equal(left, right):
                return self._term(left, context, bound) == self._term(right, context, bound)
            case Compare(operator, left, right):
                return _COMPARE[operator](self._term(left, context, bound), self._term(right, context, bound))
            case Not(body):
                return z3.Not(self._formula(body, context, bound))
            case And(parts):
                return z3.And(*(self._formula(part, context, bound) for part in parts)) if parts else z3.BoolVal(True, context)
            case Or(parts):
                return z3.Or(*(self._formula(part, context, bound) for part in parts)) if parts else z3.BoolVal(False, context)
            case Implies(premise, conclusion):
                return z3.Implies(self._formula(premise, context, bound), self._formula(conclusion, context, bound))
            case Iff(left, right) | Equisatisfiable(left, right):
                return self._formula(left, context, bound) == self._formula(right, context, bound)
            case ForAll(variables, body) | Exists(variables, body):
                inner = dict(bound)
                constants = []
                for variable in variables:
                    constants.append(z3.Const(variable.name, self.sort(variable.sort, context)))
                    inner[variable] = constants[-1]
                quantifier = z3.ForAll if isinstance(formula, ForAll) else z3.Exists
                return quantifier(constants, self._formula(body, context, inner))
        raise ValueError(f"Not a formula: {formula!r}")

    def _argument(self, argument: object, context: z3.Context, bound: dict[Variable, z3.ExprRef]) -> z3.ExprRef:
        if isinstance(argument, (Variable, Constant, Number, Application)):
            return self._term(argument, context, bound)
        return self._formula(argument, context, bound)  # type: ignore[arg-type]

    def _term(self, term: Term, context: z3.Context, bound: dict[Variable, z3.ExprRef]) -> z3.ExprRef:
        match term:
            case Variable() if term in bound:
                return bound[term]
            case Variable(name, sort) | Constant(name, sort):
                return z3.Const(name, self.sort(sort, context))
            case Number(value):
                return z3.IntVal(value, context) if isinstance(value, int) else z3.RealVal(value, context)
            case Application(symbol, arguments):
                values = [self._argument(argument, context, bound) for argument in arguments]
                if symbol.name in ARITHMETIC and len(values) == 2 and symbol.result in (INT, REAL):
                    return _ARITHMETIC[symbol.name](values[0], values[1])
                function = z3.Function(
                    symbol.name, *(self.sort(sort, context) for sort in symbol.arguments), self.sort(symbol.result, context)
                )
                return function(*values)
        raise ValueError(f"Not a term: {term!r}")

    def _back_sort(self, sort: z3.SortRef, known: dict[str, Sort]) -> Sort:
        if sort.kind() == z3.Z3_BOOL_SORT:
            return BOOL
        if sort.kind() == z3.Z3_INT_SORT:
            return INT
        if sort.kind() == z3.Z3_REAL_SORT:
            return REAL
        return known.get(sort.name(), Sort(sort.name()))

    def _back_formula(self, expression: z3.ExprRef, known: dict[str, Sort], scopes: list[tuple[Variable, ...]]) -> Formula:
        if z3.is_quantifier(expression):
            variables = tuple(
                Variable(expression.var_name(index), self._back_sort(expression.var_sort(index), known))
                for index in range(expression.num_vars())
            )
            body = self._back_formula(expression.body(), known, [*scopes, variables])
            return ForAll(variables, body) if expression.is_forall() else Exists(variables, body)
        if z3.is_true(expression) or z3.is_false(expression):
            return Truth(z3.is_true(expression))
        if z3.is_var(expression) or not z3.is_app(expression):
            raise ValueError(f"Not a formula: {expression}")
        kind = expression.decl().kind()
        children = expression.children()
        if kind == z3.Z3_OP_NOT:
            return Not(self._back_formula(children[0], known, scopes))
        if kind == z3.Z3_OP_AND:
            return And(tuple(self._back_formula(child, known, scopes) for child in children))
        if kind == z3.Z3_OP_OR:
            return Or(tuple(self._back_formula(child, known, scopes) for child in children))
        if kind == z3.Z3_OP_IMPLIES:
            return Implies(self._back_formula(children[0], known, scopes), self._back_formula(children[1], known, scopes))
        if kind in (z3.Z3_OP_EQ, z3.Z3_OP_IFF, z3.Z3_OP_OEQ) and len(children) == 2:
            if children[0].sort().kind() == z3.Z3_BOOL_SORT:
                left, right = self._back_formula(children[0], known, scopes), self._back_formula(children[1], known, scopes)
                return Equisatisfiable(left, right) if kind == z3.Z3_OP_OEQ else Iff(left, right)
            return Equal(self._back_term(children[0], known, scopes), self._back_term(children[1], known, scopes))
        if kind in _BACK_COMPARE:
            return Compare(_BACK_COMPARE[kind], self._back_term(children[0], known, scopes), self._back_term(children[1], known, scopes))
        symbol = PredicateSymbol(expression.decl().name(), tuple(self._back_sort(child.sort(), known) for child in children))
        return Atom(symbol, tuple(self._back_argument(child, known, scopes) for child in children))

    def _back_argument(self, expression: z3.ExprRef, known: dict[str, Sort], scopes: list[tuple[Variable, ...]]) -> object:
        if expression.sort().kind() == z3.Z3_BOOL_SORT:
            return self._back_formula(expression, known, scopes)
        return self._back_term(expression, known, scopes)

    def _back_term(self, expression: z3.ExprRef, known: dict[str, Sort], scopes: list[tuple[Variable, ...]]) -> Term:
        if z3.is_var(expression):
            bound = [variable for scope in scopes for variable in scope]
            index = z3.get_var_index(expression)
            if index < len(bound):
                return bound[len(bound) - 1 - index]
            return Variable(f"v{index - len(bound)}", self._back_sort(expression.sort(), known))
        if z3.is_int_value(expression):
            return Number(expression.as_long())
        if z3.is_rational_value(expression):
            fraction = expression.as_fraction()
            return Number(fraction.numerator if fraction.denominator == 1 else float(fraction))
        children = expression.children()
        kind = expression.decl().kind()
        sort = self._back_sort(expression.sort(), known)
        if kind in _BACK_ARITHMETIC and len(children) >= 2:
            operand_sorts = tuple(self._back_sort(child.sort(), known) for child in children[:2])
            symbol = FunctionSymbol(_BACK_ARITHMETIC[kind], operand_sorts, sort)
            term: Term = Application(symbol, (self._back_term(children[0], known, scopes), self._back_term(children[1], known, scopes)))
            for child in children[2:]:
                term = Application(symbol, (term, self._back_term(child, known, scopes)))
            return term
        if not children:
            return Constant(expression.decl().name(), sort)
        symbol = FunctionSymbol(expression.decl().name(), tuple(self._back_sort(child.sort(), known) for child in children), sort)
        return Application(symbol, tuple(self._back_argument(child, known, scopes) for child in children))
