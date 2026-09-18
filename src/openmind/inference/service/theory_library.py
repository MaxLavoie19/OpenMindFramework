from openmind.inference.constant.logic_constant import (
    BEST_PREFIX,
    DIFFERENCE,
    INTERSECTION,
    MEMBER,
    SELECTION_PREFIX,
    SUBSET,
    UNION,
    WORST_PREFIX,
)
from openmind.inference.model.formula import And, Atom, Compare, Equal, Exists, ForAll, Formula, Iff, Implies, Not, Or
from openmind.inference.model.sort import SetSort, Sort
from openmind.inference.model.symbol import FunctionSymbol, PredicateSymbol
from openmind.inference.model.term import Application, Variable
from openmind.inference.model.theory import Axiom, Theory


class TheoryLibrary:
    """OMF's built-in definitions, for any sort of element: nothing here knows a game.

    - `sets(element)`: membership, subset, union, intersection and difference, each defined by its members, and
      extensionality, sets with the same members being equal;
    - `selection(name, variable, clause)`: the subset of a set whose members the clause holds for;
    - `best(value)` and `worst(value)`: the highest and lowest value over a set, at least (at most) every member's value,
      and reached by a member when the set has one."""

    def set_sort(self, element: Sort) -> SetSort:
        """The sort of the sets of the element sort."""
        return SetSort(f"Set_{element.name}", element)

    def member(self, element: Sort) -> PredicateSymbol:
        return PredicateSymbol(MEMBER, (element, self.set_sort(element)))

    def subset(self, element: Sort) -> PredicateSymbol:
        sets = self.set_sort(element)
        return PredicateSymbol(SUBSET, (sets, sets))

    def operation(self, name: str, element: Sort) -> FunctionSymbol:
        """Union, intersection or difference, by name."""
        sets = self.set_sort(element)
        return FunctionSymbol(name, (sets, sets), sets)

    def sets(self, element: Sort) -> Theory:
        sets = self.set_sort(element)
        member, subset = self.member(element), self.subset(element)
        x, first, second = Variable("x", element), Variable("X", sets), Variable("Y", sets)

        def holds(where: object) -> Formula:
            return Atom(member, (x, where))

        axioms = [
            Axiom(
                SUBSET,
                ForAll((first, second), Iff(Atom(subset, (first, second)), ForAll((x,), Implies(holds(first), holds(second))))),
            ),
            Axiom(
                "extensionality",
                ForAll((first, second), Implies(ForAll((x,), Iff(holds(first), holds(second))), Equal(first, second))),
            ),
        ]
        operations = []
        for name, combined in (
            (UNION, Or((holds(first), holds(second)))),
            (INTERSECTION, And((holds(first), holds(second)))),
            (DIFFERENCE, And((holds(first), Not(holds(second))))),
        ):
            operation = self.operation(name, element)
            operations.append(operation)
            axioms.append(
                Axiom(name, ForAll((first, second, x), Iff(holds(Application(operation, (first, second))), combined)))
            )
        return Theory(f"sets of {element.name}", (member, subset, *operations), tuple(axioms))

    def selection(self, name: str, variable: Variable, clause: Formula) -> Theory:
        """The subset of any set of the variable's sort whose members the clause, read at the variable, holds for."""
        element = variable.sort
        sets = self.set_sort(element)
        selected = FunctionSymbol(SELECTION_PREFIX + name, (sets,), sets)
        whole = Variable("X", sets)
        member = self.member(element)
        definition = ForAll(
            (whole, variable),
            Iff(Atom(member, (variable, Application(selected, (whole,)))), And((Atom(member, (variable, whole)), clause))),
        )
        return Theory(f"selection {name}", (selected,), (Axiom(SELECTION_PREFIX + name, definition),))

    def best(self, value: FunctionSymbol) -> Theory:
        """The highest of a value over a set."""
        return self._extreme(BEST_PREFIX, ">=", value)

    def worst(self, value: FunctionSymbol) -> Theory:
        """The lowest of a value over a set."""
        return self._extreme(WORST_PREFIX, "<=", value)

    def _extreme(self, prefix: str, operator: str, value: FunctionSymbol) -> Theory:
        if len(value.arguments) != 1:
            raise ValueError(f"A value over a set reads one member, not {len(value.arguments)} arguments: {value.name}")
        (element,) = value.arguments
        sets = self.set_sort(element)
        extreme = FunctionSymbol(prefix + value.name, (sets,), value.result)
        whole, x = Variable("X", sets), Variable("x", element)
        member = self.member(element)
        of_whole = Application(extreme, (whole,))
        of_member = Application(value, (x,))
        bound = ForAll((whole, x), Implies(Atom(member, (x, whole)), Compare(operator, of_whole, of_member)))
        reached = ForAll(
            (whole,),
            Implies(Exists((x,), Atom(member, (x, whole))), Exists((x,), And((Atom(member, (x, whole)), Equal(of_member, of_whole))))),
        )
        name = prefix + value.name
        return Theory(name, (extreme,), (Axiom(f"{name} bound", bound), Axiom(f"{name} reached", reached)))
