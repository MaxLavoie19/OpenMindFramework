from openmind.inference.constant.logic_constant import ARITHMETIC, DIFFERENCE, INTERSECTION, MEMBER, SUBSET, UNION
from openmind.inference.model.formula import (
    And,
    Atom,
    Compare,
    Equal,
    Equisatisfiable,
    Exists,
    ForAll,
    Iff,
    Implies,
    Not,
    Or,
    Truth,
)
from openmind.inference.model.term import Application, Constant, Number, Variable

_COMPARE = {"<=": "≤", "<": "<", ">=": "≥", ">": ">"}
_RELATIONS = {MEMBER: "∈", SUBSET: "⊆"}
_OPERATIONS = {UNION: "∪", INTERSECTION: "∩", DIFFERENCE: "∖"}


class FormulaTextMapper:
    """Writes formulas and terms as text: `∀x (x ∈ A → x ∈ B)`. Membership, subset, union, intersection, difference,
    comparisons and arithmetic are written between their arguments, every other symbol as a function call; a compound
    part of a formula is wrapped in parentheses."""

    def to_text(self, node: object) -> str:
        match node:
            case Truth(value):
                return "⊤" if value else "⊥"
            case Atom(symbol, (left, right)) if symbol.name in _RELATIONS:
                return f"{self._part(left)} {_RELATIONS[symbol.name]} {self._part(right)}"
            case Atom(symbol, arguments):
                return self._call(symbol.name, arguments)
            case Equal(left, right):
                return f"{self._part(left)} = {self._part(right)}"
            case Compare(operator, left, right):
                return f"{self._part(left)} {_COMPARE[operator]} {self._part(right)}"
            case Not(body):
                return f"¬{self._part(body)}"
            case And(parts):
                return " ∧ ".join(self._part(part) for part in parts) if parts else "⊤"
            case Or(parts):
                return " ∨ ".join(self._part(part) for part in parts) if parts else "⊥"
            case Implies(premise, conclusion):
                return f"{self._part(premise)} → {self._part(conclusion)}"
            case Iff(left, right):
                return f"{self._part(left)} ↔ {self._part(right)}"
            case Equisatisfiable(left, right):
                return f"{self._part(left)} ≈ {self._part(right)}"
            case ForAll(variables, body) | Exists(variables, body):
                quantifier = "∀" if isinstance(node, ForAll) else "∃"
                return f"{quantifier}{', '.join(variable.name for variable in variables)} {self._part(body)}"
            case Variable(name) | Constant(name):
                return name
            case Number(value):
                return str(value)
            case Application(symbol, (left, right)) if symbol.name in _OPERATIONS or symbol.name in ARITHMETIC:
                return f"{self._part(left)} {_OPERATIONS.get(symbol.name, symbol.name)} {self._part(right)}"
            case Application(symbol, arguments):
                return self._call(symbol.name, arguments)
        raise ValueError(f"Not a formula or a term: {node!r}")

    def _call(self, name: str, arguments: tuple[object, ...]) -> str:
        return f"{name}({', '.join(self.to_text(argument) for argument in arguments)})" if arguments else name

    def _part(self, node: object) -> str:
        """The text of a part of a larger formula or term, in parentheses unless it reads as one piece."""
        text = self.to_text(node)
        simple = isinstance(node, (Truth, Variable, Constant, Number, Not)) or (
            isinstance(node, (Atom, Application)) and text.endswith(")") and "(" in text.split(" ", 1)[0]
        )
        return text if simple or (isinstance(node, (Atom, Application)) and " " not in text) else f"({text})"
