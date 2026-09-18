import ast
import textwrap

from openmind.rbs.model.python_rule import PythonRule


class CallOperandMapper:
    """Maps a rule that is a single call, such as all_different(a, b, cell[1, 1]), to its arguments as rules of their
    own. Each rule is parsed once and its answer kept."""

    def __init__(self) -> None:
        self._operands: dict[tuple[PythonRule, str], tuple[PythonRule, ...] | None] = {}

    def to_operands(self, rule: PythonRule, function: str) -> tuple[PythonRule, ...] | None:
        """The arguments, or None when the rule isn't one call to that function with plain positional arguments."""
        key = (rule, function)
        if key not in self._operands:
            self._operands[key] = self._parse(rule, function)
        return self._operands[key]

    def _parse(self, rule: PythonRule, function: str) -> tuple[PythonRule, ...] | None:
        try:
            tree = ast.parse(textwrap.dedent(rule.source), mode="eval")
        except SyntaxError:
            return None
        call = tree.body
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != function
            or call.keywords
            or any(isinstance(argument, ast.Starred) for argument in call.args)
        ):
            return None
        return tuple(PythonRule(ast.unparse(argument)) for argument in call.args)
