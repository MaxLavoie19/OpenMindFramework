import ast
import hashlib
import linecache
import textwrap
from collections.abc import Iterable
from types import CodeType

from openmind.rule.constant.rule_constant import DEFINITIONS, EFFECTS, RULE_FUNCTION, VALUE
from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.repository.compiled_rule_cache import CompiledRuleCache


class RuleCompiler:
    """Compiles Python rules once, keeping them in the cache it is given. Tracebacks show a rule's own source and line
    numbers."""

    def __init__(self, cache: CompiledRuleCache | None = None) -> None:
        self._compiled = CompiledRuleCache() if cache is None else cache

    def compile_value(
        self, rule: PythonRule, parameters: Iterable[str] = (), definitions: PythonRule | None = None
    ) -> CompiledRule:
        """A rule read for its value: an expression, or a script that returns one. It becomes a function of the given
        parameters it reads, so a caller passes only those."""
        names = tuple(parameters)
        key = (VALUE, rule, names, definitions)
        if (compiled := self._compiled.get(key)) is not None:
            return compiled
        filename = self._filename(rule)
        tree = self._parse(rule, filename)
        body: list[ast.stmt] = [ast.Return(tree.body)] if isinstance(tree, ast.Expression) else tree.body
        read = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        arguments = tuple(name for name in names if name in read)
        function = ast.FunctionDef(
            name=RULE_FUNCTION,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in arguments],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        code = next(
            constant
            for constant in compile(module, filename, "exec").co_consts
            if isinstance(constant, CodeType) and constant.co_name == RULE_FUNCTION
        )
        compiled = CompiledRule(rule, VALUE, code, arguments, self._definitions(definitions))
        return self._compiled.put(key, compiled)

    def compile_effects(self, rule: PythonRule, definitions: PythonRule | None = None) -> CompiledRule:
        """A script run for what it changes: what it assigns to state variables makes the next state."""
        key = (EFFECTS, rule, (), definitions)
        if (compiled := self._compiled.get(key)) is None:
            compiled = CompiledRule(rule, EFFECTS, self._module(rule), (), self._definitions(definitions))
            return self._compiled.put(key, compiled)
        return compiled

    def compile_definitions(self, rule: PythonRule) -> CompiledRule:
        """A script run once per domain; every name it leaves is visible to the domain's rules."""
        key = (DEFINITIONS, rule, (), None)
        if (compiled := self._compiled.get(key)) is None:
            compiled = CompiledRule(rule, DEFINITIONS, self._module(rule), (), None)
            return self._compiled.put(key, compiled)
        return compiled

    def _definitions(self, definitions: PythonRule | None) -> CompiledRule | None:
        return None if definitions is None else self.compile_definitions(definitions)

    def _module(self, rule: PythonRule) -> CodeType:
        filename = self._filename(rule)
        return compile(self._parse(rule, filename, expression=False), filename, "exec")

    def _parse(self, rule: PythonRule, filename: str, expression: bool = True) -> ast.Expression | ast.Module:
        source = textwrap.dedent(rule.source)
        if expression:
            try:
                return ast.parse(source, filename, mode="eval")
            except SyntaxError:
                pass
        try:
            return ast.parse(source, filename, mode="exec")
        except SyntaxError as error:
            raise SyntaxError(f"{error.msg} in rule {rule.source!r}, line {error.lineno}") from error

    def _filename(self, rule: PythonRule) -> str:
        source = textwrap.dedent(rule.source)
        filename = f"<rule {hashlib.sha1(source.encode()).hexdigest()[:12]}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        return filename
