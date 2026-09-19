from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.python_rule import PythonRule

#: What a compiled rule is kept under: how it was compiled, the rule, the parameters it takes and its definitions.
type CompiledRuleKey = tuple[str, PythonRule, tuple[str, ...], PythonRule | None]


class CompiledRuleCache:
    """The rules compiled so far, built once and given to the compiler, which keeps nothing itself. Compiled code
    doesn't travel to other processes: a copy arrives empty and its rules are compiled again."""

    def __init__(self) -> None:
        self._compiled: dict[CompiledRuleKey, CompiledRule] = {}

    def __getstate__(self) -> dict[str, object]:
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._compiled = {}

    def get(self, key: CompiledRuleKey) -> CompiledRule | None:
        return self._compiled.get(key)

    def put(self, key: CompiledRuleKey, compiled: CompiledRule) -> CompiledRule:
        self._compiled[key] = compiled
        return compiled
