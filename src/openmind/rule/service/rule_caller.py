import inspect
from collections.abc import Mapping, Sequence

from openmind.rule.model.called_rule import CalledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.state import State
from openmind.world.model.value import Value

#: What a function's qualified name holds when it was defined inside another function, so no other process can find it.
LOCAL = "<locals>"


class RuleCaller:
    """Calls a domain's rules, whichever way the project wrote them: Python source, compiled and run with the state's
    variables as its names, or a function the project gives OMF, called with the state and the parameters it needs.

    A function is called `rule(state, **parameters)`; an effects function gives the next state, where an effects script
    assigns to the state's variables. The names a generated rule reads, such as `here` and `me`, go to source rules only:
    a function reads the state itself."""

    def __init__(self, rule_compiler: RuleCompiler, rule_runner: RuleRunner) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner

    def prepare(self, rule: Rule, parameters: Sequence[str] = (), definitions: PythonRule | None = None) -> CalledRule:
        """The rule ready to call. Source says which of the parameters it reads; a function is taken to read them all."""
        if isinstance(rule, PythonRule):
            compiled = self._rule_compiler.compile_value(rule, parameters, definitions)
            return CalledRule(rule, compiled, compiled.arguments, rule.source)
        return CalledRule(rule, None, tuple(parameters), self.source(rule))

    def call(
        self,
        prepared: CalledRule,
        state: State,
        parameters: Mapping[str, object] | None = None,
        names: Mapping[str, object] | None = None,
    ) -> object:
        """The rule's value for the state."""
        if prepared.compiled is not None:
            return self._rule_runner.value(prepared.compiled, state, parameters, names)
        given = parameters or {}
        return prepared.rule(state, **{name: given[name] for name in prepared.arguments})  # type: ignore[operator]

    def value(
        self,
        rule: Rule,
        state: State,
        parameters: Mapping[str, object] | None = None,
        names: Mapping[str, object] | None = None,
        definitions: PythonRule | None = None,
    ) -> object:
        """The rule's value, prepared and called; the compiler keeps what it compiled."""
        return self.call(self.prepare(rule, tuple(parameters or ()), definitions), state, parameters, names)

    def apply(
        self, rule: Rule, state: State, parameters: Mapping[str, Value] | None = None, definitions: PythonRule | None = None
    ) -> State:
        """The state after the rule's effects: a script's assignments, or what the function gives. A function giving
        anything but a state raises TypeError."""
        if isinstance(rule, PythonRule):
            return self._rule_runner.apply(self._rule_compiler.compile_effects(rule, definitions), state, parameters)
        outcome = rule(state, **(parameters or {}))
        if not isinstance(outcome, State):
            raise TypeError(f"Effects of {self.source(rule)} gave {outcome!r} instead of a state")
        return outcome

    def check(self, rule: Rule | None) -> None:
        """Raises ValueError for a function no other process can find: a lambda, or a function defined inside another.
        Workers are started fresh and are given a rule by name."""
        if rule is None or isinstance(rule, PythonRule):
            return
        name = getattr(rule, "__qualname__", "")
        if not name or LOCAL in name or name.endswith("<lambda>"):
            raise ValueError(f"Rule {self.source(rule)} can't be found by a worker process: write it at a module's top level")

    def source(self, rule: Rule) -> str:
        """How the rule reads in a log: its Python source, or a function's module and name."""
        if isinstance(rule, PythonRule):
            return rule.source
        module = inspect.getmodule(rule)
        return f"{getattr(module, '__name__', '?')}.{getattr(rule, '__qualname__', repr(rule))}"
