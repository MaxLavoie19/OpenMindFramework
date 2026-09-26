import inspect
from collections.abc import Mapping, Sequence

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.rule.model.called_rule import CalledRule
from openmind.rule.model.clause_caller import ClauseCaller
from openmind.rule.model.clause_rule import ClauseRule
from openmind.rule.model.consequence_caller import ConsequenceCaller
from openmind.rule.model.consequence_rule import ConsequenceRule
from openmind.rule.model.domain_rule import DomainRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.state import State
from openmind.structure.model.value import Value

#: What a function's qualified name holds when it was defined inside another function, so no other process can find it.
LOCAL = "<locals>"


class RuleCaller:
    """Calls a domain's rules, whichever way the project wrote them: Python source, compiled and run with the state's
    variables as its names, or a function the project gives OMF, called with the state and the parameters it needs.

    A function is called `rule(state, **parameters)`; an effects function gives the next state, where an effects script
    assigns to the state's variables. The names a generated rule reads, such as `here` and `me`, go to source rules only:
    a function reads the state itself.

    **And a clause, which is what OMF induces.** A learned rule is kept as logic so it can be reasoned over, and
    until now that meant it could not be run: there was no branch for one here, so a constraint OMF worked out for
    itself could never decide whether a move was legal. It is answered by whatever the caller was built with to
    run clauses, which must read a position the way the learning read it — see `ClauseCaller`."""

    def __init__(
        self,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        clause_caller: ClauseCaller | None = None,
        consequence_caller: ConsequenceCaller | None = None,
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._clause_caller = clause_caller
        self._consequence_caller = consequence_caller

    def prepare(self, rule: Rule, parameters: Sequence[str] = (), definitions: PythonRule | None = None) -> CalledRule:
        """The rule ready to call. Source says which of the parameters it reads; a function reads those its signature
        names, or all of them when it takes any keyword, so a function written before a parameter was offered still
        gets only what it asks for.

        A clause reads every parameter it was prepared with, as a function does: what a clause is about is its
        readings of the whole candidate, and which parameters those came from is not something to guess at here."""
        if isinstance(rule, PythonRule):
            compiled = self._rule_compiler.compile_value(rule, parameters, definitions)
            return CalledRule(rule, compiled, compiled.arguments, rule.source)
        if isinstance(rule, ClauseRule):
            return CalledRule(rule, None, tuple(parameters), self.source(rule))
        if isinstance(rule, DomainRule):
            return CalledRule(rule, None, (), self.source(rule))
        return CalledRule(rule, None, self._accepted(rule, parameters), self.source(rule))

    def _accepted(self, rule: Rule, parameters: Sequence[str]) -> tuple[str, ...]:
        try:
            signature = inspect.signature(rule)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return tuple(parameters)
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return tuple(parameters)
        return tuple(name for name in parameters if name in signature.parameters)

    def call(
        self,
        prepared: CalledRule,
        state: State,
        parameters: Mapping[str, object] | None = None,
        names: Mapping[str, object] | None = None,
    ) -> object:
        """The rule's value for the state.

        A clause is asked whether it holds, by whatever was given to run clauses with. Given nothing, it says so:
        this used to fall through to calling the clause, which raised `TypeError: 'ClauseRule' object is not
        callable` from inside the solver, where nothing said what was really missing."""
        if prepared.compiled is not None:
            return self._rule_runner.value(prepared.compiled, state, parameters, names)
        if isinstance(prepared.rule, DomainRule):
            return list(prepared.rule.values)
        given = parameters or {}
        if isinstance(prepared.rule, ClauseRule):
            if self._clause_caller is None:
                raise ValueError(
                    f"Nothing was given to run clauses with, so {prepared.source} cannot be asked: "
                    "build the rule caller with the clause caller that reads positions the way the clause was learned"
                )
            return self._clause_caller.holds(
                prepared.rule.clause, state, {name: given[name] for name in prepared.arguments if name in given}
            )
        return prepared.rule(state, **{name: given[name] for name in prepared.arguments})  # type: ignore[operator]

    def value(
        self,
        rule: Rule,
        state: State,
        parameters: Mapping[str, object] | None = None,
        names: Mapping[str, object] | None = None,
        definitions: PythonRule | None = None,
    ) -> object:
        """The rule's value, prepared and called, under a `rule` reasoning frame; the compiler keeps what it compiled."""
        with process_debugger().frame("rule", state=state, details=_about(self, rule)):
            return self.call(self.prepare(rule, tuple(parameters or ()), definitions), state, parameters, names)

    def apply(
        self, rule: Rule, state: State, parameters: Mapping[str, Value] | None = None, definitions: PythonRule | None = None
    ) -> State:
        """The state after the rule's effects: a script's assignments, or what the function gives. A function is given
        the parameters its signature names, as a value rule's is, so one written before a parameter was offered — the
        player taking the action, say — still gets only what it asks for. A function giving anything but a state
        raises TypeError. It runs under a `rule` reasoning frame."""
        with process_debugger().frame("rule", state=state, details=_about(self, rule)):
            return self._applied(rule, state, parameters, definitions)

    def _applied(
        self, rule: Rule, state: State, parameters: Mapping[str, Value] | None, definitions: PythonRule | None
    ) -> State:
        if isinstance(rule, PythonRule):
            return self._rule_runner.apply(self._rule_compiler.compile_effects(rule, definitions), state, parameters)
        if isinstance(rule, ConsequenceRule):
            if self._consequence_caller is None:
                raise ValueError(
                    f"Nothing was given to make consequences happen with, so {self.source(rule)} cannot be applied: "
                    "build the rule caller with the consequence caller that draws changes the way they were learned"
                )
            return self._consequence_caller.after(rule.consequences, state, parameters or {}, self._acting(rule))
        given = parameters or {}
        accepted = self._accepted(rule, tuple(given))
        outcome = rule(state, **{name: given[name] for name in accepted})
        if not isinstance(outcome, State):
            raise TypeError(f"Effects of {self.source(rule)} gave {outcome!r} instead of a state")
        return outcome

    def check(self, rule: Rule | None) -> None:
        """Raises ValueError for a function no other process can find: a lambda, or a function defined inside another.
        Workers are started fresh and are given a rule by name.

        A clause passes: it is data, not a function, so a worker rebuilds it rather than having to find it. That is
        the reason a learned constraint is answered through a clause caller instead of through a closure — a
        closure over the learned clauses is exactly what this refuses, and would have to be refused."""
        if rule is None or isinstance(rule, PythonRule | ClauseRule | ConsequenceRule | DomainRule):
            return
        name = getattr(rule, "__qualname__", "")
        if not name or LOCAL in name or name.endswith("<lambda>"):
            raise ValueError(f"Rule {self.source(rule)} can't be found by a worker process: write it at a module's top level")

    def _acting(self, rule: ConsequenceRule) -> str:
        """Which action those consequences are of; they are all of one, since they were learned per action."""
        return next((one.action for one in rule.consequences), "")

    def source(self, rule: Rule) -> str:
        """How the rule reads in a log: its Python source, a clause as logic is written, or a function's module
        and name."""
        if isinstance(rule, PythonRule):
            return rule.source
        if isinstance(rule, ClauseRule | ConsequenceRule | DomainRule):
            return rule.readable
        module = inspect.getmodule(rule)
        return f"{getattr(module, '__name__', '?')}.{getattr(rule, '__qualname__', repr(rule))}"


def _about(caller: "RuleCaller", rule: Rule) -> dict[str, Value] | None:
    """What a rule frame says about its rule: its source's first line, only worked out when frames are kept."""
    if not process_debugger().tracking:
        return None
    source = caller.source(rule)
    return {"rule": source.splitlines()[0][:120] if source else ""}
