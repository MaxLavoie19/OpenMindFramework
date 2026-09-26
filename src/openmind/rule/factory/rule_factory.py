from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.clause_caller import ClauseCaller
from openmind.rule.model.consequence_caller import ConsequenceCaller
from openmind.rule.service.rule_caller import RuleCaller
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner


def create_rule_caller(
    clause_caller: ClauseCaller | None = None, consequence_caller: ConsequenceCaller | None = None
) -> RuleCaller:
    """A rule caller with its own compiler and runner: it calls a domain's rules, written as Python source, as the
    project's own functions, or as what OMF learned — a clause saying when something is refused, consequences
    saying what an action does, a domain saying what a parameter may take. A domain needs nothing to run it; the
    other two are run by what it is given, and without them they say so rather than failing obscurely.

    **Given and not defaulted, because a rule caller travels to worker processes.** Wiring the learned callers
    in here was tried and is what this paragraph is: `ClauseConstraint` holds a `CandidateReadings`, which holds
    an `EvaluablePredicates`, which holds lambdas written inside its own constructor — so every rule caller in
    the system became unpicklable at once and no game of functions could cross to a worker. The project already
    refuses a rule a worker cannot find; a service a worker cannot receive is the same fault one layer up.

    So a game that plays what it learned is built with a caller that can run it. That is also where the one
    thing OMF cannot know belongs — which of a position's models says who is acting, which a consequence needs
    before it can say whose turn it becomes."""
    return RuleCaller(RuleCompiler(), RuleRunner(StateNamespaceMapper()), clause_caller, consequence_caller)
