import logging
from dataclasses import replace

from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.state_domain import StateDomain
from openmind.csp.model.variable import Variable
from openmind.inference.constant.inference_constant import PASS
from openmind.inference.model.relaxation import Relaxation
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.players import Players

logger = logging.getLogger(__name__)

#: A branch that always happens.
CERTAIN = 1.0


class DomainRelaxer:
    """Makes a relaxed copy of a domain, to reason where the rules allow more than they do: a constraint dropped, a
    parameter given wider values, or a `pass` action that only hands the turn over. Nothing here knows a game; the copy
    is a domain like any other, so the solver, the predictor and deduction work in it unchanged, and the domain given is
    left as it is."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def relaxations(self, domain: Domain) -> tuple[Relaxation, ...]:
        """The relaxations the rules themselves allow: each constraint of each action dropped, and, where one player acts
        at a time, every player also being able to pass. Wider values for a parameter aren't among them, since the rules
        don't say what wider would mean; a project passes such a relaxation in."""
        found = [
            Relaxation(f"{definition.name} without {self._rule_caller.source(constraint)}", dropped=((definition.name, at),))
            for definition in domain.problem.actions
            for at, constraint in enumerate(definition.constraints)
        ]
        if self._hands_over(domain):
            found.append(Relaxation("a player may pass", passing=True))
        return tuple(found)

    def relax(self, domain: Domain, relaxation: Relaxation) -> Domain:
        """The domain with the relaxation applied. Passing in a domain where the players act at once, or where no
        variable names the player to act, raises ValueError."""
        actions = tuple(self._relaxed(definition, relaxation) for definition in domain.problem.actions)
        problem = replace(domain.problem, actions=actions)
        transitions = domain.transitions
        if relaxation.passing:
            if not self._hands_over(domain):
                raise ValueError(f"{domain.name} has no variable naming the player to act, so no player can pass")
            problem = replace(problem, actions=(*actions, self._passing(actions, domain)))
            handover = Branch(CERTAIN, self._handover(domain.players))
            transitions = replace(transitions, transitions=(*transitions.transitions, Transition(PASS, (handover,))))
        logger.info("Relaxed %s: %s", domain.name, relaxation.name)
        return replace(domain, problem=problem, transitions=transitions)

    def _relaxed(self, definition: ActionDefinition, relaxation: Relaxation) -> ActionDefinition:
        dropped = {at for action, at in relaxation.dropped if action == definition.name}
        constraints = tuple(
            constraint for at, constraint in enumerate(definition.constraints) if at not in dropped
        )
        wider = {parameter: rule for action, parameter, rule in relaxation.widened if action == definition.name}
        variables = tuple(
            Variable(variable.name, StateDomain(wider[variable.name])) if variable.name in wider else variable
            for variable in definition.variables
        )
        return replace(definition, variables=variables, constraints=constraints)

    def _passing(self, actions: tuple[ActionDefinition, ...], domain: Domain) -> ActionDefinition:
        """The `pass` action: no parameters, and the constraints that read no parameter, so a player can pass while the
        game goes on and not once it is over."""
        constraints: list[Rule] = []
        for definition in actions:
            names = [variable.name for variable in definition.variables]
            for constraint in definition.constraints:
                prepared = self._rule_caller.prepare(constraint, names, domain.problem.definitions)
                if not prepared.arguments and constraint not in constraints:
                    constraints.append(constraint)
        return ActionDefinition(PASS, (), tuple(constraints))

    def _handover(self, players: Players) -> PythonRule:
        """Effects that give the turn to the next player, in the players' order."""
        following = {name: players.names[(at + 1) % len(players.names)] for at, name in enumerate(players.names)}
        return PythonRule(f"{players.to_act} = {following!r}[{players.to_act}]")

    def _hands_over(self, domain: Domain) -> bool:
        return domain.players.to_act in {name for name, _ in domain.initial_state.variables}
