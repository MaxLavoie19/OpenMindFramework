import logging
from collections.abc import Mapping, Sequence

from openmind.inference.model.example import Example
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.rule.model.clause import Clause
from openmind.rule.model.rule import ConstraintRule
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ClauseConstraint:
    """What OMF has learned, run as one of the solver's constraints.

    `ClauseRule` is declared as a kind of rule and stored as one, but nothing in OMF runs it — `RuleCaller` has no
    branch for it — so until now nothing the engine learned could reach the solver. This is that bridge.

    **It needs no compiler.** Refusing a candidate is asking whether a constraint covers its readings, which is
    exactly what learning already does. One mechanism serves learning and playing, so a constraint cannot mean one
    thing while it is being learned and another once it is used — which is the kind of difference that is found
    out much later and explains nothing.

    The cost is that a position is read per candidate the solver asks about, rather than once. That is the price
    of the whole structure being the vocabulary, and it is paid here rather than designed around."""

    def __init__(self, readings: CandidateReadings | None = None, learner: RefusalLearner | None = None) -> None:
        self._readings = CandidateReadings() if readings is None else readings
        self._learner = RefusalLearner() if learner is None else learner

    def holds(self, clause: Clause, state: State, parameters: Mapping[str, Value]) -> bool:
        """Whether that one learned constraint leaves the candidate standing: `ClauseCaller`, filled.

        **One clause and not a closure over many.** A closure would be a local function, which `RuleCaller.check`
        refuses and no worker process could rebuild — so the thing that was meant to carry learned rules into the
        solver could never have travelled to where the solving happens. Asked one clause at a time, against a
        rule the knowledge base already stores one clause at a time, nothing has to be closed over: the solver
        conjoins them by holding each as its own constraint, which is what it does for declared rules too.

        The action's name is not in the parameters and is not needed: a candidate is read from its parameters and
        the position, and the clause was learned over exactly those readings."""
        candidate = Action("", tuple(sorted(parameters.items())))
        return not self._learner.refuses(
            (clause,), Example(self._readings.read(state, candidate), False, state)
        )

    def constraint(self, clauses: Sequence[Clause], action: str = "") -> ConstraintRule:
        """Those constraints as the callable the `Solver` takes: true where no constraint refuses.

        The action's name is not looked at — a candidate is read from its parameters and the position — but it is
        taken so that what the solver runs says which action it belongs to.

        This is the shape for a caller holding clauses in hand rather than as stored rules. Where they are stored,
        `holds` is what the rule caller asks, one clause at a time, and nothing has to be closed over."""

        def legal(state: State, **parameters: Value) -> bool:
            candidate = Action(action, tuple(sorted(parameters.items())))
            return not self._learner.refuses(
                clauses, Example(self._readings.read(state, candidate), False, state)
            )

        return legal

    def refusing(self, clauses: Sequence[Clause], state: State, action: Action) -> tuple[Clause, ...]:
        """Which of them refuse that action, for saying why a move was not offered.

        A move OMF will not make is the mistake nothing tells it about, so being able to ask which constraint
        turned it away is what makes that mistake findable at all."""
        case = Example(self._readings.read(state, action), False, state)
        found = tuple(one for one in clauses if self._learner.covers(one, case))
        logger.debug("%d of %d constraints refuse %s", len(found), len(clauses), action.parameters)
        return found
