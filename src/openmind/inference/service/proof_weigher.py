import logging
import random
from collections.abc import Mapping, Sequence

from openmind.inference.model.chance import Chance
from openmind.inference.model.decision_diagram import DecisionDiagram, DecisionNode
from openmind.inference.model.derivation import Derivation
from openmind.inference.model.derivation_step import GIVEN
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.justifier import Justifier

logger = logging.getLogger(__name__)


class ProofWeigher:
    """How likely a conclusion is, from every proof of it.

    The thing this exists to avoid is quiet and costly. Two proofs of the same conclusion look like two reasons,
    and treating them as independent — one minus the product of their failures — gives a number that is too high
    whenever they lean on the same doubtful clause. It never looks wrong. It just makes the agent surer than it
    should be, everywhere, and most where it has reasoned hardest.

    So the proofs are not combined. They are laid out over the clauses they use: pick a doubtful clause, ask what
    follows if it held and what follows if it did not, and carry on. In each branch that clause has one answer, so
    no branch can count it twice, and branches that come to the same thing become one node. Weighting the layout
    from the bottom up gives the chance.

    **Exact while it fits, sampled when it does not, and it says which.** Laid out fully the structure can grow
    with the proofs, so the caller's budget caps the nodes; past that the clauses are drawn at random and the
    proofs tried against the draw. A sampled answer is marked sampled and carries the spread of the sampling,
    because an estimate that passes for a count is the same failure as before in a different coat.

    It keeps nothing: built once, it is given the proofs on every call."""

    def __init__(self, justifier: Justifier | None = None, draws: random.Random | None = None) -> None:
        self._justifier = justifier
        self._draws = random.Random(0) if draws is None else draws

    def chance(self, derivations: Sequence[Derivation], budget: InferenceBudget) -> Chance:
        """How likely the thing those proofs reach is.

        No proofs is no chance at all, not a chance of nothing: a conclusion nothing reaches is unsupported rather
        than false, which is the same distinction the answer's third value makes."""
        if not derivations:
            return Chance(0.0, 0.0, 0, True)
        probabilities = self._probabilities(derivations)
        ways = self._ways(derivations)
        if any(not way for way in ways):
            return Chance(1.0, 0.0, len(derivations), True)
        nodes: dict[object, DecisionDiagram] = {}
        variables = sorted({one for way in ways for one in way})
        diagram = self._laid_out(ways, tuple(variables), nodes, budget.nodes)
        if diagram is not None:
            value = self._weighed(diagram, probabilities, {})
            logger.info(
                "Weighed %d proofs over %d doubtful clauses into %d nodes: %.4g",
                len(derivations), len(variables), len(nodes), value,
            )
            return Chance(value, 0.0, len(derivations), True)
        draws = budget.drawing
        value, spread = self._sampled(ways, probabilities, draws)
        logger.info(
            "Weighed %d proofs over %d doubtful clauses by %d draws, the layout being larger than %d nodes: about %.4g",
            len(derivations), len(variables), draws, budget.nodes, value,
        )
        return Chance(value, spread, len(derivations), False)

    def independent(self, derivations: Sequence[Derivation]) -> int:
        """How many of those proofs are reasons of their own rather than one reason found again."""
        justifier = self._justifier
        if justifier is None:
            return len({frozenset(one.rests_on) for one in derivations if one.rests_on})
        return justifier.independent(tuple(derivations))

    def _ways(self, derivations: Sequence[Derivation]) -> tuple[frozenset[str], ...]:
        """Each proof as the doubtful clauses it needs; a proof needing none holds whatever happens."""
        found: list[frozenset[str]] = []
        for derivation in derivations:
            way = frozenset(derivation.chances)
            if not any(held <= way for held in found):
                found = [held for held in found if not way < held]
                found.append(way)
        return tuple(found)

    def _probabilities(self, derivations: Sequence[Derivation]) -> Mapping[str, float]:
        """What each doubtful clause's chance is, read off the proofs that used it."""
        found: dict[str, float] = {}
        for derivation in derivations:
            for step in derivation.steps:
                if step.rule == GIVEN and step.clause.name and not step.clause.certain:
                    found[step.clause.name] = step.clause.probability
        return found

    def _laid_out(
        self,
        ways: tuple[frozenset[str], ...],
        variables: tuple[str, ...],
        nodes: dict[object, DecisionDiagram],
        most: int,
    ) -> DecisionDiagram | None:
        """The proofs laid out over the clauses, or None where that would take more nodes than allowed."""
        if not ways:
            return False
        if any(not way for way in ways):
            return True
        if not variables:
            return False
        key = (frozenset(ways), variables)
        held = nodes.get(key)
        if held is not None:
            return held
        if len(nodes) >= most:
            return None
        variable, rest = variables[0], variables[1:]
        when_held = self._laid_out(tuple(way - {variable} for way in ways), rest, nodes, most)
        if when_held is None:
            return None
        when_not = self._laid_out(tuple(way for way in ways if variable not in way), rest, nodes, most)
        if when_not is None:
            return None
        made: DecisionDiagram = when_held if when_held == when_not else DecisionNode(variable, when_held, when_not)
        nodes[key] = made
        return made

    def _weighed(
        self, diagram: DecisionDiagram, probabilities: Mapping[str, float], weighed: dict[int, float]
    ) -> float:
        """The layout weighted from the bottom up."""
        if isinstance(diagram, bool):
            return 1.0 if diagram else 0.0
        held = weighed.get(id(diagram))
        if held is not None:
            return held
        chance = probabilities.get(diagram.variable, 1.0)
        value = chance * self._weighed(diagram.when_held, probabilities, weighed) + (1.0 - chance) * self._weighed(
            diagram.when_not, probabilities, weighed
        )
        weighed[id(diagram)] = value
        return value

    def _sampled(
        self, ways: tuple[frozenset[str], ...], probabilities: Mapping[str, float], draws: int
    ) -> tuple[float, float]:
        """The chance by drawing the clauses and trying the proofs against each draw, with its spread."""
        held = 0
        for _ in range(draws):
            drawn = {name: self._draws.random() < chance for name, chance in probabilities.items()}
            if any(all(drawn.get(name, True) for name in way) for way in ways):
                held += 1
        value = held / draws
        return value, (value * (1.0 - value) / draws) ** 0.5
