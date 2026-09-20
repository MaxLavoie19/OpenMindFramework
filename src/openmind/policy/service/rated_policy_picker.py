import logging

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.policy.model.policy_valuer import PolicyValuer

logger = logging.getLogger(__name__)


class RatedPolicyPicker:
    """The policy picker that reads what a policy value model says and keeps what is worth expanding.

    A policy rated below the default policy — picking at random among the legal actions — isn't worth expanding: the
    cheapest way of playing already does better. When the default policy is the best rated, it is picked alone, since
    nothing else is then worth considering.

    A policy the model says nothing about is kept, best-rated first and unrated last: a policy nobody has measured is
    worth trying before it is dismissed."""

    def __init__(self, policy_valuer: PolicyValuer[object]) -> None:
        self._valuer = policy_valuer

    def pick(self, model: object, node: Node, policies: tuple[Policy, ...], player: str) -> tuple[Policy, ...]:
        """The policies worth expanding here, best first; empty where there are no policies."""
        if not policies:
            return ()
        rated = tuple(zip(policies, self._valuer.rate(model, node, policies, player), strict=True))
        default = next((rating for policy, rating in rated if not policy.name), None)
        kept = [
            (policy, rating)
            for policy, rating in rated
            if rating is None or default is None or not policy.name or rating >= default
        ]
        ordered = sorted(kept, key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
        best = ordered[0][0]
        if not best.name:
            logger.debug("Picking at random is the best rated policy here: nothing else is worth considering")
            return (best,)
        picked = tuple(policy for policy, _ in ordered if policy.name)
        logger.debug("Picked %d of %d policies: %s", len(picked), len(policies), ", ".join(policy.name for policy in picked))
        return picked
