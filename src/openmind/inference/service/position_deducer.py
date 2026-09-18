import logging
import math
import time
from collections.abc import Callable

from openmind.inference.model.deduction import Deduction
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)

#: Each player's proven payoffs, and the line showing them: each action with the state it led to.
type Proof = tuple[tuple[float, ...], tuple[tuple[Action, State], ...]]


class PositionDeducer:
    """Reasons about one position with nothing but the domain's own rules: its legal actions, their outcomes and the
    payoffs of finished games. It deepens one ply at a time, up to the budget's plies and within its seconds.

    - A finished game is proven: its payoffs.
    - An action is proven when every outcome with a chance above 0 is, and is worth their payoffs weighted by their
      chances.
    - A position in play is proven when every legal action is, the player to act taking the best for them (ties go to
      the solver's first), or as soon as one proven action gives that player the budget's highest payoff, which no other
      action can beat.

    Nothing is estimated: a position the plies don't reach the end of stays unproven."""

    def __init__(
        self,
        state_reader: StateReader,
        action_text_mapper: ActionTextMapper,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state_reader = state_reader
        self._action_text_mapper = action_text_mapper
        self._clock = clock

    def deduce(self, rbs: RuleBasedSystem, state: State, budget: DeductionBudget) -> Deduction:
        """The first depth that proves the position, or nothing proven when the plies or the seconds run out; a budget of
        fewer than 1 ply or no seconds, or a position without a legal action, raises ValueError."""
        if budget.plies < 1 or budget.seconds <= 0.0:
            raise ValueError(f"A deduction needs at least 1 ply and more than 0 seconds, not {budget}")
        if not rbs.actions(state):
            raise ValueError("No legal action to deduce from")
        player = rbs.players().names[self._state_reader.player_to_act(state, rbs.players())]
        deadline = self._clock() + budget.seconds
        memo: dict[tuple[State, int], Proof | None] = {}
        reached = 0
        for depth in range(1, budget.plies + 1):
            try:
                proof = self._decide(rbs, state, depth, budget.highest, deadline, memo)
            except TimeoutError:
                break
            reached = depth
            if proof is not None:
                payoffs, line = proof
                logger.info(
                    "Deduced %s for %s within %d plies: payoffs %s, along %s",
                    self._action_text_mapper.to_text(line[0][0]),
                    player,
                    depth,
                    " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, payoffs)),
                    " > ".join(self._action_text_mapper.to_text(action) for action, _ in line),
                )
                return Deduction(state, player, line[0][0], payoffs, line, depth)
        logger.debug(
            "Nothing proven for %s within %d plies%s",
            player,
            reached,
            "" if reached == budget.plies else f": the {budget.seconds} seconds ran out",
        )
        return Deduction(state, player, None, None, (), reached)

    def _decide(
        self,
        rbs: RuleBasedSystem,
        state: State,
        depth: int,
        highest: float,
        deadline: float,
        memo: dict[tuple[State, int], Proof | None],
    ) -> Proof | None:
        key = (state, depth)
        if key in memo:
            return memo[key]
        if self._clock() >= deadline:
            raise TimeoutError
        actions = rbs.actions(state)
        proof: Proof | None = None
        if not actions:
            proof = (self._state_reader.payoffs(state, rbs.players()), ())
        elif depth > 0:
            mover = self._state_reader.player_to_act(state, rbs.players())
            unproven = False
            for action in actions:
                found = self._act(rbs, state, action, depth, highest, deadline, memo)
                if found is None:
                    unproven = True
                elif proof is None or found[0][mover] > proof[0][mover]:
                    proof = found
                if proof is not None and proof[0][mover] >= highest:
                    unproven = False
                    break
            if unproven:
                proof = None
        memo[key] = proof
        return proof

    def _act(
        self,
        rbs: RuleBasedSystem,
        state: State,
        action: Action,
        depth: int,
        highest: float,
        deadline: float,
        memo: dict[tuple[State, int], Proof | None],
    ) -> Proof | None:
        proofs: list[tuple[float, State, Proof]] = []
        for outcome, probability in rbs.outcomes(state, action).outcomes:
            if probability <= 0.0:
                continue
            found = self._decide(rbs, outcome, depth - 1, highest, deadline, memo)
            if found is None:
                return None
            proofs.append((probability, outcome, found))
        payoffs = tuple(
            math.fsum(probability * found[0][index] for probability, _, found in proofs)
            for index in range(len(rbs.players().names))
        )
        _, likeliest, found = max(proofs, key=lambda item: item[0])
        return payoffs, ((action, likeliest), *found[1])
