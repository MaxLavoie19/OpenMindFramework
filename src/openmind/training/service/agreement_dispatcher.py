import logging
from collections.abc import Sequence

from openmind.heuristic.service.rule_position_valuer import RulePositionValuer
from openmind.heuristic.service.successor_move_rater import SuccessorMoveRater
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.rbs_factory import create_rule_heuristic
from openmind.training.model.agreement import Agreement
from openmind.training.model.decided import Decided
from openmind.training.service.agreement_scorer import AgreementScorer

logger = logging.getLogger(__name__)


def scored_alone(name: str, model: object, decisions: Sequence[Decided]) -> Agreement:
    """One heuristic put to those decisions, in whatever process is running this.

    **At module level because pickle stores a function by name.** A worker starts as a fresh interpreter and
    everything reaches it by pickling, and pickling a function writes down its module and qualified name to be
    imported again on the other side. The judging built its raters as lambdas inside a function, whose
    qualified name is `<locals>.<lambda>` and which nothing can import — so what is dispatched has to be a
    function that has a name.

    **Nothing else forced the shape.** Measured, the services all pickle: the rule heuristic in 900 bytes, the
    valuer in 993, the rater in 1083, and the lambda's captured defaults pickle on their own. They are rebuilt
    here rather than sent because it costs the same either way and leaves each worker's reading caches its
    own instead of copying a parent's."""
    valuer = RulePositionValuer(create_rule_heuristic())
    rater = SuccessorMoveRater(valuer)
    return AgreementScorer().scored(
        decisions, {name: lambda node, actions, player: rater.rate(model, node, actions, player)}
    )[0]


class AgreementDispatcher:
    """Every heuristic put to the same decisions, one per worker process.

    **Judging is the one thing in the run that grows without bound and blocks everything while it grows.** It
    costs candidates times decisions times legal moves, and answering one decision means valuing the position
    each legal move leads to. Measured: three heuristics over one four-hundred-ply game is about thirty-six
    thousand position valuations and eleven minutes, and it ran on the thread that learns the constraints — so
    the learner did three positions in fifteen minutes while the judging held it.

    **Nothing here is shared, which is what makes it dispatchable at all.** What one heuristic expected of a
    game has nothing to do with what another expected of it: no ordering, no accumulation, no store. So the
    split is one heuristic per call, and the axis it splits on — how many candidates there are — is exactly the
    one that grows.

    **The decisions cross whole to every worker, because they are small.** One game's decisions pickle to a
    quarter of a megabyte and come back in ten milliseconds, against minutes of valuing; splitting them instead
    would save nothing and would leave each worker with a part of a heuristic's score to be summed somewhere.

    A heuristic that ends its worker is not fatal to the round: the call runs again in a fresh one, and a call
    that fails twice leaves that heuristic unjudged this round rather than stopping the judging."""

    def __init__(self, task_runner: TaskRunner) -> None:
        self._task_runner = task_runner

    def scored(
        self, decisions: Sequence[Decided], models: Sequence[tuple[str, object]]
    ) -> tuple[Agreement, ...]:
        """Each named heuristic measured over those decisions, in the order they were given.

        `models` is one ruleset per heuristic, as data rather than as a loaded service, because a worker
        rebuilds what reads it. With one worker this runs here, which is what `TaskRunner` does with a single
        worker or a single call — so a run that asked for no workers behaves as it did before."""
        if not models or not decisions:
            return ()
        names = [name for name, _ in models]
        found = self._task_runner.map(
            scored_alone, names, [model for _, model in models], [decisions] * len(models), droppable=True
        )
        kept = tuple(one for one in found if isinstance(one, Agreement))
        if len(kept) != len(found):
            logger.warning(
                "%d of %d heuristics could not be judged this round and are left unjudged rather than "
                "stopping the round",
                len(found) - len(kept),
                len(found),
            )
        for one in sorted(kept, key=lambda held: -(held.mass - held.offered)):
            logger.info(
                "%s expected %.2f of what happened where knowing nothing would have expected %.2f, over %d "
                "decisions of %d; it had nothing to say in %d and nothing to choose between in %d",
                one.holder, one.mass, one.offered, one.decided,
                len(decisions), one.declined, one.undecided,
            )
        return kept
