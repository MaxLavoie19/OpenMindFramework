import pickle

import pytest

from openmind.heuristic.model.node import Node
from openmind.parallel.service.task_runner import TaskRunner
from openmind.training.model.decided import Decided
from openmind.training.service.agreement_dispatcher import AgreementDispatcher, scored_alone
from openmind.world.model.action import Action
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

STEP = Action("step", (("by", 1),))
JUMP = Action("jump", (("by", 2),))
OFFERED = (STEP, JUMP)


def decision(taken: Action = STEP, paid: float = 1.0) -> Decided:
    return Decided(Node(State((("at", 0),))), OFFERED, taken, "white", paid)


def test_what_is_dispatched_can_be_pickled_and_a_lambda_cannot() -> None:
    """The whole reason the worker function sits at module level. Pickle writes a function down as its module
    and qualified name and imports it again on the other side, so a lambda built inside a function — whose
    qualified name is `<locals>.<lambda>` — has no name to write down.

    Not because closures capture state: the captured values pickle perfectly well on their own."""
    rater = object()

    assert pickle.loads(pickle.dumps(scored_alone)) is scored_alone

    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(lambda node, actions, player, through=rater: None)

    assert pickle.dumps(rater), "and what it captured was never the problem"


def test_a_decision_crosses_to_a_worker_whole() -> None:
    """Measured at a quarter of a megabyte for a whole game against minutes of valuing, which is why the
    decisions are sent entire to every worker rather than split between them."""
    found = [decision() for _ in range(50)]

    assert pickle.loads(pickle.dumps(found)) == found


def test_one_worker_runs_here_and_answers_the_same(monkeypatch) -> None:
    """A run that asked for no workers must behave as it did before, so a single worker is not a special case
    to write but what `TaskRunner` already does."""
    calls = []

    def rating(name, model, decisions, among=0, reading=1.0):
        calls.append(name)
        return _agreement(name)

    monkeypatch.setattr("openmind.training.service.agreement_dispatcher.scored_alone", rating)
    dispatcher = AgreementDispatcher(TaskRunner(1))

    found = dispatcher.scored([decision()], [("first", object()), ("second", object())])

    assert [one.holder for one in found] == ["first", "second"]
    assert calls == ["first", "second"], "and it ran in this process"


def test_nothing_to_judge_or_nobody_to_judge_it_is_nothing(monkeypatch) -> None:
    dispatcher = AgreementDispatcher(TaskRunner(1))

    assert dispatcher.scored([], [("a heuristic", object())]) == ()
    assert dispatcher.scored([decision()], []) == ()


def test_a_heuristic_whose_worker_died_leaves_the_round_standing(monkeypatch) -> None:
    """A round that stops because one candidate of two hundred blew up would lose the other hundred and
    ninety-nine judgings with it, and those are the evidence everything else is decided on."""
    from openmind.parallel.model.dropped_call import DroppedCall

    dispatcher = AgreementDispatcher(TaskRunner(1))
    monkeypatch.setattr(
        dispatcher._task_runner,  # noqa: SLF001
        "map",
        lambda *arguments, **named: [_agreement("survived"), DroppedCall(1, None)],
    )

    found = dispatcher.scored([decision()], [("survived", object()), ("died", object())])

    assert [one.holder for one in found] == ["survived"]


def _agreement(holder: str):
    from openmind.training.model.agreement import Agreement

    return Agreement(holder, mass=0.5, decided=1, declined=0, undecided=0, offered=0.5)
