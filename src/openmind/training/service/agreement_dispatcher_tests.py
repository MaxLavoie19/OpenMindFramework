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

    def rating(name, model, decisions, among=0, reading=1.0, seconds=0.0):
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


def test_a_whole_round_s_budget_divides_down_to_one_valuing(monkeypatch) -> None:
    """**One heuristic in a pool can hold a round on its own.** Measured on a run's own store: of ten
    heuristics, nine valued a position in 0.2 to 0.6 ms and one took 277 ms — fourteen hundred times the
    others — and that one was essentially the whole of a 195-second round. Handed the same share as everybody
    else, it reads what it can afford in that and the round is the round.

    **The allowance is per valuing and is the same for every heuristic, not a slice of the round divided among
    them.** They are judged in parallel, one to a worker, so a round takes what the slowest takes rather than
    the sum — and divided by the pool it would be a quota that shrank as the pool grew, starving a ruleset
    that holds one slow rule even where its other rules left room for it. Within the allowance a ruleset takes
    what is worth the most for its cost while the total fits, so a long rule is balanced out by short ones."""
    given = []

    def rating(name, model, decisions, among=0, reading=1.0, seconds=0.0):
        given.append(seconds)
        return _agreement(name)

    monkeypatch.setattr("openmind.training.service.agreement_dispatcher.scored_alone", rating)
    dispatcher = AgreementDispatcher(TaskRunner(1), among=2, seconds=12.0)

    dispatcher.scored([decision(), decision(), decision()], [("one", object()), ("two", object())])

    assert given == [pytest.approx(2.0), pytest.approx(2.0)], "twelve seconds over three decisions of two moves"


def test_a_judging_given_no_budget_is_not_bounded(monkeypatch) -> None:
    """Nought leaves the judging as it was, which is what a caller that asks for nothing gets."""
    given = []

    def rating(name, model, decisions, among=0, reading=1.0, seconds=0.0):
        given.append(seconds)
        return _agreement(name)

    monkeypatch.setattr("openmind.training.service.agreement_dispatcher.scored_alone", rating)

    AgreementDispatcher(TaskRunner(1)).scored([decision()], [("one", object())])

    assert given == [0.0]
