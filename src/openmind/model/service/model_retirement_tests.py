from pathlib import Path

import pytest

from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_model_retirement

pytestmark = pytest.mark.log_level("INFO")


def base(tmp_path: Path) -> tuple[KnowledgeBase, str]:
    knowledge = create_knowledge_base("retirements", tmp_path)
    return knowledge, knowledge.ensure_context("chess").id


def record(name: str) -> ModelRecord:
    return ModelRecord(name, ("position value",), "chess", "rules", "mechanism-1", "ruleset-1")


def test_a_heuristic_that_has_shown_nothing_yet_is_not_retired(tmp_path: Path) -> None:
    """The difference between a heuristic shown to be useless and one nobody has asked enough. Coverage
    governs how long before a thing may be dropped and never what it is worth, so the count gates the
    retirement and never the number it is judged on."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()

    gone = retirement.standing(knowledge, context, {"barely asked": (-5.0, 3)}, least_decided=50)

    assert gone == ()
    assert not retirement.retired(knowledge, context, "barely asked")


def test_a_heuristic_at_or_below_knowing_nothing_is_retired(tmp_path: Path) -> None:
    """Nought is the line and nobody chose it: a heuristic at nought expected exactly what a heuristic with no
    opinion would have expected, and below it, it is actively wrong about what wins."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()

    gone = retirement.standing(knowledge, context, {"worse than nothing": (-0.4, 80)}, least_decided=50)

    assert gone == ("worse than nothing",)
    assert retirement.retired(knowledge, context, "worse than nothing")


def test_a_heuristic_that_saw_something_is_kept(tmp_path: Path) -> None:
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()

    assert retirement.standing(knowledge, context, {"it saw something": (0.3, 80)}, least_decided=50) == ()


def test_what_a_heuristic_has_shown_is_read_over_every_judging_and_not_the_last(tmp_path: Path) -> None:
    """One judging is a handful of games and a candidate can sit below ignorance in one of them by luck. What
    accumulates is what it has shown in total, so a retirement rests on all the evidence there has ever
    been — and a heuristic that is well ahead survives a bad round."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()
    for _ in range(4):
        retirement.standing(knowledge, context, {"mostly good": (0.5, 20)}, least_decided=50)

    gone = retirement.standing(knowledge, context, {"mostly good": (-0.4, 20)}, least_decided=50)

    assert gone == (), "one bad round does not undo four good ones"
    held, seen = retirement.shown(knowledge, context, "mostly good", 0.0, 0)
    assert held == pytest.approx(1.6)
    assert seen == 100


def test_enough_bad_rounds_do_retire_it(tmp_path: Path) -> None:
    """And accumulating is not a reprieve. A heuristic that keeps coming out below the line crosses it in
    total, which is the whole point of reading the total."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()
    gone: tuple = ()
    for _ in range(12):
        gone = retirement.standing(knowledge, context, {"steadily wrong": (-0.2, 20)}, least_decided=50)
        if gone:
            break

    assert gone == ("steadily wrong",)


def test_a_heuristic_shown_wrong_over_enough_decisions_is_not_judged_again(tmp_path: Path) -> None:
    """The saving this is for. Judging costs candidates times decisions times moves, so a candidate that goes
    on being asked after it has been shown wrong is the cost the retirement was meant to remove.

    Wrong by a wide margin and over four thousand decisions: nothing it could still turn out to be is above
    nought, so no amount of not knowing brings it back."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()
    retirement.shown(knowledge, context, "gone", worth=-400.0, decisions=4000)
    retirement.retire(knowledge, context, "gone", "it showed nothing")

    kept = retirement.keeping(knowledge, context, [record("gone"), record("still here")])

    assert [one.name for one in kept] == ["still here"]


def test_a_heuristic_retired_on_almost_nothing_is_asked_again(tmp_path: Path) -> None:
    """**Retirement is a bound and not a door**, because what a heuristic showed over a handful of decisions is
    not evidence that it is useless — and the reason it showed nothing may be that the pool it came from had
    no variety for it to be different from.

    Here it was wrong, but barely and over three decisions. What it could still be worth is well above nought,
    so it is the next best thing to try and it comes back. The one above was wrong over four thousand and does
    not."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()
    retirement.shown(knowledge, context, "hardly asked", worth=-0.1, decisions=3)
    retirement.retire(knowledge, context, "hardly asked", "it showed nothing")

    kept = retirement.keeping(knowledge, context, [record("hardly asked"), record("still here")])

    assert [one.name for one in kept] == ["hardly asked", "still here"]


def test_what_a_retired_heuristic_could_still_be_worth_narrows_as_it_is_asked_more(tmp_path: Path) -> None:
    """The standard error is the whole of what brings one back, so it has to shrink with the evidence. Two
    heuristics that have shown the same per decision, one over three decisions and one over three thousand:
    only the first is still an open question."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()
    retirement.shown(knowledge, context, "scant", worth=-0.3, decisions=3)
    retirement.shown(knowledge, context, "plenty", worth=-300.0, decisions=3000)

    scant = retirement.worth_asking_again(knowledge, context, "scant", among=20)
    plenty = retirement.worth_asking_again(knowledge, context, "plenty", among=20)

    assert scant > plenty, "the same showing over less evidence leaves more room to be wrong about"
    assert plenty < 0.0, "and over enough of it, there is no room left"


def test_retiring_the_same_heuristic_twice_does_not_add_to_its_record(tmp_path: Path) -> None:
    """Once it is out, it is out. Counting the rounds it was not asked about as more evidence against it would
    make a retirement compound itself."""
    knowledge, context = base(tmp_path)
    retirement = create_model_retirement()
    retirement.standing(knowledge, context, {"gone": (-1.0, 80)}, least_decided=50)
    before = retirement.shown(knowledge, context, "gone", 0.0, 0)

    retirement.standing(knowledge, context, {"gone": (-1.0, 80)}, least_decided=50)

    assert retirement.shown(knowledge, context, "gone", 0.0, 0) == before


def test_a_retirement_outlives_the_run_that_decided_it(tmp_path: Path) -> None:
    """Retired and not deleted, because the store is append-only and that is not an accident: what was learned
    about a heuristic survives, and a restart does not start asking it again."""
    knowledge, context = base(tmp_path)
    create_model_retirement().standing(knowledge, context, {"gone": (-1.0, 80)}, least_decided=50)

    again = create_knowledge_base("retirements", tmp_path)

    assert create_model_retirement().retired(again, again.context_named("chess").id, "gone")
