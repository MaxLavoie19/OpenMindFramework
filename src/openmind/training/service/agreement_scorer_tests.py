from dataclasses import replace

import pytest

from openmind.heuristic.model.node import Node
from openmind.training.model.decided import Decided
from openmind.training.service.agreement_scorer import AgreementScorer
from openmind.world.model.action import Action
from openmind.world.model.state import State

STEP = Action("step", (("by", 1),))
JUMP = Action("jump", (("by", 2),))
WAIT = Action("wait", ())
OFFERED = (STEP, JUMP, WAIT)


def decision(taken: Action, paid: float, offered: tuple[Action, ...] = OFFERED) -> Decided:
    return Decided(Node(State((("at", 0),))), offered, taken, "white", paid)


def rates(**named: float | None):
    """A heuristic that always says the same thing, whatever the position."""
    return lambda node, actions, player: tuple(named.get(one.name) for one in actions)


def test_a_heuristic_that_expected_what_the_winner_did_gathers_more_than_one_that_did_not() -> None:
    """The whole point: siding with whoever won is what a heuristic is scored for."""
    decisions = (decision(STEP, paid=1.0),)

    found = {one.holder: one for one in AgreementScorer().scored(
        decisions,
        {"knows": rates(step=10.0, jump=0.0, wait=0.0), "wrong": rates(step=0.0, jump=10.0, wait=0.0)},
    )}

    assert found["knows"].mass > found["knows"].offered, "it expected more than ignorance would"
    assert found["wrong"].mass < found["wrong"].offered, "and the one that backed the loser expected less"
    assert found["knows"].mass > 5 * found["wrong"].mass


def test_expecting_the_loser_is_worth_nothing_and_a_draw_is_worth_half() -> None:
    """What the game paid is the weight, on the scale a played model's accuracy already uses."""
    scorer, sure = AgreementScorer(), {"sure": rates(step=10.0, jump=0.0, wait=0.0)}

    won = scorer.scored((decision(STEP, paid=1.0),), sure)[0].mass
    drew = scorer.scored((decision(STEP, paid=0.5),), sure)[0].mass
    lost = scorer.scored((decision(STEP, paid=0.0),), sure)[0].mass

    assert won > drew > lost
    assert lost == 0.0


def test_knowing_nothing_is_the_line_the_score_is_read_against() -> None:
    """A mass on its own says nothing: what matters is whether it beat having no opinion at all."""
    decisions = (decision(STEP, paid=1.0), decision(JUMP, paid=1.0))

    found = AgreementScorer().scored(decisions, {"sure": rates(step=10.0, jump=10.0, wait=0.0)})[0]

    assert found.offered > 0, "three actions were on offer, so ignorance expected a third of them"
    assert found.mass > found.offered, "it ruled out the one that was never taken"


def test_a_heuristic_with_nothing_to_say_is_set_aside_rather_than_marked_wrong() -> None:
    """Not firing is not the same as being wrong. A rule that fires rarely and is right when it does is a
    rule worth keeping, and counting its silence as error is what drives it out."""
    decisions = (decision(STEP, paid=1.0),)

    found = AgreementScorer().scored(decisions, {"silent": rates()})[0]

    assert found.declined == 1
    assert found.decided == 0
    assert found.mass == 0.0


def test_rating_everything_alike_is_an_opinion_that_separates_nothing() -> None:
    """Told apart from a decline because they are different states — one could not read the position, the
    other read it and found nothing to choose — and counted against neither."""
    decisions = (decision(STEP, paid=1.0),)

    found = AgreementScorer().scored(decisions, {"flat": rates(step=1.0, jump=1.0, wait=1.0)})[0]

    assert found.undecided == 1
    assert found.decided == 0


def test_a_rule_that_fires_rarely_is_not_marked_down_for_its_coverage() -> None:
    """The specialist case, which folding coverage into the score would destroy: a fork rule sees nothing in
    most positions and is right in the few it does. Its score is over what it answered; how often it answered
    is a separate number the caller may weigh as it likes."""
    quiet = lambda node, actions, player: (  # noqa: E731
        (10.0, 0.0, 0.0) if actions[0].parameters == (("by", 9),) else (None, None, None)
    )
    fires = Action("step", (("by", 9),))
    decisions = (
        decision(fires, paid=1.0, offered=(fires, JUMP, WAIT)),
        decision(STEP, paid=1.0),
        decision(STEP, paid=1.0),
    )

    found = AgreementScorer().scored(decisions, {"fork": quiet})[0]

    assert found.decided == 1 and found.declined == 2
    assert found.mass > 2 * found.offered, "it is judged on what it answered, not on what it kept quiet about"


def test_an_action_it_never_saw_is_not_a_mistake_of_its_own() -> None:
    """Where the move played is not among the actions it was shown, nothing it said bears on what happened."""
    decisions = (decision(WAIT, paid=1.0),)
    unseen = lambda node, actions, player: (1.0, 2.0, None)  # noqa: E731

    found = AgreementScorer().scored(decisions, {"partial": unseen})[0]

    assert found.decided == 1, "wait was on offer and it rated the others, so it did decide"
    assert found.mass >= 0.0


def test_scoring_nothing_scores_every_heuristic_at_nothing() -> None:
    found = AgreementScorer().scored((), {"any": rates(step=1.0)})

    assert found[0].decided == 0 and found[0].mass == 0.0


def test_a_rule_that_fires_on_one_move_and_says_nothing_else_still_has_an_opinion() -> None:
    """The single-firing shape, which is most of what a rule is. Read as the least rating given, its silence
    would swallow the one thing it said and the rule would be recorded as knowing nothing."""
    decisions = (decision(STEP, paid=1.0),)

    found = AgreementScorer().scored(decisions, {"fork": rates(step=10.0)})[0]

    assert found.decided == 1, "it said one thing, which is an opinion"
    assert found.mass > 2 * found.offered, "and it expected far more of what happened than ignorance would"


def test_a_rule_that_only_fires_against_a_move_discourages_it() -> None:
    """A rule may push down as well as up: blundering mate is a rule whose whole content is that one move is
    ruinous and the rest unremarkable. It must be able to say so by speaking only of the bad one."""
    scorer = AgreementScorer()
    blunders = {"mate": rates(jump=-100.0)}

    avoided = scorer.scored((decision(STEP, paid=1.0),), blunders)[0]
    taken = scorer.scored((decision(JUMP, paid=1.0),), blunders)[0]

    assert avoided.decided == 1 and taken.decided == 1
    assert avoided.mass > taken.mass, "it expected the move it did not warn against"
    assert taken.mass < taken.offered / 3, "it warned against what happened and gets far less than ignorance"


def test_asking_about_every_action_is_what_a_caller_that_asks_for_nothing_gets() -> None:
    """A run that never asked for a sample must score exactly as it scored."""
    found = AgreementScorer().scored((decision(STEP, paid=1.0),), {"sure": rates(step=10.0, jump=0.0, wait=0.0)})[0]

    assert found.offered == pytest.approx(1 / 3), "ignorance over all three"


def test_a_sample_always_holds_the_move_that_was_played() -> None:
    """The whole question is what the heuristic made of what happened. A sample that could leave the played
    move out would be asking a different question and calling the answer a score."""
    scorer = AgreementScorer(among=2)

    for taken in (STEP, JUMP, WAIT):
        asked = scorer._asked(decision(taken, paid=1.0))  # noqa: SLF001

        assert taken in asked.offered
        assert len(asked.offered) == 2


def test_the_baseline_moves_with_the_sample() -> None:
    """What keeps a sample honest. Ignorance expects one in however many were asked about, so a sample of two
    is read against a half and not against a third — otherwise every heuristic would look better simply for
    having been asked less."""
    sampled = AgreementScorer(among=2).scored((decision(STEP, paid=1.0),), {"sure": rates(step=10.0, jump=0.0, wait=0.0)})[0]

    assert sampled.offered == pytest.approx(0.5)


def test_every_heuristic_is_asked_about_the_same_alternatives() -> None:
    """Drawn afresh per heuristic, two candidates would be scored on different questions and the difference
    would be recorded as skill."""
    scorer = AgreementScorer(among=2)
    one = decision(STEP, paid=1.0)

    assert scorer._asked(one).offered == scorer._asked(one).offered  # noqa: SLF001


def test_a_decision_with_fewer_actions_than_the_sample_is_asked_whole() -> None:
    """Nothing to sample, and taking a sample of five from three would either repeat an action or raise."""
    asked = AgreementScorer(among=9)._asked(decision(STEP, paid=1.0))  # noqa: SLF001

    assert asked.offered == OFFERED


def test_a_heuristic_is_judged_on_agreeing_with_what_the_search_settled_on() -> None:
    """**Distilling, put where heuristics are chosen between.** A search reads a heuristic at its leaves, looks
    ahead, and settles somewhere better. Selecting the heuristic whose shallow reading most resembles that is
    what moves a whole pool toward playing deeply without the looking ahead.

    Scored as the overlap of the two distributions, so a heuristic ranking the same moves a little differently
    is nearly right — which agreeing with the one move played cannot express."""
    searched = replace(decision(STEP, paid=1.0), searched=((STEP, 0.7), (JUMP, 0.3)))

    found = {one.holder: one for one in AgreementScorer().scored(
        (searched,),
        {"close": rates(step=7.0, jump=3.0, wait=0.0), "far": rates(step=1.0, jump=9.0, wait=0.0)},
    )}

    assert found["close"].mass > found["far"].mass, "the one that settled where the search settled"
    assert found["close"].decided == found["far"].decided == 1


def test_a_heuristic_agreeing_with_the_search_exactly_is_worth_more_than_one_that_only_tops_it() -> None:
    """What a distribution says and a single move cannot. Both pick the move the search liked best; one
    shares the search's whole opinion and the other is certain where the search was not."""
    searched = replace(decision(STEP, paid=1.0), searched=((STEP, 0.6), (JUMP, 0.4)))

    found = {one.holder: one for one in AgreementScorer().scored(
        (searched,),
        {"same": rates(step=6.0, jump=4.0, wait=0.0), "sure": rates(step=100.0, jump=0.0, wait=0.0)},
    )}

    assert found["same"].mass > found["sure"].mass


def test_what_was_played_is_judged_on_where_nothing_searched() -> None:
    """A game somebody else played carries no verdict of ours, and the move they made is what there is."""
    found = AgreementScorer().scored(
        (decision(STEP, paid=1.0),), {"one": rates(step=10.0, jump=0.0, wait=0.0)}
    )[0]

    assert found.decided == 1
    assert found.mass > found.offered
