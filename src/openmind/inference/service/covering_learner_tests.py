from openmind.inference.service.covering_learner import Covering, CoveringLearner


def sliding_pieces():
    """Two kinds that slide and one that does not, with one kind represented by a single action.

    That single action is what a residue is made of. Learned on its own there is nothing left to contradict what
    separates it, so the column it happens to be on becomes a condition; learned among the others it is one more
    sliding piece."""
    legal = [
        {"kind": "bishop", "slides": True, "mine": True, "column": 1},
        {"kind": "bishop", "slides": True, "mine": True, "column": 2},
        {"kind": "bishop", "slides": True, "mine": True, "column": 3},
        {"kind": "rook", "slides": True, "mine": True, "column": 4},
    ]
    illegal = [
        {"kind": "rook", "slides": True, "mine": False, "column": 4},
        {"kind": "rook", "slides": True, "mine": False, "column": 5},
        {"kind": "pawn", "slides": False, "mine": True, "column": 1},
        {"kind": "pawn", "slides": True, "mine": True, "column": 2},
    ]
    return [(readings, True) for readings in legal] + [(readings, False) for readings in illegal]


def test_a_rule_may_cover_what_another_rule_already_covers():
    learner = CoveringLearner()

    rules = learner.learn(sliding_pieces(), grow=1.0, positions=1, overlapping=True)

    assert sum(rule.covers for rule in rules) > 4


def test_a_leftover_action_is_grown_among_the_others_rather_than_on_its_own():
    """Covering it alone names the column it stands on, which explains that action and nothing else."""
    learner = CoveringLearner()

    alone = learner.learn(sliding_pieces(), grow=1.0, positions=1)
    among = learner.learn(sliding_pieces(), grow=1.0, positions=1, overlapping=True)

    assert ("column", "==", 4) in alone[-1].conditions
    assert ("column", "==", 4) not in among[-1].conditions


def test_learning_stops_once_every_action_is_covered_and_not_at_a_count():
    """Nothing caps how many rules a game takes, so what stops it is having covered the game."""
    learner = CoveringLearner()
    examples = sliding_pieces()

    rules = learner.learn(examples, grow=1.0, positions=1, overlapping=True)

    assert len(rules) <= sum(1 for _, legal in examples if legal)
    assert all(learner.covers(rules, readings) for readings, legal in examples if legal)


def test_every_legal_action_is_covered_either_way():
    learner = CoveringLearner()
    examples = sliding_pieces()

    for overlapping in (False, True):
        rules = learner.learn(examples, grow=1.0, positions=1, overlapping=overlapping)
        assert learner.scored(rules, examples) == (4, 0, 0), [rule.readable for rule in rules]


def test_a_loose_rule_and_its_exclusion_together_say_what_a_tight_rule_says():
    """The layered shape: a draft that lists as much as it can, and what forbids the rest stated under it."""
    learner = CoveringLearner()
    examples = sliding_pieces()

    drafts = learner.learn(examples, grow=1.0, positions=1, overlapping=True, loosely=0.5)
    assert learner.scored(drafts, examples) == (4, 0, 1)

    told = learner.excluding(drafts, examples, grow=1.0, positions=1)
    assert learner.scored(told, examples) == (4, 0, 0)


def sliding_and_leaping():
    return (
        Covering((("mine", "==", True), ("clear", "==", True), ("kind", "==", "rook"), ("straight", "==", True)), 3, 0),
        Covering((("mine", "==", True), ("clear", "==", True), ("kind", "==", "bishop"), ("diagonal", "==", True)), 3, 0),
        Covering((("mine", "==", True), ("kind", "==", "knight"), ("leaps", "==", True)), 2, 0),
    )


def test_what_the_rules_share_is_said_once_as_a_principle():
    learner = CoveringLearner()

    principles = learner.distilled(sliding_and_leaping())

    assert len(principles) == 1
    assert principles[0].conditions == (("mine", "==", True),)


def test_a_family_within_a_family_becomes_a_principle_of_its_own():
    """Needing an unobstructed way is a rule about sliding, not three facts about three pieces."""
    learner = CoveringLearner()

    principles = learner.distilled(sliding_and_leaping())
    sliding = [one for one in principles[0].under if one.conditions == (("clear", "==", True),)]

    assert len(sliding) == 1
    assert {one.conditions for one in sliding[0].under} == {
        (("kind", "==", "rook"), ("straight", "==", True)),
        (("kind", "==", "bishop"), ("diagonal", "==", True)),
    }


def test_a_principle_covers_what_the_rules_it_was_drawn_from_covered():
    learner = CoveringLearner()
    rules = sliding_and_leaping()
    actions = [
        {"mine": True, "clear": True, "kind": "rook", "straight": True, "diagonal": False, "leaps": False},
        {"mine": True, "clear": True, "kind": "bishop", "straight": False, "diagonal": True, "leaps": False},
        {"mine": True, "clear": False, "kind": "knight", "straight": False, "diagonal": False, "leaps": True},
        {"mine": True, "clear": False, "kind": "rook", "straight": True, "diagonal": False, "leaps": False},
        {"mine": False, "clear": True, "kind": "rook", "straight": True, "diagonal": False, "leaps": False},
    ]
    principles = learner.distilled(rules)

    for readings in actions:
        assert learner.covers(principles, readings) == learner.covers(rules, readings), readings


def test_a_principle_counts_what_it_covers_of_the_evidence():
    learner = CoveringLearner()
    examples = [
        ({"mine": True, "clear": True, "kind": "rook", "straight": True, "diagonal": False, "leaps": False}, True),
        ({"mine": True, "clear": True, "kind": "bishop", "straight": False, "diagonal": True, "leaps": False}, True),
        ({"mine": False, "clear": True, "kind": "rook", "straight": True, "diagonal": False, "leaps": False}, False),
    ]

    principles = learner.distilled(sliding_and_leaping(), examples)

    assert (principles[0].covers, principles[0].wrongly) == (2, 0)


def test_a_specialization_is_counted_with_the_conditions_it_inherits():
    """It only ever answers for actions its principle covers, so it is not wrong about the rest."""
    learner = CoveringLearner()
    rules = (
        Covering((("mine", "==", True), ("kind", "==", "rook")), 1, 0),
        Covering((("mine", "==", True), ("kind", "==", "bishop")), 1, 0),
    )
    examples = [
        ({"mine": True, "kind": "rook"}, True),
        ({"mine": True, "kind": "bishop"}, True),
        ({"mine": False, "kind": "rook"}, False),
        ({"mine": False, "kind": "rook"}, False),
    ]

    principles = learner.distilled(rules, examples)

    assert (principles[0].covers, principles[0].wrongly) == (2, 0)
    assert [(one.covers, one.wrongly) for one in principles[0].under] == [(1, 0), (1, 0)]


def test_a_rule_is_grown_where_the_evidence_is_thickest_before_it_is_seeded():
    """Seeding every rule spends them on whichever family the leftover action happens to belong to, and the large
    families no rule has reached are never asked about."""
    learner = CoveringLearner()
    examples = sliding_pieces()
    first = [one for one in examples if one[0].get("kind") == "rook" and one[1]]
    reordered = first + [one for one in examples if one not in first]

    rules = learner.learn(reordered, grow=1.0, positions=1, overlapping=True)

    assert ("kind", "==", "bishop") in rules[0].conditions


def test_a_rule_that_holds_nothing_but_the_principle_leaves_nothing_to_specialize():
    learner = CoveringLearner()
    rules = (
        Covering((("mine", "==", True),), 4, 0),
        Covering((("mine", "==", True), ("kind", "==", "rook")), 3, 0),
    )

    principles = learner.distilled(rules)

    assert principles[0].conditions == (("mine", "==", True),)
    assert principles[0].under == ()


def test_what_forbids_every_specialization_is_learned_at_the_principle():
    """A prohibition of the game is stated once, where everything it applies to can be seen."""
    learner = CoveringLearner()
    rules = (
        Covering((("mine", "==", True), ("kind", "==", "rook")), 2, 1),
        Covering((("mine", "==", True), ("kind", "==", "bishop")), 2, 1),
    )
    examples = [
        ({"mine": True, "kind": "rook", "attacked": False}, True),
        ({"mine": True, "kind": "rook", "attacked": True}, False),
        ({"mine": True, "kind": "bishop", "attacked": False}, True),
        ({"mine": True, "kind": "bishop", "attacked": True}, False),
    ]

    told = learner.excluding(learner.distilled(rules), examples, grow=1.0, positions=1)

    assert told[0].excluding[0].conditions == (("attacked", "==", True),)
    assert all(not one.excluding for one in told[0].under)
    assert learner.scored(told, examples) == (2, 0, 0)


def test_a_rule_does_not_cover_what_its_own_exclusion_rejects():
    learner = CoveringLearner()
    rules = (Covering((("kind", "==", "rook"),), 2, 0, (Covering((("blocked", "==", True),), 1, 0),)),)

    assert learner.covers(rules, {"kind": "rook", "blocked": False})
    assert not learner.covers(rules, {"kind": "rook", "blocked": True})


def test_another_rule_covering_it_is_not_bound_by_that_exclusion():
    """Exclusions belong to the rule they were learned under: what forbids one way of being legal says nothing
    about another."""
    learner = CoveringLearner()
    rules = (
        Covering((("kind", "==", "rook"),), 2, 0, (Covering((("blocked", "==", True),), 1, 0),)),
        Covering((("mine", "==", True),), 2, 0),
    )

    assert learner.covers(rules, {"kind": "rook", "blocked": True, "mine": True})


def test_what_the_action_leads_to_is_asked_for_only_once_a_rule_covers_it():
    learner = CoveringLearner()
    rules = (
        Covering((("kind", "==", "rook"),), 2, 0, (Covering((("attacked", "==", True),), 1, 0),)),
        Covering((("kind", "==", "pawn"),), 2, 0),
    )
    asked = []

    def after():
        asked.append(True)
        return {"attacked": True}

    assert not learner.covers(rules, {"kind": "rook"}, after)
    assert len(asked) == 1
    assert learner.covers(rules, {"kind": "pawn"}, after)
    assert len(asked) == 1


def test_a_rule_covers_where_what_it_leads_to_cannot_be_told():
    learner = CoveringLearner()
    rules = (Covering((("kind", "==", "rook"),), 2, 0, (Covering((("attacked", "==", True),), 1, 0),)),)

    assert learner.covers(rules, {"kind": "rook"}, lambda: None)


def test_exclusions_are_learned_from_the_actions_their_own_rule_covers():
    learner = CoveringLearner()
    examples = [
        ({"kind": "rook", "blocked": False}, True),
        ({"kind": "rook", "blocked": True}, False),
        ({"kind": "rook", "blocked": False}, True),
        ({"kind": "pawn", "blocked": True}, False),
    ]
    rules = (Covering((("kind", "==", "rook"),), 2, 1),)

    told = learner.excluding(rules, examples, grow=1.0, positions=1)

    assert told[0].excluding[0].conditions == (("blocked", "==", True),)
    assert learner.covers(told, {"kind": "rook", "blocked": False})
    assert not learner.covers(told, {"kind": "rook", "blocked": True})


def test_an_exception_to_a_prohibition_allows_the_action_again():
    """A prohibition may have exceptions, which is how a game says en passant: a pawn may not take an empty
    square, except the one a pawn just passed."""
    learner = CoveringLearner()
    rules = (
        Covering(
            (("kind", "==", "pawn"),),
            2,
            0,
            (Covering((("empty", "==", True),), 2, 0, (Covering((("passed", "==", True),), 1, 0),)),),
        ),
    )

    assert learner.covers(rules, {"kind": "pawn", "empty": False, "passed": False})
    assert not learner.covers(rules, {"kind": "pawn", "empty": True, "passed": False})
    assert learner.covers(rules, {"kind": "pawn", "empty": True, "passed": True})


def test_a_rule_is_answered_until_it_is_wrong_about_nothing_it_has_seen():
    learner = CoveringLearner()
    rules = (Covering((("kind", "==", "pawn"),), 2, 2),)
    examples = [
        ({"kind": "pawn", "across": True, "empty": False, "passed": False}, True),
        ({"kind": "pawn", "across": True, "empty": True, "passed": False}, False),
        ({"kind": "pawn", "across": True, "empty": True, "passed": True}, True),
        ({"kind": "pawn", "across": False, "empty": True, "passed": False}, True),
        ({"kind": "pawn", "across": True, "empty": True, "passed": False}, False),
        ({"kind": "pawn", "across": True, "empty": True, "passed": True}, True),
    ]

    told = learner.excluding(rules, examples, grow=1.0, positions=1)

    assert learner.scored(told, examples) == (4, 0, 0), told[0].readable


def test_a_level_that_answers_for_as_much_as_the_one_above_is_not_taken_further():
    """What stops the alternation is making no progress, which is why it always stops."""
    learner = CoveringLearner()
    rules = (Covering((("kind", "==", "pawn"),), 1, 1),)
    examples = [
        ({"kind": "pawn", "across": True}, True),
        ({"kind": "pawn", "across": True}, False),
    ]

    told = learner.excluding(rules, examples, grow=1.0, positions=1)

    assert all(not one.excluding for one in told[0].excluding)


def test_a_rule_that_covers_nothing_illegal_is_given_no_exclusion():
    learner = CoveringLearner()
    examples = [({"kind": "rook", "blocked": False}, True), ({"kind": "pawn", "blocked": True}, False)]
    rules = (Covering((("kind", "==", "rook"),), 1, 0),)

    assert learner.excluding(rules, examples, positions=1) == rules


def test_a_rule_reads_with_what_it_excludes():
    rule = Covering((("kind", "==", "rook"),), 2, 0, (Covering((("blocked", "==", True),), 1, 0),))

    assert rule.readable == "kind == 'rook', except where blocked == True"
