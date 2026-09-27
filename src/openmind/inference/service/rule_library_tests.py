from openmind.inference.service.rule_library import RuleLibrary
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number, Variable


def refusing(*conditions):
    return Clause((Literal("refused", ()), *(one.denied for one in conditions)))


def unequal_offsets():
    return refusing(
        Literal("from nothing", (Constant("x"), Variable("W1"), Variable("D1"))),
        Literal("from nothing", (Constant("y"), Variable("W2"), Variable("D2"))),
        Literal("other_than", (Variable("D1"), Variable("D2"))),
    )


def onto_your_own():
    return refusing(
        Literal("turn", (Variable("Mover"),)),
        Literal("lands on", (Constant("self"), Constant("x"), Constant("y"), Constant("grid"), Variable("Mine"))),
    )


def a_postcode(row, column):
    return refusing(Literal("self", (Number(row), Number(column))))


def test_the_same_rule_learned_twice_is_one_rule():
    """Two sets arriving at the same rule separately arrive at the same pointer, because the pointer is made from
    what the rule says. Sharing needs no comparing pass and cannot be missed."""
    library = RuleLibrary()

    first = library.told("white bishop", unequal_offsets())
    second = library.told("black bishop", unequal_offsets())

    assert first == second
    assert len(library.all()) == 1
    assert library.pointed_at(first) == 2


def test_how_widely_a_rule_holds_is_read_rather_than_declared():
    """A rule every set names is a rule of the game; one a single set names is that mover's own. Nobody draws
    that line."""
    library = RuleLibrary()
    for mover in ("bishop", "rook", "queen"):
        library.told(mover, onto_your_own())
    library.told("bishop", unequal_offsets())

    shared = dict(library.shared())

    assert shared[library.put(onto_your_own())] == 3
    assert shared[library.put(unequal_offsets())] == 1


def test_a_rule_naming_a_square_never_gets_cheaper():
    """The whole pressure, and nothing has to recognise a postcode for it to work. A rule serving six movers is a
    sixth as dear per use; one that can only ever serve one goes on costing full price."""
    shared, apiece = RuleLibrary(), RuleLibrary()
    movers = ("bishop", "rook", "queen", "knight", "king", "pawn")
    for number, mover in enumerate(movers):
        shared.told(mover, onto_your_own())
        apiece.told(mover, a_postcode(number + 1, 1))

    assert shared.cost() < apiece.cost()
    assert len(shared.all()) == 1 and len(apiece.all()) == len(movers)


def test_what_nothing_points_at_is_forgotten():
    """The other half of the pricing. A theory that has been improved is cheaper than one merely added to, so
    repairing a set makes the whole thing smaller."""
    library = RuleLibrary()
    pointer = library.told("bishop", unequal_offsets())
    library.told("rook", onto_your_own())

    library.untold("bishop", pointer)

    assert library.at(pointer) is None
    assert len(library.all()) == 1


def test_a_rule_two_sets_need_survives_one_of_them_dropping_it():
    """Forgetting is by what points, not by who asked."""
    library = RuleLibrary()
    pointer = library.told("bishop", onto_your_own())
    library.told("rook", onto_your_own())

    library.untold("bishop", pointer)

    assert library.at(pointer) is not None
    assert library.pointed_at(pointer) == 1


def test_a_set_gives_back_its_rules_and_not_its_pointers():
    """Whoever asks what a mover refuses by wants rules. The pointers are the library's business."""
    library = RuleLibrary()
    library.told("bishop", unequal_offsets())
    library.told("bishop", onto_your_own())

    assert set(library.rules("bishop")) == {unequal_offsets(), onto_your_own()}
    assert library.rules("knight") == ()


def test_pointing_twice_is_pointing_once():
    library = RuleLibrary()
    library.told("bishop", unequal_offsets())
    pointer = library.told("bishop", unequal_offsets())

    assert len(library.of("bishop")) == 1
    assert library.pointed_at(pointer) == 1
