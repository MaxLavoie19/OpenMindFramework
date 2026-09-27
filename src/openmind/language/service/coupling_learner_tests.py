from dataclasses import dataclass

from openmind.language.model.coupling import Coupling
from openmind.language.model.happening import Happening
from openmind.language.model.role import Role
from openmind.language.model.shape import Shape
from openmind.language.service.coupling_learner import CouplingLearner
from openmind.language.service.syntax_learner import SyntaxLearner
from openmind.structure.model.grid import Grid
from openmind.structure.model.record import Record
from openmind.statement.model.change import Moved, Removed
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Thing(Record):
    sort: str


def a_position(**things):
    """A four by four board with those things on it, named by row and column: t12 is row 1, column 2."""
    held = [[None] * 4 for _ in range(4)]
    for name, sort in things.items():
        held[int(name[1]) - 1][int(name[2]) - 1] = Thing(sort)
    return State.of(grid=Grid.of(held))


def carrying(source, target, taking=False):
    changes = [Removed("grid", target)] if taking else []
    return Happening((*changes, Moved("grid", source, target)))


def seen():
    """A little notation of made-up rules, in two forms.

    Either the sort's letter and the row landed on, or the sort's letter, the row set out from, `x`, and the row
    landed on. So the same digit says where a carrying ended in one form and where it began in the other — which
    is exactly what a place counted from the start of a string cannot tell apart, and what a place of a shape
    can."""
    found = []
    for sort, letter in (("hopper", "H"), ("walker", "W")):
        for column in range(1, 5):
            for row in range(2, 5):
                state = a_position(**{f"t1{column}": sort})
                found.append((state, f"{letter}{row}", carrying((1, column), (row, column))))
            for source in range(1, 5):
                for row in range(1, 5):
                    if source == row:
                        continue
                    state = a_position(**{f"t{source}{column}": sort})
                    found.append(
                        (state, f"{letter}{source}x{row}", carrying((source, column), (row, column), taking=True))
                    )
    return found * 4


def a_grammar(sightings=None):
    """The shapes those notations fall into, learned from the strings and nothing else."""
    return SyntaxLearner().learn([notation for _, notation, _ in sightings or seen()])


def learned(**how):
    return CouplingLearner().learn(seen(), a_grammar(), least=4, **how)


def test_a_symbol_belongs_to_what_it_tells_us_about():
    """Nothing says a letter names a sort or a digit a row. A symbol belongs to a part when knowing it stands in
    a place narrows that part, measured in bits over many sightings."""
    found = {(one.symbol, one.part) for one in learned() if one.value == "hopper"}

    assert ("H", "from sort") in found
    assert not any(symbol == "W" for symbol, _ in found)


def test_a_symbol_that_tells_us_nothing_is_not_a_symbol():
    """However often it turns up. This is what lets the pieces of a notation be found rather than declared, since
    nothing has to decide beforehand what any of them mean.

    Asked of a symbol and a part that really are unrelated, rather than by raising the threshold past what any
    part has to say. A sort's letter stands in half of every sighting there is and settles nothing about which
    row a carrying ended on — the rows are spread the same among the hoppers as among everything — so it carries
    no bits about that part and is not paired with it. A threshold high enough to exclude every saying excludes
    the good ones too, by the ceiling of what the part holds, and would go on passing if the measure broke."""
    found = {(one.symbol, one.part) for one in learned()}

    assert ("H", "from sort") in found
    assert ("H", "to place 1") not in found


def test_a_symbol_that_settles_its_part_is_a_symbol_however_seldom_it_stands():
    """How rare a word is and what it means are two questions, and only the second decides whether it is a word.

    Measured against the same notation with one landing row thinned to a twentieth of the sightings. The digit
    naming that row settles where a carrying ended exactly as the common digits do, and is learned beside them.

    **Pinned as the three digits carrying the same number, which is the claim.** Each of them settles where the
    carrying ended, and they are alike in everything but how often they are said — so a measure of what they say
    must give them one answer. Averaged over the sightings a symbol is *absent* from it gives three: 0.98 for
    the two common digits and 0.60 for the thinned one, because what that quantity can reach is capped by how
    often the symbol turns up. Asked where each stands, all three come to 1.45.

    That cap is what silenced chess's a- and h-files. Eight files put every letter within a hair of the line,
    and the two rarest fell under it while being right nine hundred and ninety-seven times in a thousand — each
    of the eight sitting exactly at its own ceiling, which is what said the number was measuring frequency
    rather than meaning.

    Thinned only to a seventh, not further: below that the syntax learner gives the rare digit a sort of its
    own, and a part its shape settles is said by nothing at all — which is a different matter, decided
    separately, and would let this pass for the wrong reason."""
    thinned = [one for one in seen() if one[1][-1] != "4"]
    thinned += [one for number, one in enumerate(one for one in seen() if one[1][-1] == "4") if number % 3 == 0]
    found = CouplingLearner().learn(thinned, a_grammar(thinned), least=4)
    said = {
        symbol: max(
            (one.telling for one in found if one.part == "to place 1" and one.symbol == symbol), default=0.0
        )
        for symbol in "234"
    }

    assert ("4", "to place 1", 4) in {(one.symbol, one.part, one.value) for one in found}
    assert said["4"] == said["2"] == said["3"]


def test_the_same_character_in_two_roles_says_two_things():
    """A digit says where a carrying ended in one shape and where it began in another. Counted from the start of
    the string those are one saying, right some of the time and wrong the rest; as places of shapes they are two,
    each right always.

    Asked as *nearly* always rather than exactly always, because how sure a saying is is now counted with a case
    assumed either way and so never quite reaches one — a symbol right in every one of twenty sightings comes to
    about 0.95, which is the point: nothing is certain from nothing, and a bare share said a symbol seen twice
    was certain."""
    found = [one for one in learned() if one.symbol == "3" and one.surely > 0.9]

    assert {one.part for one in found} >= {"to place 1", "from place 1"}
    assert all(one.value == 3 for one in found)


def test_what_the_notation_leaves_out_is_reported_rather_than_missed(caplog):
    """A part nothing says anything about is a part the notation does not carry. In chess that is where a move
    starts, left out because the rules make it recoverable — so the measure says what the rules have to supply."""
    import logging

    with caplog.at_level(logging.INFO):
        learned()

    assert "nothing says anything about" in caplog.text


def test_places_that_say_the_same_things_are_made_one_role():
    """The trailing square of a plain carrying and of a taking one say the same thing about where it ended.
    Said once it is a rule; said per shape it is a table that grows with the shapes."""
    plain = Shape(("HW", "1234"))
    taking = Shape(("HW", "1234", "x", "1234"))
    family = [
        Coupling("3", "to place 1", 3, Role(((plain, 1),)), 0.6, 1.0, 100),
        Coupling("3", "to place 1", 3, Role(((taking, 3),)), 0.4, 1.0, 300),
    ]

    found = CouplingLearner().said_alike(family)

    assert len(found) == 1
    assert found[0].where.places == ((plain, 1), (taking, 3))
    assert found[0].telling == (0.6 * 100 + 0.4 * 300) / 400
    assert found[0].seen == 400


def test_a_place_that_says_nearly_the_same_is_left_alone():
    """Nearly is not the same. A place that says most of what another says is a different place, and joining them
    would put a saying where it was never measured."""
    plain = Shape(("HW", "1234"))
    taking = Shape(("HW", "1234", "x", "1234"))
    family = [
        Coupling("3", "to place 1", 3, Role(((plain, 1),)), 0.6, 1.0, 100),
        Coupling("3", "to place 1", 3, Role(((taking, 3),)), 0.4, 1.0, 300),
        Coupling("3", "at place 1", 3, Role(((taking, 3),)), 0.4, 1.0, 300),
    ]

    found = CouplingLearner().said_alike(family)

    assert {len(one.where.places) for one in found} == {1}


def test_a_family_running_in_step_is_said_once():
    """Four digits naming four rows are four sayings and a list. Said as one correspondence they are a rule, and
    a fifth digit reads as a fifth row on a board that has one."""
    family = [Coupling(str(number), "to place 1", number, None, 0.6) for number in range(1, 5)]

    said = CouplingLearner().said_as_one(family)

    assert len(said) == 1
    assert said[0].says("5") == 5


def test_a_family_with_no_order_to_run_in_is_left_alone():
    """Two things run in step when there is a step to run in. The order of a set of names is the order somebody
    spelled them in — B, N and R sorting as bishop, knight and rook is a fact about English, not about chess."""
    kinds = [
        Coupling("B", "from sort", "bishop", None, 0.6),
        Coupling("N", "from sort", "knight", None, 0.6),
        Coupling("R", "from sort", "rook", None, 0.6),
    ]

    assert not CouplingLearner().said_as_one(kinds)
