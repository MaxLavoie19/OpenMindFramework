from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, DEFINITIONS, EFFECTS, POSITION, VALUES
from openmind.knowledge.constant.task_constant import SIMULATION
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.source import Source
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.model.python_rule import PythonRule

SOURCE = Source("a test")


def a_rule(name: str, kind: str, source: str = "True", **held) -> RuleRecord:
    return RuleRecord(name, kind, PythonRule(source), SOURCE, id=name, **held)


def an_rbs(*rules: tuple[RuleRecord, float]) -> RuleBasedSystem:
    return RuleBasedSystem("a game", "context-1", Ruleset(SIMULATION, "context-1", SIMULATION, SOURCE), rules)


def a_game() -> RuleBasedSystem:
    """A little game of two actions, so that what belongs to which is something to get wrong."""
    return an_rbs(
        (a_rule("place is legal, 1", CONSTRAINT, action="place"), 1.0),
        (a_rule("what cell can be in place", VALUES, action="place", parameter="cell"), 1.0),
        (a_rule("what mark can be in place", VALUES, action="place", parameter="mark"), 1.0),
        (a_rule("what place leads to", EFFECTS, action="place"), 1.0),
        (a_rule("pass is legal, 1", CONSTRAINT, action="pass"), 1.0),
        (a_rule("what they lead to together", EFFECTS), 1.0),
        (a_rule("a heuristic", POSITION), 0.25),
    )


def test_the_actions_are_the_ones_its_constraints_and_values_are_about():
    """Nothing declares an action's name: what actions a game has falls out of the rules that speak of one, which
    is why a game with no constraint and no values rule has no actions at all."""
    assert a_game().action_names() == ("place", "pass")
    assert an_rbs((a_rule("a heuristic", POSITION), 1.0)).action_names() == ()


def test_each_action_keeps_its_own_parameters_and_its_own_constraints():
    game = a_game()

    assert sorted(game.values("place")) == ["cell", "mark"]
    assert game.values("pass") == {}
    assert len(game.constraints("place")) == 1
    assert len(game.constraints("pass")) == 1


def test_what_the_players_do_together_is_the_effects_rule_belonging_to_no_action():
    """A game where several act at once has something that happens after all of them, and it is told from an
    action's own effects by having no action."""
    game = a_game()

    assert len(game.effects("place")) == 1
    assert len(game.effects(None)) == 1


def test_an_effects_rule_carries_the_chance_it_happens():
    """A game with chance in it has nowhere else to say so: the outcome and how often it comes about are one rule."""
    rbs = an_rbs(
        (a_rule("what roll leads to, 1", EFFECTS, action="roll", probability=0.25), 1.0),
        (a_rule("what roll leads to, 2", EFFECTS, action="roll", probability=0.75), 1.0),
    )

    assert sorted(chance for chance, _ in rbs.effects("roll")) == [0.25, 0.75]


def test_a_rule_weighs_what_the_ruleset_lists_it_at_and_a_rule_it_does_not_list_weighs_nothing():
    """A heuristic's value is the sum of its rules' weighted readings, so a rule the ruleset never listed has to
    add nothing rather than one."""
    kept = a_rule("a heuristic", POSITION)
    game = a_game()

    assert game.weight(kept) == 0.25
    assert game.weight(a_rule("never listed", POSITION)) == 0.0


def test_rules_come_back_in_the_order_they_were_linked():
    """Order is not presentation for effects: what happens first is what a game said happened first."""
    assert [rule.name for rule in a_game().of(EFFECTS)] == ["what place leads to", "what they lead to together"]


def test_the_definitions_a_rule_sees_are_found_by_name():
    rbs = an_rbs(
        (a_rule("the names its rules see", DEFINITIONS, "x = 1"), 1.0),
        (a_rule("the names its effects see", DEFINITIONS, "y = 2"), 1.0),
    )

    assert rbs.definitions("the names its effects see").source == "y = 2"


def test_a_ruleset_with_one_definitions_script_gives_it_whatever_it_is_called():
    """A game that declared one script and a service asking for a name it does not use should not silently see
    no definitions at all — which reads as every rule failing, for no visible reason."""
    rbs = an_rbs((a_rule("whatever it is called", DEFINITIONS, "x = 1"), 1.0))

    assert rbs.definitions("some other name").source == "x = 1"


def test_a_ruleset_with_several_scripts_and_no_match_gives_none_rather_than_guessing():
    rbs = an_rbs(
        (a_rule("one script", DEFINITIONS, "x = 1"), 1.0),
        (a_rule("another script", DEFINITIONS, "y = 2"), 1.0),
    )

    assert rbs.definitions("neither of them") is None
