from openmind.language.mapper.expression_meaning_mapper import ExpressionMeaningMapper
from openmind.language.mapper.meaning_json_mapper import MeaningJsonMapper
from openmind.language.model.glossary import Glossary


def test_a_look_ahead_holds_whose_moves_and_what_it_reads_after_them() -> None:
    source = (
        "here.best(other, lambda v2: v2.worst(me, lambda v1: (sum(1 for at in v1.color if v1.color[at] == other)) "
        "- (sum(1 for at in v2.color if v2.color[at] == other))))"
    )

    assert ExpressionMeaningMapper().to_meaning(source) == {
        "look ahead": "highest",
        "moves of": "opponent",
        "of": {
            "look ahead": "lowest",
            "moves of": "me",
            "of": {
                "change in": {
                    "count": "places",
                    "where": [
                        {
                            "compare": "equals",
                            "left": {"reading": "color", "at": {"place": "there"}},
                            "right": {"player": "opponent"},
                        }
                    ],
                }
            },
        },
    }


def test_a_reading_on_an_earlier_position_holds_who_moved_to_reach_it() -> None:
    meaning = ExpressionMeaningMapper().to_meaning("here.best(other, lambda v2: v2.best(me, lambda v1: v2.x[me] + here.x[other]))")

    assert meaning["of"]["of"] == {  # type: ignore[index]
        "arithmetic": "plus",
        "left": {"reading": "x", "at": {"player": "me"}, "position": {"after moves of": ["opponent"]}},
        "right": {"reading": "x", "at": {"player": "opponent"}, "position": {"now": True}},
    }


def test_the_glossary_names_indices_and_meanings_round_trip_through_json() -> None:
    glossary = Glossary({"color": {(5, 4): "d5"}})
    mapper, json_mapper = ExpressionMeaningMapper(), MeaningJsonMapper()

    meaning = mapper.to_meaning("(here.color[5, 4] == me) * (here.color[5, 5] == None)", glossary)

    assert meaning == {
        "arithmetic": "times",
        "left": {"compare": "equals", "left": {"reading": "color", "at": {"label": "d5"}}, "right": {"player": "me"}},
        "right": {"compare": "equals", "left": {"reading": "color", "at": {"indices": [5, 5]}}, "right": {"empty": True}},
    }
    assert json_mapper.from_json(json_mapper.to_json(meaning)) == meaning
    assert json_mapper.to_json({"number": 3}) == '{"number": 3}'
