import pytest

from openmind.rhetoric.model.distance import Distance
from openmind.rhetoric.model.ethos import Ethos
from openmind.rhetoric.model.pathos import Pathos
from openmind.rhetoric.model.position import Position
from openmind.rhetoric.model.speaker import Speaker
from openmind.rhetoric.service.distance_measurer import DistanceMeasurer


def teacher() -> Speaker:
    """A chess teacher who knows a lot, shows a little less to a nervous student, and doesn't care about the weather,
    speaking to a student who knows less and cares a lot about their favourite opening."""
    effective = Ethos((Position("knows chess", 0.9, 0.8), Position("the weather is nice", 1.0, 0.1)))
    shown = Ethos((Position("knows chess", 0.6), Position("the weather is nice", 1.0)))
    perceived = Pathos((Position("knows chess", -0.2, 0.9), Position("the weather is nice", -1.0, 0.05)))
    return Speaker("teacher", effective, (("student", shown),), (("student", perceived),))


def test_identity_and_audience_distances_are_signed_and_weighed_by_importance() -> None:
    distances = DistanceMeasurer().distances(teacher())

    assert [(d.member, d.question, d.kind) for d in distances] == [
        ("student", "knows chess", "identity"),
        ("student", "the weather is nice", "identity"),
        ("student", "knows chess", "audience"),
        ("student", "the weather is nice", "audience"),
    ]
    identity, weather_identity, audience, weather = distances
    assert (identity.value, identity.problematicity) == pytest.approx((0.3, 0.24))
    assert (weather_identity.value, weather_identity.problematicity) == (0.0, 0.0)
    assert (audience.value, audience.problematicity) == pytest.approx((0.8, 0.72))
    assert (weather.value, weather.problematicity) == pytest.approx((2.0, 0.1))
    assert weather.value > audience.value and weather.problematicity < audience.problematicity


def test_a_question_only_one_side_answers_has_no_distance() -> None:
    speaker = Speaker(
        "ann",
        Ethos((Position("fish is good", 1.0),)),
        (("joe", Ethos((Position("fish is good", 1.0), Position("joe cheated", 1.0))),),),
        (("joe", Pathos((Position("joe cheated", -1.0),))),),
    )

    assert DistanceMeasurer().distances(speaker) == (
        Distance("joe", "fish is good", "identity", 0.0, 0.0),
        Distance("joe", "joe cheated", "audience", 2.0, 2.0),
    )
