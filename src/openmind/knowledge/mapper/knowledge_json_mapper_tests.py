from datetime import datetime

from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.context import Context
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.mechanism import Mechanism
from openmind.knowledge.model.opinion import Opinion
from openmind.knowledge.model.source import Source
from openmind.knowledge.model.task import Task

AT = datetime(2026, 9, 18, 14, 30)
CAMERA = Source("camera", (("device", "front"), ("frame", 12)), AT)


def test_everything_the_knowledge_base_keeps_goes_out_and_comes_back_equal() -> None:
    mapper = KnowledgeJsonMapper()
    experience = DirectExperience("seen", "chess", "e4", CAMERA, AT, (("topic", "openings"),), "e000001")
    belief = Belief(
        "white opened with the king's pawn",
        "chess",
        True,
        ("black",),
        0.9,
        0.8,
        None,
        (Evidence(True, 0.9, Source("mechanism-reader", (("frame", 12),), AT, ("experience-1",))),),
        (("topic", "openings"), ("date", "2026-09-18")),
        "b000001",
    )
    opinion = Opinion("the king's gambit", "chess", "reckless", AT, (Source("belief", (("id", "b000001"),)),), (), "o000001")
    task = Task("review the game", "chess", (belief,), Belief("expected time", "chess", 90.0), "running", (), "t000001")
    context = Context("context-960", "chess/960", "context-life", ("context-chess",), (("family", "chess"),))
    mechanism = Mechanism("mechanism-reader", "move reader", 0.9, (("kind", "decoder"),))

    assert mapper.experience_from_data(mapper.experience_to_data(experience)) == experience
    assert mapper.belief_from_data(mapper.belief_to_data(belief)) == belief
    assert mapper.opinion_from_data(mapper.opinion_to_data(opinion)) == opinion
    assert mapper.task_from_data(mapper.task_to_data(task)) == task
    assert mapper.context_from_data(mapper.context_to_data(context)) == context
    assert mapper.mechanism_from_data(mapper.mechanism_to_data(mechanism)) == mechanism
