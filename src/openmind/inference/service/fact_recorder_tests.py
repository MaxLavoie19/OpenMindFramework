import tempfile
from pathlib import Path

from openmind.inference.model.fact import Fact
from openmind.inference.service.fact_recorder import FactRecorder
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base


def a_knowledge_base(directory):
    return create_knowledge_base("chess", Path(directory))


def a_chain():
    """A rook reaches fourteen, a queen takes in a rook, so a queen is worth at least fourteen."""
    reaches = Fact("reaches", ("rook",), 14.0, ("source and target share a row or a column == True",))
    worth = Fact("is worth at least", ("rook",), 14.0, (), (reaches,))
    takes = Fact("takes in", ("queen", "rook"), None, ("anything a rook may do",))
    return (Fact("is worth at least", ("queen",), 14.0, (), (takes, worth)), reaches, worth, takes)


def test_what_was_concluded_is_written_down_as_beliefs():
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain())

        named = {one.variable: one.value for one in kept}
        assert named["reaches rook"] == 14.0
        assert named["is worth at least queen"] == 14.0


def test_a_conclusion_points_at_the_conclusions_it_was_drawn_from():
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain())
        by_name = {one.variable: one for one in kept}
        queen = by_name["is worth at least queen"]
        rests_on = queen.evidence[0].source.rests_on

        assert set(rests_on) == {by_name["takes in queen rook"].id, by_name["is worth at least rook"].id}


def test_a_chain_is_written_down_premises_first():
    """Written out of order, a conclusion points at nothing and the chain is lost."""
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain()[:1])
        by_name = {one.variable: one for one in kept}

        assert "reaches rook" in by_name
        assert by_name["is worth at least rook"].evidence[0].source.rests_on == (by_name["reaches rook"].id,)


def test_what_was_written_down_can_be_read_back():
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)
        context = knowledge.ensure_context("chess")

        FactRecorder().record(knowledge, "chess", a_chain())
        held = knowledge.belief("reaches rook", context.id)

        assert held is not None
        assert held.value == 14.0
        assert held.evidence[0].source.mechanism == knowledge.mechanism_named("inference").id


def test_how_sure_it_is_is_the_caller_s_to_say():
    """Reasoning adds no doubt of its own; what the rules were worth is a question about the rules."""
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain(), certainty=0.6)

        assert all(one.certainty == 0.6 for one in kept)
