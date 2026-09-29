import tempfile
from pathlib import Path

from openmind.inference.model.fact import Fact
from openmind.inference.service.fact_recorder import FactRecorder
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base


def a_knowledge_base(directory):
    return create_knowledge_base("chess", Path(directory))


def a_chain():
    """A fact drawn from two others, kept with what it rests on, so the chain survives the run.

    The numbers here are a fixture and not a claim: what a thing is worth is never read off its rules."""
    reaches = Fact("reaches", ("narrow",), 14.0, ("source and target share a row or a column == True",))
    worth = Fact("at least", ("narrow",), 14.0, (), (reaches,))
    takes = Fact("takes in", ("wide", "narrow"), None, ("anything a rook may do",))
    return (Fact("at least", ("wide",), 14.0, (), (takes, worth)), reaches, worth, takes)


def test_what_was_concluded_is_written_down_as_beliefs():
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain())

        named = {one.variable: one.value for one in kept}
        assert named["reaches narrow"] == 14.0
        assert named["at least wide"] == 14.0


def test_a_conclusion_points_at_the_conclusions_it_was_drawn_from():
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain())
        by_name = {one.variable: one for one in kept}
        queen = by_name["at least wide"]
        rests_on = queen.evidence[0].source.rests_on

        assert set(rests_on) == {by_name["takes in wide narrow"].id, by_name["at least narrow"].id}


def test_a_chain_is_written_down_premises_first():
    """Written out of order, a conclusion points at nothing and the chain is lost."""
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain()[:1])
        by_name = {one.variable: one for one in kept}

        assert "reaches narrow" in by_name
        assert by_name["at least narrow"].evidence[0].source.rests_on == (by_name["reaches narrow"].id,)


def test_what_was_written_down_can_be_read_back():
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)
        context = knowledge.ensure_context("chess")

        FactRecorder().record(knowledge, "chess", a_chain())
        held = knowledge.belief("reaches narrow", context.id)

        assert held is not None
        assert held.value == 14.0
        assert held.evidence[0].source.mechanism == knowledge.mechanism_named("inference").id


def test_how_sure_it_is_is_the_caller_s_to_say():
    """Reasoning adds no doubt of its own; what the rules were worth is a question about the rules."""
    with tempfile.TemporaryDirectory() as directory:
        knowledge = a_knowledge_base(directory)

        kept = FactRecorder().record(knowledge, "chess", a_chain(), certainty=0.6)

        assert all(one.certainty == 0.6 for one in kept)
