import pytest

from openmind.language.service.structured_text_decoder import StructuredTextDecoder
from openmind.language.service.structured_text_encoder import StructuredTextEncoder
from openmind.rhetoric.model.position import Position
from openmind.rhetoric.model.tell import Tell

FISH = Tell(
    "joe",
    "disgust",
    (Position("do I want his fish", -1.0, 0.2), Position("is his fish disgusting", 1.0, 0.9)),
)


def test_the_encoder_says_the_positions_and_strips_the_tone_addressee_and_importance() -> None:
    assert StructuredTextEncoder().encode(FISH) == (
        "[{question: do I want his fish, answer: no}, {question: is his fish disgusting, answer: yes}]"
    )


def test_the_decoder_hears_the_positions_the_encoder_said() -> None:
    text = StructuredTextEncoder().encode(FISH)

    assert StructuredTextDecoder().decode(text) == (
        Position("do I want his fish", -1.0),
        Position("is his fish disgusting", 1.0),
    )


def test_other_answers_are_numbers_and_special_characters_are_escaped() -> None:
    tricky = Tell("ann", "calm", (Position("is 3:2 {or} [more], really?", 0.25),))
    encoder, decoder = StructuredTextEncoder(), StructuredTextDecoder()

    text = encoder.encode(tricky)

    assert text == "[{question: is 3\\:2 \\{or\\} \\[more\\]\\, really?, answer: 0.25}]"
    assert decoder.decode(text) == (Position("is 3:2 {or} [more], really?", 0.25),)
    assert decoder.decode(encoder.encode(Tell("ann", "calm", ()))) == ()


@pytest.mark.parametrize(
    "text",
    ["Your fish is disgusting", "[{question: is it}]", "[{answer: no}]", "[question: x, answer: y]", "[{question: x, answer: maybe}]"],
)
def test_text_that_isn_t_structured_language_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        StructuredTextDecoder().decode(text)
