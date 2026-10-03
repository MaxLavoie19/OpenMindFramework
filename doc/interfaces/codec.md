# Interfaces: the codec, natural language in and out

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

The codec is OMF's human-machine interface. An **encoder** puts what an agent means into words; a **decoder** turns
words back into what was meant. `doc/architecture.md` ("Encoders and decoders", "Rhetoric") gives the design in
prose; this file gives it a shape.

## The first job: a teacher teaches a student chess in words

- **The teacher** is an OMF agent from the chess project, with Stockfish as a teller.
- **The student** is an OMF agent with no chess module, no engine and no notation.
- **Everything the student learns of chess reaches it as natural language**, the board included. The teacher may
  describe the 8×8 board, the pieces, how they move and where they stand, in words. No JSON, XML, PGN or FEN.
- **The teacher is also the student's environment** (my reading, to confirm). It describes each position in words.
- **The student says its moves, and the two negotiate until they agree** on what was meant.
- **The student's play is the payoff.** Its games say whether it understood, as for everything else in OMF.
- **The second job is reading.** The student goes on learning chess from texts, and these too are free of
  notation: no form of non-natural language reaches the student in this test.

**The codec conveys exactly the message it is given.** An LLM can talk about chess fluently and still not hold a
coherent thought about it, so the codec's knowledge of chess is never trusted.
- **Nothing is made up.**
  - Given "all guys in my family are ginger", "I'm ginger", "my brother is ginger" and "my father is ginger", the
    codec may not say "my uncle is ginger". It wasn't provided, and there may be no uncle.
  - A rule is conveyed as a rule, and its instances are only the ones given.
- **Certainty is part of the message.** A factoid held at 99% must not come out sounding doubtful, or the reverse.
- **Leaving things out is allowed.** Not every example or factoid has to be said, only what is needed.
- **Pragmatic inference is allowed.** A listener may read part of a message from word use, but not distort it.

## Why it is shaped this way

The reasons come from a shallow literature pass of 2026-10-02 (nine claims checked, eight standing):
- **The symbolic side decides the content; the LLM only words it.**
  - Planning first and letting a neural model realize the plan cut omissions from 41 to 6 on 440 facts
    (Moryossef, Goldberg & Dagan 2019).
  - The same realizer kept every planned entity in only 66.7% of unseen inputs, so coverage is checked on every
    output, not trusted.
- **Meanings are compared, never texts.**
  - Round-trip text scores barely tracked quality: BLEU r = −0.04 (Somers 2005); per-sentence Kendall τ 0.12 to
    0.27 (Moon, Cho & Park 2020).
  - A meaningless text can come back perfectly: "tit for tat" through nonsense Portuguese.
  - The `Prover` is the sound outside verifier that works where self-checking fails (Stechly et al. 2024).
- **A decoder writes valid logic that is often wrong.**
  - On FOLIO, GPT-4 was 93.9% syntactically valid but reached the right answer 63.8% of the time (Han et al.).
  - The most common error is one symbol used with several arities (Olausson et al., LINC).
- **Several drafts are ranked by a listener, rather than one draft passed or blocked.** Human success rose from
  66% with one draft to 85% with a thousand, levelling off after a hundred (Andreas & Klein 2016).
- **Listeners are layered: generic, archetype, individual, each pulled toward its parent.** Regularizing toward the
  parent kept the loss on unseen contexts to 2% (Hawkins et al. 2020).
  - Archetypes written as a persona description come out as caricatures (Santurkar et al. 2023).
  - So an archetype is fitted from evidence, never prompted into being.
- **Stage directions are soft.**
  - A described target beats a label: squared error on the six-level CEFR vocabulary scale fell from 3.66 to 0.28
    (Malik et al. 2024).
  - Readability trades against faithfulness.
  - An accent written as spelling is rated worse by speakers of that dialect and carries stereotypes the model can't
    see, so an accent in text is word choice only.
- **A model checking its own family's output shares its errors** (about 60% agreement when both are wrong, Kim et al.
  2025). Maxime allows one model to play several roles; measured accuracy is what will show whether that hurts.

## Models

```python
@dataclass(frozen=True, slots=True)
class Factoid:
    """Information, true or not, presented as fact.

    A clause from the statement domain, which carries no epistemics: the same factoid is the same whether the
    speaker believes it, doubts it or is lying. What the listener makes of it is the listener's knowledge base's
    business, where it arrives as a told claim."""

    clause: Clause


@dataclass(frozen=True, slots=True)
class Question:
    """A goal clause asked of the listener."""

    goal: Clause


@dataclass(frozen=True, slots=True)
class Inference:
    """A conclusion with the steps that reached it, so the listener can follow and not only be told."""

    derivation: Derivation


@dataclass(frozen=True, slots=True)
class Rule:
    """A rule offered to the listener, as a clause with a head and a body."""

    clause: Clause


type Part = Factoid | Question | Inference | Rule


@dataclass(frozen=True, slots=True)
class StageDirection:
    """How a message is said, not what it says.

    Open-ended: language, tone, vocabulary level, register, persona, accent, emphasis, framing, a phrasing to keep.
    The description says the target in words, such as "short words, as for a child of eleven", because a bare
    label barely moves an LLM while a described target does.

    Directions are measured, not judged by an LLM's impression:
    - tone, by sentiment analysis;
    - persona and register, by the in-group markers they carry, such as dog whistles or liturgical dialogue.
      Those markers are themselves factoids in the knowledge base, about what makes someone part of a group.

    An accent in text is word choice and idiom only; how it sounds is for a voice encoder."""

    aspect: str
    value: str
    description: str


@dataclass(frozen=True, slots=True)
class Message:
    """What a speaker means to convey: the parts, and how to say them.

    Several parts travel together, because a message usually negotiates more than one question at once."""

    parts: tuple[Part, ...]
    directions: tuple[StageDirection, ...] = ()


@dataclass(frozen=True, slots=True)
class Rendering:
    """One draft of a message in words, and which encoder wrote it."""

    text: str
    message: Message
    encoder: str


@dataclass(frozen=True, slots=True)
class Interpretation:
    """One reading of a text, with how likely the decoder holds it.

    A decoder returns several. That is where ambiguity, puns and sarcasm come from, and why a text is never read as
    one meaning by default."""

    parts: tuple[Part, ...]
    directions: tuple[StageDirection, ...]
    probability: float


@dataclass(frozen=True, slots=True)
class Listener:
    """Who a text is for: generic, an archetype, or one individual.

    It names a holder in the knowledge base, so what a listener knows is read from the beliefs held by it: its
    vocabulary, the topics it knows and how well, what was said to it before. The parent is the level above, and a
    listener with little evidence of its own leans on it.

    An archetype is not written in. It splits off from its parent once there is enough data to train a model
    distinct from the parent's."""

    holder: tuple[str, ...]
    parent: Listener | None = None


@dataclass(frozen=True, slots=True)
class Understanding:
    """What a listener would take from a draft, judged against what was meant.

    Every judgement here is the Prover's, made on meanings: never a comparison of texts.
    - covered: the intended parts the readings entail;
    - extra: what the readings hold that the message doesn't entail, such as an instance of a rule about
      something the message never said exists;
    - distorted: intended parts the readings contradict. A certainty is read back as a range, that listener's own,
      and it is distorted when the range doesn't hold the certainty meant: 99% read as a toss-up."""

    rendering: Rendering
    listener: Listener
    interpretations: tuple[Interpretation, ...]
    covered: tuple[Part, ...]
    extra: tuple[Part, ...]
    distorted: tuple[Part, ...]
```

## Ports

```python
class Encoder(Protocol):
    """Puts a message into words for a listener.

    The content is decided before the encoder is called. It only words it, starting from the controlled English
    `ClauseTextMapper` gives a clause, under the stage directions. It returns several drafts because drafts are
    ranked, and the count comes from the budget."""

    @property
    def name(self) -> str: ...

    def encode(self, message: Message, listener: Listener, count: int) -> tuple[Rendering, ...]: ...


class Decoder(Protocol):
    """Reads a text as a listener would, as several interpretations with their probabilities.

    A decoder for a listener is conditioned on what that listener knows and nothing more. The generic decoder is
    the root listener's."""

    @property
    def name(self) -> str: ...

    def decode(self, text: str, listener: Listener) -> tuple[Interpretation, ...]: ...
```

**Every encoder and decoder is a `Mechanism`** in the knowledge base. What a decoder produces enters as evidence
whose source is that decoder, and its accuracy is measured against what really happened next: whether the listener
answered, played or acted as the reading predicted. A decoder fitted to an archetype earns trust or loses it.

**Backends are local.** One model may play several roles, or one model per role and per listener; each pairing is
its own mechanism, so which is better is measured rather than chosen.

## Services

```python
class UnderstandingJudge:
    """Decodes a draft as one listener would, and says what got across.

    The decoder never sees the intended message or the stage directions; it reads only the text, as the listener
    will. Comparing the readings with the message is the Prover's job."""

    def judge(self, rendering: Rendering, listener: Listener) -> Understanding: ...


class ValidationGate:
    """Chooses the draft to send, or none.

    Each draft is judged by every listener the message reaches.

    For now the gate has one job: no made-up factoids get through.
    - A draft is not sent if any of its readings, however unlikely, finds something extra or distorted in it.
      Drawing a line at "unlikely enough" would need a constant.
    - The drafts that remain are ranked by how much of what is needed they cover.
    - If none remains, nothing is sent.

    The rhetorical framework comes later, and with it the weighing of goals."""

    def choose(self, message: Message, listeners: Sequence[Listener]) -> Rendering | None: ...
```

**The teacher's gate reads through its model of the student.**
- The listener it judges with is the student as the teacher believes the student to be: the beliefs the teacher
  holds about the student.
- It never reads the student's own knowledge base.
- Before the teacher has evidence about this student, that model is the generic listener.

**What the student hears enters as told, never as truth.** A decoded factoid or rule is a claim from the teacher, in
the student's knowledge base, weighed by the teacher's measured reliability.

## The student assembles concepts; it invents none

**The vocabulary chess is described in is already OMF's.** The student needs no novel concept. It needs to assemble
the concepts it has into a meaningful model of chess, and that assembling is the concept invention to build.

So the student's decoder is held to a grammar made from OMF's vocabulary, its predicates and their arities, like
every decoder. That removes the commonest decoding error, one symbol used with several arities, by construction.
What it hears is mapped onto concepts it has.

What each kind of knowledge asks of the codec:
- **Pieces** are described in natural language and assembled from the vocabulary.
- **Constraints** convert easily into natural language and back.
- **Move generation is the new challenge.** A piece's moves are generated by code, so understanding "a knight
  jumps two squares one way and one the other" means writing that code from the words. **Open**, see
  `doc/open-questions.md` 49.

## What this replaces

`language/service/text_encoder.py`'s `TextEncoder.encode(Meaning) -> str | None` is the only natural-language port
today. It takes a JSON meaning rather than clauses, and nothing calls it. `Encoder` replaces it.
`inference/mapper/clause_text_mapper.py` stays as the controlled English an encoder starts from.

## Training

How the encoders and decoders are trained to convey exactly the message is in `doc/interfaces/codec-training.md`.
