# Interfaces: training the codec to convey exactly the message

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

The codec is in `doc/interfaces/codec.md`. A local LLM can talk about chess fluently without holding a coherent
thought about it. This file is how an encoder and a decoder are trained until they carry exactly what they are
given, adding nothing and distorting nothing.

## The signal is the Prover

One training pass:
1. A message is taken from the teacher's knowledge base.
2. The encoder words it.
3. A decoder reads the words back, without seeing the message.
4. The Prover compares what was read with what was meant, as in `UnderstandingJudge`:
   - **extra:** what the readings hold that the message doesn't entail. This disqualifies the pass.
   - **distorted:** the intended parts the readings contradict, a certainty read back as another included. This
     disqualifies the pass.
   - **covered:** the needed parts the readings entail. Rewarded.

**Making up and distorting aren't weighed against covering.** They aren't costs to trade; they rule a draft out.

- **Leaving out is not a fault.** Examples and factoids beyond what is needed may go unsaid.
- **Certainty is checked like any other content.** A model once turned 99% certainty into wording that sounded
  almost unusual, and that is a distortion.

**Texts are never compared.** Round-trip text scores barely track quality, and judging oneself makes things worse
(Somers 2005; Moon, Cho & Park 2020; Huang et al. 2024). A sound outside verifier is what works (Stechly et al.
2024), and the Prover is one.

**What is needed, for chess:** everything sent to the teacher's encoder, until rhetoric exists to choose.

**How close a certainty must come back:**
- People word certainty with different nuances, so a decoder reads a certainty back as a range, that listener's own.
- The range must hold the certainty meant. 99% may come back as "almost surely" or "no doubt", never as a toss-up.
- Each listener's range is learned from its replies, like the rest of its model.

## Where the messages come from

From the teacher's own knowledge base, which already holds what it would teach:
- **The rules of chess**, as rule clauses.
- **Positions described in words.** Where each piece stands, as factoids about squares named in words.
- **Moves and their worth.** A move with the engine's judgement of it, as a teller's claim, never as truth.
- **Its own heuristics**, as rules, with how well grounded each is.

Stage directions vary across the same content, so the encoder learns to change how a thing is said without
changing what is said.

## Keeping the words natural

**An encoder and decoder trained together can agree on a private code.** It round-trips perfectly and is not
language: "tit for tat" through nonsense Portuguese came back perfectly. A student who reads such text learns
nothing a human could have taught it.

Two guards:
- **Anchor the encoder to its base model's language,** so the trained text stays close to what the untrained model
  would say.
- **Judge drafts with a decoder that wasn't trained alongside this encoder**, such as a frozen copy of the base
  model, as well as the one that was.

**Open:** the literature on language drift in emergent communication (Lee, Cho & Kiela 2019; Lazaridou, Potapenko &
Tieleman 2020) has not been read in this pass. It is to be read before the code step; it may change both guards.

## Where it runs

**Open, to research before the code step, with no dependency added now:**
- **Fine-tuning:** low-rank adaptation of a local open model, on cinamon's V100 or on the GX10.
- **Holding a decoder to a grammar** built from the knowledge base's predicates: llama.cpp grammars, Outlines or
  XGrammar.

## How it is seen

Every pass logs, under `data/log/`:
- the message;
- the draft;
- each reading with its probability;
- the covered, extra and distorted parts, in columns with a header.

The logs are the review loop.
