# To do

## Next, in order

The knowledge base's core is built (`doxastic/README.md`): records kept word for word with their provenance, looked up
by subject, name, keyword, claim, teller or source, and claims weighed with the evidence for them and against them kept
apart. Nothing feeds it yet. In order:

1. **The training's own evidence:** a signal's agreements and disagreements as `counted` records, a deduction's proofs
   as `proved`, a relaxed domain's as `relaxed`. `SignalRecord`'s reliability then comes out of the knowledge base.
2. **What was seen and what was played:** what a player sees, from `StateObserver`, as `seen`; what self-play played and
   what came of it, as `played`.
3. **What was told:** claims that can be lies, in the hidden-information games below and in rhetoric, as `told`, with
   `EvidenceWeigher` discounting a teller by how reliable they have been — where this meets ethos.
4. **Another store behind `RecordStore`,** SQLite or otherwise, once the JSON lines are too slow to load.

## Games that erode the simplicity of the games so far

So far OpenMind has dealt with discrete-time, deterministic, complete-information, zero-sum games. Each game below
erodes that simplicity a bit, in this order, toward the rhetorical agent.

| Game | What it erodes | Why it matters |
|---|---|---|
| Repeated prisoner's dilemma | non-zero-sum; simultaneous choices, hidden until both are made | cooperation and betrayal without a rule forbidding either |
| Centipede game | non-zero-sum, in turns | perfect play grabs at once, yet passing makes both gain |
| Trust game | non-zero-sum, in turns | investing in someone who can betray |
| Alternating-offers bargaining (Rubinstein) | non-zero-sum, a shrinking pie | negotiation at its simplest |
| Stag hunt, chicken, battle of the sexes | simultaneous choices | coordination, escalation or backing down, conflicting preferences |
| Public goods game | several players, simultaneous choices | profiting from others' contributions |
| Kuhn poker | hidden cards, chance | the smallest game where bluffing is part of perfect play |
| Liar's dice | hidden dice, claims that can be false and challenged | lies beaten by a better player, not by a rule |
| Cheat (Bullshit, I Doubt It) | hidden cards, face-down plays whose claimed rank can be a lie, challenges | predicting lies from what players have shown; the test bed of semi-determinized MCTS (Bitan and Kraus, 2017, arXiv:1709.09451) |
| Kriegspiel | chess with the opponent's pieces hidden | hidden information in chess |
| Lewis signaling game | a sender and a receiver, meanings not given | the encoder and decoder as a game |
| Hanabi | cooperative, hidden cards, limited hints | what you say matters; reading intent |
| Werewolf, The Resistance (Avalon) | hidden roles, many players | finding liars through what people say |
| Diplomacy | seven players, negotiation, promises and betrayal | the closest to rhetoric |

## Relaxed domains

"Sometimes fewer constraints lead to good insights." At the end of a round, the agent takes a few positions, relaxes the
rules and toys with them: free to teleport its pieces, it might learn checkmate geometry, how to cast a mate net, and
how not to compromise its pieces. Relaxations are hypotheses like signals: what they teach is kept only when it helps
in real games.

| Relaxation | In chess | In tic-tac-toe |
|---|---|---|
| Drop an explicit constraint rule | only "no payoff set": play on after the game ends | "the cell is empty": marks can be overwritten |
| Widen a variable's domain | any from–to square instead of the legal moves: teleport, since python-chess's `push` doesn't check legality and checkmate is still detected after | any cell |
| Change turn order | the other side passes, or one side plays two or three moves in a row | the same |
| Edit the position | fewer pieces, pieces moved (the "what if" views) | fewer marks |

How it would run: pick positions (those where signals disagreed most with the coming winner), relax the domain, deduce
and walk back inside it within a budget, turn proofs into seeds (`DeductionInducer`) and relaxed goal distances into
candidate signals, keep only what fits real games, and choose relaxations like arms, by how often their insights
survive. First step: a `Relaxation` model and a relaxed-domain builder for each kind above.
