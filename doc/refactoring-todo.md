# Refactoring to do

The rework of OMF into the design in `doc/architecture.md`. Each component goes through two stops: its interfaces are
approved, then its code is approved.

## Done

- [x] Explore the current code and map it to the new design
- [x] Plan the rework: component by component, what doesn't fit deleted
- [x] Theory of mind is emergent (KB + doxastic + agent models' heuristics); rhetoric stays as a built-in module
- [x] Settle `doc/open-questions.md` with Maxime (two points deferred: how a coding agent plans without tree search,
      how running modules exchange work)
- [x] Write the new design in `doc/architecture.md` — approved
- [x] Deletions: evaluation (not a thing of its own), z3 prover and formula models and mappers, rule explainer and
      ollama model, duplicate expression-sentence mapper, the entrypoints except play, solve and dashboard, the
      TheoryOfMind protocol (not a thing of its own); every module imports and every test collects (tests not run: few work as they should yet). The built-in games and their entrypoints stay,
      as examples and tutorials.

## To do

- [x] rule + world + knowledge: doxastic source (method + parameters), certainty, accuracy, precision; opinions kept apart
      from beliefs; tasks with drifting
      values and expected times; contexts as compartments in a hierarchy (parent sets child's goals and budget,
      knowledge crosses contexts only on purpose); Rule moved down to break the rbs cycle — interfaces approved (`doc/interfaces/rule-world-knowledge.md`)
- [x] rule + world + knowledge — code approved
- [x] epistemology (foundherentism: anchors in observations and fundamental rules, coherence between observations,
      models and rules, conflicts become tasks, no self-supported claims, deduction and induction, confidence from
      method and data) — interfaces approved (`doc/interfaces/epistemology.md`)
- [x] epistemology — code approved (with GUIDs and mechanisms in knowledge)
- [x] structure: data structures with their methods and actions, importable by game definitions; the grid first —
      interfaces approved (`doc/interfaces/structure.md`)
- [x] structure — code approved: the state is named data models everywhere; tic-tac-toe and sudoku declared with the
      grid. Noted: the tic-tac-toe inference test went from 4 to 9.7 minutes (profile when asked)
- [x] debug and logs (per-session verbosity, minimal or targeted; warnings: conflicting rules, observations a frozen
      rule can't explain; debugger: interrupts, stack, conditional breakpoints; the dashboard is its viewer) —
      interfaces approved (`doc/interfaces/debug.md`)
- [x] debug and logs — code approved
- [x] game (definition API, registry, runtime; frozen rules by default, rules or rulesets declared open; action durations and cooldowns) + csp + predictor — interfaces approved (`doc/interfaces/game.md`)
- [x] game + csp + predictor — code approved; `openmind-play tictactoe` and `openmind-solve sudoku` run. Rules live
      in rulesets, weighted per ruleset; an RBS is a model one stateless service runs; all players play at once, and
      the constraints say who can act. The tests of the code still to be reworked were deleted. Noted: a memory-guard
      test left a thread spinning, which had been slowing every suite; the suite now runs in 17 seconds
- [ ] Adapt OpenMindChess to the game API
- [x] heuristic + model (registry per task, measured precision and processing time) — interfaces approved
      (`doc/interfaces/heuristic-model.md`)
- [x] heuristic + model — code approved: models are records measured as they run; a heuristic reads a node whose
      features are extracted once and shared
- [ ] policy + optimizer + utility (a policy is tuned to a subset of the actions; the default policy; the CSP gives
      valid values; optimizers from cheap to costly: a random picker, listing candidates, proposing solutions as for
      NLP or coding agents; a policy value heuristic; a policy takes its move's utility; a move's utility is the
      expectation over outcomes, binned when continuous, by a binning model; several goals weighed together; soft goals
      valued by the judge's opinion) — interfaces approved (`doc/interfaces/policy.md`)
- [ ] policy + optimizer + utility — code approved
- [ ] agent_model + search (SDMCTS, PUCT prior only, hypotheses from KB beliefs, the minimizer as one opponent model among others,
      simultaneous + chance) — interfaces approved
- [ ] agent_model + search — code approved
- [ ] budget + agent loop (one loop per level of the context hierarchy, a parent delegating to a child; each level
      sees an abstraction of the state and fetches specifics on demand; planning model per level, SDMCTS only where
      it fits; perceive, process, plan, communicate, act; a trainable time management policy choosing the
      model and search size at each level: theory of mind, inferences, state value, action value, predictor, SDMCTS
      nodes) — interfaces approved
- [ ] budget + agent loop — code approved; tic-tac-toe, rock paper scissors and prisoner's dilemma played
- [ ] codec + rhetoric (any modality: text, image, sound, temperature, radar, …; detectors as decoders of a state
      (fork, pin, …), grounding emergent from decoders + epistemology; interpretations as a distribution;
      Meyer's tactics; balanced goals; validation gate; making the case for
      soft goals, such as a feature's quality-for-effort ratio) — interfaces approved
- [ ] codec + rhetoric — code approved
- [ ] Continuous next-best-task loop (tasks in the KB with drifting values and expected times, deferred tasks kept
      pending; ponder, self-play, review games, study literature, train, play; games started by a player take
      priority; roles: player, coach) — interfaces approved
- [ ] Continuous next-best-task loop — code approved
- [ ] Coach role: game analysis and feedback during and between games — interfaces approved
- [ ] Coach role — code approved
- [ ] dashboard to visualize and debug (tasks, time management choices, tactics and SDMCTS tree, beliefs and sources,
      models, games) — interfaces approved
- [ ] dashboard — code approved
- [ ] inference bootstrap (relaxed problems, guiding principles into heuristics; a trainable bootstrapper learning
      which bootstrapping pays off) — interfaces approved
- [ ] inference bootstrap — code approved
- [ ] training (self-play, game study; heuristics and predictor learned as DNN, decision tree, lookup table, …; the
      time management policy trained from outcomes) — interfaces approved
- [ ] training — code approved
- [ ] No-learning entrypoint for production (frozen models; no new rules, tactics or facts; no persisted beliefs)
- [ ] Tutorials, examples (the built-in games) and instructions for coding agents using OMF

## Games to implement

A wide variety of games, each eroding the simplicity of the ones before (imported from `doc/todo.md`). Tic-tac-toe
with its variants, sudoku, the repeated prisoner's dilemma and rock paper scissors already exist as rules.

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
