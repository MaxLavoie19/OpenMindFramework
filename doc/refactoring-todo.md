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

## To do

- [ ] Deletions: evaluation, z3 prover and formula mappers, rule explainer and ollama model, duplicate
      expression-sentence mapper, the if-chain game registration, the entrypoints except play and solve,
      the TheoryOfMind protocol; `pytest` green
- [ ] world + knowledge: doxastic source (method + parameters), certainty, accuracy, precision; opinions kept apart
      from beliefs; tasks with drifting
      values and expected times; contexts as compartments in a hierarchy (parent sets child's goals and budget,
      knowledge crosses contexts only on purpose); Rule moved down to break the rbs cycle — interfaces approved
- [ ] world + knowledge — code approved
- [ ] epistemology (foundherentism: anchors in observations and fundamental rules, coherence between observations,
      models and rules, conflicts become tasks, no self-supported claims, deduction and induction, confidence from
      method and data) — interfaces approved
- [ ] epistemology — code approved
- [ ] structure: data structures with their methods and actions, importable by game definitions; the grid first —
      interfaces approved
- [ ] structure — code approved; tic-tac-toe and sudoku declared with the grid
- [ ] debug and logs (per-session verbosity, minimal or targeted; warnings: conflicting rules, observations a frozen
      rule can't explain; debugger: interrupts, stack, conditional breakpoints; the dashboard is its viewer) —
      interfaces approved
- [ ] debug and logs — code approved
- [ ] game (definition API, registry, runtime; frozen rules by default, rules or rulesets declared open; action durations and cooldowns) + csp + predictor — interfaces approved
- [ ] game + csp + predictor — code approved; `openmind-play tictactoe` and `openmind-solve sudoku` run
- [ ] Adapt OpenMindChess to the game API
- [ ] heuristic + model (registry per task, measured precision and processing time) — interfaces approved
- [ ] heuristic + model — code approved
- [ ] tactic + optimizer + utility (default tactic; the CSP gives valid values; optimizers from cheap to costly: a random
      picker, listing candidates, proposing solutions as for NLP or coding agents; a tactic value heuristic; a tactic
      takes its move's utility; a move's utility is the expectation over outcomes, binned when continuous, by a binning
      model; several goals weighed together; soft goals valued by the judge's opinion) — interfaces approved
- [ ] tactic + optimizer + utility — code approved
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
- [ ] Tutorials, examples and instructions for coding agents using OMF
