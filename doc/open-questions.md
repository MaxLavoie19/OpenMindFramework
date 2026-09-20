# Open questions

These are gaps, contradictions and unknowns in `doc/architecture.md`. Each one waits for Maxime's decision. None is
settled until it is struck from this file and the design says so.

## A. Parts of the design written by Claude, not by Maxime

Claude filled these blanks in `doc/architecture.md` without being asked. For each one, the options are:
- keep it as written;
- change it;
- strike it and leave the point open.

1. **Default tactic.** When a game has no tactics, the doc makes "any legal action" the default.
   - **Decided (Maxime):**
     - The default tactic is an unnamed tactic whose sub-goal must be populated. At worst, it plays randomly.
     - An agent with free time deduces that it needs to prepare a tactic. It analyses the rules, facts and so on to
       emit credible heuristics.
2. **Accuracy and precision.**
   - The doc defines accuracy as how often beliefs from the same source and method turned out right.
   - It defines precision as how narrow a belief is, such as the spread of an estimate.
   - **Decided (Maxime):** keep.
3. **Certainty.** The doc defines certainty as "how strongly it is held".
   - **Decided (Maxime):** certainty is a degree of how strongly a belief is held. It should depend on evidence, but a
     belief might not have a known, proper epistemic justification.
4. **Goal weights are beliefs.** The Utility section calls an agent's own goal weights beliefs. They might instead be
   opinions, preferences, or something else.
   - **Decided (Maxime):** they are preferences.
5. **Time management levels.** The doc adds two levels you didn't name, the tactic value heuristic and binning, and it
   puts the choice of the next task at the policy's top level.
   - **Decided (Maxime):** keep all.
6. **The bootstrap time management policy.** The doc says it starts from a simple bootstrap policy, today's
   `MovePlanner`. What that policy is remains unsaid.
   - **Decided (Maxime):** keep: start from `MovePlanner`, generalized.
7. **CSP and continuous actions.** The doc says the CSP gives valid ranges for continuous actions and checks the
   proposals of generative optimizers.
   - **Decided (Maxime):** keep both.
8. **Hierarchy.**
   - The doc says knowledge crosses contexts only on purpose, through inheritance, trying a model elsewhere, or testing
     a principle in several games.
   - It says every level runs the same machinery.
   - It says the next-best-task choice runs at the top.
   - It says a child reports its outcome to its parent.
   - **Decided (Maxime):**
     - Keep all four.
     - The goal of a sub-task is provided by the parent, such as coaching rather than winning.
9. **State abstraction.** The doc says abstractions are made by models: decoders, summaries, relaxations.
   - **Decided (Maxime):** keep.
10. **Planning in a coding agent.** The doc lists the alternatives to tree search as: pick a tactic by its value, follow
    a plan, propose a plan.
    - **Decided (Maxime):** struck. How a coding agent plans without tree search stays open.
11. **Soft goals.** The doc says the judge's opinion updates the agent's beliefs about that judge's opinions.
    - **Decided (Maxime):**
      - An agent can edit its own opinions, which are its direct experience. It can't edit someone else's.
      - It can form and update beliefs about someone else's opinion.
12. **Epistemology.** These points came from Claude:
    - the crossword analogy;
    - anchors can be revised;
    - conflicts become tasks;
    - the independence of supports matters;
    - the three examples of confidence;
    - confidence is re-evaluated when a method's accuracy is re-measured.
    - **Decided (Maxime):** keep all.
      - Revisable anchors are limited by B2: frozen rules are never revised.
      - Only misperceptions, deduced rules and open rules can be revised.
13. **Search and hidden information.** The doc weighs each hypothesis by the knowledge base's belief in it.
    - **Decided (Maxime):** keep.
14. **The package map.** The packages, their names, their order and their boundaries are all Claude's.
    - **Decided (Maxime):** it is provisional. Each package's boundaries are confirmed at its own interfaces stop.

## B. Contradictions

1. **Expected utility vs fuzzy distance.**
   - **Decided (Maxime):**
     - Values are not numbers as such. They are gauged in words, like temperature (freezing … burning).
     - Each word's meaning is learned per agent or in general, and depends on context and audience.
     - Whether another agent's valuation was guessed right is a negotiation: a neglected factor, or an over-valued or
       under-valued one.
   - How the search operates on words was decided in B5.

   The original question:
   - The design computes a move's utility as a sum of value × probability, which is arithmetic.
   - Rhetoric's distance, by your earlier point, is not arithmetic: one insult doesn't cancel two compliments, so it
     needs fuzzy logic.
   - Soft goals are opinions with qualifiers such as "good" or "expensive".
   - How do fuzzy or qualitative values enter an expected utility?
   - Options:
     - Map every qualifier to a number, learned per holder. This is simple, but it reintroduces arithmetic.
     - Keep expected utility for numeric goals only. Compare fuzzy goals with fuzzy operators, then combine both with a
       fuzzy aggregation. This is coherent with fuzzy logic, but there are two mechanisms.
     - Make utility itself fuzzy everywhere, so bins carry fuzzy values. There is one mechanism, but it departs from
       AlphaZero-style numeric backups in the search.
     - Something else.
2. **Anchors that can be wrong.**
   - **Decided (Maxime):**
     - Rules hardcoded by an application are frozen by default.
     - Deduced rules are edited freely when a model that better explains the data is found.
     - A rule or a ruleset can be declared open for modification.
     - Conflicts with a frozen rule become warnings in a debug and logs module, such as a missing en passant.

   The original question:
   - Fundamental rules are anchors.
   - A game's definition, given by the programmer, is treated as truth by the CSP and the predictor.
   - Can coherence ever override a given rule, for example after observing a move the rules call illegal? Options:
     - Given rules are never revised. Observations that contradict them are flagged.
     - Given rules are revisable like any anchor, with very high certainty.
     - It depends on the source: rules from the programmer are fixed, and rules decoded from literature are revisable.
3. **The zero-sum assumption vs agent models.**
   - In a zero-sum game, the opponent may be assumed to minimize OMF's utility.
   - But agent models describe opponents with limited skill and their own preferences.
   - **Decided (Maxime):**
     - Conflicting models are the norm.
     - The minimizer suffers from projection: it plays what the agent itself would play. That fits Stockfish, not a
       low-level agent.
     - The model that better predicts an agent's actions and leads to better results is preferred.
     - Every model is selected on accuracy, cost and, if necessary, explainability.
   - Which wins, and when? Options:
     - Assume minimization when no agent model exists, and use the model otherwise.
     - Blend both, weighted by confidence in the model.
     - Let the time management policy choose, as another model of the opponent.
4. **Payoff vs utility vs rhetorical gain.**
   - Earlier (2026-09-14) you said a player's gain is progress toward the distance they want. "Winning makes him above."
   - The design defines payoff as what the game gives, and utility as the agent's own valuation over goals.
   - Are payoffs rhetorical gains, a special case of utility, or a separate thing?
   - **Decided (Maxime):**
     - Both hold, at different levels.
     - Inside a game's context, the payoff is what counts.
     - At the parent level, the result becomes rhetorical gain: progress toward the wanted distance, such as "winning
       makes him above".

5. **Words in the search.**
   - SDMCTS backs up and averages values, and utility is "value × probability, summed".
   - Both are defined on numbers. With values gauged in words, what is:
     - the average of "warm" and "freezing"?
     - "hot" at a 30 % chance?
   - **Decided (Maxime):**
     - Words are fuzzy sets over a hidden scale, and the search computes with fuzzy arithmetic.
     - Rejecting arithmetic for distance was a mistake. It is arithmetic, but fuzzy: best-effort estimates from
       incomplete information, varying by agent, time and context.

## C. Unknowns to investigate

1. **Where the hierarchy lives.** Contexts are in `knowledge`, but no package owns levels, delegation, or the
   parent–child exchange of goals, budget and outcome.
   - **Decided (Maxime):**
     - Contexts stay in `knowledge`.
     - `agent` runs one loop per level and owns delegation: goals and budget go down, outcomes come back up.
2. **Opinions held by the agent itself.** Can OMF hold opinions, such as "this move is ugly", or only beliefs about the
   opinions of others?
   - **Answered in A11:** yes, an agent holds and edits its own opinions.
3. **How a tactic is defined.**
   - What is a tactic in data? A rule, a model, or a goal given to the optimizer?
   - Who defines one? The programmer, inference, or both?
   - Known so far (A1): a tactic has a sub-goal, and an agent can prepare tactics itself from the rules and facts.
   - **Decided (Maxime):**
     - A tactic has a move generator and an evaluator, both consistent with its goal.
     - Example: "flee" ignores attacking moves and weighs directions by the distance each one puts from the enemy.
4. **Rhetorical tactics.**
   - The permutation of four ethos and pathos terms with six operations gives many tactics.
   - Which pairs of terms are meaningful? For example, is effective ethos against projective ethos affirming or hiding
     identity?
   - **Decided (Maxime):** all 36 are pre-defined (6 pairs × 6 operations). Each tactic learns to perform its specific
     job.
5. **What "literature" is.** Imported games only, or also texts decoded into rules?
   - **Decided (Maxime):** literature is:
     - imported games;
     - texts decoded into rules;
     - the output of other models, such as engine evaluations.
   - Annotated games were offered and not chosen.
6. **The budget's unit when levels overlap.** A parent and a child both run in real time. Does the child's budget come
   out of the parent's, or run in parallel?
   - **Decided (Maxime):** the parent decides: either carved out of its own budget or run alongside, as part of what it
     gives the child.
7. **Task value.** What is a task's value measured in: the utility gained, the precision gained, or the time saved?
   - **Decided (Maxime):** several measures: utility gained, precision gained and time saved, weighed together by
     preferences.
8. **Debug vs dashboard vs logs.**
   - There is now a debug and logs module, a dashboard "to visualize and debug", and the existing convention that
     services log their decisions.
   - Where are the boundaries between them?
   - **Decided (Maxime):**
     - `debug` owns logging and warnings, and the dashboard is its viewer.
     - Log level depends on the session: minimal or targeted verbosity.
     - It has debugger support: interrupts, a stack, conditional breakpoints (for example, on evaluating a position that
       allows en passant).
9. **The debugger's stack and the real-time clock.**
   - What does "stack" mean in OMF: the Python call stack, OMF's reasoning chain, or both?
   - What happens to clocks and budgets while execution is paused at a breakpoint?
   - **Decided (Maxime):**
     - The stack is both the Python call stack and OMF's reasoning stack, linked.
     - What time does during a pause is decided by the debug session.
10. **Measurements taken across a pause.**
    - When the session lets time run during a pause, processing times and game results measured across the pause are
      distorted.
    - Are they kept for training the time management policy and the models' measured processing times?
    - **Decided (Maxime):** the debug session decides.
11. **"Tactic" means two things.**
    - In OMF, a tactic is a general direction that guides the optimizer (charge, kite, retreat; testing, debugging).
    - In chess, "tactics" are patterns such as a fork, a pin, a skewer or an x-ray, which detectors recognize.
    - The two will meet in the same code and logs.
    - **Decided (Maxime):** they are different idioms. Nothing is renamed.
12. **Detectors: where they come from and where they live.**
    - Are they written by the programmer, inferred, learned, or all three?
    - Are they a package of their own?
    - Are they a model family, with accuracy, cost and explainability like the others?
    - **Decided (Maxime):**
      - Detectors are decoders.
      - Grounding emerges from decoders plus epistemology, with no grounding module.
      - The goal is to prevent hallucinated analyses, such as ones that invent pieces or ignore the move played.
13. **Interconnecting modules.**
    - Rhetoric often needs notions from epistemology, and other modules will need each other in the same way.
    - How do modules connect?
    - **Decided (Maxime):**
      - The question is how running modules exchange work, not how code imports code.
      - Rhetoric always depends on epistemology. It is the only way to prevent hallucinations.
    - **Decided (Maxime):**
      - The idea of forbidding imports between modules is dropped.
      - Modules should be able to be populated and retrieved.
      - The details are worked out as we go, and no constraint is added until one is needed.
    - **Deferred:** how running modules exchange work (direct calls, a message bus, tasks).
14. **Documentation.**
    - Maxime: OMF needs many `README.md` files, instructions for coding agents, tutorials and examples.
    - Open:
      - Are they written with each component, or as a step of their own?
      - Who are coding-agent instructions for: agents working on OMF, agents writing games with OMF, or both?
    - **Decided (Maxime):**
      - READMEs are written with each component.
      - Tutorials and examples get a step of their own, once enough works end to end.
      - Coding-agent instructions are for agents using OMF: writing games and applications with it.
15. **What "no learning" freezes.**
    - Maxime: a no-learning entrypoint runs trained models frozen, for production.
    - **Decided (Maxime):** by default it freezes:
      - trained models;
      - rules, tactics and facts: no new ones;
      - beliefs: none are persisted. Those a task needs are kept in the task's instance context.
16. **Simulation and the predictor.**
    - Maxime: the integrator runs the game; OMF only simulates it. A context holds rulesets, one per purpose (the
      simulation, the position heuristic, the move heuristic, one per known agent's play style). A ruleset is one model
      of its task: any other model may replace it (a DNN, a lookup table, …; in chess's early game a lookup table is
      more performant, accurate and explainable), or it may rationalize another model's decision.
    - **Decided (Maxime):**
      - The simulation is how the SDMCTS expands: the CSP provides legal actions, the heuristics choose between them,
        the predictor provides the possible outcomes from the state and the set of actions.
      - A ruleset lists the ids of its rules; a rule can belong to several rulesets (e.g. similar rulesets per tactic).
      - Rules and rulesets can each be open or frozen. An integrator might freeze the chess simulation ruleset. A
        heuristic can be modified at any time and modifies copies of frozen rules.
      - A ruleset record: id, name, context, task, tags, open, rule ids; it may grow (its role, …).
      - At a high level an agent chooses between playing the best move, offering a draw, abandoning, or planning
        (running the simulation to explore viable options).
      - A heuristic changes copies of a frozen ruleset, or has an open ruleset of its own. A frozen ruleset has only
        frozen rules.
      - The game's context declares who acts: one player at a time or simultaneously, and in complex game states (Magic:
        The Gathering's phases) who may intervene and how. When it's black's turn, white can only plan.
      - Payoffs come from the predictor: an end state declares a payoff per player.
      - A game declares the rules for offering a draw and abandoning.
      - A rule's weight lives in the relation between a ruleset and the rule: a rule weighs x in a given ruleset.
      - Maxime first proposed a weight (a, b), contributing a·x^b. For now weights are linear (a·x); some RBS would do
        better with non-linear weights, left for later (and what a·x^b is for negative x and fractional b with it).
      - Every ruleset–rule relation carries a weight, even where it isn't used yet.
      - A ruleset belongs to a context, and a rule to the rulesets that list it.
      - Who acts in complex game states (Magic: The Gathering's phases): the constraints determine the legal actions
        for each player in each phase. The players allowed to act are a variable part of the state (turn = black; turn
        = Bob, phase = defender declaration lets Alice assign defenders).
      - All players play at the same time, all the time; in a game played in turns, a player has no options outside
        their turn. Turn structure doesn't belong in OMF: `Players.to_act` goes.
      - OMF recognizes the actions that offer a draw or abandon by the effect of performing them.
      - The producer of fitted heuristic terms links them into their ruleset.
    - Open, found while coding the game step:
      - A rule listed in two rulesets of the same context with different weights.
        - **Decided (Maxime):** services are stateless, instantiated once at build and injected where necessary. The
          RBS is a model, passed into a service that runs its rules, inference and so on.
        - **Decided (Maxime):** caches are injected (a cache built once and passed to the stateless services); one
          RBS per ruleset.
      - Running out of time: with no timeout rule, OMF's own referees (self-play, `openmind-play`) log the flag and
        let the game go on. What should they do? The game could declare it with its own clock variable, once durations
        are used.
      - `rule`'s services use the debugger, which uses `knowledge`, which uses `rule`'s models: a cycle between
        packages.
      - How a rationalizing ruleset is measured: by how often it agrees with the model it explains. Deferred to the
        training step.

17. **The heuristic and model step.**
    - **Decided (Maxime):**
      - The time management policy picks a *set* of models solving each step — outcome prediction, move value per
        player, state value per player, … — from the time available, the models available and what each one trades.
      - Every piece of information OMF produces is stored in the knowledge base, for introspection: a model's measured
        accuracy and processing time among them.
      - A model has a record of its own: an id, a name, its task, its context and where its data lives; the data itself
        lives in `data/` or the integrator's storage.
      - Every task is named now, with a port of its own.
      - Every reading is timed, and the timings build each model's processing-time profile, which a specialized build
        (a competitive chess solver with its parts built in) is later generated from.
      - A heuristic is given a node — the state, its game and its extracted features — not a bare state. The node
        extracts a feature on demand, a possibly memoized call, and shares it with every model after. The search builds
        nodes for its tree; anything else valuing a position, training included, builds one too.
      - A win always carries the payoff value, whatever the heuristic.

18. **The policy and optimizer step.**
    - **Decided (Maxime):**
      - "Tactic" was the wrong word for it: it is a **policy**, in the spirit of proximal policy optimization. The
        actions are too many to weigh one by one, so the agent first picks a policy tuned to a subset of them —
        nonsensical, fleeing, attacking — and then picks the action that serves it best. Chess's own tactics (a fork,
        a pin) keep the word; rhetoric's tactics are policies too, each narrowing the messages to those negotiating
        one distance, on one axis, in one direction. A message carries several policies at once, since several
        questions are usually negotiated together.
      - Values are quantitative in some domains and qualitative in others: centipawns in chess, "sounds selfish" in
        rhetoric. The words are not a stage every value passes through.
      - Preferences live in the knowledge base, so that the agent can know and understand its own choices.
      - All three optimizers. The random picker is a policy of its own, the one that saves time when time is short;
        with time to spare, an agent follows several complementary policies and learns when each one pays.
      - A policy is a combination of specialised heuristics: a position value heuristic valuing the states it is
        after, and a move value heuristic valuing the moves that serve it. A fleeing policy values safe states and
        moves that put distance between the soldier and the enemy.
      - The time management policy keeps its name.
      - A policy is manufactured (an agent is told to learn freeze, fight and flight) or learned by policy
        optimization.
      - A policy isn't necessarily one model: it performs two tasks, which one network with two heads or two rulesets
        can perform.
      - A policy picker is a port of its own: it says which policies are worth expanding and which aren't worth
        considering. Random is rarely worth expanding; when it is, nothing else is worth considering.
      - One model record can name several tasks, such as a network with a head for each of a policy's two.
      - Control systems belong here: multi-input multi-output systems, model predictive control and model reference
        adaptive control are optimizer models. An optimizer gives the next best step from the current position; model
        predictive control's plan is the series of steps the optimizers give as the search explores, and an adaptive
        one keeps what it adapted as its own data.
      - An optimizer can be a special kind of CSP: equations computing the perfect value, a PID holding a variable on
        target, or a constraint solver working the values out one at a time. What an optimizer gives is checked
        against the constraints unless it is the solver itself.
      - The optimizer is an alternative to expanding: rather than listing every action and sorting them, it produces
        the one action that serves the goal, as an artillery piece calculates its firing solution instead of
        enumerating every ballistic one.

19. **The agent model and search step.**
    - **Decided (Maxime):**
      - Planning gives a strategy, not a plan: a mixed strategy, a move distribution per state — in state A play these
        actions at these probabilities, in state B those. A plan is a series of actions and states, while a strategy
        covers the responses and the paths that invalidate it. A strategy covers what is likely to be met — deep in
        the main line, and the unlikely result of the action being taken — but not implausible states, unless
        strategizing is so far ahead that those are what is left to improve. A fast model can play pre-moves from the
        strategy while the planner keeps strategizing. A thread of its own dispatches a strategy's actions —
        several synchronous ones at once, such as motors moving into position, and asynchronous ones such as API calls
        — and the state is kept current for both the actor and the planner, so the actor answers what is happening and
        the planner prunes what can no longer be reached. A plan can be read off a strategy; a surprise means strategizing
        again from there.
      - The existing MCTS is worth neither keeping nor editing; a MCTS and a SDMCTS are implemented later.
      - Planning is a task and a search is one model of it: OMF enforces none. Minimax suits tic-tac-toe, MCTS chess,
        SDMCTS poker or Stratego, and a conversation is improvised rather than planned. `mcts` goes, and the searches
        become models of the planning task.
      - An agent model is whatever the modelling calls for — a chess player's biases, a conversational partner's topics
        and how well they know them, a network such as Maia — and OMF doesn't enforce how it is represented. The same
        goes for hypotheses about hidden information: a hidden Markov model and a ruleset express them differently.
      - The caller passes the agent model: a named opponent's where there is one, a generic one otherwise.

20. **The budget and agent loop step.**
    - **Decided (Maxime):**
      - The integrator pushes observations: OMF holds the current state, the decoders turn what arrives into models,
        and the actor and the planner read the same one.
      - The planner keeps its tree between moves: strategizing carries on, and what can no longer be reached is
        pruned.
      - The time management policy decides both the set of models each step runs with and how much to explore,
        starting from a simple bootstrap policy and trainable later.
      - Dispatching actions is a port an integrator can fill — a robot's control loop, a game client's connection —
        with OMF's own thread as the default. It waits where the strategy has nothing prepared for the state it is
        in, which is the surprise case the planner is already strategizing from.
      - The actor thread opens no debugger frames yet: the planner stays debuggable and acting shows in the logs, until
        the dashboard can show threads.
      - `openmind-play` becomes the integrator: it pushes what the human played, dispatches what OMF chose, and prints.
        It is the worked example of integrating OMF.
      - Planning makes no sense where guesses can't be educated (no time, no model, no guiding principle): depth is
        wasted on random moves, and breadth might stumble on a win. The time management policy chooses on that.
      - The trailing player should gamble, the leading one shouldn't ("Optimal strategy in Guess Who?: beyond binary
        search"); in chess it also buys time to think on the opponent's turn. Where the imbalance belongs is the
        utility of a mixed strategy, judging the whole distribution rather than its mean: a high-risk line can have a
        clearly low expected utility and still be the likeliest way to win. Maxime's intuition, not yet analysed; a
        risk preference and a bold policy would both work too.

21. **The context hierarchy step.**
    - **Decided (Maxime):**
      - A parent delegates a goal and a budget, and the child runs its own agent loop in its own context with its own
        models, reporting back what came of it. It is a call, not a task queue.
      - The parent decides per delegation whether the child's seconds are carved out of its own or run alongside.
        Noted and not dug further: a child running alongside also takes a core the parent could have used, and nothing
        weighs cores.
      - A level sees an abstraction of the state, made by a model of the `abstraction` task: every level abstracts from
        what the world holds, each in its own way, and a level with no abstraction model sees the state as it is.
      - Delegating is an action of the parent's game, declared with its constraints, duration and cooldown, and
        dispatched into the hierarchy. A parent could play the child's level out itself, but not well: a level is
        solved by the models that suit it — minimax for tic-tac-toe, MCTS for chess, SDMCTS for poker — and those live
        in the child's context.
      - Fetching specifics on demand is an action too: a level holding a file's path reads the file as a move of its
        own game, whose effect puts the content in that level's state, so the abstractor stays stateless.
      - A child keeps its parent informed after every step, not only when it is done: it hands up where its level
        stands and the parent abstracts that into its own state. A coach delegating "move toward interesting
        positions" hears of each move as it is played, in time to comment on what the student chose or warn them to
        pay attention. What lands in the parent is its own state, never the child's game state applied to it: OMF
        runs no games.
