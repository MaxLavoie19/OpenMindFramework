# Interfaces: the marker wall, a problem to test the solver with

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

Depends on [circuit-constraint.md](circuit-constraint.md) and [search-trail.md](search-trail.md). Nothing in the
framework is specific to this problem, and nothing about this problem is declared inside it.

## The problem

Maxime staples colored markers to a wall and wants to know, from the colors of the few markers in view, which
stretch of wall he is looking at. Seven distinct colors. A marker every metre. The markers may be in one row or
in several stacked rows, and the question is how many metres can be told apart for a given number of markers in
view.

Settled: spacing one metre, so positions and metres are the same number; the markers are always read in a known
direction; the sweep is one to three rows against windows one to four columns wide.

## The counting answer, which needs no solver

A window of `m` markers can show at most `7**m` distinct color patterns, so `7**m` bounds the positions that can
be told apart. The bound is attained, because a de Bruijn sequence of order `m` over seven symbols contains every
pattern exactly once, and one exists for every alphabet and every order — the de Bruijn graph is connected and
every node has equal in-degree and out-degree, so it has an Eulerian circuit.

With `R` rows and a window `k` columns wide, a column of `R` markers is one symbol from an alphabet of `Q = 7**R`:

```
    positions            N = Q**k = 7**(R*k)
    metres               N, at a marker every metre
    columns to staple    M = N + k - 1           (a wall with two ends)
    markers to staple    R * M
```

**Coverage depends only on `R*k`, the total markers in view.** So extra rows do not lower the stapling per metre
— they cost `R` markers per metre — they lower how wide a field of view is needed for the same reach. Four
markers in one row and two rows of two both reach 2 401 m; the second needs half the wall in frame and twice the
markers.

## The encoding

Nodes are the `N` window patterns; there is an edge from `p` to `q` exactly when `p`'s last `k-1` columns are
`q`'s first `k-1`. Every node has out-degree and in-degree `Q`. A de Bruijn sequence is a Hamiltonian cycle
through that graph, which is `circuit` and nothing else.

**The numbering is the declarer's**, as the positional convention requires. Most significant column first, so
the edge relation is a shift:

```
    a color                 0 .. 6
    a column of R markers   c = sum over rows of  color_r * 7**r          0 .. Q-1
    a window of k columns   p = sum over columns of c_i * Q**(k-1-i)      0 .. N-1
    the successors of p     (p % Q**(k-1)) * Q + x   for x in 0 .. Q-1
```

So the declaration is:

- one parameter `successor_<index>` per node, `index` written out in full with no abbreviation
- each parameter's values a `DomainRule` of its `Q` out-neighbours — **`DomainRule` and not `PythonRule`, for
  exactly the reason its docstring gives**: 117 649 distinct sources would mean 117 649 compiles and 117 649
  cached namespaces, where values kept as values are read straight off. Measured at 0.4 s for all 117 649 through
  the real `RuleCaller`.
- one `circuit(successor_0, ..., successor_<N-1>)` constraint, and nothing else

Reading the answer back: walk the cycle from node nought, take each node's first column `p // Q**(k-1)`, which
gives `N` columns cyclically; then repeat the first `k-1` to make a wall with two ends, `M` columns. Each column
index unpacks to `R` colors by `(c // 7**r) % 7`.

## Where it lives, and a departure from advice

`test/integration/marker_wall/`, driving `Solver` directly through `create_solver()`. Nothing in `src/`, no game
declared, no entry point, no registration under `openmind.domains`.

**This departs from what the design pass recommended**, which was a registered `debruijn` domain in
`src/openmind/agent/factory/` so the large cells could be run through `openmind-solve` and get its statistics
reporting and log-level flags for free. The cost of staying out is a small runner of its own. It is worth paying:
Maxime asked for a test of the solver, not a domain, and OMF is being kept free of domains — chess lives in its
own project for the same reason.

```
    test/integration/marker_wall/marker_wall_circuit.py         builds the encoding
    test/integration/marker_wall/marker_wall_circuit_tests.py   solves the small cells and checks them
```

The large cells are a backgrounded run, not suite tests, or the suite stops being runnable. An operational note
that matters at this size: `BacktrackingSearch` logs a line per assignment at DEBUG, which is about 120 000
records on the largest cell, so it runs at WARNING.

## What is reachable, and what is not

Time goes as the domain removals, with removals about `N*Q` where the search does not backtrack. Memory goes as
the domain elements, `N*Q` of them, measured at about 123 bytes each held as sets. The factor between a
standalone measurement and the same work inside OMF is put at three to eight and is the guess most worth checking
first.

Measured through OMF's own solver, with a budget of six hundred seconds and eight gigabytes a cell. The last
three were not run and their costs are arithmetic.

```
    rows   window     positions N   out-degree Q      metres        domains    in OMF   verdict
       1        1               7              7         7 m         < 1 MB    0.00 s   yes
       1        2              49              7        49 m         < 1 MB    0.00 s   yes
       2        1              49             49        49 m         < 1 MB    0.00 s   yes
       1        3             343              7       343 m         < 1 MB    0.01 s   yes
       3        1             343            343       343 m          14 MB    0.12 s   yes
       1        4           2 401              7     2 401 m           2 MB    0.38 s   yes, and the one that backtracks
       2        2           2 401             49     2 401 m          14 MB    0.17 s   yes
       2        3         117 649             49   117 649 m         709 MB   10.88 s   yes
       3        2         117 649            343   117 649 m       4 963 MB   59.02 s   yes
       2        4       5 764 801             49     5 765 km      34 744 MB        --   no: memory
       3        3      40 353 607            343    40 354 km        1.70 TB        --   no: the answer is too large
       3        4  13 841 287 201            343    13 841 km         584 TB        --   no: the answer is too large
```

**An estimate of mine that was wrong, and is corrected here.** Before it was built, the cost of running inside OMF
rather than standalone was put at three to eight times, and that number sat in all three of these documents. It is
about **one**: 10.88 s against 8.83 s at two rows by three, and 59.02 s against 64.25 s at three rows by two,
where OMF is the *faster* of the two. The heaviest cell that runs uses a tenth of its budget, not five sixths.

What the standalone run did predict exactly is the search itself. Every assignment and dead-end count matches
between the two implementations — 5, 40, 47, 292, 341, 21 899, 2 350, 115 246, 117 304 — which is the strongest
evidence available that both are right, since neither was written from the other.

Nine of the twelve cells. **The last two were never reachable by any means**, because the sequence has one column
per position, so their answers are 40 million and 13.8 billion columns of markers — that is the size of the
answer, not a limit of the solver, and no amount of work on the CSP changes it. The third, two rows by four
columns, is out by about three times on time and two and a half on memory.

**The cell to watch is one row by four**, not one of the big ones. It has the thinnest graph — out-degree 7 over
2 401 nodes — and it is the only cell measured that backtracks at all, at 11 520 dead ends
([circuit-constraint.md](circuit-constraint.md) has the numbers). It still solves in a fifth of a second, but it
is where the search's behaviour is least predictable, so it is the cell whose logs are worth reading.

The capacity table stays complete for all twelve, because it is closed form.

## How it is judged

The closed form is a real oracle, so the assertions pin desired behaviour rather than whatever the solver
happens to do:

- a solution is found
- the successors form one cycle visiting all `N` nodes
- the sequence read off it is `M = N + k - 1` columns
- every one of the `N` window patterns appears in it exactly once

Checked in the suite at one row by two and one row by three, where the whole thing can also be read by eye: the
49-column answer to one row by two must contain every ordered pair of the seven colors once.

The sweep is reported as a table with a header, one row per cell — rows, window, positions, metres, columns,
markers, solved or not reached, assignments, dead ends, values pruned, seconds — and cells the budget does not
reach say so while still showing their closed-form capacity. Ten minutes a cell, which is Maxime's budget and the
only limit in play; nothing in the code caps anything.

## Open

- **Whether the sweep's runner belongs in `test/` beside the test or somewhere else.** It is not a test, it is a
  measurement that takes an hour and a half, and `test/` is where it can see the encoding.
- **Whether one row by one column is worth running at all.** Seven markers, seven metres, a cycle through the
  complete graph on seven nodes. It tests the encoding's edges more than the solver.
- **Whether the two-ends form or the cyclic form is what Maxime actually wants on the wall.** The plan produces
  the two-ends form, `M = N + k - 1` columns, because a wall has ends. If the wall wraps — a room, a pillar — the
  cyclic form is `N` columns and `k-1` markers fewer.
- **Nothing here checks that seven colors are distinguishable in situ.** Under mixed lighting, at distance, on a
  phone camera, seven is at the practical limit, and a misread color puts the answer somewhere else entirely
  rather than nearby. The encoding has no redundancy and detects no error. Worth knowing before stapling
  2 401 of them.
