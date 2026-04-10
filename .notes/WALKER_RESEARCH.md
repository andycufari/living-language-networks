# LLN Walker Research Session Notes

**Date**: 2026-04-08
**Model**: v16 blend (100K vocab, 117.5M edges, 34.5M PMI, 11.9M trigrams)
**Goal**: Can LLN generate grammatical English that stays on topic?

---

## The Core Discovery

**Grammar and semantics both exist in the topology. They live in different structures.**

- **Semantics** lives in PMI edges (34.5M associations). The activation field correctly identifies topic-relevant vocabulary.
- **Grammar** lives in forward edges (117.5M bigrams) + trigrams (11.9M). High-weight bigram chains encode clause structure, verb argument patterns, and function word sequencing.

The original walker could access semantics but not grammar. We proved grammar exists by building a walker that follows it, producing perfect Gutenberg-quality English. The unsolved problem is combining them.

---

## Walker Iterations

### Original Walker (v11)
- **Scoring**: `step_score = (norm_w * tri_mult) + (proximity * target_pmi)`
- **Result**: High semantic relevance, zero grammar
- **Diagnosis**: `proximity * target_pmi` ~ 75, grammar signal ~ 3. Pull is 25x louder than grammar. Beam search does shortest-path-to-target, grammar is accidental.

### Fix 1: Log-Compressed Pull
- **Change**: `proximity * log1p(target_pmi)` instead of raw target_pmi
- **Result**: Grammar improved. Miss rate increased (walker is pickier about paths).
- **Diagnosis**: Pull reduced from 75 to ~10. Grammar now audible. But still additive.

### Fix 2: Hard Trigram Gate
- **Change**: Trigram penalty 0.5 -> 0.3 when pair exists but next word never observed
- **Result**: Marginal improvement. Not the bottleneck.

### Fix 3: Grammar-First with Semantic Fence
- **Architecture change**: Semantic field becomes territory constraint, not destination. Walker moves by grammar inside fence of activated + function + punctuation words.
- **Result**: Perfect English grammar. Zero semantic content. "They seem to know you want to do it."
- **Key proof**: Grammar IS in the topology. Trigram + edge weight structure encodes clause patterns.
- **Failure mode**: Function words outweigh content 10:1 in edge mass. Walker surfs Gutenberg dialogue highways.

### Fix 4: Cadence Walker (Forced Content Rhythm)
- **Architecture change**: After max_streak (3) consecutive function words, force content word landing.
- **Result**: 0-3 content words per output. Forcing fails ~80% (content not 1-hop reachable from function positions).
- **Bright spots**: "Dark clouds" -> "distant galaxies". "Music played softly" -> "hear", "fingers", "trembled".

### Fix 4b: Cadence + Bridge Search
- **Architecture change**: When 1-hop forcing fails, mini beam search (depth=3, width=3) finds grammatical path to nearest content word. Emits entire bridge path.
- **Result**: 1-5 content words per output. Bridge paths are grammatical: "most distant galaxies", "hand trembled", "very proud".
- **Remaining issues**: OpenWebText artifacts (org, pdf, com) in content fence. Bridge beam too narrow for some positions.
- **Status**: Best architecture. Proves the concept.

---

## The Fundamental Tradeoff

```
                    SEMANTIC RELEVANCE
                    |
                    |   Original Walker
                    |   *  (word salad, on topic)
                    |
                    |          o Fix 4b (bridge)
                    |        o Fix 4 (cadence)
                    |
                    |   Fix 1+2 *
                    |
                    |                    Fix 3 *
                    |                    (perfect grammar, off topic)
                    +-------------------------------------->
                              GRAMMATICAL QUALITY
```

---

## Key Architectural Findings

1. **No negation channel**: PMI activation is purely additive. "not X" and "X" produce identical fields.
2. **No syntactic binding**: No positional encoding. "cat that dog chased ran" and "dog that cat ran chased" activate identically.
3. **Grammar/semantics scale mismatch**: Function words have 10x edge mass vs content words. Can't be fixed with additive scoring.
4. **Cadence = English rhythm**: Real English is func-func-CONTENT-func-func-CONTENT. Enforcing this mechanically produces the best hybrid results.

---

## Next Steps

1. **Corpus artifact filter**: Remove web tokens (org, pdf, com, html, cache, server) from content fence
2. **Bridge tuning**: depth=4, width=5 for better content reachability
3. **Two-phase generation**: Phase A = content skeleton (original PMI targeting). Phase B = grammar infill (fenced walker). Full Wernicke/Broca separation.
4. **Content-biased normalization**: Normalize edge weights within fenced subgraph so content words become locally competitive

---

## Files

| File | Description |
|------|-------------|
| generate.py | Core generator with original walker + fenced walker (--fenced flag) |
| fix4_cadence.py | Cadence walker with 1-hop forcing |
| fix4b_cadence_bridge.py | Cadence + bridge search (best current) |
| WALKER_RESEARCH.md | This document |

---

*"The grammar is in the edges. The meaning is in the PMI. The rhythm is in the cadence."*

---

## Session 2026-04-09: Path Memory Experiment

**Hypothesis** (Andy): Each step of the walker can't choose without knowing
**where to go** (activated targets) AND **where it came from** (memory of
prior path). The shipped walker has targets but no memory — the beam search
only knows the tokens inside the current sub-walk, not the full sentence so far.

### Phase 1: Topology diagnostic (`graph_distance_debug.py`)

BFS on top-50 forward edges between random content-token pairs in v16.

- **Mean content-to-content distance: 5.17 hops**
- **42% of pairs unreachable within 6 hops**
- **Graph is NOT small-world** — longer reach than most language corpora

Implications:
- Sliding window for memory must be at least N≈5-6 (graph diameter)
- 1-hop density alone is too sparse a signal
- 2-hop reach gives meaningful differentiation

### Phase 2: Minimal walker (`path_coherent_walker.py`, variant A)

Pure greedy max-weight walker, no targets, no fence, no PMI field. Just:
`score(candidate) = edge_weight`, with visited-cap to prevent infinite loops.

Result: **every prompt collapses to the same grammatical attractor**:
```
"the same time , and the same time , and the same time , and a few days ."
```

Finding: Mantra #1 is correct ("weight IS grammar") but *greedy max* weight
is NOT grammar — it's the language's center of gravity (highest-count bigrams).
Without target-seeking, the walker funnels into generic high-frequency phrases.

**The shipped walker's target-seeking is NOT overlay — it's the mechanism that
breaks out of the max-weight attractor.**

### Phase 3: Verify assumptions (`verify_assumptions.py`)

Before iterating further, verified claims I'd made about v16 topology
against actual graph data. **Corrected wrong assumptions**:

- ❌ "`. → "` is top weight" → actually rank 2 (`The` wins at 27.9M > `"` 22.6M)
- ❌ "Dialogue dominates v16 output" → `burned`, `fire`, `door` top forward
  edges contain no dialogue markers at all. The `"` that appears in outputs
  comes from `.` reachability, not from content-word positions.
- ❌ "`public safety` is a Fineweb stock phrase" → `PMI(fire, public) = 0.0`,
  but `PMI(fire, safety) = 3.67`. The walker correctly targeted `safety`; the
  word `public` entered via max-weight beam path `, and the public safety`.
- ✓ Confirmed: `PMI(burned, alive) = 5.99`, `PMI(burned, brightly) = 2.45`,
  `PMI(door, opened) = 5.51`, `PMI(river, flows) = 5.56`. Content PMI is real.

**Lesson: do not speculate about v16 output quirks without running
verify_assumptions.py first.**

### Phase 4: Target walker + path memory (`target_walker_with_memory.py`)

Forked shipped `walk_to_target` to take `full_path_window` (last N tokens of
the overall sentence, across all prior walks) and added a memory term:
```
step_score += memory_alpha * density
  where density = connections_to_full_path_window / window_size
```

**A/B results on 8 default prompts** (memory_alpha=2.0, window=8, v16):

| Prompt | A (shipped) hits/toks | B (memory) hits/toks | Notes |
|---|---|---|---|
| The fire burned | 7/14 | **13/20** | Extended with "glowed bright lights flashing red hot" — fire domain |
| The king | 5/14 | 5/14 | Identical (shipped already works) |
| She opened the door | 5/20 | 5/20 | Identical |
| The army marched | 3/7 | **8/20** | Extended with "force, and the two miles inland. The company" |
| Dark clouds | 1/3 | **10/20** | Extended with "star forming galaxies collide beams" — astronomy! |
| The river flows | 5/8 | **13/20** | Extended with "blood levels of a major rivers draining fluid" |
| Scientists discovered | 8/18 | 9/20 | Minimal change |
| The old man walked | 6/20 | 6/20 | Identical |

**Findings:**

1. **Path memory works as a RESCUE signal** on sink-dominated prompts.
   Sink-dominated prompts (TOPOLOGY_DEBUG_V16.md proved Dark clouds and
   The fire burned are sink-heavy) jumped from 1-3 tokens to 10-13 token
   hits.

2. **Memory does NOT distort working walks.** Prompts where the shipped
   walker already reached the cap (The king, She opened the door, The old
   man walked) produced identical output.

3. **Mechanism is reach, not rank.** Memory doesn't help pick better
   content targets (targeting already works). It helps the beam reach
   targets that would otherwise be pruned as unreachable. The bonus is
   enough to tip the beam toward paths that eventually find the content
   target.

4. **Cost**: ~100x slower (0.6s → 60-400s per prompt). Unoptimized. The
   expensive part is building window forward-edge sets per beam step.
   Optimization is deferred — the finding matters more than speed right now.

### Phase 5: Memory refinement needed (open)

Raw path memory counts ALL connections, treating function words and content
words equally on both sides. This has two known problems:

1. **Path side**: `the`, `,`, `.` in the window contribute meaningless
   "memory" (they connect to everything). The `content_only=True` mode
   added to `target_walker_with_memory.py` filters the path side using
   `top-150 by in_degree + punctuation` as noise_set.

2. **Candidate side** (STILL UNFIXED): function-word candidates still get
   memory bonuses when they connect to content path tokens. Memory should
   not apply to function candidates at all — grammar alone (raw weight)
   should decide function-word positions. Memory should only nudge CONTENT
   candidates.

**The correct mechanism is multiplicative-for-content-only:**
```
if candidate is content:
    step_score *= (1 + memory_alpha * content_path_density)
# else: step_score unchanged, grammar primacy preserved
```

### Phase 6: Grammar is being drowned by existing overrides (open)

Per Andy's insight: "we're controlling too much outside the weighted path".
The shipped walker already has THREE semantic overrides that pull the beam
away from raw weight:

```
step_score = (log1p(w) * 0.1 * tri_mult)         # grammar, ~3-8
           + (proximity_direct * 3.0)             # direct target reach
           + (proximity_overlap * 0.3 * overlap)  # neighborhood overlap
           + (log1p(target_pmi))                  # pull strength, 5-10
```

Grammar is `0.1 * log1p(w) * tri_mult` — the `0.1` scalar is already
reducing grammar's voice. Meanwhile proximity_direct * 3.0 and unbounded
log1p(target_pmi) can dominate grammar 3:1 on strong targets.

**Open hypothesis: grammar quality will improve by INCREASING grammar_weight
and DECREASING the semantic override coefficients, even before adding memory.**

The next experiment expose all five coefficients as tunable parameters and
run manual scans to find the configuration that maximizes fluent English
output. No automation — read glass-box traces, form hypothesis, adjust one
coefficient, re-run.

### Phase 7: Coefficient tuning results (executed 2026-04-09)

Ran 7 configurations on 8 default prompts (v16 model, max_tokens=20,
max_chains=15). All configurations use the refined v2 walker with
content-only memory on both sides + multiplicative-for-content-only
nudge.

| Config | Tokens | Hits | vs shipped | Notes |
|---|---|---|---|---|
| **shipped / v2_baseline** | **104** | **40** | anchor | max_chains=15, memory_alpha=0 |
| v2_grammar_first (g=0.3, pd=1.5, p=0.5, m=1.5) | 76 | 27 | -27% / -33% | Collapsed. Too much grammar starves the walker. |
| v2_memory_grammar (g=0.15, pd=2.5, p=0.8, m=1.0) | 96 | 43 | -8% / +8% | Weak overall BUT produced literary gem |
| v2_grammar_up (g=0.2) | 110 | 43 | +6% / +8% | Shorter but cleaner clauses |
| v2_pull_half (p=0.5) | 119 | 44 | +14% / +10% | Same pattern as grammar_up (ratio-equivalent) |
| v2_memory_1 (m=1.0) | 130 | 51 | +25% / +28% | Good rescue, some over-nudging |
| **v2_memory_half (m=0.5)** | **136** | **56** | **+31% / +40%** | **Sweet spot** |

(g=grammar_weight, pd=proximity_direct, p=pull_strength, m=memory_alpha;
defaults 0.1 / 3.0 / 1.0 / 0 match shipped exactly.)

**v2_baseline exactly reproduces shipped**: character-identical outputs,
identical token and hit counts. Framework verified before tuning.

### Key findings

**1. memory_alpha=0.5 is the Pareto optimum discovered so far.**
Rescues sink-dominated prompts (Dark clouds 3→19, Army marched 7→20,
Fire burned 14→11 but with better grammar) without over-nudging content
candidates. At alpha=1.0, memory wins too often and fragments clauses
("current flowing stream channels"). At alpha=0.5, grammar keeps primacy
and memory only breaks ties.

**2. Grammar_weight and pull_strength are ratio-equivalent.** Doubling
grammar (0.1→0.2) and halving pull (1.0→0.5) produced nearly identical
outputs. The beam compares scores relatively, so only the ratio matters.
The four semantic overrides (grammar, prox_direct, prox_overlap, pull)
collapse to effectively ONE composite "semantic pull vs grammar" knob.

**3. Content-only memory KILLED the known grammatical traps:**
- "public safety" trap (The fire burned): GONE in v2_memory_*
- `"` unmatched-quote trap (She opened the door): GONE in v2_memory_*
- "the same time" attractor loop: not observed in any memory variant

Mechanism: these traps have function-word components that would get
memory bonuses under the naive scheme, but under content-only memory,
the function-word candidates score on grammar alone and the content
path's memory pull selects candidates that have topical connection back.

**4. Pushing all semantics down at once collapses the walker** (v2_grammar_first).
But it produces one spectacular output en route:
```
"Dark clouds" -> "hung heavily laden dust storm burst asunder"
```
7 tokens, 6 hits, real literary English, topologically perfect.
This proves the topology CAN produce this quality — the walker just
needs the right configuration to let it through consistently.

### Qualitative comparison on hardest prompts

**"Dark clouds"** — previously 3 tokens of drift:
- shipped: `"of this matter"` [3 tok, 1 hit]
- v2_memory_half: `"of the world . It doesn't matter how much less energy . It is quite clear blue smoke cleared"` [19 tok, 5 hits]
- v2_memory_grammar: `"hung heavily laden dust storm burst asunder"` [7 tok, 6 hits] ← **real English**

**"The king"** — previously drifted to generic narrative:
- shipped: `"of the most powerful , and I heard a little boy whom he answered"` [14 tok, 5 hits]
- v2_memory_half: `"had heard my lord hath commanded thee thy life . It is my boy whom he answered quietly replied Mr"` [20 tok, 11 hits] ← **archaic royal register**

**"The river flows"**:
- shipped: `"south bank deposits . The main stream flowing"` [8 tok, 5 hits]
- v2_memory_half: `"south bank of the water flow . The main stream flowing blood"` [12 tok, 6 hits] ← **cleaner clause boundary**

### Open questions for next session

- **Can we reproduce the "Dark clouds → hung heavily laden" output reliably?**
  Was it v2_memory_grammar's specific (g=0.15, pd=2.5, p=0.8, m=1.0) or
  an artifact of beam ordering? Needs deterministic verification.
- **Window size** kept at 8 throughout — should test 4, 6, 12.
- **Should memory use the CURRENT walk's path_set in addition to full
  history?** Currently only uses last N of (prompt + generated before
  this walk), not the tokens from this walk in progress.
- **Is v2_memory_half the true optimum, or is the search space larger?**
  We only ran 7 configurations. Pareto frontier could have better points.
- **Does `find_targets` need updating for v16 topology?** TOPOLOGY_DEBUG_V16
  showed content words became sinks in v16 — target selection may be
  picking unreachable targets regardless of walker quality.

### Phase 8: Window size sweep (executed 2026-04-09)

Fixed v2_memory_half coefficients (α=0.5), varied `window_size` across 4, 6, 8, 12.

| Window | Tokens | Hits | Dark clouds hits |
|---|---|---|---|
| 4 | 136 | 52 | 2 |
| 6 | 130 | 50 | 2 |
| **8** (default) | **136** | **56** | **5** |
| 12 | 131 | 53 | 2 |

**Finding: window=8 is the peak**, especially visible on sink-dominated
prompts. Window=8 found 5 Dark clouds hits while 4/6/12 dropped to 2.

**Why window=8 > window=6 despite graph diameter ~5.17?** The window
includes prompt tokens AND generated tokens. At window=6, prompt (3-4
tokens) + early generation exhausts the budget, leaving no mid-walk
memory. Window=8 provides room for prompt anchor AND mid-walk content.

**Window=12 is wasted.** 2x slower, no improvement. Path tokens beyond
diameter become noise.

### Phase 9: Sink-mode inversion (executed 2026-04-09)

**Problem identified from TOPOLOGY_DEBUG_V16:** In v16, content words
became global sinks (fire 0.43, door 0.22, extinguisher 0.11). The
shipped `find_targets` penalizes sinks (`score *= 0.2` when `tokens_remaining > 5`),
effectively killing all high-quality content targets early in generation.

**Test:** Three sink modes with v2_memory_half coefficients:
- `penalty` — shipped behavior
- `neutral` — ignore pr_ratio, pure PMI ranking
- `reward` — INVERT: reward deep sinks as topical attractors

| Config | Tokens | Hits | Fire burned key phrase |
|---|---|---|---|
| memhalf (penalty) | 136 | 56 | "alive. It is my eyes flashed brightly" |
| memhalf + sink_neutral | **142** | 54 | "alive. **It provides protection for** your eyes flashed" |
| memhalf + sink_reward | 135 | **56** | "alive. **It provides protection for** your eyes flashed" |

**Key finding:** sink inversion unlocks deep content sinks like `protection`
(pr_ratio=0.189) that penalty mode was suppressing. "It provides protection"
is the best fire-domain phrase we've seen — it's genuinely grammatical AND
topically specific.

**Tradeoffs:**
- Dark clouds drifts to color vocabulary ("green leafy green energy") under
  inversion, whereas penalty+memory gave "blue smoke cleared" (arguably
  closer to weather).
- Hit counts are statistically equivalent between penalty and reward (both 56).
- The improvement is prompt-dependent — sinks that were previously unreachable
  due to the penalty are now accessible, but some prompts lose specificity.

**Verdict:** sink_reward is the new recommended config. Rationale:
1. Matches best hit count (56) of all configurations tested
2. Unlocks "protection" and similar deep sinks = better topical specificity
3. Architecturally aligned with TOPOLOGY_DEBUG_V16 finding (v16 content = sinks)
4. Co-design insight: path memory + sink reward work together. Memory makes
   sinks reachable in the walker; sink reward makes them selectable as targets.

### Final recommended configuration

```python
generate_v2(
    grammar_weight=0.1,
    proximity_direct=3.0,
    proximity_overlap=0.3,
    pull_strength=1.0,
    memory_alpha=0.5,
    window_size=8,
    sink_mode='reward',
)
```

This is the best v2 walker configuration as of 2026-04-09.

### Phase 10: Rank-based vocabulary tiers (executed 2026-04-09)

**Problem (Andy's insight):** absolute in_degree thresholds are harmful.
The codebase had at least 5 ad-hoc thresholds: `> 5000`, `> 10000`,
`> 15000`, `> 20000` — each encoding some notion of "common word" with
different cutoffs. None are justified. They silently misbehave across
models/corpora.

**Evidence** (`measure_vocab_ranks.py` on v16):

- `discovered` has in_degree=15990 but rank=628 (clearly content).
  The `in_degree > 15000` filter in `find_targets` was silently
  excluding "discovered" as a target — breaking "Scientists discovered"
  prompts.
- `fire` at in_degree=14790 was ONE unit away from being filtered.
  In a future 2x-corpus model it would cross the threshold and
  generation quality for fire prompts would collapse silently.
- Top-150 by in_degree captures the natural knee in the distribution
  (from 99087 at rank 1 to 31653 at rank 150). Everything above rank
  150 is true function words + punctuation + top-frequency particles.

**Fix: VocabRank module** (`.notes/vocab_rank.py`)

Three tiers by rank:
- **function** (rank 0..149): punctuation, articles, particles. Never
  targets. Skipped from PMI activation expansion.
- **semi-function** (rank 150..499): common content with function-like
  behavior ("also", "often", "much"). Fence-eligible. Noise for density.
- **content** (rank 500..): real semantic tokens. Density-worthy.

Exposed as CLI args: `--n-function 150 --n-semi-function 500`.
Model-agnostic.

**Replaced call sites in `target_walker_v2.py`:**
- `activate()` → `activate_v2()` — uses `vocab_rank.is_function()`
  instead of `in_degree > 20000`
- `find_targets_v2()` — uses `vocab_rank.is_function()` instead of
  `in_degree > 15000`
- `generate_v2()` — content counting uses rank instead of `in_degree <= 20000`
- `build_content_mask()` — delegated to `vocab_rank.noise_set`

`generate.py` (shipped) is NOT touched — all changes are in `.notes/`.

**v2_baseline still matches shipped exactly** (104/40, identical outputs).
Proves the rank fix preserves default behavior when sink_mode=penalty
(which uses the shipped `find_targets` with its own in_degree filter).

**Rank-based memhalf_sink_reward results:**

Default prompts (16 configs previously, this is the new one):

| Config | Tokens | Hits | vs shipped |
|---|---|---|---|
| shipped | 104 | 40 | anchor |
| old memhalf_sink_reward (in_deg 15000) | 135 | 56 | previous best |
| **rank memhalf_sink_reward** | 138 | 43 | +3 tok, -13 hits vs old |

Weather prompts:

| Config | Tokens | Hits |
|---|---|---|
| shipped | 66 | 26 |
| old memhalf_sink_reward | 92 | 38 |
| **rank memhalf_sink_reward** | **119** | **45** |

**Rank-based is a clean win on weather (+29% tok, +18% hits vs previous
best) but mixed on defaults (slight token gain, hit loss).**

**Why the default regression?** The filter change in `activate()`
affects which PMI neighbors get expanded, which affects the target list,
which affects the walker. "The fire burned" truncated from 14 to 5 tokens
because target discovery shifted. The constants were compensating for
each other — removing one exposed tuning issues in the others.

**Qualitative wins from rank-based:**
- "She opened the door → opens fire. She paused abruptly closed the door.
  She smiled and he looked at the next day." — **full grammatical clause,
  door-domain coherent.**
- "Scientists discovered → until recently. Some believe they say, I hope
  we call him. But it's hard working families will" — much more natural
  than any previous output.
- "A cold front → door of water supply lines in front rank of an air" —
  11 tokens vs 6 previously, stays in cold/water vocabulary.

**Qualitative losses:**
- "The fire burned → alive. So let off" — truncated, lost "protection"
  target that old filter was reaching.
- Some outputs have mid-sentence repetitions ("the second round... the
  same direction towards the same direction").

### Phase 10a: The activate() filter is principled

**Andy's correction** (critical): activate() operates on the STATIC full
graph where the in_degree distribution is known and fixed. The
`in_degree > 20000` filter there is principled — it's measuring against
global topology to skip hubs whose neighborhoods are too broad.

The rank-based fix belongs DOWNSTREAM of activation: inside find_targets
and inside density/memory calculation, where we're working with the
induced subnetwork and global in_degree no longer reflects local behavior.

**Reverted:** `activate_v2` removed. `generate_v2` now calls the shipped
`activate()` directly, preserving its in_degree > 20000 filter unchanged.

**Kept:** rank-based filter in `find_targets_v2` and noise_set in memory
density calculation. These are the places where global in_degree was
genuinely harmful.

### Phase 10b: Calibration of n_semi_function

The rank-based find_targets filter needed calibration. Test:

| n_semi_function | Defaults tok/hit | Weather tok/hit | Fire burned |
|---|---|---|---|
| 500 (natural knee) | 140 / 48 | **119 / 45** | 7/5 |
| 700 (≈ in_deg 15000) | **135 / 50** | 90 / 34 | 9/7 ← restored |

**Finding: no single cutoff wins both.** Default prompts prefer higher
cutoff (700) because semi-common words dilute the topical target pool.
Weather prompts prefer lower cutoff (500) because thin activation fields
need more targets available.

**Decision:** default to `n_semi_function=500`, expose as CLI arg so
caller can override. Use `--n-semi-function 700` for default prompts,
`--n-semi-function 500` for weather/thin-field prompts.

This is the first concrete proof that **a single fixed constant cannot
work for all prompts**. Different subnetwork topologies need different
target filter aggressiveness.

### Phase 10c: VocabRank simplified to 2 tiers

Removed the separate `n_target_exclude` tier. The rank system is now
just two cutoffs:

- `n_function` (default 150): true function words. Never targets. Also
  excluded from density/memory as noise (upper bound of noise_set).
- `n_semi_function` (default 500): noise tier cutoff. Excluded from
  density AND from target list via `is_noise()`.

Single `is_noise()` method used in three places: find_targets filter,
density computation exclude list, noise_set construction. One cutoff
to tune per subnetwork instead of two or three.

Mantra #5: simplify always wins.

### Final rank-based configuration

```python
# Default for weather and mixed prompts
generate_v2(
    n_function=150,
    n_semi_function=500,
    memory_alpha=0.5,
    sink_mode='reward',
    window_size=8,
    # ... rest shipped defaults
)

# Tighter target filter for default prompts (better hit count, ~same tokens)
generate_v2(
    n_function=150,
    n_semi_function=700,  # ← tuned for defaults
    ...
)
```

Both are rank-based and model-agnostic. Both expose the cutoff as a
CLI argument (`--n-function`, `--n-semi-function`). Neither relies on
absolute in_degree thresholds anywhere inside the tunable path.

### Why this matters for Phase 11 (adaptive)

Phase 10 proved that:
1. **Different subnetworks need different cutoffs.** Defaults want 700,
   weather wants 500. No single constant is optimal.
2. **The cutoffs are tunable per-call** now, not hardcoded. Phase 11 can
   compute the right cutoff from the subnetwork topology and pass it in
   without code changes.
3. **Rank-based is the right representation.** Absolute thresholds would
   still require retuning across corpora; rank-based only requires
   retuning across prompts within the same corpus.

Phase 11 will add `derive_config_from_subnet(activated, graph)` that
measures subnetwork properties (field_size, sink_fraction, PMI density)
and returns a dict of coefficients including `n_semi_function`, ready
to pass into `generate_v2`.

### Files added this session

| File | Description |
|------|-------------|
| measure_vocab_ranks.py | Diagnostic: vocab in_degree distribution by rank |
| vocab_rank.py | VocabRank — 2-tier rank-based classification |
| target_walker_v2.py | Rank-based find_targets, preserved shipped activate |
| tune_manual.py | Exposes --n-function / --n-semi-function CLI args |

### Files added this session

| File | Description |
|------|-------------|
| graph_distance_debug.py | Topology diagnostic: content-to-content distance, small-worldness |
| path_coherent_walker.py | Minimal walker baseline (pure weight, no targets) |
| verify_assumptions.py | Check claims about edge weights / PMI pairs before speculating |
| target_walker_with_memory.py | First path-memory experiment (all-tokens, additive) |
| target_walker_v2.py | Refined: content-only selectivity on both sides, multiplicative nudge |
| tune_manual.py | Manual coefficient runner — no automation, just config + side-by-side output |

