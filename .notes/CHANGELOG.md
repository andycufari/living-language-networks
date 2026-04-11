# LLN Walker Changelog

## v1 — Original Walker (shipped as generate.py, commit a8396f4)

**Architecture**: activate → find_targets → walk_to_target (beam search) → deplete → repeat
**Scoring**: `step_score = (log1p(w) * 0.1 * tri_mult) + (proximity * log1p(target_pmi))`

Strengths:
- Glass box: every decision traceable
- Proved grammar/semantics separability in the topology
- Beat custom GPT-2 on PMI relevance (30/40 prompts)

Weaknesses (measured):
- Walked through 6-9x heavier edges than real sentences (stock phrase attractor)
- Rank 24-81 mid-sentence (should be 105-140 per corpus measurement)
- 83-100% trigram coverage (real sentences: 65-72% — over-attesting)
- pr_ratio 0.01 (real: 0.03-0.05 — too deep into sinks)
- Known traps: "public safety" detour, unmatched `"` quotes, "and the same time" loops
- 5 absolute in_degree thresholds (5000, 10000, 15000, 20000) — corpus-specific, silently broke on edge cases
- Weather prompts catastrophically broken: 6 of 8 produced <5 tokens

**File**: `.notes/generate_v1.py`

---

## v2 — Path Memory + Rank-Based Filters (never shipped)

**Changes from v1**:
- Added path memory: beam scores candidates by topological connection to
  full prior path (prompt + all generated tokens), not just current walk
- Content-only memory: function words excluded on both sides (path and candidate)
- Multiplicative nudge: `score *= (1 + memory_alpha * density)` on content candidates only
- Rank-based VocabRank replacing absolute in_degree thresholds
- Sink mode: reward/neutral/penalty for target selection
- 7 tunable CLI coefficients

**Results**: improved defaults (+35% tok, +20% hits) and weather (+80% tok, +73% hits).
Fixed "public safety" trap and unmatched quote trap.

**Why not shipped**: no single fixed configuration beat v1 on ALL prompts.
"The fire burned" regressed from 14/7 to 7/5. Required per-prompt tuning.

**Files**: `.notes/target_walker_v2.py`, `.notes/vocab_rank.py`, `.notes/tune_manual.py`

---

## v3 — Profile Walker (shipped as generate.py)

**Key discovery** (SENTENCE_ANATOMY.md): real sentences have a measurable
topological wave shape. Measured from 545 corpus sentences:

- Rank stays 105-140 mid-sentence (not 24 like v1, not 500+ like v2)
- Forward weight stays 700K-1.5M (not 6-9M like v1's max-weight)
- Trigram coverage ~65-72% (not 90-100% like v1's over-attesting)
- pr_ratio 0.03-0.05 (not 0.01 like v1's deep sinks)

**Architecture**: same two-phase as v1 (find_targets → walk_to_target).
Only the beam score changes: minimize topological distance from the
measured sentence profile at each normalized position, plus topical
bonus for reaching the target.

```python
score = -profile_distance(candidate, target_profile_at_position) + topical_bonus
```

This is the opposite of v1's "maximize grammar + pull." v3 says
"be normal at this position, be topical." The sentence shape IS the grammar.

**Results** (15 unique prompts, fresh side-by-side):

| | main (v1) | v3 |
|---|---|---|
| Total tokens | 167 | **236** (+41%) |
| Total hits | 65 | **91** (+40%) |
| Prompts won | 0 | **14** |
| Ties | 1 | 1 |
| Prompts lost | 0 | **0** |

**Zero regressions.** 14 wins, 1 tie, 0 losses across all 15 prompts.

Hero prompt preserved and improved:
```
v1: alive , and the public safety , and his eyes flashed brightly glowing cheeks    [14/7]
v3: alive with his own personal safety training camp with his eyes flashed brightly glowing cheeks    [15/8]
```

Weather rescued (5 prompts from 1 token to 8-19 tokens):
- Heavy rain: 1→8, A cold front: 1→19, The wind howled: 1→14

Traps eliminated: no unmatched quotes, no stock-phrase loops, no "public safety" detour.

**Why it works**: v1 maximized scoring terms (higher = better on every axis).
This produced outputs that were MORE EXTREME than real English on every
measurable dimension. v3 matches the measured wave — moderate weights,
moderate ranks, moderate trigram density. This IS how English works
topologically.

**Tuned weights** (commit f08b956): rank=0.5, fwd=1.5, topical=2.5.
Weight scale auto-normalization for cross-model portability.

**File**: `generate.py` (replaced v1)

---

## Current status (2026-04-11)

**Output quality**: phrase salad — between word salad and broken sentences.

The walker produces recognizable multi-word units but cannot connect them
into grammatical sentences. Specific problems:

1. No clause structure (no subject-verb-object awareness)
2. Register drift (archaic mixes with modern mid-sentence)
3. Topical drift (OpenWebText vocabulary leaks into unrelated prompts)
4. No sentence ending (outputs hit token cap mid-phrase)

```
Quality spectrum:
  Word salad:        "army infantry artillery cavalry troops"
  Phrase salad:      "his brother officers came forward . An"     ← HERE
  Broken sentences:  "The officers came forward but the army"
  Correct sentences: "The army's officers came forward at dawn."
```

**What's needed for "broken sentences"** (next quality level):
- Structural position awareness (know WHAT ROLE each position plays, not just what rank)
- Clause boundary detection (know when to emit `.` and start a new clause)
- Register locking (once archaic, stay archaic)
- Two-pass architecture (content skeleton first, grammatical infill second)

**The approach is viable.** The gap from "phrase salad" to "broken sentences"
is smaller than the gap from "word salad" to "phrase salad" — and we crossed
that gap in this session.

---

## Research artifacts

All intermediate experiments preserved in `.notes/`:

| File | Role |
|---|---|
| `generate_v1.py` | Original shipped walker (archived) |
| `target_walker_v2.py` | v2 with memory + rank filters |
| `vocab_rank.py` | Rank-based vocabulary classification |
| `profile_walker.py` | v3 standalone (before merge to generate.py) |
| `tune_manual.py` | Named-config A/B runner |
| `sentence_anatomy.py` | Corpus sentence measurement tool |
| `graph_distance_debug.py` | BFS topology diagnostic |
| `verify_assumptions.py` | Edge weight / PMI pair checker |
| `measure_vocab_ranks.py` | Vocab in_degree rank distribution |
| `path_coherent_walker.py` | Minimal walker baseline (pure weight collapse demo) |
| `fix4_cadence.py` | Cadence walker experiment (2026-04-08) |
| `fix4b_cadence_bridge.py` | Cadence + bridge search experiment |
| `WALKER_RESEARCH.md` | Full research log (10+ phases) |
| `SENTENCE_ANATOMY.md` | Corpus sentence profile data |
| `V2_VS_MAIN.md` | v2 vs v1 comparison document |
| `V3_RESULTS.md` | v3 results + tuning sweep |
