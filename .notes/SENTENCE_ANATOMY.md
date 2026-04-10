# Sentence Anatomy — what real sentences look like in LLN v16

**Date**: 2026-04-09
**Model**: v16 blend (100K vocab, 117.5M edges, 34.5M PMI, 11.9M trigrams)
**Sample**: 545 clean sentences (180 fineweb-edu, 159 gutenberg, 206 openwebtext)
**Positions measured**: 6910
**Script**: `.notes/sentence_anatomy.py`

---

## Summary: what real sentences look like, position by position

Sentences are 5-20 tokens, normalized to 0.0-1.0. Each row is aggregated
across ~600 positions from ~545 sentences.

| norm pos | median rank | mean pr | mean fwd_weight | trigram% |
|---|---|---|---|---|
| 0.00 (start) | **334** | 0.05 | 433K | 0% |
| 0.10 | **129** | 0.03 | 1119K | 49% |
| 0.20 | **120** | 0.03 | 1019K | 62% |
| 0.30 | **134** | 0.03 | 1026K | 67% |
| 0.40 | **138** | 0.04 | 1472K | 69% |
| 0.50 | **114** | 0.04 | 1229K | 65% |
| 0.60 | **132** | 0.03 | 1079K | 67% |
| 0.70 | **105** | 0.04 | 687K | 70% |
| 0.80 | **136** | 0.03 | 1384K | 68% |
| 0.90 | **105** | 0.03 | 808K | 72% |

**Reading the profile:**

1. **Position 0 is unique**: rank 334, higher pr (0.05) — the sentence-starter slot (capitalized sources like "The", "She", "Having")
2. **Positions 0.1-0.9 are remarkably stable**: median rank oscillates between 105 and 140, pr stays at 0.03-0.04, forward weight stays in the 700K-1.5M band
3. **Trigram coverage ramps up**: 0% at position 0 (no history yet), climbs to 49% at 0.1, stabilizes around 65-72% from 0.3 onward

**Per-source consistency**: fineweb-edu, gutenberg, and openwebtext all
produce profiles within 20% of each other on every feature. The
sentence shape is **corpus-invariant** — it's a property of English, not
a property of a specific text source.

---

## Walker outputs (8 prompts each)

### Main (generate.py at commit a8396f4)

| norm pos | median rank | mean pr | mean fwd_weight | trigram% |
|---|---|---|---|---|
| 0.00 | 421 | 0.01 | 87K | 13% |
| 0.10 | **1176** | 0.02 | **4466K** | 64% |
| 0.20 | 438 | 0.01 | **8026K** | 83% |
| 0.30 | **24** | 0.01 | **6502K** | 92% |
| 0.40 | **81** | 0.01 | **8131K** | 91% |
| 0.50 | **33** | 0.01 | **9131K** | 92% |
| 0.60 | **52** | 0.01 | **6667K** | 100% |
| 0.70 | **30** | 0.01 | **9835K** | 100% |
| 0.80 | 1226 | 0.02 | 80K | 100% |
| 0.90 | 1572 | 0.03 | 3168K | 81% |

### v2 (target_walker_v2.py, memory_half + sink_reward)

| norm pos | median rank | mean pr | mean fwd_weight | trigram% |
|---|---|---|---|---|
| 0.00 | 578 | 0.02 | 143K | 25% |
| 0.10 | 1416 | 0.02 | 566K | 71% |
| 0.20 | 1004 | 0.01 | 322K | 53% |
| 0.30 | 523 | 0.01 | 3790K | 94% |
| 0.40 | 567 | 0.01 | 330K | 86% |
| 0.50 | 536 | 0.01 | 952K | 94% |
| 0.60 | 672 | 0.01 | 434K | 88% |
| 0.70 | 1077 | 0.02 | 7449K | 93% |
| 0.80 | 354 | 0.01 | 1701K | 100% |
| 0.90 | 529 | 0.02 | 3877K | 86% |

---

## The four big findings

### Finding 1: Walkers walk through MUCH heavier edges than real sentences

| | fwd_weight range (middle positions 0.3-0.7) |
|---|---|
| **Real sentences** | 700K - 1.5M |
| **Main walker** | 6.5M - 9.8M |
| **v2 walker** | 330K - 7.4M (highly variable) |

**Main walks through the heaviest paths in the graph, 6-9x heavier
than real sentences use.** This is the "stock phrase attractor"
quantified: the walker picks `, and the` and similar max-weight chains
because they have the strongest edges, but real sentences use a more
moderate weight band.

**v2 is closer to the real range sometimes** (position 0.4 = 330K) but
wildly variable — it jumps from 330K to 7.4M across adjacent positions.
Real sentences are smooth; v2 spikes.

**Mantra #1 needs refinement**: "weight IS grammar" is true, but grammar
uses MODERATE weight, not maximum. The walker should prefer candidates
in a weight band near the expected profile, not simply maximize.

### Finding 2: Walkers dive into top-30 rank slots in the middle

| | median rank at position 0.3-0.7 |
|---|---|
| **Real sentences** | 105 - 140 |
| **Main walker** | 24 - 81 |
| **v2 walker** | 523 - 1077 |

**Main's middle positions hit rank 24-81** — that's `.`, `,`, `the`,
`of`, `and`, `to`. Real sentences DO NOT stay in that tiny pool. They
live in the rank 100-150 range, which is where semi-content and
connective phrases live (`when`, `which`, `said`, `went`, `took`,
`how`, `over`).

**v2 over-corrects** — it's at rank 500-1000, deeper into content
vocabulary than real sentences. Real sentences don't sustain this depth
either.

**The target zone is rank 100-150.** Neither walker hits it. Main is too
shallow, v2 is too deep.

### Finding 3: Walkers over-use attested trigrams

| | trigram coverage (positions 0.3-0.9) |
|---|---|
| **Real sentences** | 67% - 72% |
| **Main walker** | 81% - 100% |
| **v2 walker** | 85% - 100% |

**Both walkers produce outputs with HIGHER trigram coverage than real
English.** This is counterintuitive but important: real sentences
constantly include trigrams that are NOT in our trigram set (because
the min_count filter excluded rare ones, or they're genuinely rare in
the corpus).

The walker's trigram scoring is acting as a filter that forces
over-representation of the strongest phrases. Real writing is ~30%
rare trigrams. Our walker is ~0-20% rare trigrams.

**This is the "sound of a walker":** everything connects via strong
trigrams, which means everything sounds like a stock phrase.
Real writing has variability the walker is erasing.

### Finding 4: Walkers walk through deeper sinks

| | mean pr_ratio (middle positions) |
|---|---|
| **Real sentences** | 0.03 - 0.05 |
| **Main walker** | 0.01 |
| **v2 walker** | 0.01 |

**Real sentences flow through tokens with pr 0.03-0.05** (moderate
sinks that still push forward a bit). Walker outputs sit at pr 0.01
(deep sinks, dead-end tokens).

Both walkers select tokens that receive much more flow than they emit.
Real sentences select tokens that have a bit more forward flow to
propagate through.

This is consistent with Finding 1: deep sinks tend to be the ends of
strong collocations. `(in/on/at) → the → (stuff)` — "the" is a sink at
pr 0.01 because it receives from everything. The walker picks it
because it has the maximum edge weight from many prior tokens.

---

## Position 0 is unique and everyone agrees

| | median rank at pos 0 |
|---|---|
| Real sentences | 334 |
| Main walker | 421 |
| v2 walker | 578 |

Position 0 is the sentence-starter slot. Both real and walker outputs
agree: this position is filled by rank 300-600 tokens (`The`, `She`,
`Having`, `He`, etc.). Walkers get position 0 right because they start
from the prompt's last token, which is already a valid starter.

**Everything after position 0 diverges.** The walker doesn't know how to
sustain the rank-100-to-150 band that real sentences live in.

---

## What this implies for v3

### The scoring objective should be PROFILE MATCHING, not maximization

Current walkers maximize a sum of terms:
```
score = grammar + proximity + pull + memory
```

Every term is "more is better." The result is an output that is
**more extreme than real English on every measurable axis**: heavier
weights, shallower ranks, deeper sinks, stronger trigrams.

**Proposed v3 objective**: at each position, prefer candidates whose
topological features are CLOSE to the expected profile at that
normalized position in the sentence:

```python
# Target profile for this position (interpolated from corpus anatomy)
target = profile[norm_position]
# target = {rank: 130, pr: 0.04, fwd_w: 1e6, trigram_valid: 0.68}

# Candidate score: negative distance to target profile
rank_dist    = abs(log(candidate.rank + 1) - log(target.rank + 1))
pr_dist      = abs(candidate.pr - target.pr) * 20
fwd_dist     = abs(log(candidate.fwd_w + 1) - log(target.fwd_w + 1))
trigram_bonus = 1.0 if candidate.trigram_valid == (target.trigram_valid > 0.5) else 0.5

distance = rank_dist + pr_dist + fwd_dist
score    = -distance * trigram_bonus + topical_bonus
```

**Maximize this score = be normal at this position, be topical.**

This is a completely different framing from everything we've tried:
- Not "weight IS grammar" (maximize weight)
- Not "pull toward target" (maximize proximity)
- Not "memory stays topical" (maximize content density)
- **"Be normal. Be topical."** The sentence shape is the grammar.

### Why this might work

1. **Real sentences have a measurable shape.** We just measured it.
   Rank 100-140, pr 0.03-0.05, fwd_w 700K-1.5M, trigram% 65-70%.
2. **The walker's failure modes are all "too extreme."** Too common
   (rank 24), too heavy (9M weight), too deep sink (pr 0.01), too
   strong trigram (100%). Profile-matching penalizes extremes.
3. **This is corpus-invariant.** The profile is consistent across
   fineweb, gutenberg, openwebtext. It's not a tunable constant —
   it's a measurement of English.
4. **It naturally handles sinks.** Real sentences DO walk through
   sinks (pr 0.03-0.05), just not deep ones (pr 0.01). Profile
   matching prefers moderate sinks automatically, no sink_mode flag.
5. **It naturally handles function/content balance.** Rank 105-140
   is exactly the semi-function zone (what VocabRank calls tier 2).
   Real sentences oscillate in this band, which is neither function
   nor content — it's the connective tissue.

### Why this might NOT work (honest risks)

1. **Profile is an average.** Individual sentences vary around it. A
   walker that matches the average might produce bland "average English"
   that has no character.
2. **Topicality still needs to be encoded.** The profile alone would
   produce generic sentences. We still need activation/PMI as a second
   objective.
3. **No sentence-start detection.** The walker would need to know
   "I'm at position 0 of a new sentence" vs "I'm mid-phrase." This is
   the structural position awareness problem, still unresolved.
4. **Not all positions are equivalent.** A sentence "I went home"
   is 4 tokens; "She carefully opened the old wooden door" is 9 tokens.
   Normalizing to 0-1 loses length-specific structure.

### Plan for v3

1. Build `.notes/profile_walker.py`
2. Load sentence anatomy profile from disk (cached from this run)
3. Copy v1's `find_targets` → `walk_to_target` structure
4. Replace the beam score inside `walk_to_target` with profile-matching
5. Keep topical bonus (PMI to target)
6. Run on same 16 prompts, compare to main and v2

**This session did not modify the walker.** Only measurement. v3 is the
next session's work, informed by this data.

---

## Files added

| File | Description |
|---|---|
| `.notes/sentence_anatomy.py` | Measurement script — samples corpus, computes profiles |
| `.notes/SENTENCE_ANATOMY.md` | This document |

## Next session

Build `profile_walker.py` implementing profile-matching score. A priori
hypothesis: profile matching reduces output quality variance (no more
"9M weight" spikes) and produces outputs closer to real English on
every measured axis. Test on same 16 prompts, document results, decide
if v3 replaces v2 as the research baseline.
