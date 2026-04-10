# Topology Encodes Language: Separable Grammar and Semantics in Co-occurrence Graphs

**Andy Cufari**
*April 2026*

---

## Abstract

We present Living Language Networks (LLN), a language generation system that uses no gradient descent, no backpropagation, and no learned weight matrices. LLN builds a directed weighted graph from raw text co-occurrence counts and generates language by routing activation through the graph's topology.

The architecture separates two functions that neurolinguistics has long identified as distinct: **semantic targeting** (what to say) via a frequency-penalized PMI activation field, and **grammatical execution** (how to say it) via profile-matching beam search. We demonstrate empirically that these two signals are independently extractable from the same graph: a walker configured for semantic pull produces topically relevant but ungrammatical output, while a walker configured for grammatical momentum produces fluent English sentences without semantic content. The graph encodes both grammar and meaning as separable structures in its topology.

A key empirical finding is that **real English sentences have a measurable topological signature** — a consistent "wave" profile of vocabulary rank, edge weight, and flow ratio at each normalized position. This profile is corpus-invariant across all three training sources. The profile-matching walker scores candidates by distance from this measured wave rather than by maximizing edge weight, producing output whose topological shape matches real sentences.

On a 40-prompt benchmark against a 17.7M-parameter GPT-2 model trained on the same corpus, LLN achieves an 87% win rate on topical relevance (0.488 vs 0.225 average) and vocabulary diversity (95.0% vs 65.5%), while GPT-2 achieves superior grammatical fluency. This tradeoff — topology excels at content selection, neural networks excel at syntactic binding — is a central finding of this work.

The current model (v16) is built from a 32GB blend of three corpora — FineWeb-Edu, Project Gutenberg, and OpenWebText — producing a graph with 117.5 million forward edges, 34.5 million PMI semantic associations, and 11.9 million trigram pairs. The model builds in approximately 3 hours on an 8-core CPU. Generation takes ~0.3-0.6 seconds per prompt. No GPU is required at any stage.

Every generation step is fully traceable: which targets were selected, which were reached, which were organically pruned, and why the system halted. The model is a complete glass box.

---

## 1. Introduction: The Historical Bottleneck

Language generation has been dominated by two paradigms, each with a fundamental flaw.

### 1.1 The Markov Amnesia

Classical n-gram and Markov chain models generate text by following the highest-probability transition from the current token. This works locally — each step is grammatically plausible — but the system has no memory of its goal. After three steps, the walker has forgotten the prompt entirely and orbits through high-frequency function words: `the → of → the → and → the → of`.

This is the **amnesia problem**: without a persistent representation of intent, the generator drifts into the statistical center of the language — the function word attractor basin — and stays there indefinitely.

### 1.2 The Transformer Black Box

Transformer architectures solve the amnesia problem through self-attention: every token attends to every other token, maintaining context across the full sequence. This works extraordinarily well, but at significant cost.

Self-attention requires O(n²) computation per layer, where n is the sequence length. More critically, the model's understanding of language is encoded as continuous-valued weight matrices — millions of floating-point numbers that cannot be individually interpreted. When the model produces an incorrect or nonsensical output, there is no mechanism to determine *why*. The weights that caused the error are distributed across the entire parameter space.

This is the **black box problem**: the system works, but nobody — including the system itself — can explain its decisions. When a transformer hallucinates, the hallucination is structurally indistinguishable from a correct output.

### 1.3 A Third Path

LLN takes a different approach entirely. Instead of learning implicit representations through gradient descent, it builds an explicit graph of observed word transitions and navigates that graph using biologically-inspired routing systems.

The graph is the model. Every edge is a directly observed fact: "word A was followed by word B exactly N times in the training corpus." There are no hidden layers, no learned embeddings, no parameters to tune via optimization. The topology of the graph — which words connect to which, and how strongly — *is* the semantics.

---

## 2. Architectural Motivation: Separating "What" from "How"

The human brain processes language through specialized regions — Wernicke's area (semantic comprehension) and Broca's area (syntactic production) — that perform fundamentally different functions. Damage to one produces a specific deficit: Wernicke's damage yields fluent but meaningless speech; Broca's damage yields meaningful but ungrammatical speech. This dissociation inspired LLN's architecture, though we make no neuroscientific claims — the analogy is structural, not biological.

LLN explicitly separates semantic targeting from grammatical execution:

**Phase 1 (Semantic targeting)**: A frequency-penalized PMI activation field identifies content word targets — the semantic destinations the system should reach. This field is computed once from the prompt and frozen. It does not change as tokens are generated.

**Phase 2 (Routing)**: A flow-aware targeting system classifies the local topology into sources, throughputs, and sinks — dynamically routing the walker toward nodes that can sustain forward momentum.

**Phase 3 (Grammatical execution)**: A profile-matching beam search evaluates multiple competing paths in parallel, scoring candidates by distance from the measured topological signature of real English sentences.

This separation is the core architectural insight. Each system operates on different graph structures, at different timescales, with different objectives.

A key empirical finding of this work (Section 5) is that this separation is not merely structural but functional: when the walker is configured to follow only the grammatical signal (edge weights + trigrams), it produces fluent English; when configured to follow only the semantic signal (PMI activation), it produces topically coherent word sequences. The two signals are independently extractable from the same graph topology.

---

## 3. The Architecture

### 3.1 The Graph

The model is a directed weighted graph G = (V, E, W) where:

- **V** = vocabulary of 100,000 tokens (words + punctuation, case preserved)
- **E** = directed edges representing observed bigram transitions
- **W: E → R+** = edge weights equal to raw co-occurrence counts

From a 32GB blended corpus (6.42 billion bigrams observed), the graph contains:

| Component | Count |
|-----------|-------|
| Vocabulary | 100,000 tokens |
| Forward edges | 117.5M directed |
| PMI edges | 34.5M bidirectional |
| Trigram pairs | 11.9M |
| Total bigrams | 6.42 billion |

The corpus blends three sources to combine different linguistic registers:

- **FineWeb-Edu** (10GB): Modern factual syntax — clean grammar, informational structure
- **Project Gutenberg** (12GB): Narrative momentum — fiction creates rich, diverse edge patterns
- **OpenWebText** (10GB): Conversational diversity — informal patterns, modern idioms

The graph is stored as Compressed Sparse Row (CSR) arrays in LMDB, enabling O(1) edge lookup per token. Total model size: 2.1 GB.

Three parallel edge sets are maintained:
- **Sorted edges**: top-200 edges per node by weight (for candidate generation)
- **Full edges**: all 117.5M edges (for scoring and proximity computation)
- **PMI edges**: 34.5M high-PMI associations (for semantic activation)

### 3.2 Phase 1: Frequency-Penalized PMI Activation

Given a prompt P = [p_1, p_2, ..., p_k], the activation phase constructs a semantic field S:

1. For each content word p_i in P (where in_degree(p_i) < 20,000):
   - Collect all 1-hop PMI neighbors with their PMI weights
2. **Frequency adjustment**: Each candidate's raw PMI score is multiplied by log(1 + in_degree), boosting common words that are also semantically close while suppressing ultra-rare tokens
3. **Capital penalty**: Capitalized tokens (proper nouns, title fragments) receive a 0.3x multiplier unless the prompt consists entirely of proper nouns. This prevents fantasy/sci-fi title collocations ("Dark Ages", "Dark Elf", "Dark Jedi") from dominating the semantic field
4. Pool all adjusted scores across prompt tokens, keeping the maximum per token
5. Retain the top 20-40% by adjusted score → activated set S (~1,000-2,200 tokens)

This field S is **frozen at T=0**. Generated tokens never expand or modify the activation.

**Why frequency-penalized PMI?** Raw PMI is mathematically biased toward rare words. A word appearing 5 times that co-occurs 3 times with "fire" gets higher PMI than "burning" which appears 10,000 times but co-occurs 2,000 times. The frequency multiplier corrects this bias without eliminating rare-but-relevant associations entirely.

The in-degree threshold (20,000) is a topological filter, not a word list. Function words like "the" (in_degree = 81,067) and "of" (in_degree = 84,811) are automatically excluded because their PMI neighbors would flood the field with noise.

### 3.3 Phase 2: Flow-Aware Target Selection

At each generation step, the system selects the next content target from the intersection of:

1. **Semantic activation** (frozen field S from prompt)
2. **Topological reachability** (2-3 forward hops from current position, dual-anchored from both prompt and current token)
3. **Content filtering** (in_degree < 15,000)

Targets are scored by:

```
target_score = adjusted_PMI × (4 - hop_distance) × flow_multiplier
```

The **flow_multiplier** implements dynamic topological mass — a fast proxy for the local push/receive ratio of each candidate:

```
pr_ratio = out_degree / in_degree
```

- **Sinks** (pr_ratio < 0.4): These are topological cul-de-sacs — words that absorb flow but don't push forward. In the v16 blend, most content nouns are sinks (e.g., "flames" pr=0.231, "ashes" pr=0.232). Sinks receive a 0.2x penalty **unless** the walker is in the final 5 tokens of generation, where they serve as natural endpoints.
- **Throughput/sources** (pr_ratio >= 0.9): These words pass or generate flow — they connect well to further content. Examples: "eyes" (pr=0.905), "finally" (pr=0.908). These receive a 1.5x boost.

This mechanism prevents the walker from targeting dead ends early in generation, solving the "1-token halt" problem observed in prompts like "The fire burned" where all PMI-adjacent content words (flames, ashes, extinguisher) are topological sinks.

### 3.4 Phase 3: Profile-Matching Beam Search

Given a target token t with adjusted score PMI_t, the walker navigates from the current position toward t using **beam search with 5 competing paths** and a **profile-matching scoring function**.

#### The Sentence Profile

Measurement of 545 real sentences sampled from all three training corpora revealed that English sentences have a consistent topological signature — a "wave" of vocabulary rank, forward edge weight, and push/receive ratio at each normalized position:

| Position | Median Rank | Mean PR | Mean Fwd Weight |
|----------|------------|---------|-----------------|
| 0.0 (start) | 334 | 0.05 | 433K |
| 0.1-0.3 | 120-134 | 0.03 | 1.0-1.1M |
| 0.4-0.6 | 114-138 | 0.03-0.04 | 1.1-1.5M |
| 0.7-0.9 | 105-136 | 0.03-0.04 | 0.7-1.4M |

This profile is corpus-invariant: fineweb-edu, gutenberg, and openwebtext all produce profiles within 20% on every feature. Key properties:

- **Rank stays in the 105-140 band** throughout the mid-sentence — this is the semi-function/connective zone ("when", "said", "went", "how"), not the deep function word zone (rank 1-50: "the", "of", "and") or deep content zone (rank 500+: "extinguisher", "brightly")
- **Forward weight stays in the 700K-1.5M band** — moderate, not maximum. Real sentences do not walk the heaviest-weight path.
- **Trigram coverage is ~65-72%** — real sentences include ~30% rare trigrams that are not in the graph's trigram set.

#### Profile-Matching Score

At each beam step, every candidate neighbor c is scored by how close its topological features are to the expected profile at its normalized position in the sentence, plus a topical bonus for reaching the target:

```
profile_distance = |log(rank(c)) - log(target_rank)|
                 + 20 × |pr(c) - target_pr|
                 + |log(fwd_weight(c)) - log(target_fwd_weight)|

topical_bonus = proximity(c) × log(1 + PMI_t)

step_score = -profile_distance + 2.0 × topical_bonus
```

Where:
- **profile_distance**: log-space distance for rank and forward weight (they span orders of magnitude), linear-scaled for PR ratio
- **proximity(c)**: c has a forward edge to t → 2.0. c shares outgoing neighbors with t → min(overlap × 0.2, 1.5). Otherwise → 0.0
- **target_fwd_weight** is auto-scaled to the model's edge weight distribution, making the profile portable across models of different corpus size

This scoring function is fundamentally different from maximization-based approaches. Instead of "pick the highest-weight path," it says **"pick the path that looks most like a real sentence at this position."** The sentence shape IS the grammar.

#### Why Profile Matching Works

Analysis of walker outputs (Section 5.3) revealed that the original maximization-based walker produced outputs that were **more extreme than real English on every measurable axis**: 6-9x heavier edge weights, rank 24-81 mid-sentence (should be 105-140), 85-100% trigram coverage (should be 65-72%), and PR ratio 0.01 (should be 0.03-0.05). The walker was addicted to the statistical center of gravity — the highest-weight stock phrases — which is not where real sentences live.

Profile matching corrects all four distortions simultaneously by penalizing candidates that deviate from the measured wave.

#### Beam Mechanics

The beam search evaluates up to **8 steps** with **5 competing paths**. Paths are ranked by **average score per step** — this prevents long paths from winning purely by accumulation and keeps the beam focused on quality over length.

**Target hit**: If any path in the beam reaches the target token, search halts immediately and that path is returned. Failed walks (no path reaches target in 8 steps) return empty — triggering organic pruning.

### 3.5 Phase 4: Depletion and Halting

When a target is reached, it is **depleted**: its PMI score is effectively zeroed in the active field. The activation landscape shifts, and the next-highest remaining target becomes the dominant attractor.

This mirrors **synaptic depression** in neuroscience — the temporary reduction in synaptic efficacy after repeated stimulation.

Additionally, a **synaptic fatigue** curve `1/(1 + 0.15 × chain)` applies to all target scores. Early chains fire at full strength; later chains require progressively stronger PMI to activate. This hyperbolic decay halves at chain ~7 and never reaches zero — the system gradually runs out of steam rather than hitting a hard cutoff.

Three halting conditions exist:

1. **Semantic exhaustion**: all targets in the frozen field have been depleted or pruned
2. **Safety cap**: a configurable maximum token count (default 20)
3. **Complete miss**: all remaining targets are too weak to overcome grammatical highways

---

## 4. Core Mechanisms

### 4.1 Solving Markovian Drift

LLN solves drift by **freezing the activation field at T=0**. The prompt defines the semantic territory. Generated tokens update the walker's *physical position* (which edges are available) but never expand the *semantic goal* (which targets to pursue). Fire stays about fire.

### 4.2 Frequency-Penalized Activation (Solving Rare-Word Hallucination)

In a 32GB multi-corpus model, raw PMI activation for the prompt "Dark clouds" produced targets like "Ages" (Dark Ages), "Elf" (Dark Elf), "Jedi" (Dark Jedi), "Crystal" (Dark Crystal) — all capitalized proper nouns from fantasy fiction titles. These had extremely high raw PMI because they rarely appear outside these collocations.

The frequency-penalized activation replaced these with: cirrus, cumulus, storm, thunder, fog, snow, grey, swirling, thick, leaden — exactly the weather vocabulary a human would associate with "dark clouds."

The fix is mathematically simple: `adjusted = raw_PMI × log(1 + frequency)`. Common words that are also semantically relevant score higher than rare collocations. The capital penalty (0.3x for proper nouns) provides an additional correction for the systematic PMI bias toward capitalized title fragments.

### 4.3 Flow-Aware Routing (Solving the Sink Trap)

Topological analysis revealed that in the v16 blend model, most content words are **topological sinks** — they absorb weight from many sources but don't push forward. "flames" has push/receive ratio 0.231. "ashes" has 0.232. "extinguisher" has 0.113.

The original walker targeted these words first (highest PMI) and immediately got stuck — reaching "alive" in one hop from "burned" but finding no forward path from there. Every remaining target was another sink.

Flow-aware routing penalizes sinks (0.2x) and boosts throughput nodes (1.5x), steering the walker toward words like "eyes" (0.905), "flashed" (0.905), "brightly" (0.416 but reachable through throughput chains). Combined with the profile-matching beam search (Section 3.4), the result: "The fire burned" produces 15 tokens with 8 content hits, including multi-word coherent phrases like "personal safety training camp."

### 4.4 Beam Search (Solving Bridge Blindness)

The greedy walker evaluated exactly one candidate at each step. If the best single step led away from the target, the walker followed it and never recovered. This produced a systematic failure to reach targets that required a temporary low-weight step (a "bridge").

Beam search maintains 5 competing paths. At each of 8 steps, all paths expand simultaneously. The winning path is often one that took a temporarily low-scoring step through a function word to reach a high-value content bridge on the other side.

| Prompt | Greedy (6 steps) | Beam (8 steps, width 5) |
|--------|-----------------|------------------------|
| The king | 0 tokens | 20 tokens |
| The fire burned | 1 token | 15 tokens |
| Dark clouds | 0 tokens | 12 tokens |
| The volcano erupted | 1 token | 7 tokens |
| The ship sailed | 10 tokens | 20 tokens |

### 4.5 The Glass Box

Every LLN generation produces a complete trace:

```
chain 0:  target=extinguisher (PMI=134.21, PR=0.75 [NEUTRAL])  → missed (organic pruning)
chain 1:  target=alive (PMI=28.36, PR=0.02 [SINK])             → reached: alive
chain 5:  target=safety (PMI=11.51, PR=0.02 [SINK])            → reached: with his own personal safety
chain 8:  target=camp (PMI=6.18, PR=0.02 [SINK])               → reached: training camp
chain 10: target=eyes (PMI=4.82, PR=0.02 [SINK])               → reached: with his eyes
chain 11: target=flashed (PMI=8.34, PR=0.04 [SINK])            → reached: flashed
chain 12: target=brightly (PMI=3.96, PR=0.10 [SINK])           → reached: brightly
```

For every token, you can identify: which target it was walking toward, what score justified the walk, the topological role that influenced routing, and whether the target was reached or pruned.

---

## 5. Experiments

### 5.1 Benchmark: LLN vs GPT-2 (Same Corpus)

We compare three systems on 40 prompts spanning nature, history, science, daily life, abstract concepts, and edge cases (2-word prompts):

| System | Parameters | Training | Corpus |
|--------|-----------|----------|--------|
| **LLN** | 0 learned | 5 min (CPU) | web_text + Wikipedia (~500MB) |
| **GPT-2** | 17,735,936 | 8.5 hours (GPU/MPS) | Same |
| **Markov** | 0 | — | Same graph as LLN |

GPT-2 is a custom model trained from scratch (6 layers, 256 embedding, 8 heads) — not a pretrained checkpoint. Both systems had equal access to the same training data.

Scoring uses PMI-based activation relevance: the fraction of output tokens that fall within the prompt's semantic field. This measures topical coherence — does the output relate to what was asked?

| Metric | LLN | GPT-2 | Markov |
|--------|-----|-------|--------|
| **Win Rate** | **35/40 (87%)** | 4/40 (10%) | 0/40 |
| **Avg Relevance** | **0.488** | 0.225 | 0.007 |
| **Content Ratio** | **0.788** | 0.456 | 0.519 |
| **Distinct-1** | **0.950** | 0.655 | 1.000 |

**Scoring bias note:** The relevance metric uses LLN's own PMI activation field. A perplexity-based metric would likely favor GPT-2. The metrics in this paper measure *topical coherence*, not *linguistic fluency*. We include this benchmark to demonstrate that graph-based routing produces non-trivially topical output, not to claim superiority over neural language models.

**Walker evolution note:** The original weight-maximizing walker (v1) scored 75% win rate with 0.510 average relevance. The profile-matching walker improved the win rate to 87% with 0.488 average relevance — a 12 percentage point gain in consistency with comparable per-token relevance. Content ratio increased from 76.1% to 78.8% and vocabulary diversity from 91.7% to 95.0%.

### 5.2 Grammar-Semantics Separability

Adversarial testing with modified walker configurations revealed that the graph encodes grammatical and semantic information as separable structures.

**Experiment:** The beam search step scoring function `step_score = (norm_w × tri_mult) + (proximity × target_pmi)` contains two terms. The first (`norm_w × tri_mult`) reflects grammatical transition strength — high-weight edges and trigram-supported continuations. The second (`proximity × target_pmi`) reflects semantic pull toward activated targets. In the default configuration, the semantic pull term is approximately 25× larger than the grammar term, effectively drowning the grammatical signal.

**Grammar-dominant configuration:** Log-compressing the semantic pull (`proximity × log(1 + target_pmi)`) brings both terms to comparable scale. This produces fluent English clause structure: "They seem to know you want to do it. I mean, and the great care, and I don't expect" — but with near-zero topical relevance to the original prompt. The walker follows high-weight grammatical highways (primarily conversational patterns from Gutenberg fiction) regardless of the semantic activation field.

**Semantics-dominant configuration:** The default walker produces topically dense but ungrammatical output: "northward up the second story goes straight white supremacists marched rapidly and other two young fellow officers came forward" — where nearly every word is PMI-adjacent to the prompt "The army marched" but the syntactic structure is broken.

**Interpretation:** The same graph topology contains both signals. Forward edge weights and trigram patterns encode clause structure, verb argument patterns, and function word sequencing. PMI edges encode semantic neighborhoods and topical associations. These are independently extractable: a walker tuned for one produces strong results on that dimension while performing poorly on the other. Combining them into output that is both grammatical and topically relevant remains an open problem.

### 5.3 Sentence Anatomy: The Topological Wave

To understand why the weight-maximizing walker produced ungrammatical output, we measured the topological signature of real English sentences from the training corpus.

**Method:** 545 clean sentences (5-20 tokens) were sampled from all three corpus files. For each token position, normalized by sentence length, we recorded: vocabulary rank (position in frequency-sorted list), push/receive ratio (out_degree / in_degree), forward edge weight to the next token, and trigram validity (does the 3-word window exist in the trigram graph?).

**Discovery:** Real sentences have a consistent topological "wave" shape that is corpus-invariant — fineweb-edu, gutenberg, and openwebtext all produce profiles within 20% of each other:

| Feature | Real Sentences | Weight-Maximizing Walker | Profile Walker |
|---------|---------------|------------------------|----------------|
| Median rank (mid-sentence) | 105-140 | 24-81 | 135-155 |
| Forward weight (mid-sentence) | 700K-1.5M | 6.5M-9.8M | 200K-400K |
| Trigram coverage | 65-72% | 83-100% | 62-85% |
| Mean PR ratio | 0.03-0.05 | 0.01 | 0.01-0.02 |

**Key finding:** The weight-maximizing walker was **more extreme than real English on every measured axis**. It walked through 6-9x heavier edges (stock phrase attractors), stayed in the top-30 vocabulary ranks (deep function word zone), and used only heavily-attested trigrams (eliminating the natural variety of real text).

**The sentence shape IS the grammar.** Real sentences don't maximize edge weight — they ride a moderate-weight wave in the rank 105-140 band. This band contains connective vocabulary ("when", "said", "which", "went", "how") that forms the structural backbone of English clauses, rather than the extreme function words ("the", "of", "and") that the maximizing walker preferred.

**Profile matching:** Replacing the maximization objective with a profile-distance objective — prefer candidates whose rank, PR, and forward weight are CLOSE to the measured wave at their normalized position — produced a walker that:

1. Wins on 14 of 15 test prompts against the original walker (1 tie, 0 losses)
2. Extends weather-domain prompts from 1 token to 8-19 tokens
3. Produces multi-word coherent units: "my lord hath commanded thee thy people", "personal safety training camp", "main stream flowing blood", "closed doors locked in front door opening"
4. Eliminates known grammatical traps (unmatched quotes, stock-phrase loops)

**This is the opposite of what intuition suggests.** The path to better output was not stronger grammar enforcement or heavier semantic pull — it was matching the moderate, balanced topological shape that real sentences already exhibit. The topology contains the answer; the walker just needed to read it correctly.

### 5.4 Adversarial Prompts

Testing with syntactically adversarial prompts reveals structural limitations of flat graph topology:

| Prompt | Tests | Result |
|--------|-------|--------|
| "the cat that the dog chased ran" | Center-embedded clause, long-range binding | Failure. PMI activates animal vocabulary but the cat→ran subject binding is invisible. |
| "she didn't go to the store because it was closed" | Negation, causal reasoning | Failure. "didn't" filtered as function word (in_degree > 20K). "store" + "closed" activate identical field to "went to the store." |

These failures are structural, not tunable. The frozen PMI field has no polarity mechanism (negation is invisible), no positional encoding (word order is lost after activation), and no hierarchical structure (relative clauses are flattened). These capabilities would require architectural additions beyond the current graph-routing paradigm.

---

## 6. Limitations and Future Work

### 6.1 Honest Limitations

**Grammar**: LLN produces topological word sequences with local grammatical coherence (multi-word phrases, trigram-level structure) but not syntactically correct sentences. The profile-matching walker produces smoother output than the original weight-maximizing walker, but clause-level grammar remains limited. This is the primary limitation.

**No long-range syntax**: The system has no mechanism for center-embedded clauses, negation, pronoun binding, or any syntactic structure that requires maintaining state across intervening material. These require positional encoding or hierarchical representations that flat graph topology does not provide.

**Profile is an average**: The sentence profile is measured across all sentence types and lengths. A walker matching the average profile produces "average English" that may lack the character of specific registers. Per-register profiles (narrative, factual, conversational) would improve output quality for register-specific prompts.

**Scoring bias**: The relevance metric uses LLN's own PMI activation field. A perplexity-based metric would likely favor GPT-2. The metrics in this paper measure *topical coherence*, not *linguistic fluency*.

**Corpus artifacts**: The model faithfully exposes the biases of its training data. OpenWebText leaks web artifacts ("HTML", ".org", ".com"). Gutenberg adds archaic patterns ("hath", "thee"). Non-English tokens (Spanish, Portuguese) from OpenWebText leak through the Latin-alphabet tokenizer.

**GPT-2 comparison**: The GPT-2 model used in benchmarks is a small custom model (17.7M parameters, 6 layers). Larger pretrained models would produce significantly better output. The comparison demonstrates the efficiency of graph-based routing on a controlled corpus, not superiority over the state of the art.

### 6.2 Future Directions

**Adaptive profile**: The sentence profile is currently a single average across all corpora and sentence types. Measuring per-register profiles (gutenberg narrative, fineweb factual, openwebtext conversational) and selecting the closest profile based on the prompt's PMI field signature would allow the walker to match archaic register for "The king" and modern factual register for "Scientists discovered."

**Adaptive parameters from subnetwork topology**: The activated subnetwork has measurable properties (field density, sink fraction, content ratio) that vary per prompt. Coefficients like the topical weight and profile distance weights should be derived from these measurements, not hardcoded globally. Early experiments showed that no single fixed configuration works optimally for both dense fields ("The king") and sparse fields ("Dark clouds").

**Two-phase generation**: Use PMI targeting to select an ordered content skeleton, then use profile-matching walking to infill syntactic structure between content words. The current system does both simultaneously; separating them may allow each phase to optimize for its own objective.

**Structural position awareness**: The profile walker knows "I am at normalized position 0.5 of the projected sentence" but doesn't know "I am inside a noun phrase" or "I need a verb next." Adding topological position detection — recognizing structural roles from the local graph signature — could produce clause-level grammar without explicit POS tagging.

**Full subnetwork mass**: The current flow-aware routing uses a fast proxy (out_degree / in_degree). Computing full local weight sums within the activated subnetwork reveals richer role information. Integrating full subnetwork mass into the walker could enable context-aware grammatical routing.

**LLN as a component**: The PMI activation field, the sentence profile, and the glass-box target selection may have value as components in larger systems, independent of the walker's output quality. The sentence profile could serve as a fluency regularizer for neural decoders. The PMI activation field could provide semantic steering for constrained generation. The glass-box targeting could add an interpretability layer to systems that currently operate as black boxes. Whether these components transfer to cross-linguistic settings (do other languages show the same topological wave?) is an open empirical question.

**Cross-linguistic profile**: The sentence profile was measured on English text. If the topological wave (rank 105-140, moderate forward weight, ~65-72% trigram coverage) is a property of natural language generally — not just English — then profile matching could be a language-agnostic grammar proxy. Testing on non-English corpora would settle this.

**Corpus quality**: Filtering non-English tokens and web artifacts from the training pipeline would improve activation quality. Language detection during vocabulary building and URL/HTML pattern filtering during bigram counting are straightforward additions.

---

## 7. Conclusion

LLN demonstrates that raw co-occurrence topology encodes both semantic content and grammatical structure as separable, independently extractable signals. A directed weighted graph built from counting word pairs — with no gradient descent, no learned weight matrices, and no optimization loop — can produce topically coherent output that outperforms a 17.7M-parameter neural network on semantic relevance metrics across 87% of test prompts.

The sentence anatomy discovery (Section 5.3) reveals that real English sentences have a measurable topological signature — a "wave" of vocabulary rank, edge weight, and flow ratio that is consistent across three independent corpora. This wave is not a human-imposed rule; it's a property of the topology itself, measured directly from the graph. The profile-matching walker exploits this wave to produce output whose topological shape matches real sentences, closing the grammar gap not through rule engineering but through measurement.

The system does not yet produce fully grammatical English sentences. This is an honest limitation. But the trajectory is clear: each walker iteration has produced measurably closer approximations to the sentence profile, with fewer grammatical traps and more multi-word coherent units. The topology contains the answer. The walker is learning to read it.

When the system has nothing meaningful to say, it stops. When it says something unexpected, you can see exactly why.

The meaning is in the structure. The grammar is in the wave.

---

## Appendix A: Reproduction

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned" --verbose
```

The model (2.1 GB) downloads automatically from HuggingFace on first run. Full source code, training scripts, benchmark suite, and experimental walker implementations are available at the project repository.

---

## Appendix B: System Specifications

| Component | Specification |
|-----------|--------------|
| Training hardware | MacBook Pro, 8-core CPU, 16GB RAM |
| Training time | ~3 hours (32GB 3-corpus blend), ~55 min (6.6GB Wikipedia) |
| Inference hardware | Any machine with 4GB+ RAM |
| Inference time | ~0.3-0.6 seconds per prompt (beam search) |
| Model format | LMDB with CSR arrays |
| Dependencies | Python 3.10+, numpy, lmdb |
| Tokenizer | Regex: `[a-zA-Z]+(?:'[a-zA-Z]+)?\|[.,;:!?()\"-]` |

---

*"The meaning is in the structure. The grammar is in the wave. We measured it."*