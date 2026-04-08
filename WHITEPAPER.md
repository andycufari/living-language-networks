# Topology is Semantics: Zero-Parameter Language Generation via Biologically-Inspired Graph Routing

**Andy Cufari**
*April 2026*

---

## Abstract

We present Living Language Networks (LLN), a language generation system that uses zero learned parameters. Instead of gradient descent over millions of weights, LLN builds a directed weighted graph from raw text co-occurrence counts and generates language by routing activation through the graph's topology.

On a 40-prompt benchmark against a 17.7M-parameter GPT-2 model trained on the same corpus, LLN achieves **75% win rate** on topical relevance (0.510 vs 0.225), **76.1% content word ratio** (vs 45.6%), and **91.7% vocabulary diversity** (vs 65.5%) — while producing zero degenerate outputs.

The current model (v16) is trained on a 32GB blend of three corpora — FineWeb-Edu, Project Gutenberg, and OpenWebText — producing a graph with 117.5 million forward edges, 34.5 million PMI semantic associations, and 11.9 million trigram pairs. The model builds in approximately 3 hours on an 8-core CPU. Generation takes ~0.3-0.6 seconds per prompt. No GPU is required at any stage.

The architecture introduces five mechanisms that solve long-standing problems in unconstrained graph walks: **frequency-penalized PMI activation** prevents rare-word hallucination, **flow-aware target selection** avoids topological dead ends, **beam search path competition** finds grammatical bridges across low-weight gaps, **target depletion** eliminates Markovian drift, and **anchored activation** prevents the semantic field from expanding beyond the original prompt. Together, these produce a system where sentence length emerges naturally from the topology — generation halts when semantic energy is exhausted, without a hardcoded maximum.

Every generation step is fully traceable: which targets were selected, which were reached, which were organically pruned, and why the system halted. The model is a glass box.

---

## 1. Introduction: The Historical Bottleneck

Language generation has been dominated by two paradigms, each with a fundamental flaw.

### 1.1 The Markov Amnesia

Classical n-gram and Markov chain models generate text by following the highest-probability transition from the current token. This works locally — each step is grammatically plausible — but the system has no memory of its goal. After three steps, the walker has forgotten the prompt entirely and orbits through high-frequency function words: `the → of → the → and → the → of`.

This is the **amnesia problem**: without a persistent representation of intent, the generator drifts into the statistical center of the language — the function word attractor basin — and stays there indefinitely.

### 1.2 The Transformer Black Box

Transformer architectures solve the amnesia problem through self-attention: every token attends to every other token, maintaining context across the full sequence. This works extraordinarily well, but at significant cost.

Self-attention requires O(n^2) computation per layer, where n is the sequence length. More critically, the model's understanding of language is encoded as continuous-valued weight matrices — millions of floating-point numbers that cannot be individually interpreted. When the model produces an incorrect or nonsensical output, there is no mechanism to determine *why*. The weights that caused the error are distributed across the entire parameter space.

This is the **black box problem**: the system works, but nobody — including the system itself — can explain its decisions. When a transformer hallucinates, the hallucination is structurally indistinguishable from a correct output.

### 1.3 A Third Path

LLN takes a different approach entirely. Instead of learning implicit representations through gradient descent, it builds an explicit graph of observed word transitions and navigates that graph using biologically-inspired routing systems.

The graph is the model. Every edge is a directly observed fact: "word A was followed by word B exactly N times in the training corpus." There are no hidden layers, no learned embeddings, no parameters to tune. The topology of the graph — which words connect to which, and how strongly — *is* the semantics.

---

## 2. Biological Inspiration: Wernicke and Broca

The human brain processes language through two specialized regions that perform fundamentally different functions.

**Wernicke's area**, located in the posterior temporal lobe, handles semantic comprehension. It determines *what* to say — the concepts, the meaning, the topic. Damage to Wernicke's area produces fluent speech that is semantically empty: grammatically correct sentences about nothing.

**Broca's area**, located in the inferior frontal gyrus, handles syntactic production. It determines *how* to say it — word order, grammatical structure, phrase construction. Damage to Broca's area produces semantically coherent but syntactically broken speech: meaningful words without grammatical connectors.

LLN explicitly separates these two functions.

**Phase 1 (Wernicke)**: A frequency-penalized PMI activation field identifies content word targets — the semantic destinations the system should reach. This field is computed once from the prompt and frozen. It does not change as tokens are generated.

**Phase 2 (Routing)**: A flow-aware targeting system classifies the local topology into sources, throughputs, and sinks — dynamically routing the walker toward nodes that can sustain forward momentum.

**Phase 3 (Broca)**: A beam search walker evaluates multiple competing grammatical paths in parallel, finding bridges across low-weight topological gaps that a greedy walker would miss.

This separation is the core architectural insight. The semantic system and the grammatical system operate on different representations, at different timescales, with different objectives.

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

### 3.4 Phase 3: Beam Search Grammar Walk

Given a target token t with adjusted score PMI_t, the walker navigates from the current position toward t using **beam search with 5 competing paths**.

At each step, every path in the beam is expanded by evaluating the top-K forward edges from its current token. Each candidate neighbor c is scored:

```
step_score = (norm_weight(c) × trigram_mult(c)) + (proximity(c) × PMI_t)
```

Where:

- **norm_weight(c)** = log(1 + w_c) / log(1 + w_max), normalized to [0, 1]
- **trigram_mult(c)**: trigram (prev, current) → c. Exists with count N: 1.0 + log(1 + N). Pair exists but c absent: 0.5. No data: 1.0
- **proximity(c)**: c has a forward edge to t → 3.0. c shares outgoing neighbors with t → min(overlap × 0.3, 2.0). Otherwise → 0.0
- **PMI_t**: the target's adjusted PMI score, modulating pull strength

The beam search evaluates up to **8 steps** (increased from 6 in the greedy walker). Paths are ranked by **average score per step** — this prevents long paths from winning purely by accumulation and keeps the beam focused on quality over length.

**Why beam search?** The greedy walker gets trapped by local minima. If the highest-scoring single step leads away from the target, the greedy walker follows it and never recovers. Beam search maintains 5 alternative paths simultaneously, allowing the system to explore a "low-weight bridge" (e.g., stepping through a function word with weak edge weight) that ultimately leads to the target. This is the difference between a chess player evaluating only the best move versus evaluating the best 5 moves — the winner often requires a temporary sacrifice.

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

## 4. Core Breakthroughs

### 4.1 Solving Markovian Drift

LLN solves drift by **freezing the activation field at T=0**. The prompt defines the semantic territory. Generated tokens update the walker's *physical position* (which edges are available) but never expand the *semantic goal* (which targets to pursue). Fire stays about fire.

### 4.2 Frequency-Penalized Activation (Solving Rare-Word Hallucination)

In a 32GB multi-corpus model, raw PMI activation for the prompt "Dark clouds" produced targets like "Ages" (Dark Ages), "Elf" (Dark Elf), "Jedi" (Dark Jedi), "Crystal" (Dark Crystal) — all capitalized proper nouns from fantasy fiction titles. These had extremely high raw PMI because they rarely appear outside these collocations.

The frequency-penalized activation replaced these with: cirrus, cumulus, storm, thunder, fog, snow, grey, swirling, thick, leaden — exactly the weather vocabulary a human would associate with "dark clouds."

The fix is mathematically simple: `adjusted = raw_PMI × log(1 + frequency)`. Common words that are also semantically relevant score higher than rare collocations. The capital penalty (0.3x for proper nouns) provides an additional correction for the systematic PMI bias toward capitalized title fragments.

### 4.3 Flow-Aware Routing (Solving the Sink Trap)

Our topological analysis (TOPOLOGY_DEBUG_V16.md) revealed that in the v16 blend model, most content words are **topological sinks** — they absorb weight from many sources but don't push forward. "flames" has push/receive ratio 0.231. "ashes" has 0.232. "extinguisher" has 0.113.

The old walker targeted these words first (highest PMI) and immediately got stuck — reaching "alive" in one hop from "burned" but finding no forward path from there. Every remaining target was another sink.

Flow-aware routing penalizes sinks (0.2x) and boosts throughput nodes (1.5x), steering the walker toward words like "eyes" (0.905), "flashed" (0.905), "brightly" (0.416 but reachable through throughput chains). The result: "The fire burned" went from 1 token to 15 tokens.

### 4.4 Beam Search (Solving Bridge Blindness)

The greedy walker evaluated exactly one candidate at each step — the locally highest-scoring neighbor. If the best single step led away from the target, the walker followed it and never recovered. This produced a systematic failure to reach targets that required a temporary low-weight step (a "bridge").

Beam search maintains 5 competing paths. At each of 8 steps, all paths expand simultaneously. The winning path is often one that took a temporarily low-scoring step through a function word to reach a high-value content bridge on the other side.

The impact is measurable across all prompts:

| Prompt | Greedy (6 steps) | Beam (8 steps, width 5) |
|--------|-----------------|------------------------|
| The king | 0 tokens | 20 tokens |
| The fire burned | 1 token | 15 tokens |
| Dark clouds | 0 tokens | 12 tokens |
| The volcano erupted | 1 token | 7 tokens |
| The ship sailed | 10 tokens | 20 tokens |

Generation time increases from ~0.1s to ~0.3-0.6s per prompt — an acceptable tradeoff for dramatically improved reachability.

### 4.5 Organic Pruning

Not all targets are reachable, even with beam search. When no path in the beam can reach the target within 8 steps, the walk fails. The target is depleted and the system moves on. This creates a continuous, organic decay threshold: as the field depletes and only weak targets remain, the system naturally halts.

### 4.6 The Glass Box

Every LLN generation produces a complete trace:

```
chain 0:  target=extinguisher (PMI=134.21, 55 remaining) → missed (organic pruning)
chain 1:  target=alive (PMI=28.36, 54 remaining)         → reached: alive
chain 5:  target=safety (PMI=11.51, 33 remaining)        → reached: , and the public safety
chain 8:  target=camp (PMI=6.18, 36 remaining)           → reached: training camp
chain 10: target=eyes (PMI=4.82, 44 remaining)           → reached: with his eyes
chain 11: target=flashed (PMI=8.84, 47 remaining)        → reached: flashed
chain 12: target=brightly (PMI=3.96, 43 remaining)       → reached: brightly
chain 13: target=glowing (PMI=6.93, 65 remaining)        → reached: glowing
chain 14: target=cheeks (PMI=7.96, 71 remaining)         → reached: cheeks
```

For every token, you can identify: which target it was walking toward, what score justified the walk, and whether the target was reached or pruned.

---

## 5. Benchmark Results

### 5.1 Experimental Setup

We compare three systems on 40 prompts spanning nature, history, science, daily life, abstract concepts, and edge cases (2-word prompts):

| System | Parameters | Training | Corpus |
|--------|-----------|----------|--------|
| **LLN** | 0 | 5 min (CPU) | web_text + Wikipedia (~500MB) |
| **GPT-2** | 17,735,936 | 8.5 hours (GPU/MPS) | Same |
| **Markov** | 0 | — | Same graph as LLN |

GPT-2 is a custom model trained from scratch (6 layers, 256 embedding, 8 heads) — not a pretrained checkpoint. Both systems had equal access to the same training data.

Scoring uses PMI-based activation relevance: the fraction of output tokens that fall within the prompt's semantic field. This measures topical coherence — does the output relate to what was asked?

### 5.2 Results

| Metric | LLN | GPT-2 | Markov |
|--------|-----|-------|--------|
| **Win Rate** | **30/40 (75%)** | 8/40 (20%) | 0/40 |
| **Avg Relevance** | **0.510** | 0.225 | 0.007 |
| **Content Ratio** | **0.761** | 0.456 | 0.519 |
| **Distinct-1** | **0.917** | 0.655 | 1.000 |
| **Distinct-2** | **0.925** | 0.817 | 1.000 |
| **Prompt Echo** | **0.028** | 0.199 | 0.000 |
| **Avg Tokens** | 10.3 | 18.5 | 20.0 |
| **Avg Gen Time** | **0.041s** | 0.243s | 0.000s |

Note: Benchmarks were conducted on an earlier model (v13, Wikipedia). The v16 blend model with beam search produces longer, denser output — a new benchmark round is pending.

### 5.3 Notable Outputs

| Prompt | LLN | GPT-2 |
|--------|-----|-------|
| the emperor declared war | crimes committed himself Emperor Justinian II's illegitimate | on the throne of the Empire, and the emperor was the first emperor |
| deep in the forest | fires | a large, narrow, narrow, narrow, narrow, narrow |
| the government decided to | write down upon reaching reforms enacted policies implemented | build a new building, which was built in the late 19th century |
| the doctor examined the patient | confidentiality | patient's body was examined by the surgeon |

### 5.4 GPT-2 Degenerate Modes

Despite 8.5 hours of GPU training, GPT-2 produces degenerate loops on 3 of 40 prompts:

- "deep in the forest" → `a large, narrow, narrow, narrow, narrow, narrow, narrow`
- "the sun set behind" → `the sun was the sun's sun. The sun was the sun's`
- "the ancient city of" → `the city is the largest city in the city. The city is the largest city`

LLN never produces degenerate output. It either generates semantically justified content or halts.

---

## 6. Limitations and Future Work

### 6.1 Honest Limitations

**Grammar**: LLN produces topological word sequences, not syntactically correct sentences. Output like "northward up the second story goes straight white supremacists marched rapidly" contains the right content words in plausible proximity, but does not constitute a grammatical English sentence. GPT-2 produces significantly better syntax.

**Scoring bias**: The relevance metric uses LLN's own PMI activation field. A perplexity-based metric would likely favor GPT-2. The metrics in this paper measure *topical coherence*, not *linguistic fluency*.

**Corpus artifacts**: The model faithfully exposes the biases of its training data. OpenWebText leaks web artifacts ("HTML video streaming"). Gutenberg adds archaic patterns ("hath", "thee"). These are data issues, not algorithmic ones.

**Sink-dominated prompts**: Some prompts (e.g., "The volcano erupted") activate semantic fields where nearly all content words are topological sinks. Flow-aware routing mitigates this but doesn't fully solve it — these prompts still produce shorter output (7 tokens vs 20 for prompts with richer throughput topology).

**GPT-2 comparison**: The GPT-2 model used in benchmarks is a small custom model (17.7M parameters, 6 layers). Larger pretrained models would produce significantly better output. The comparison demonstrates the efficiency of graph-based routing, not superiority over the state of the art.

### 6.2 Future Directions

**Hybrid architecture**: The most promising direction is using LLN as a semantic pre-processor for a small transformer. LLN would provide a topologically-validated word cloud — the content targets, in order, with connector positions marked — and a lightweight language model would format this into syntactically correct English.

**Live learning**: Because the model is a simple co-occurrence graph, new text can be incorporated by incrementally updating edge weights and recomputing PMI. No retraining is needed. This enables models that learn from conversation in real time.

**Full subnetwork mass**: The current flow-aware routing uses a fast proxy (out_degree / in_degree). Our research (TOPOLOGY_DEBUG_V16.md) shows that computing full local weight sums within the activated subnetwork reveals much richer role information — "the" shifts from a balanced node globally to a weight black hole in military context. Integrating full subnetwork mass computation into the walker could enable truly context-aware grammatical routing.

**Multi-hop PMI propagation**: Current activation uses 1-hop PMI from prompt words. Propagating activation through 2-3 PMI hops (with appropriate decay) would produce richer semantic fields for short prompts while maintaining precision.

---

## 7. Conclusion

LLN demonstrates that topical language generation does not require learned parameters. A directed weighted graph, built from raw co-occurrence statistics in hours on a CPU, can outperform a neural network with 17.7 million learned weights on the task of producing topically relevant output.

The key insight is architectural: by separating semantic routing (PMI activation) from grammatical execution (beam search walking), by implementing biologically-inspired mechanisms like synaptic depression (target depletion) and executive attention (PMI-modulated proximity), and by using topological analysis to dynamically classify the landscape (flow-aware routing), we achieve coherent multi-step generation without the amnesia of Markov chains or the opacity of transformers.

The model is a glass box. Every decision is traceable. Every output is justified by observable graph structure. When the system has nothing meaningful to say, it stops.

Topology is semantics. The meaning is in the structure.

---

## Appendix A: Reproduction

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned" --verbose
```

The model (2.1 GB) downloads automatically from HuggingFace on first run. Full source code, training scripts, and benchmark suite are available at the project repository.

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

*"Language is emerging from the topology. We are just cleaning the dirt."*
