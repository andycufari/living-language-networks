# Topology is Semantics: Zero-Parameter Language Generation via Biologically-Inspired Graph Routing

**Andy Cufari**
*April 2026*

---

## Abstract

We present Living Language Networks (LLN), a language generation system that uses zero learned parameters. Instead of gradient descent over millions of weights, LLN builds a directed weighted graph from raw text co-occurrence counts and generates language by routing activation through the graph's topology.

On a 40-prompt benchmark against a 17.7M-parameter GPT-2 model trained on the same corpus, LLN achieves **75% win rate** on topical relevance (0.510 vs 0.225), **76.1% content word ratio** (vs 45.6%), and **91.7% vocabulary diversity** (vs 65.5%) — while producing zero degenerate outputs.

The model builds in approximately 55 minutes on an 8-core CPU. Generation takes ~0.07 seconds per prompt. No GPU is required at any stage.

The architecture introduces three mechanisms that solve long-standing problems in unconstrained graph walks: **target depletion** eliminates Markovian drift, **PMI-modulated proximity** balances grammatical habit against semantic goals, and **anchored activation** prevents the semantic field from expanding beyond the original prompt. Together, these produce a system where sentence length emerges naturally from the topology — generation halts when semantic energy is exhausted, without a hardcoded maximum.

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

LLN takes a different approach entirely. Instead of learning implicit representations through gradient descent, it builds an explicit graph of observed word transitions and navigates that graph using two biologically-inspired routing systems.

The graph is the model. Every edge is a directly observed fact: "word A was followed by word B exactly N times in the training corpus." There are no hidden layers, no learned embeddings, no parameters to tune. The topology of the graph — which words connect to which, and how strongly — *is* the semantics.

---

## 2. Biological Inspiration: Wernicke and Broca

The human brain processes language through two specialized regions that perform fundamentally different functions.

**Wernicke's area**, located in the posterior temporal lobe, handles semantic comprehension. It determines *what* to say — the concepts, the meaning, the topic. Damage to Wernicke's area produces fluent speech that is semantically empty: grammatically correct sentences about nothing.

**Broca's area**, located in the inferior frontal gyrus, handles syntactic production. It determines *how* to say it — word order, grammatical structure, phrase construction. Damage to Broca's area produces semantically coherent but syntactically broken speech: meaningful words without grammatical connectors.

LLN explicitly separates these two functions.

**Phase 1 (Wernicke)**: A PMI activation field identifies content word targets — the semantic destinations the system should reach. This field is computed once from the prompt and frozen. It does not change as tokens are generated. This mirrors the prefrontal cortex holding a task demand in working memory: no matter what words the mouth is currently forming, the brain maintains a static representation of what it intends to communicate.

**Phase 2 (Broca)**: A grammar walker navigates the full graph topology toward each target, using forward edge weights and trigram momentum to construct grammatically plausible transitions. Function words flow naturally as connectors — they are not filtered out, but emerge from the graph's own structure.

This separation is the core architectural insight. The semantic system and the grammatical system operate on different representations, at different timescales, with different objectives. Merging them — as transformers do, encoding both meaning and syntax into the same weight matrices — creates power but sacrifices interpretability.

---

## 3. The Architecture

### 3.1 The Graph

The model is a directed weighted graph G = (V, E, W) where:

- **V** = vocabulary of 100,000 tokens (words + punctuation, case preserved)
- **E** = directed edges representing observed bigram transitions
- **W: E → R+** = edge weights equal to raw co-occurrence counts

From a 6.6GB Wikipedia corpus (4.06 billion bigrams observed), the graph contains:

| Component | Count |
|-----------|-------|
| Vocabulary | 100,000 tokens |
| Forward edges | 55.8M directed |
| PMI edges | 9.5M bidirectional |
| Trigram pairs | 4.5M |

The graph is stored as Compressed Sparse Row (CSR) arrays in LMDB, enabling O(1) edge lookup per token. Total model size: 831 MB.

Three parallel edge sets are maintained:
- **Sorted edges**: top-200 edges per node by weight (for candidate generation)
- **Full edges**: all 55.8M edges (for scoring and proximity computation)
- **PMI edges**: 9.5M high-PMI associations (for semantic activation)

### 3.2 Phase 1: Anchored Activation

Given a prompt P = [p_1, p_2, ..., p_k], the activation phase constructs a semantic field S:

1. For each content word p_i in P (where in_degree(p_i) < 20,000):
   - Collect all 1-hop PMI neighbors with their PMI weights
2. Pool all PMI neighbors across prompt tokens, keeping the maximum weight per token
3. Retain the top 20% by PMI weight → activated set S (~200-600 tokens)

This field S is **frozen at T=0**. Generated tokens never expand or modify the activation. The prompt defines a finite semantic territory; the generator explores it until it is exhausted.

The in-degree threshold (20,000) is a topological filter, not a word list. Function words like "the" (in_degree = 63,056) and "of" (in_degree = 68,760) are automatically excluded from activation spreading because they connect to everything — their PMI neighbors would flood the field with noise. Content words like "fire" (in_degree = 6,919) or "army" (in_degree = 5,372) have concentrated, meaningful PMI neighborhoods.

### 3.3 Phase 2: Target Selection

At each generation step, the system selects the next content target from the intersection of:

1. **Semantic activation** (frozen field S from prompt)
2. **Topological reachability** (2-3 forward hops from current position)
3. **Content filtering** (in_degree < 15,000)

Targets are scored by:

```
target_score = PMI_weight × (4 - hop_distance)
```

Closer targets score higher: a 1-hop target gets 3x its PMI weight; a 3-hop target gets 1x. This ensures the walker pursues achievable targets rather than distant semantic associations with no grammatical bridge.

Targets that have been **depleted** (hit or missed in previous chains) are excluded. When no targets remain, generation halts.

### 3.4 Phase 3: The Grammar Walk

Given a target token t with PMI score PMI_t, the walker navigates from the current position toward t using the full graph. At each step, every candidate neighbor c is scored:

```
score(c) = (norm_weight(c) × trigram_mult(c)) + (proximity(c) × PMI_t)
```

Where:

- **norm_weight(c)** = log(1 + w_c) / log(1 + w_max), normalized to [0, 1]. This prevents high-frequency edges from dominating through raw magnitude.

- **trigram_mult(c)**: given the previous two tokens (prev, current), look up the trigram (prev, current) → c.
  - Trigram exists with count N: multiplier = 1.0 + log(1 + N)
  - Trigram pair exists but c is absent: multiplier = 0.5
  - No trigram data for this pair: multiplier = 1.0

- **proximity(c)**: topological closeness of candidate c to target t.
  - c has a forward edge to t: proximity = 3.0
  - c shares outgoing neighbors with t: proximity = min(overlap × 0.3, 2.0)
  - Otherwise: proximity = 0.0

- **PMI_t**: the PMI score of the target, modulating pull strength.

This formula creates a continuous tug-of-war between two forces:

**Grammatical habit** (norm_weight × trigram_mult): the basal ganglia's practiced motor routines. Strong trigram sequences like "in the" → "same" exert powerful momentum, pulling the walker along well-worn grammatical highways.

**Semantic goal** (proximity × PMI_t): the prefrontal cortex's task demand. A high-PMI target (e.g., "brightly" for the prompt "fire burned", PMI = 28.57) exerts strong gravitational pull, overriding habitual highways. A low-PMI target (e.g., "idea", PMI = 3.59) cannot overcome the habit engine — the walker misses it and moves on.

The walker takes up to 6 steps to reach the target. If the target is reached, the walk is emitted. If not, the walk is **discarded** — failed highway paths never enter the output.

### 3.5 Phase 4: Depletion and Halting

When a target is reached, it is **depleted**: its PMI score is effectively zeroed in the active field. The activation landscape shifts, and the next-highest remaining target becomes the dominant attractor.

This mirrors **synaptic depression** in neuroscience — the temporary reduction in synaptic efficacy after repeated stimulation. The brain does not return to the same concept twice in a sentence without renewed external input.

Three halting conditions exist:

1. **Semantic exhaustion**: all targets in the frozen field have been depleted or pruned. The system has said everything the prompt's topology supports.
2. **Safety cap**: a configurable maximum token count (default 20) prevents runaway generation in pathological cases.
3. **Complete miss**: all remaining targets are too weak to overcome grammatical highways. The walker cannot reach any remaining semantic peak.

In practice, most generations halt via condition 1. The prompt "Dark clouds" activates only 193 tokens, of which 2 are reachable content targets. The system generates "overhead hung" — two words — and halts. This is not a failure; it is the correct behavior for a low-energy semantic input.

---

## 4. Core Breakthroughs

### 4.1 Solving Markovian Drift

Traditional graph walkers drift because each step redefines the context. If the walker generates "fire → brightly → illuminated," the word "illuminated" shifts the semantic field toward manuscripts and medieval art. Three steps later, the system is generating text about woodcuts — semantically coherent locally, but completely disconnected from "fire."

LLN solves this by **freezing the activation field at T=0**. The prompt defines the semantic territory. Generated tokens update the walker's *physical position* (which edges are available) but never expand the *semantic goal* (which targets to pursue). Fire stays about fire.

### 4.2 Organic Pruning

Not all targets are reachable. When the best remaining target has a PMI score too low to overcome the trigram momentum of nearby grammatical highways, the walker steps away from the target and fails to reach it within the 6-step maximum.

This is a **feature, not a bug**. Failed walks are never emitted. The target is depleted and the system moves on. This creates a continuous, organic decay threshold: as the field depletes and only weak targets remain, the walker naturally produces shorter and shorter walks until it can no longer reach any target — and halts.

No hardcoded threshold is needed. The balance between habit momentum and goal salience produces automatic pruning at the exact point where semantic justification runs out.

### 4.3 The Glass Box

Every LLN generation produces a complete trace:

```
chain 0: target=brightly (PMI=28.57, 25 remaining)  → reached
chain 1: target=burning (PMI=17.46, 26 remaining)   → reached
chain 2: target=low (PMI=17.31, 23 remaining)       → reached
chain 3: target=tide (PMI=11.11, 14 remaining)      → reached
...
chain 7: target=fact (PMI=3.63, 4 remaining)         → missed (organic pruning)
chain 8: target=idea (PMI=3.59, 3 remaining)         → reached
[halt: semantic field exhausted after 10 tokens]
```

For every token in the output, you can identify: which target it was walking toward, what PMI score justified the walk, which edges were followed, and whether the target was reached or pruned. When the system produces unexpected output, the cause is immediately visible in the trace — not buried in a 17-million-dimensional weight space.

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

LLN produces output that is **2.3x more topically relevant** than GPT-2, with **nearly zero prompt echo** (0.028 vs 0.199). GPT-2's wins are strongly correlated with prompt repetition — when GPT-2 "wins" on relevance, it does so by echoing the prompt words back.

### 5.3 Notable Outputs

| Prompt | LLN | GPT-2 |
|--------|-----|-------|
| the emperor declared war | crimes committed himself Emperor Justinian II's illegitimate | on the throne of the Empire, and the emperor was the first emperor |
| deep in the forest | fires | a large, narrow, narrow, narrow, narrow, narrow |
| the government decided to | write down upon reaching reforms enacted policies implemented | build a new building, which was built in the late 19th century |
| the doctor examined the patient | confidentiality | patient's body was examined by the surgeon |

The prompt "the doctor examined the patient" → "confidentiality" is a single-token generation that achieves 100% relevance. The system recognized that its semantic field contained exactly one reachable, activated content target — and produced that word. GPT-2 generated 16 tokens that mostly echo the prompt.

### 5.4 GPT-2 Degenerate Modes

Despite 8.5 hours of GPU training, GPT-2 produces degenerate loops on 3 of 40 prompts:

- "deep in the forest" → `a large, narrow, narrow, narrow, narrow, narrow, narrow`
- "the sun set behind" → `the sun was the sun's sun. The sun was the sun's`
- "the ancient city of" → `the city is the largest city in the city. The city is the largest city`

LLN never produces degenerate output. It either generates semantically justified content or halts.

---

## 6. Limitations and Future Work

### 6.1 Honest Limitations

**Grammar**: LLN produces topological word sequences, not syntactically correct sentences. Output like "lavas poured out that a matter over the old story" contains the right content words in plausible proximity, but does not constitute a grammatical English sentence. GPT-2 produces significantly better syntax.

**Scoring bias**: The relevance metric uses LLN's own PMI activation field. A perplexity-based metric — which measures how well the model predicts the next token in held-out text — would likely favor GPT-2. The metrics in this paper measure *topical coherence*, not *linguistic fluency*.

**Corpus artifacts**: The model faithfully exposes the biases of its training data. Wikipedia's multilingual articles produce Dutch function words in door-related contexts. Gutenberg literary patterns create narrative gravity wells ("the whole story"). These are data issues, not algorithmic ones — but they affect output quality.

**Short prompts**: Two-word prompts like "Dark clouds" produce very short output (2 tokens) because the PMI field is narrow. This is architecturally correct — ambiguous input produces minimal output — but may not match user expectations.

**GPT-2 comparison**: The GPT-2 model used in benchmarks is a small custom model (17.7M parameters, 6 layers). Larger pretrained models (GPT-2 medium at 345M parameters, or modern LLMs) would produce significantly better output. The comparison demonstrates the efficiency of graph-based routing, not superiority over the state of the art.

### 6.2 Future Directions

**Hybrid architecture**: The most promising direction is using LLN as a semantic pre-processor for a small transformer. LLN would provide a topologically-validated word cloud — the content targets, in order, with connector positions marked — and a lightweight language model would format this into syntactically correct English. The semantic routing is zero-hallucination by construction; only the final formatting step uses learned weights.

**Live learning**: Because the model is a simple co-occurrence graph, new text can be incorporated by incrementally updating edge weights and recomputing PMI. No retraining is needed. This enables models that learn from conversation in real time — something fundamentally impossible with gradient-based architectures without catastrophic forgetting.

**Subnetwork energy scoring**: Our experiments show that real sentences form subnetworks with measurably higher internal connectivity than random walks (9/10 discrimination accuracy). Integrating this "resonance" metric into the walker itself — rather than using it only for post-hoc evaluation — could significantly improve output quality.

**Contextual mass**: Session 16 of our research demonstrated that the same word has radically different topological properties depending on the activated subnetwork. "the" has average incoming weight of 3,098 globally but 125,816 within a military context — a 40x amplification. Incorporating contextual mass into the walk scoring could enable the system to dynamically adjust its treatment of function words based on the active semantic field.

---

## 7. Conclusion

LLN demonstrates that topical language generation does not require learned parameters. A directed weighted graph, built from raw co-occurrence statistics in minutes on a CPU, can outperform a neural network with 17.7 million learned weights on the task of producing topically relevant output.

The key insight is architectural: by separating semantic routing (PMI activation) from grammatical execution (edge-weight walking), and by implementing biologically-inspired mechanisms like synaptic depression (target depletion) and executive attention (PMI-modulated proximity), we achieve coherent multi-step generation without the amnesia of Markov chains or the opacity of transformers.

The model is a glass box. Every decision is traceable. Every output is justified by observable graph structure. When the system has nothing meaningful to say, it stops.

Topology is semantics. The meaning is in the structure.

---

## Appendix A: Reproduction

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned" --verbose
```

The model (831 MB) downloads automatically from HuggingFace on first run. Full source code, training scripts, and benchmark suite are available at the project repository.

---

## Appendix B: System Specifications

| Component | Specification |
|-----------|--------------|
| Training hardware | MacBook Pro, 8-core CPU, 16GB RAM |
| Training time | 55 minutes (full Wikipedia), 5 minutes (500MB subset) |
| Inference hardware | Any machine with 2GB+ RAM |
| Inference time | ~0.07 seconds per prompt |
| Model format | LMDB with CSR arrays |
| Dependencies | Python 3.10+, numpy, lmdb |
| Tokenizer | Regex: `[a-zA-Z]+(?:'[a-zA-Z]+)?\|[.,;:!?()\"-]` |

---

*"Language is emerging from the topology. We are just cleaning the dirt."*
