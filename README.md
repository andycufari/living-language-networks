# LLN: Living Language Network

**Language generation from pure graph topology. No neural networks. No gradient descent.**

A directed weighted graph, a PMI activation field, and a biologically-inspired walker that separates *what to say* from *how to say it* — and lets you watch every decision.

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned" --verbose
```

```
chain 0:  target=extinguisher (PMI=134.21, PR=0.75 [NEUTRAL])  → missed (organic pruning)
chain 1:  target=alive (PMI=28.36, PR=0.02 [SINK])             → reached: alive
chain 2:  target=department (PMI=20.83, PR=0.03 [SINK])        → reached: . On the state police department
chain 5:  target=safety (PMI=11.51, PR=0.02 [SINK])            → reached: of them to their own personal safety
chain 6:  target=protection (PMI=9.93, PR=0.02 [SINK])         → reached: and for you will provide protection
```

Every word is traceable. Every target, every hit, every miss, every halt reason — visible. The model downloads automatically from HuggingFace on first run (~2.1 GB).

---

## What is this?

LLN builds a directed weighted graph from raw text: nodes are words, edges encode "word A followed word B", weights are co-occurrence counts. No training loop, no backpropagation, no weight matrices.

Generation uses a biologically-inspired two-system architecture:

1. **Activate** (Wernicke's area) — Frequency-penalized PMI identifies content word targets: WHAT to talk about
2. **Route** — Flow-aware target selection classifies nodes as sinks/throughputs/sources, avoids topological dead ends
3. **Walk** (Broca's area) — Profile-matching beam search: candidates scored by distance from the measured topological signature of real English sentences. The walker rides the sentence "wave" — matching the rank, weight, and flow profile that corpus measurement revealed
4. **Deplete** — Hit a target, zero its activation, the landscape shifts to the next semantic peak
5. **Halt** — Semantic field exhausted, stop naturally

The separation of semantic targeting (PMI activation) from grammatical execution (beam search walking) is the core architectural insight. Each system operates on different graph structures, at different timescales, with different objectives.

---

## The Key Finding

**Grammar and semantics are independently encoded in co-occurrence topology.**

Adversarial testing with different walker configurations revealed that the same graph contains two separable signals:

- **Semantic signal** lives in PMI edges (34.5M associations). Activating the PMI field for "fire burned" correctly identifies fire-related vocabulary: flames, ashes, extinguisher, smoke, brightly.
- **Grammatical signal** lives in forward edges (117.5M bigrams) + trigrams (11.9M). Following high-weight edge chains produces fluent English clause structure: "They seem to know you want to do it. I mean, and the great care."

When the walker prioritizes semantic pull, it produces topically relevant word sequences with broken grammar. When it prioritizes grammatical momentum, it produces fluent English about nothing in particular. Combining these signals — generating grammatical sentences that stay on topic — is an active area of research (see [Experimental Walkers](#experimental-walkers) below).

This separability is not obvious. It suggests that raw co-occurrence statistics encode both *what words mean* (PMI neighborhoods) and *how words combine* (transition probabilities and trigram patterns) as distinct, extractable structures in the same graph.

---

## The Glass Box

When a Transformer hallucinates, you can't easily find the neuron responsible. When LLN outputs something strange, you turn on `--verbose`:

```bash
python generate.py --prompt "The fire burned" --verbose
```

```
chain 0:  target=extinguisher (PMI=134.21, PR=0.75 [NEUTRAL])  → missed (organic pruning)
chain 1:  target=alive (PMI=28.36, PR=0.02 [SINK])             → reached: alive
chain 2:  target=department (PMI=20.83, PR=0.03 [SINK])        → reached: . On the state police department
chain 5:  target=safety (PMI=11.51, PR=0.02 [SINK])            → reached: of them to their own personal safety
chain 6:  target=protection (PMI=9.93, PR=0.02 [SINK])         → reached: and for you will provide protection
```

Every target selection shows the PMI score that justified the walk and the topological role (SINK/THROUGHPUT/SOURCE) that determined routing priority. You can trace exactly why "extinguisher" was attempted and failed (organic pruning, no beam path reached it in 8 steps), why "alive" succeeded (1-hop from "burned"), and how the walker bridged to "personal safety" and "provide protection" — fire-domain vocabulary reached through profile-matching grammar paths. The model is a glass box.

---

## Examples (v16, 3-corpus blend, profile walker)

| Prompt | Output | Tokens |
|--------|--------|--------|
| The king | , so much more powerful and he heard my lord hath commanded thee thy name of my boy whom he | 20 |
| She opened the door | opened fire . She paused , his eyes closed the " She laughed softly in front door opening the city | 20 |
| The fire burned | alive . On the state police department of them to their own personal safety and for you will provide protection | 20 |
| The river flows | north bank of her the way to go far south of water flow of its main stream flowing blood flow | 20 |
| The ship sailed | northward across the way to go far south west of him to go straight to come forth a big leagues | 20 |
| Dark clouds | that this matter of an increase energy | 7 |
| The army marched | northward to go straight white supremacists and so much more advanced rapidly , his brother officers came forward . An | 20 |
| Scientists discovered | that we believe I hope you would expect to think you find the people want more easily identified as they | 20 |
| The volcano erupted | violently against the world war against the s largest of such a long - based violence | 16 |
| The old man walked | beside him to go straight away , so much more advanced rapidly . No woman in my dear lady , his old gentleman who have the old chap . With a young man's best friend of any age of my | 40 |

These outputs are topologically guided word sequences. The profile-matching walker produces multi-word coherent units ("my lord hath commanded thee thy", "his brother officers came forward", "main stream flowing blood flow") by matching the measured topological signature of real English sentences, rather than maximizing edge weight. See [Known Limitations](#known-limitations) for an honest assessment.

**About token counts:** Generation runs with a soft cap of 20 tokens by default. Some prompts halt naturally before it (the walker stops when the semantic field is exhausted). "The old man walked" was run at 40 tokens to show extended generation. The cap is configurable via `--max-tokens`.

---

## Benchmark: LLN vs GPT-2

40-prompt blind evaluation on a shared corpus (details in [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md)):

| Metric | LLN | GPT-2 (17.7M params) |
|--------|-----|---------------------|
| Win Rate | **35/40 (87%)** | 4/40 (10%) |
| Avg Relevance | **0.488** | 0.225 |
| Content Ratio | **78.8%** | 45.6% |
| Diversity (Distinct-1) | **0.950** | 0.655 |
| Parameters | 0 learned | 17,735,936 |
| Build Time | ~3h CPU | 8.5h GPU |

**Important context:** The GPT-2 model is a custom 17.7M-parameter model trained from scratch (6 layers, 256 embedding, 8 heads) on the same ~500MB corpus — not a pretrained GPT-2 checkpoint. Both systems had equal access to the same training data. Transformers improve dramatically with scale; at 500MB, the GPT-2 model hasn't seen enough data to generalize fully. The benchmark measures **data efficiency** — which architecture extracts more useful structure from limited data — not superiority over large-scale language models. Relevance is measured by PMI activation overlap, applied equally to all systems. A perplexity-based metric would favor GPT-2.

LLN wins on *what* it says (topical relevance, lexical diversity, content density) and *how consistently* it says it (87% win rate, 95% vocabulary diversity). GPT-2 trained on 500MB collapses into repetitive loops on most prompts (average distinct-1: 0.655 vs LLN's 0.950). The benchmark demonstrates data efficiency: a graph that directly encodes every observed bigram extracts more useful structure from limited data than a transformer that must approximate the same patterns through 17.7 million learned parameters.

---

## Ongoing Research

Adversarial testing reveals that the graph encodes grammar and semantics as separable structures (see whitepaper, Section 5.2). The profile-matching walker addresses this by scoring candidates against the measured topological signature of real English sentences — matching the "wave" of rank, weight, and flow that corpus analysis revealed. Improving long-range syntactic structure and sentence-boundary detection are active areas of work.

---

## The Model

| Property | Value |
|----------|-------|
| Vocabulary | 100,000 tokens |
| Forward edges | 117.5M directed |
| PMI edges | 34.5M |
| Trigram pairs | 11.9M |
| Total bigrams | 6.42 billion |
| Corpus | FineWeb-Edu (10GB) + Gutenberg (12GB) + OpenWebText (10GB) |
| Build time | ~3 hours CPU |
| Model size | 2.1 GB |

### Why blend three corpora?

Each corpus contributes a different dimension to the graph's topology:

- **FineWeb-Edu** (10GB) — Modern factual syntax. Clean grammar, informational structure. The backbone of well-formed word transitions.
- **Gutenberg** (12GB) — Narrative momentum. Fiction creates rich, diverse edge patterns. Words like "whispered", "brightly", "glowing" gain strong PMI connections.
- **OpenWebText** (10GB) — Conversational diversity. Informal patterns, contractions, modern idioms. Fills gaps between the formal registers of the other two.

The blend produces a graph with 3.3x more observed bigrams than any single corpus, and 4.2x more PMI semantic edges. This density is critical: a denser graph means more paths exist between semantic targets.

---

## How to use

```bash
# Single prompt
python generate.py --prompt "The king ruled"

# Multiple prompts (model loads once)
python generate.py --prompt "The fire burned" "The king" "Dark clouds"

# Interactive mode (model loads once, type prompts in a loop)
python generate.py --interactive

# With verbose tracing
python generate.py --prompt "The ship sailed" --verbose

# Custom model
python generate.py --model path/to/your/model.lmdb --prompt "Hello world"

# Control length
python generate.py --prompt "The experiment showed" --max-tokens 30

# Run all default demo prompts
python generate.py
```

---

## Train your own model

Build a graph from any English text:

```bash
# Single file
python train.py --input corpus.txt --output model/

# Multiple files (blended corpus)
python train.py --input fineweb.txt gutenberg.txt openwebtext.txt --output model/

# Smaller vocabulary (faster, less memory)
python train.py --input corpus.txt --output model/ --vocab-size 50000

# Large files: chunked processing with checkpointing (crash-safe)
python train.py --input bigfile.txt --output model/ --chunk-size 2

# Then generate with your model
python generate.py --model model/ --prompt "Your prompt here"
```

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | One or more text files |
| `--output` | `model/` | Output LMDB directory |
| `--vocab-size` | 100000 | Keep top N tokens by frequency |
| `--min-freq` | 10 | Minimum token frequency to include |
| `--topk` | 200 | Top-K edges per node (for beam search) |
| `--pmi-min-count` | 5 | Minimum bigram count for PMI edges |
| `--tri-min-count` | 3 | Minimum trigram count to keep |
| `--chunk-size` | 0 | Process in N-GB chunks with checkpointing (0 = all in RAM) |

### Training time examples

| Corpus | Size | Build Time | Model Size |
|--------|------|-----------|------------|
| Wikipedia (subset) | 500 MB | ~5 min | 179 MB |
| Wikipedia (full) | 6.6 GB | ~55 min | 831 MB |
| 3-corpus blend | 32 GB | ~3 hours | 2.1 GB |

No GPU needed. The chunked trainer processes arbitrarily large corpora on machines with 8GB+ RAM.

---

## Known limitations

**Grammar:** Output is topological word sequences, not syntactically correct sentences. The walker produces local grammatical fragments (trigram-level coherence) connected by semantic transitions. Grammar quality is below any neural language model, including small GPT-2 variants. This is the primary limitation.

**No long-range syntax:** The system cannot process center-embedded clauses ("the cat that the dog chased ran"), negation ("she didn't go"), or pronoun binding. These require positional encoding or hierarchical structure that flat graph topology does not provide.

**No word-sense disambiguation:** "bank" (river) and "bank" (financial) share a single node. The PMI field activates neighbors from all senses simultaneously. There is no attention mechanism to select the contextually appropriate sense.

**Scoring bias:** The relevance metric in benchmarks uses LLN's own PMI activation field. A perplexity-based metric would favor GPT-2. The benchmarks measure topical coherence, not linguistic fluency.

**Corpus artifacts:** The model faithfully exposes biases in training data. OpenWebText leaks web fragments (HTML, URLs, non-English tokens). Gutenberg adds archaic patterns ("hath", "thee"). These are data issues, not algorithmic ones.

**Sink-dominated prompts:** Some prompts (e.g., "The volcano erupted") activate semantic fields where most content words are topological sinks (high in-degree, low out-degree). These prompts produce shorter output because the walker can reach sinks but can't chain forward from them.

---

## License

MIT, Andy Cufari, 2026