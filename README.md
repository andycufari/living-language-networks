# LLN: Living Language Network

**Zero-parameter language generation from pure graph topology. With live learning.**

No neural networks. No gradient descent. No learned weights.
A directed weighted graph, a PMI activation field, a biologically-inspired walker, and an episodic memory that learns in real time.

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned"
# → alive . The government of the state police department s own personal safety training camp with his eyes flashed brightly
```

The model downloads automatically from HuggingFace on first run (~2.1 GB).

---

## What is this?

LLN builds a graph from raw text: nodes are words, edges are "word A followed word B", weights are co-occurrence counts. That's it. No training loop, no backpropagation, no parameters to learn.

Generation works like the brain's language system:

1. **Activate** (Wernicke) Frequency-penalized PMI finds content word targets (WHAT to talk about)
2. **Route** Flow-aware target selection avoids topological dead ends (WHERE to aim)
3. **Walk** (Broca) Beam search across competing paths builds the grammar bridge (HOW to get there)
4. **Remember** (Hippocampus) A Delta Graph overlay provides O(1) short-term episodic memory, allowing real-time learning that overrides base habits
5. **Deplete** Hit a target, zero its activation, landscape shifts to next peak
6. **Halt** Semantic field exhausted, stop naturally

Every decision is traceable. You can see exactly why each word was chosen.

---

## Results

### Examples (v16, 3-corpus blend)

| Prompt | Output | Tokens |
|--------|--------|--------|
| The king | a more powerful voice heard my lord hath commanded thee thy life . The most famous letter addressed a | 20 |
| She opened the door | opened fire on the door swung open . She paused abruptly closed his lips , she said Jack Ruby smiled | 20 |
| The fire burned | alive . The government of the state police department s own personal safety training camp with his eyes flashed brightly | 20 |
| The river flows | south bank of the most beautiful valley bottoms of water flow . The main stream flowing blood | 17 |
| The ship sailed | northward along the right to go straight to him so forth a big leagues farther inland navigation channel the whole | 20 |
| Dark clouds | that this matter is called a small village green leafy green energy | 12 |
| The army marched | northward up the second story goes straight white supremacists marched rapidly and other two young fellow officers came forward an | 20 |
| Scientists discovered | they would not believe that he began studying all I hope you don't expect to think you find you want | 20 |
| The volcano erupted | violently excited crowd s largest ethnic violence | 7 |
| The old man walked | beside him go straight away so very much more advanced rapidly . A woman in my old lady . A young gentleman , the old chap , and a young man's best friend of the same age . My dear | 40 |

**About token counts:** Generation runs with a soft cap of 20 tokens by default. Some prompts hit it, some halt naturally before it (the walker stops when the semantic field is exhausted). The volcano, at 7 tokens, is an example of a sink-dominated prompt where the topology runs dry fast. "The old man walked" was run at 40 tokens to show how longer prompts with rich topology keep going. The cap is configurable via `--max-tokens`.

### LLN vs GPT-2: Same Corpus, Same Prompts

40-prompt blind evaluation (details in [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md)):

| Metric | LLN | GPT-2 (17.7M params) |
|--------|-----|---------------------|
| **Win Rate** | **30/40 (75%)** | 8/40 (20%) |
| **Avg Relevance** | **0.510** | 0.225 |
| **Content Ratio** | **76.1%** | 45.6% |
| **Diversity (Distinct-1)** | **0.917** | 0.655 |
| **Parameters** | **0** | 17,735,936 |
| **Build Time** | **~3h CPU** | 8.5h GPU |

LLN wins on topical relevance and lexical diversity. GPT-2 wins on grammar. This is the tradeoff: LLN generates topologically correct word sequences, not grammatically correct sentences.

---

## The Glass Box

```bash
python generate.py --prompt "The fire burned" --verbose
```

```
chain 0:  target=extinguisher (PMI=134.21, PR=0.75 [NEUTRAL]) → missed (organic pruning)
chain 1:  target=alive (PMI=28.36, PR=0.02 [SINK])            → reached: alive
chain 5:  target=safety (PMI=11.51, PR=0.02 [SINK])           → reached: , and the public safety
chain 8:  target=camp (PMI=6.18, PR=0.02 [SINK])              → reached: training camp
chain 10: target=eyes (PMI=24.08, PR=0.02 [SINK])             → reached: with his eyes
chain 11: target=flashed (PMI=41.71, PR=0.04 [SINK])          → reached: flashed
chain 12: target=brightly (PMI=19.82, PR=0.10 [SINK])         → reached: brightly
```

Every target, every hit, every miss, every halt reason is visible. The `PR` tag shows the topological mass classification (SINK/THROUGHPUT/SOURCE/NEUTRAL) that determines routing priority.

---

## Live Learning: Zero-Gradient O(1) Memory

LLN features an episodic memory overlay (a "Delta Graph"). It can learn new facts instantly without backpropagation, and route through them without catastrophic forgetting of the base graph.

```bash
python living.py
```

```
>>> GENERATE: The terrifying monster
  chain 0: target=ordeal (PMI=30.44, PR=0.14 [SINK])     → missed
  chain 1: target=truck (PMI=24.18, PR=0.05 [SINK])      → reached: truck
  ...
  → truck . One important aspect of these little fellow creatures
  (Base graph only. No knowledge of space monsters)

>>> LEARN: The glorflax is a terrifying space monster that lurks behind dark nebula clouds
  Learned 12 bigrams
  Content words linked: monster, terrifying, space, lurks, nebula, clouds, dark, prey

>>> GENERATE: The terrifying monster
  chain 0: target=lurks (PMI=3220.98, PR=0.23 [SINK])    → reached: lurks
  ...
  → lurks behind every aspect of these little fellow creatures
  (Delta Graph routes through "lurks behind", learned 0.01s ago!)

>>> FORGET
  Short-term memory cleared.

>>> GENERATE: The terrifying monster
  → truck . One important aspect of these little fellow creatures
  (Back to base graph. Zero catastrophic forgetting.)
```

The base graph (117.5M edges from 32GB of text) acts as **semantic memory**: deep, slow, permanent. The Delta Graph acts as **episodic memory**: fast, volatile, overriding. Just like the hippocampus overrides cortical habits with recent experience.

**Try it yourself:**

```bash
python living.py

# Teach it something it doesn't know
>>> LEARN: The purple elephant danced gracefully through the moonlit garden

# Now generate (it routes through the learned edges)
>>> GENERATE: The purple elephant

# Clear memory (back to base behavior)
>>> FORGET
```

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

- **FineWeb-Edu** (10GB) Modern factual syntax. Clean grammar, informational structure. Provides the backbone of well-formed word transitions.
- **Gutenberg** (12GB) Narrative momentum. Fiction creates rich, diverse edge patterns. Words like "whispered", "brightly", "glowing" gain strong PMI connections. This is where the prose comes alive.
- **OpenWebText** (10GB) Conversational diversity. Informal patterns, contractions, modern idioms. Fills gaps between the formal registers of the other two.

The blend produces a graph with 3.3x more observed bigrams than any single corpus, and 4.2x more PMI semantic edges. This density is critical: a denser graph means the beam search walker can find more grammatical bridges between semantic targets.

---

## How to use

```bash
# Single prompt
python generate.py --prompt "The king ruled"

# With verbose tracing
python generate.py --prompt "The ship sailed" --verbose

# Interactive live learning mode
python living.py

# Custom model
python generate.py --model path/to/your/model.lmdb --prompt "Hello world"

# Control length
python generate.py --prompt "The experiment showed" --max-tokens 30
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

- Output is topological word sequences, not grammatically correct sentences
- Grammar quality is below GPT-2. LLN wins on topic, not on syntax
- Some corpus artifacts leak through (OpenWebText HTML fragments, Gutenberg archaisms)
- Sink-dominated prompts (volcano/eruption) still produce shorter output
- Live learning requires in-vocab words (OOV words like "glorflax" are skipped)

---

## License

MIT, Andy Cufari, 2026
