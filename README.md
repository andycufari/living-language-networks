# LLN — Living Language Network

**Zero-parameter language generation from pure graph topology.**

No neural networks. No gradient descent. No learned weights.
A directed weighted graph, a PMI activation field, and a biologically-inspired walker with beam search.

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned"
# → alive , and the public safety training camp with his eyes flashed brightly glowing cheeks
```

The model downloads automatically from HuggingFace on first run (~2.1 GB).

---

## What is this?

LLN builds a graph from raw text: nodes are words, edges are "word A followed word B", weights are co-occurrence counts. That's it — no training loop, no backpropagation, no parameters to learn.

Generation works like the brain's dual language system:

1. **Activate** (Wernicke) — Frequency-penalized PMI finds content word targets (WHAT to talk about)
2. **Route** — Flow-aware target selection avoids topological dead ends (WHERE to aim)
3. **Walk** (Broca) — Beam search across competing paths builds the grammar bridge (HOW to get there)
4. **Deplete** — Hit a target, zero its activation, landscape shifts to next peak
5. **Halt** — Semantic field exhausted, stop naturally

Every decision is traceable. You can see exactly why each word was chosen.

---

## Results

### Examples (v16 — 3-corpus blend)

| Prompt | Output | Tokens |
|--------|--------|--------|
| The king | a more powerful voice heard my lord hath commanded thee thy life . The most famous letter addressed a | 20 |
| She opened the door | opened fire on the door swung open . She paused abruptly closed the " " She laughed softly closed doors | 20 |
| The fire burned | alive , and the public safety training camp with his eyes flashed brightly glowing cheeks | 15 |
| The river flows | south bank of the most beautiful valley bottoms of water flow . The main stream flowing blood | 17 |
| The ship sailed | northward along the right to go straight to him so forth a big leagues farther inland navigation channel the whole | 20 |
| Dark clouds | that this matter is called a small village green leafy green energy | 12 |
| The army marched | northward up the second story goes straight white supremacists marched rapidly and other two young fellow officers came forward an | 20 |
| Scientists discovered | they would not believe that he began studying all I hope you don't expect to think you find you want | 20 |
| The volcano erupted | violently excited crowd s largest ethnic violence | 7 |

Sentence length emerges from the topology. No hardcoded maximum.

### LLN vs GPT-2 — Same Corpus, Same Prompts

| Metric | LLN | GPT-2 (17.7M params) |
|--------|-----|---------------------|
| **Win Rate** | **30/40 (75%)** | 8/40 (20%) |
| **Avg Relevance** | **0.510** | 0.225 |
| **Content Ratio** | **76.1%** | 45.6% |
| **Diversity (Distinct-1)** | **0.917** | 0.655 |
| **Parameters** | **0** | 17,735,936 |
| **Build Time** | **~3h CPU** | 8.5h GPU |

---

## The Glass Box

```bash
python generate.py --prompt "The fire burned" --verbose
```

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

Every target, every hit, every miss, every halt reason — visible.

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

- **FineWeb-Edu** (10GB) — Modern factual syntax. Clean grammar, informational structure. Provides the backbone of well-formed word transitions.
- **Gutenberg** (12GB) — Narrative momentum. Fiction creates rich, diverse edge patterns. Words like "whispered", "brightly", "glowing" gain strong PMI connections. This is where the prose comes alive.
- **OpenWebText** (10GB) — Conversational diversity. Informal patterns, contractions, modern idioms. Fills gaps between the formal registers of the other two.

The blend produces a graph with 3.3x more observed bigrams than any single corpus, and 4.2x more PMI semantic edges. This density is critical — a denser graph means the beam search walker can find more grammatical bridges between semantic targets.

---

## How to use

```bash
# Single prompt
python generate.py --prompt "The king ruled"

# With verbose tracing
python generate.py --prompt "The ship sailed" --verbose

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
- Grammar quality is below GPT-2 — LLN wins on topic, not on syntax
- Some corpus artifacts leak through (OpenWebText HTML fragments, Gutenberg archaisms)
- Volcano/eruption-type prompts where all semantic targets are topological sinks still produce shorter output

---

## License

MIT — Andy Cufari, 2026
