# LLN — Living Language Network

**Zero-parameter language generation from pure graph topology.**

No neural networks. No gradient descent. No learned weights.
A directed weighted graph, a PMI activation field, and a biologically-inspired walker.

```bash
pip install numpy lmdb huggingface_hub
python generate.py --prompt "The fire burned"
# → brightly burning low tide the whole story that some idea
```

The model downloads automatically from HuggingFace on first run (~831 MB).

---

## What is this?

LLN builds a graph from raw text: nodes are words, edges are "word A followed word B", weights are co-occurrence counts. That's it — no training loop, no backpropagation, no parameters to learn.

Generation works like the brain's dual language system:

1. **Activate** (Wernicke) — PMI finds content word targets (WHAT to talk about)
2. **Walk** (Broca) — edge weights + trigrams build the grammar path (HOW to say it)
3. **Deplete** — hit a target, zero its activation, landscape shifts to next peak
4. **Halt** — semantic field exhausted, stop naturally

Every decision is traceable. You can see exactly why each word was chosen.

---

## Results

### LLN vs GPT-2 — Same Corpus, Same Prompts

| Metric | LLN | GPT-2 (17.7M params) |
|--------|-----|---------------------|
| **Win Rate** | **30/40 (75%)** | 8/40 (20%) |
| **Avg Relevance** | **0.510** | 0.225 |
| **Content Ratio** | **76.1%** | 45.6% |
| **Diversity (Distinct-1)** | **0.917** | 0.655 |
| **Parameters** | **0** | 17,735,936 |
| **Build Time** | **5 min CPU** | 8.5h GPU |

### Examples

| Prompt | Output | Tokens |
|--------|--------|--------|
| Dark clouds | overhead hung | 2 (natural halt) |
| The fire burned | brightly burning low tide | 10 |
| Scientists discovered | something she might possibly tell how you say we don't think | 16 |
| The volcano erupted | lavas poured out | 14 |
| The army marched | northward...came forward the young king himself ! A hundred miles | 18 |

Sentence length emerges from the topology. No hardcoded maximum.

---

## The Glass Box

```bash
python generate.py --prompt "The fire burned" --verbose
```

```
chain 0: target=brightly (PMI=28.57, 25 remaining)  → reached
chain 1: target=burning (PMI=17.46, 26 remaining)   → reached
chain 2: target=low (PMI=17.31, 23 remaining)       → reached
chain 3: target=tide (PMI=11.11, 14 remaining)      → reached
...
chain 7: target=fact (PMI=3.63, 4 remaining)        → missed (organic pruning)
[halt: semantic field exhausted after 10 tokens]
```

Every target, every hit, every miss, every halt reason — visible.

---

## The Model

| Property | Value |
|----------|-------|
| Vocabulary | 100,000 tokens |
| Forward edges | 55.8M directed |
| PMI edges | 9.5M |
| Trigram pairs | 4.5M |
| Corpus | Wikipedia (6.6GB) |
| Build time | ~55 min CPU |
| Model size | 831 MB |

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

# Multiple files (Wikipedia + books)
python train.py --input wikipedia.txt gutenberg.txt --output model/

# Smaller vocabulary (faster, less memory)
python train.py --input corpus.txt --output model/ --vocab-size 50000

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

### Training time examples

| Corpus | Size | Build Time | Model Size |
|--------|------|-----------|------------|
| Wikipedia (subset) | 500 MB | ~5 min | 179 MB |
| Wikipedia (full) | 6.6 GB | ~55 min | 831 MB |
| Wikipedia + Gutenberg | 7.7 GB | ~65 min | 789 MB |

No GPU needed. Runs on any machine with 8GB+ RAM.

---

## Known limitations

- Output is topological word sequences, not grammatically correct sentences
- Short prompts (2 words) produce broad, sometimes unfocused output
- Grammar quality is below GPT-2 — LLN wins on topic, not on syntax
- Model reflects Wikipedia/Gutenberg biases (19th century literary patterns)

---

## License

MIT — Andy Cufari, 2026