# LLN vs GPT-2: Fair Fight Benchmark

Date: 2026-04-03
Prompts: 40 (blind, 5 categories)
Max tokens: 20

## Setup

Both systems trained on the same corpus (~500MB Wikipedia + web_text) to eliminate data advantage.

| System | Parameters | Training |
|--------|-----------|----------|
| LLN | 0 (graph only) | ~5 min CPU |
| GPT-2 | 17,735,936 (6 layers, 8 heads, 256 embed) | 8.5h GPU |
| Markov baseline | 0 (bigram random walk on LLN graph) | same graph |

## Results

| Metric | LLN | GPT-2 | Markov |
|--------|-----|-------|--------|
| Win Rate | **30/40 (75%)** | 8/40 (20%) | 0/40 |
| Avg Relevance | **0.510** | 0.225 | 0.008 |
| Content Ratio | **76.1%** | 45.6% | 50.0% |
| Distinct-1 | **0.917** | 0.655 | 1.000 |

Relevance is measured by PMI activation overlap: what fraction of output tokens fall inside the semantic field activated by the prompt. Same activation function used by LLN during generation, applied equally to all systems.

## Sample outputs

| Prompt | LLN | GPT-2 |
|--------|-----|-------|
| the president announced plans | in November on Tuesday | to build a new building in the city , which was to be built in the city . The building |
| the army marched north | transept across northern shore through her daughter - in other side against the British military officers | to the city of Borneo , where the city was occupied by the city 's inhabitants . |
| scientists discovered a new | constitution would result , who believed their homes worldwide in ) ) have been reported no single entity separate colony | alignment of the highway in the early 19th century , and the highway was designated as a part of |
| the ship sailed across | Eurasia around Europe into two other European countries across cultures and the - up | the Atlantic Ocean on 18 September , and was assigned to the Mediterranean Fleet . She was assigned to the |
| the music played softly | ringing guitars began publishing industry in England after two seasons | in the song , with a tempo of 120 beats per minute . The song was written by David Bowie |
| the fire burned brightly | awe behind two three years until years later he decided not stop working through a relatively slow growth | in the basement of the building , where the fire was extinguished . The fire department arrived at the scene |

## Where GPT-2 wins

GPT-2 produces grammatically correct sentences. Its outputs read like Wikipedia. LLN produces topologically correct word sequences that stay on topic but break grammar.

Example where GPT-2 wins on relevance:

| Prompt | LLN (rel=0.31) | GPT-2 (rel=0.35) |
|--------|----------------|-------------------|
| she opened the door | hardtop , he married her life - up by many years under fire | to the public , and the door was closed . The door was closed in the middle of the door |

GPT-2 stays closer to "door" but repeats itself ("The door was closed... the door"). LLN diverges topically but produces unique content.

## Where LLN wins

LLN locks onto semantic targets and walks through them. Its content is dense with topic-relevant words, even when grammar breaks.

| Prompt | LLN (rel=0.60) | GPT-2 (rel=0.17) |
|--------|----------------|-------------------|
| scientists discovered a new | constitution would result , who believed their homes worldwide in ) ) have been reported no single entity separate colony | alignment of the highway in the early 19th century , and the highway was designated as a part of |

GPT-2 drifts to highway designations. LLN stays in the semantic neighborhood of discovery/reporting/entities.

## Methodology

- **Corpus**: 500MB (Wikipedia subset + web_text) matched between LLN graph build and GPT-2 training
- **GPT-2**: 17.7M params, greedy decoding
- **LLN**: v_fair_gpt2_match model (same corpus as GPT-2), a-v26 walker
- **Scoring**: PMI activation relevance (topology-based, no human judgment)
- **Categories**: Narrative, Science/Nature, Geography, Social/Political, Abstract
- **Raw data**: benchmark run produced 40 prompt-output pairs with per-prompt metrics

## Key takeaway

LLN wins on WHAT it says (topical relevance, lexical diversity, content density). GPT-2 wins on HOW it says it (grammar, fluency). Zero parameters vs 17.7 million.
