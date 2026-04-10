"""
Verify assumptions about v16 topology — don't guess, measure.

Questions to answer with data:
  1. What are the top 20 forward edges from '.' ?  Where does '"' rank?
  2. What are the top 20 forward edges from '"' ?  Is 'She' there?
  3. What are the top 10 forward edges from 'burned' ?  Fire-domain or not?
  4. What are the top 10 forward edges from 'door' ?  Sink or throughput?
  5. Is 'public safety' a real PMI pair for 'fire' in v16?
  6. What's the PR (push/receive) of '"' globally? Sink/source/throughput?
  7. Which of the words in "opened fire , and the other . " She paused"
     are content vs function?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from generate import LLNGraph


def top_forward(graph, word, n=20):
    if word not in graph.word_to_idx:
        print(f"  '{word}' NOT IN VOCAB")
        return
    idx = graph.word_to_idx[word]
    tgt, wgt = graph.get_forward_edges(idx, top_k=n)
    in_d = graph.in_degree[idx]
    out_d = graph.out_degree(idx)
    print(f"\n  '{word}' (idx={idx}, in_deg={in_d}, out_deg={out_d})")
    print(f"  Top {n} forward edges:")
    for i in range(len(tgt)):
        t = int(tgt[i])
        w = float(wgt[i])
        word_t = graph.idx_to_word[t]
        print(f"    {i+1:3d}. {word_t:20s} w={w:10.0f}  in_deg={graph.in_degree[t]}")


def global_pr(graph, word):
    """Global push/receive ratio — out_weight vs in_weight."""
    if word not in graph.word_to_idx:
        return None
    idx = graph.word_to_idx[word]
    # Out-weight: sum of all forward edge weights
    s = int(graph.full_off[idx])
    e = int(graph.full_off[idx + 1])
    out_wgt = float(np.sum(graph.full_wgt[s:e]))
    # In-weight: we'd need to sum all edges landing on idx — expensive.
    # Use in_degree as proxy for receive count.
    return {'idx': idx, 'out_deg': e - s, 'out_wgt': out_wgt,
            'in_deg': int(graph.in_degree[idx])}


def pmi_pair(graph, a, b):
    """Is b a PMI neighbor of a?"""
    if a not in graph.word_to_idx or b not in graph.word_to_idx:
        return None
    a_idx = graph.word_to_idx[a]
    b_idx = graph.word_to_idx[b]
    tgt, wgt = graph.get_pmi_neighbors(a_idx)
    for i in range(len(tgt)):
        if int(tgt[i]) == b_idx:
            return float(wgt[i])
    return 0.0


def main():
    model_path = '/Users/CM64XD/DEUS/RESEARCH/lln/data/models/v16_mixed_100k.lmdb'
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    # Build reverse vocab
    graph.word_to_idx = {w: i for i, w in enumerate(graph.idx_to_word)}
    print(f"done ({time.time() - t0:.1f}s)\n")

    # Q1: After '.', what's next?
    print("=" * 60)
    print("Q1: Top forward edges from '.'")
    top_forward(graph, '.', n=20)

    # Q2: After '"', what's next?
    print("\n" + "=" * 60)
    print('Q2: Top forward edges from \'"\'')
    top_forward(graph, '"', n=20)

    # Q3: After 'burned'
    print("\n" + "=" * 60)
    print("Q3: Top forward edges from 'burned'")
    top_forward(graph, 'burned', n=15)

    # Q4: After 'door'
    print("\n" + "=" * 60)
    print("Q4: Top forward edges from 'door'")
    top_forward(graph, 'door', n=15)

    # Q5: After 'fire'
    print("\n" + "=" * 60)
    print("Q5: Top forward edges from 'fire'")
    top_forward(graph, 'fire', n=15)

    # Q6: After 'brightly'
    print("\n" + "=" * 60)
    print("Q6: Top forward edges from 'brightly'")
    top_forward(graph, 'brightly', n=15)

    # Q7: PMI pairs
    print("\n" + "=" * 60)
    print("Q7: PMI pair evidence")
    for a, b in [('fire', 'safety'), ('fire', 'public'),
                 ('burned', 'alive'), ('burned', 'brightly'),
                 ('door', 'opened'), ('opened', 'fire'),
                 ('opened', 'door'), ('.', '"'), ('"', 'She'),
                 ('river', 'bank'), ('river', 'flows')]:
        w = pmi_pair(graph, a, b)
        print(f"  PMI({a}, {b}) = {w}")

    # Q8: Global stats on special tokens
    print("\n" + "=" * 60)
    print("Q8: Global stats on key tokens")
    for w in ['"', '.', ',', 'the', 'and', 'She', 'She', 'fire', 'burned',
              'brightly', 'door', 'safety', 'public']:
        s = global_pr(graph, w)
        if s:
            print(f"  {w:15s} idx={s['idx']:6d}  out_deg={s['out_deg']:6d}  "
                  f"out_wgt={s['out_wgt']:12.0f}  in_deg={s['in_deg']:6d}")

    graph.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
