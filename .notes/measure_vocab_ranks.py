"""
Measure vocab in_degree distribution by rank.

Before committing to rank-based thresholds, understand the shape of the
distribution. What's the in_degree at rank 50? 150? 500? 1000?
Where do real content words fall? Where are the natural "knees" in the curve?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from generate import LLNGraph


def main():
    model_path = '/Users/CM64XD/DEUS/RESEARCH/lln/data/models/v16_mixed_100k.lmdb'
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)\n")

    # Sort vocab by in_degree descending
    in_deg = graph.in_degree[:graph.vocab_size]
    ranked = np.argsort(-in_deg)

    print("=" * 70)
    print("  VOCAB IN_DEGREE DISTRIBUTION BY RANK (v16)")
    print("=" * 70)

    # Print at key ranks
    checkpoints = [1, 10, 20, 50, 100, 150, 200, 300, 500, 1000,
                    2000, 5000, 10000, 20000, 50000, 99999]
    print(f"\n  {'rank':>6s} {'in_degree':>10s}  {'word':20s}")
    print(f"  {'-'*6} {'-'*10}  {'-'*20}")
    for r in checkpoints:
        if r >= graph.vocab_size:
            continue
        idx = int(ranked[r])
        print(f"  {r:>6d} {int(in_deg[idx]):>10d}  {graph.idx_to_word[idx]}")

    # Where do specific content words land?
    print(f"\n  Content words from our prompts:")
    print(f"  {'word':20s} {'in_deg':>10s} {'rank':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*8}")

    rank_lookup = np.empty(graph.vocab_size, dtype=np.int32)
    for r in range(graph.vocab_size):
        rank_lookup[int(ranked[r])] = r

    test_words = [
        # prompt content
        'fire', 'burned', 'king', 'door', 'army', 'marched', 'clouds',
        'Dark', 'river', 'flows', 'Scientists', 'discovered',
        # weather
        'rain', 'storm', 'Lightning', 'wind', 'Thunder', 'sky',
        # targets we want to reach
        'brightly', 'glowing', 'flames', 'ashes', 'extinguisher',
        'protection', 'safety', 'alive', 'flashed',
        # classic function words
        'the', 'a', 'of', 'and', 'in', 'to', 'was', 'is',
        # semi-function words
        'also', 'often', 'really', 'very', 'much', 'few',
        # punctuation
        '.', ',', '"', ';',
    ]

    word_to_idx = {w: i for i, w in enumerate(graph.idx_to_word)}
    for w in test_words:
        if w in word_to_idx:
            idx = word_to_idx[w]
            print(f"  {w:20s} {int(in_deg[idx]):>10d} {int(rank_lookup[idx]):>8d}")
        else:
            print(f"  {w:20s} {'NOT IN VOCAB':>20s}")

    # Look for "knees" in the distribution — points where rate of change shifts
    print(f"\n  In-degree at log-spaced ranks:")
    for r in [1, 2, 5, 10, 20, 50, 100, 150, 200, 500, 1000, 2000, 5000, 10000]:
        if r < graph.vocab_size:
            idx = int(ranked[r])
            print(f"    rank {r:>6d}: in_deg={int(in_deg[idx]):>8d}  ({graph.idx_to_word[idx]})")

    # Decile stats
    print(f"\n  In-degree deciles (percentile of vocab):")
    for pct in [50, 75, 90, 95, 99, 99.5, 99.9]:
        r = int(graph.vocab_size * (1 - pct/100))
        if r >= 1:
            idx = int(ranked[r])
            print(f"    top {pct:>5.1f}% cutoff at rank {r:>6d}: "
                  f"in_deg={int(in_deg[idx]):>8d}  ({graph.idx_to_word[idx]})")

    graph.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
