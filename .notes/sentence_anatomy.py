"""
Sentence Anatomy — measure the topological signature of real sentences.

Goal: understand what a real sentence looks like at each position, so we
can later build a walker that matches the pattern. Pure measurement,
no walker changes.

For 500-1000 real sentences sampled from the training corpus, at each
normalized position (0.0, 0.1, ..., 1.0), we measure:

  - in_degree[token]         : how many edges point to this token
  - out_degree[token]        : how many edges leave this token
  - pr_ratio                 : out_deg / in_deg (sink/throughput/source)
  - rank (from VocabRank)    : position in frequency-sorted vocab
  - forward_weight_to_next   : edge weight from this token to next
  - backward_weight_from_prev: edge weight from prev token to this one
  - trigram_valid            : does (p-2, p-1, p) exist in trigram graph?

Aggregated by normalized position. Then the same features are computed
for 8 walker outputs (main and v2) so we can see where walker outputs
diverge from real-sentence shape.

Usage:
    python .notes/sentence_anatomy.py --model path/to/v16.lmdb
    python .notes/sentence_anatomy.py --sample-size 1000
    python .notes/sentence_anatomy.py --walker-outputs  # include walker comparison
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import re
import time
import numpy as np
from collections import defaultdict

from generate import LLNGraph, download_model
from vocab_rank import VocabRank


CORPUS_FILES = [
    '/Users/CM64XD/DEUS/RESEARCH/lln/corpus/fineweb-edu.txt',
    '/Users/CM64XD/DEUS/RESEARCH/lln/corpus/gutenberg_en_clean.txt',
    '/Users/CM64XD/DEUS/RESEARCH/lln/corpus/openwebtext.txt',
]

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
BAD_SENTENCE_PATTERNS = [
    re.compile(r'https?://'),       # URLs
    re.compile(r'<[a-zA-Z/][^>]*>'), # HTML tags
    re.compile(r'[{}\[\]<>]'),      # code / markup
    re.compile(r'[^\x00-\x7f]'),    # non-ASCII
]

MIN_SENTENCE_TOKENS = 5
MAX_SENTENCE_TOKENS = 20


def is_clean_sentence(text):
    """Filter: English prose, no code / URLs / HTML / non-ASCII."""
    if len(text) < 20 or len(text) > 300:
        return False
    # Check alphabetic ratio
    alpha = sum(1 for c in text if c.isalpha() or c.isspace())
    if alpha / len(text) < 0.85:
        return False
    for pat in BAD_SENTENCE_PATTERNS:
        if pat.search(text):
            return False
    return True


def sample_sentences_from_file(path, n_sentences, seek_tries=500):
    """Sample sentences from a large text file by random offset seek.

    Reads ~2KB chunks from random offsets, splits on sentence boundaries,
    returns the first clean sentence found. Repeats until n_sentences
    are collected or seek_tries is exhausted.
    """
    sentences = []
    size = os.path.getsize(path)
    tries = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while len(sentences) < n_sentences and tries < seek_tries * n_sentences:
            tries += 1
            offset = random.randint(0, size - 4096)
            f.seek(offset)
            f.readline()  # discard partial line
            chunk = f.read(4096)
            if not chunk:
                continue
            # Split into sentences
            parts = SENTENCE_SPLIT.split(chunk)
            for part in parts[1:-1]:  # skip first/last (likely partial)
                part = part.strip()
                if is_clean_sentence(part):
                    sentences.append(part)
                    if len(sentences) >= n_sentences:
                        break
    return sentences


def compute_position_features(graph, vocab_rank, token_indices):
    """For each position in a sentence, compute topological features.

    Returns a list of dicts, one per position.
    """
    n = len(token_indices)
    features = []
    for p in range(n):
        t = token_indices[p]
        in_deg = int(graph.in_degree[t])
        out_deg = int(graph.out_degree(t))
        pr_ratio = out_deg / max(in_deg, 1)
        rank = vocab_rank.get_rank(t)

        # Forward weight to next token
        fwd_w = 0.0
        if p < n - 1:
            t_next = token_indices[p + 1]
            s = int(graph.full_off[t])
            e = int(graph.full_off[t + 1])
            for j in range(s, e):
                if int(graph.full_tgt[j]) == t_next:
                    fwd_w = float(graph.full_wgt[j])
                    break

        # Backward weight from prev token
        bwd_w = 0.0
        if p > 0:
            t_prev = token_indices[p - 1]
            s = int(graph.full_off[t_prev])
            e = int(graph.full_off[t_prev + 1])
            for j in range(s, e):
                if int(graph.full_tgt[j]) == t:
                    bwd_w = float(graph.full_wgt[j])
                    break

        # Trigram validity (p-2, p-1, p)
        trigram_valid = False
        if p >= 2:
            tri_score = graph.trigram_score(
                token_indices[p - 2],
                token_indices[p - 1],
                t)
            trigram_valid = tri_score > 1.0

        features.append({
            'position': p,
            'norm_position': p / max(n - 1, 1),
            'token': t,
            'word': graph.idx_to_word[t],
            'in_degree': in_deg,
            'out_degree': out_deg,
            'pr_ratio': pr_ratio,
            'rank': rank,
            'fwd_weight': fwd_w,
            'bwd_weight': bwd_w,
            'trigram_valid': trigram_valid,
        })
    return features


def aggregate_by_norm_position(all_features, n_bins=10):
    """Aggregate per-sentence features by normalized position bin."""
    bins = [[] for _ in range(n_bins)]
    for sentence_features in all_features:
        for f in sentence_features:
            bin_idx = min(int(f['norm_position'] * n_bins), n_bins - 1)
            bins[bin_idx].append(f)

    result = []
    for bin_idx, features in enumerate(bins):
        if not features:
            result.append(None)
            continue
        result.append({
            'bin': bin_idx,
            'bin_pos': bin_idx / n_bins,
            'n_samples': len(features),
            'mean_in_degree': np.mean([f['in_degree'] for f in features]),
            'mean_out_degree': np.mean([f['out_degree'] for f in features]),
            'mean_pr_ratio': np.mean([f['pr_ratio'] for f in features]),
            'median_pr_ratio': np.median([f['pr_ratio'] for f in features]),
            'mean_rank': np.mean([f['rank'] for f in features]),
            'median_rank': np.median([f['rank'] for f in features]),
            'mean_fwd_weight': np.mean([f['fwd_weight'] for f in features]),
            'mean_bwd_weight': np.mean([f['bwd_weight'] for f in features]),
            'trigram_valid_rate': np.mean(
                [1 if f['trigram_valid'] else 0 for f in features]),
        })
    return result


def print_profile(name, profile):
    print(f"\n  === {name} ===")
    print(f"  {'pos':>5s} {'n':>6s} {'in_deg':>8s} {'out_deg':>8s} "
          f"{'pr':>6s} {'rank':>7s} {'fwd_w':>9s} {'bwd_w':>9s} {'tri%':>6s}")
    print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*7} "
          f"{'-'*9} {'-'*9} {'-'*6}")
    for bucket in profile:
        if bucket is None:
            continue
        print(f"  {bucket['bin_pos']:>5.2f} {bucket['n_samples']:>6d} "
              f"{bucket['mean_in_degree']:>8.0f} {bucket['mean_out_degree']:>8.0f} "
              f"{bucket['mean_pr_ratio']:>6.2f} {bucket['median_rank']:>7.0f} "
              f"{bucket['mean_fwd_weight']:>9.0f} {bucket['mean_bwd_weight']:>9.0f} "
              f"{bucket['trigram_valid_rate']*100:>5.1f}%")


def measure_corpus(graph, vocab_rank, n_per_file=300):
    """Sample sentences from all corpus files, measure, aggregate."""
    print(f"\nSampling {n_per_file} sentences from each corpus file...")
    all_features = []
    per_source = {}
    for path in CORPUS_FILES:
        source = os.path.basename(path).replace('.txt', '')
        print(f"  {source}...", end=" ", flush=True)
        t0 = time.time()
        sentences = sample_sentences_from_file(path, n_per_file)
        print(f"got {len(sentences)} sentences ({time.time() - t0:.1f}s)")

        source_features = []
        for s in sentences:
            tokens = graph.tokenize(s)
            if not (MIN_SENTENCE_TOKENS <= len(tokens) <= MAX_SENTENCE_TOKENS):
                continue
            features = compute_position_features(graph, vocab_rank, tokens)
            source_features.append(features)
            all_features.append(features)
        print(f"    {len(source_features)} passed token-length filter")
        per_source[source] = source_features
    return all_features, per_source


# ═══════════════════════════════════════════════════════════════════
# Walker output comparison
# ═══════════════════════════════════════════════════════════════════

WALKER_OUTPUTS = {
    'main': {
        "The fire burned": "alive , and the public safety , and his eyes flashed brightly glowing cheeks",
        "The king": "of the most powerful , and I heard a little boy whom he answered",
        "She opened the door": "opened fire , and the other . \" She paused , and the other . \" She laughed softly in",
        "The army marched": "northward to the whole . The main",
        "Dark clouds": "of this matter",
        "The river flows": "south bank deposits . The main stream flowing",
        "Scientists discovered": "that I believe I hope you think you find that they want to be more easily identified genes",
        "The old man walked": "beside a woman , and the old lady , and the old gentleman , and his friend of the same",
    },
    'v2': {
        "The fire burned": "alive . His eyes flashed brightly glowing",
        "The king": "of a young woman in America great . My old woman named Mary Jane had heard the same day",
        "She opened the door": "opens fire . She paused abruptly closed the door . She smiled gravely . He sat down , and I",
        "The army marched": "northward march north south . The young officer . He spoke slowly the whole day . But the past few",
        "Dark clouds": "hung suspended matter . But the rest of our solar energy level rise",
        "The river flows": "south bank . These systems , the past few days , however great , and the United States . The",
        "Scientists discovered": "that I believe I hope you don't expect to think me know . They ll find a few days .",
        "The old man walked": "slowly down beside him . He went straight to a woman . My lady , and the old gentleman ,",
    },
    'v3': {
        "The fire burned": "alive with his own personal safety training camp with his eyes flashed brightly glowing cheeks",
        "The king": "of its most powerful than ever heard my lord hath commanded thee thy people who had a small boy whom",
        "She opened the door": "open fire of her eyes closed doors locked in front door opening the city hall porter",
        "The army marched": "northward along the right to go straight white supremacists and so much more advanced rapidly than three other two young",
        "Dark clouds": "that this matter of an increase energy",
        "The river flows": "south of water flow of its main stream flowing blood",
        "Scientists discovered": "that we believe that she began studying the only hope you might expect to think you find you want more",
        "The old man walked": "beside him go straight off the very much more advanced rapidly than a woman was his friend of any age",
    },
}


def measure_walker_outputs(graph, vocab_rank):
    """Measure walker outputs (prompt + generated) using same profile."""
    results = {}
    for walker_name, prompt_dict in WALKER_OUTPUTS.items():
        all_features = []
        for prompt, generated in prompt_dict.items():
            full_text = prompt + " " + generated
            tokens = graph.tokenize(full_text)
            if len(tokens) < 3:
                continue
            features = compute_position_features(graph, vocab_rank, tokens)
            all_features.append(features)
        results[walker_name] = all_features
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sentence anatomy diagnostic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=300,
                        help="Sentences per corpus file (default 300 = 900 total)")
    parser.add_argument("--walker-outputs", action="store_true",
                        help="Also measure walker outputs for comparison")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    model_path = args.model if args.model else download_model()
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    vr = VocabRank(graph)

    # 1. Measure real corpus sentences
    all_features, per_source = measure_corpus(graph, vr, n_per_file=args.sample_size)
    print(f"\n  Total sentences used: {len(all_features)}")
    print(f"  Total positions: {sum(len(f) for f in all_features)}")

    # Global profile
    print_profile("REAL SENTENCES (all sources combined)",
                  aggregate_by_norm_position(all_features, args.n_bins))

    # Per-source profiles
    for source, features in per_source.items():
        if features:
            print_profile(f"REAL SENTENCES ({source})",
                          aggregate_by_norm_position(features, args.n_bins))

    # 2. Walker outputs (if requested)
    if args.walker_outputs:
        walker_results = measure_walker_outputs(graph, vr)
        for name, features in walker_results.items():
            print_profile(f"WALKER OUTPUT ({name})",
                          aggregate_by_norm_position(features, args.n_bins))

    graph.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
