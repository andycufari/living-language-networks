#!/usr/bin/env python3
"""LLN — Train a language graph from raw text.

Reads a text corpus and builds a directed weighted graph stored in LMDB.
No neural networks, no gradient descent — just counting word pairs.

Six phases:
  1. Vocabulary — scan text, count frequencies, keep top N
  2. Bigrams — count all (word_A, word_B) consecutive pairs
  3. PMI — Pointwise Mutual Information (meaningful associations)
  4. Trigrams — (prev, cur) -> next patterns
  5. CSR — Compressed Sparse Row arrays for fast lookup
  6. LMDB — write everything to disk

Usage:
    python train.py --input corpus.txt --output model/
    python train.py --input corpus.txt --output model/ --vocab-size 50000
    python train.py --input wiki.txt gutenberg.txt --output model/

Requirements: numpy, lmdb
"""
import os
import sys
import time
import json
import struct
import argparse
import re
import numpy as np
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════
# Tokenizer (embedded — no external dependencies)
# ═══════════════════════════════════════════════════════════════════

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[.,;:!?()\"-]")

def tokenize(text):
    """Tokenize text into words + punctuation. Case preserved."""
    return TOKEN_PATTERN.findall(text)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Build Vocabulary
# ═══════════════════════════════════════════════════════════════════

def build_vocab(input_files, vocab_size=100000, min_freq=10):
    """Scan all files, count token frequencies, keep top N."""
    print(f"[1/6] Building vocabulary...")
    freq = Counter()
    total_tokens = 0

    for path in input_files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Scanning {os.path.basename(path)} ({size_mb:.0f} MB)...")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                tokens = tokenize(line)
                freq.update(tokens)
                total_tokens += len(tokens)

    # Filter by min_freq, keep top vocab_size
    filtered = [(w, c) for w, c in freq.most_common() if c >= min_freq][:vocab_size]
    word_to_idx = {w: i for i, (w, _) in enumerate(filtered)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    print(f"  {total_tokens:,} tokens, {len(word_to_idx):,} vocabulary (min_freq={min_freq})")
    return word_to_idx, idx_to_word, total_tokens


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Count Bigrams
# ═══════════════════════════════════════════════════════════════════

def count_bigrams(input_files, word_to_idx):
    """Count all consecutive word pairs."""
    print(f"[2/6] Counting bigrams...")
    bigrams = defaultdict(lambda: defaultdict(int))
    total = 0

    for path in input_files:
        print(f"  Processing {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            prev_idx = None
            for line in f:
                tokens = tokenize(line)
                for tok in tokens:
                    idx = word_to_idx.get(tok)
                    if idx is not None:
                        if prev_idx is not None:
                            bigrams[prev_idx][idx] += 1
                            total += 1
                        prev_idx = idx
                    else:
                        prev_idx = None  # break on unknown token

    n_unique = sum(len(targets) for targets in bigrams.values())
    print(f"  {total:,} bigrams, {n_unique:,} unique edges")
    return bigrams, total, n_unique


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Compute PMI
# ═══════════════════════════════════════════════════════════════════

def compute_pmi(bigrams, word_to_idx, total_bigrams, min_count=5):
    """Compute Pointwise Mutual Information edges."""
    print(f"[3/6] Computing PMI...")
    vocab_size = len(word_to_idx)

    # Token frequencies (as source and target)
    src_freq = defaultdict(int)
    tgt_freq = defaultdict(int)
    for src, targets in bigrams.items():
        for tgt, count in targets.items():
            src_freq[src] += count
            tgt_freq[tgt] += count

    # PMI = log(P(A,B) / P(A)*P(B))
    pmi_edges = defaultdict(lambda: defaultdict(float))
    n_pmi = 0
    n_filtered = 0

    for src, targets in bigrams.items():
        for tgt, count in targets.items():
            if count < min_count:
                n_filtered += 1
                continue
            p_ab = count / total_bigrams
            p_a = src_freq[src] / total_bigrams
            p_b = tgt_freq[tgt] / total_bigrams
            if p_a > 0 and p_b > 0:
                pmi = np.log(p_ab / (p_a * p_b))
                if pmi > 0:
                    pmi_edges[src][tgt] = pmi
                    pmi_edges[tgt][src] = pmi  # PMI is symmetric
                    n_pmi += 1

    print(f"  {n_pmi:,} PMI edges, {n_filtered:,} filtered (min_count={min_count})")
    return pmi_edges


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Count Trigrams
# ═══════════════════════════════════════════════════════════════════

def count_trigrams(input_files, word_to_idx, min_count=3):
    """Count (prev, cur) -> next patterns."""
    print(f"[4/6] Counting trigrams...")
    vocab_size = len(word_to_idx)
    trigrams = defaultdict(lambda: defaultdict(int))
    total = 0

    for path in input_files:
        print(f"  Processing {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            prev2 = None
            prev1 = None
            for line in f:
                tokens = tokenize(line)
                for tok in tokens:
                    idx = word_to_idx.get(tok)
                    if idx is not None:
                        if prev2 is not None and prev1 is not None:
                            key = prev2 * vocab_size + prev1
                            trigrams[key][idx] += 1
                            total += 1
                        prev2 = prev1
                        prev1 = idx
                    else:
                        prev2 = None
                        prev1 = None

    # Filter by min_count
    filtered = {}
    n_pairs = 0
    n_edges = 0
    for key, targets in trigrams.items():
        kept = {t: c for t, c in targets.items() if c >= min_count}
        if kept:
            filtered[key] = kept
            n_pairs += 1
            n_edges += len(kept)

    print(f"  {total:,} trigrams, {n_pairs:,} pairs, {n_edges:,} edges (min_count={min_count})")
    return filtered, n_pairs, n_edges


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Build CSR Arrays
# ═══════════════════════════════════════════════════════════════════

def build_csr(bigrams, pmi_edges, vocab_size, topk=200):
    """Build Compressed Sparse Row arrays for fast lookup."""
    print(f"[5/6] Building CSR arrays...")

    def _build(edges_dict, sort_desc=True, top_k=None):
        offsets = np.zeros(vocab_size + 1, dtype=np.int32)
        all_targets = []
        all_weights = []

        for src in range(vocab_size):
            targets = edges_dict.get(src, {})
            if not targets:
                offsets[src + 1] = offsets[src]
                continue

            pairs = sorted(targets.items(), key=lambda x: -x[1])
            if top_k and len(pairs) > top_k:
                pairs = pairs[:top_k]

            for tgt, wgt in pairs:
                all_targets.append(tgt)
                all_weights.append(wgt)

            offsets[src + 1] = offsets[src] + len(pairs)

        return (offsets,
                np.array(all_targets, dtype=np.int32),
                np.array(all_weights, dtype=np.float32))

    # Sorted: top-K per node
    s_off, s_tgt, s_wgt = _build(bigrams, top_k=topk)
    print(f"  Sorted: {len(s_tgt):,} edges (topk={topk})")

    # Full: all forward edges
    f_off, f_tgt, f_wgt = _build(bigrams, top_k=None)
    print(f"  Full: {len(f_tgt):,} edges")

    # PMI
    p_off, p_tgt, p_wgt = _build(pmi_edges, top_k=None)
    print(f"  PMI: {len(p_tgt):,} edges")

    return {
        'sorted': (s_off, s_tgt, s_wgt),
        'full': (f_off, f_tgt, f_wgt),
        'pmi': (p_off, p_tgt, p_wgt),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Write LMDB
# ═══════════════════════════════════════════════════════════════════

def write_lmdb(output_path, word_to_idx, idx_to_word, csr, trigrams,
               total_bigrams, n_unique_edges, n_pmi_edges, vocab_size, topk,
               input_files):
    """Write everything to LMDB."""
    print(f"[6/6] Writing LMDB to {output_path}...")
    import lmdb

    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)

    map_size = 4 * 1024 * 1024 * 1024  # 4GB
    env = lmdb.open(output_path, map_size=map_size, max_dbs=1)

    with env.begin(write=True) as txn:
        # Metadata
        meta = {
            'vocab_size': vocab_size,
            'total_bigrams': total_bigrams,
            'unique_edges': n_unique_edges,
            'pmi_edges': n_pmi_edges,
            'topk': topk,
            'corpora': [os.path.basename(f) for f in input_files],
            'version': 'lln-1.0',
        }
        txn.put(b'metadata', json.dumps(meta).encode())

        # Vocabulary
        words_blob = b''
        offsets = [0]
        for i in range(vocab_size):
            w = idx_to_word[i].encode('utf-8')
            words_blob += w
            offsets.append(len(words_blob))

        txn.put(b'vocab_words', words_blob)
        txn.put(b'vocab_offsets', np.array(offsets, dtype=np.int32).tobytes())

        # CSR arrays
        for name, key_prefix in [('sorted', 'csr_sorted'), ('full', 'csr_full'), ('pmi', 'csr_pmi')]:
            off, tgt, wgt = csr[name]
            txn.put(f'{key_prefix}_offsets'.encode(), off.tobytes())
            txn.put(f'{key_prefix}_targets'.encode(), tgt.tobytes())
            txn.put(f'{key_prefix}_weights'.encode(), wgt.tobytes())

        # Mass: (in_degree - out_degree) / (in_degree + out_degree + 1)
        full_tgt = csr['full'][1]
        in_deg = np.bincount(full_tgt[full_tgt < vocab_size], minlength=vocab_size).astype(np.float32)
        full_off = csr['full'][0]
        out_deg = np.diff(full_off).astype(np.float32)
        mass = np.clip((in_deg - out_deg) / (in_deg + out_deg + 1), -5, 5)
        txn.put(b'mass', mass.tobytes())

        # Trigrams (v2 format: n_entries, then key(u64) + n_tgt(u32) + [next(u32), count(u32)]*)
        tri_parts = [struct.pack('<I', len(trigrams))]
        for key, targets in trigrams.items():
            tri_parts.append(struct.pack('<Q', key))
            tri_parts.append(struct.pack('<I', len(targets)))
            for next_idx, count in targets.items():
                tri_parts.append(struct.pack('<II', next_idx, count))
        txn.put(b'trigrams_v2', b''.join(tri_parts))

    env.close()
    print(f"  Done. Model saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLN — Train a language graph from raw text")
    parser.add_argument("--input", nargs='+', required=True,
                        help="Input text file(s)")
    parser.add_argument("--output", default="model/",
                        help="Output LMDB directory (default: model/)")
    parser.add_argument("--vocab-size", type=int, default=100000,
                        help="Vocabulary size (default: 100000)")
    parser.add_argument("--min-freq", type=int, default=10,
                        help="Minimum token frequency (default: 10)")
    parser.add_argument("--topk", type=int, default=200,
                        help="Top-K edges per node for sorted CSR (default: 200)")
    parser.add_argument("--pmi-min-count", type=int, default=5,
                        help="Minimum bigram count for PMI (default: 5)")
    parser.add_argument("--tri-min-count", type=int, default=3,
                        help="Minimum trigram count (default: 3)")
    args = parser.parse_args()

    # Validate inputs
    for path in args.input:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            sys.exit(1)

    t0 = time.time()

    # Phase 1: Vocabulary
    word_to_idx, idx_to_word, total_tokens = build_vocab(
        args.input, vocab_size=args.vocab_size, min_freq=args.min_freq)
    vocab_size = len(word_to_idx)

    # Phase 2: Bigrams
    bigrams, total_bigrams, n_unique = count_bigrams(args.input, word_to_idx)

    # Phase 3: PMI
    pmi_edges = compute_pmi(bigrams, word_to_idx, total_bigrams,
                            min_count=args.pmi_min_count)
    n_pmi = sum(len(t) for t in pmi_edges.values())

    # Phase 4: Trigrams
    trigrams, n_tri_pairs, n_tri_edges = count_trigrams(
        args.input, word_to_idx, min_count=args.tri_min_count)

    # Phase 5: CSR
    csr = build_csr(bigrams, pmi_edges, vocab_size, topk=args.topk)

    # Phase 6: Write
    write_lmdb(args.output, word_to_idx, idx_to_word, csr, trigrams,
               total_bigrams, n_unique, n_pmi, vocab_size, args.topk,
               args.input)

    elapsed = time.time() - t0
    print(f"\nTotal build time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Vocab: {vocab_size:,} | Edges: {n_unique:,} | PMI: {n_pmi:,} | Trigrams: {n_tri_pairs:,}")


if __name__ == "__main__":
    main()
