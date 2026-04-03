#!/usr/bin/env python3
"""LLN — Living Language Network Generator

Zero-parameter language generation from graph topology.
Dual-system cognitive routing: PMI activation (Wernicke) + grammar walk (Broca).

Usage:
    python generate.py --prompt "The fire burned"
    python generate.py --prompt "Scientists discovered" --max-tokens 20
    python generate.py --prompt "Dark clouds" --verbose

The model is automatically downloaded from HuggingFace on first run.

Requirements: numpy, lmdb, huggingface_hub
    pip install numpy lmdb huggingface_hub
"""
import numpy as np
import lmdb
import json
import struct
import time
import argparse
import os

# HuggingFace model config
HF_REPO = "cufa64/lln_v13_parallel_100k"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def download_model(model_dir=DEFAULT_MODEL_DIR):
    """Download model from HuggingFace if not present locally."""
    data_mdb = os.path.join(model_dir, "data.mdb")
    if os.path.exists(data_mdb):
        return model_dir

    print(f"Downloading model from HuggingFace ({HF_REPO})...")
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(model_dir, exist_ok=True)

        # Download data.mdb
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="data.mdb",
            local_dir=model_dir,
            repo_type="model",
        )
        print(f"  Downloaded to {model_dir}")

        # Create lock.mdb (LMDB needs it)
        lock_path = os.path.join(model_dir, "lock.mdb")
        if not os.path.exists(lock_path):
            with open(lock_path, 'wb') as f:
                f.write(b'\x00' * 8192)

        return model_dir
    except ImportError:
        print("  huggingface_hub not installed. Install with: pip install huggingface_hub")
        print("  Or download manually from https://huggingface.co/andycufari/lln-v13-wikipedia")
        raise
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  You can manually place the LMDB model at: {model_dir}/data.mdb")
        raise


# ═══════════════════════════════════════════════════════════════════
# Graph — LMDB CSR reader
# ═══════════════════════════════════════════════════════════════════

class LLNGraph:
    """Reads the LMDB language network model.

    The model is a directed weighted graph stored as Compressed Sparse Row
    arrays. Nodes = words. Edges = "word A followed word B in training text."
    Edge weight = raw co-occurrence count.

    Three edge sets:
      sorted: top-200 edges per node (for fast beam search)
      full:   all edges (for scoring and lookup)
      pmi:    semantic associations (Pointwise Mutual Information)
    """

    def __init__(self, model_path):
        self.env = lmdb.open(model_path, readonly=True, max_dbs=1,
                             map_size=4 * 1024 * 1024 * 1024)
        with self.env.begin() as txn:
            meta = json.loads(txn.get(b'metadata'))
            self.vocab_size = meta['vocab_size']

            # Vocabulary
            vocab_blob = txn.get(b'vocab_words')
            vocab_off = np.frombuffer(txn.get(b'vocab_offsets'), dtype=np.int32)
            self.idx_to_word = []
            for i in range(self.vocab_size):
                s, e = vocab_off[i], vocab_off[i + 1]
                self.idx_to_word.append(vocab_blob[s:e].decode('utf-8', errors='replace'))
            self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}

            # Forward edges (sorted = top-200 per node, by weight desc)
            self.fwd_off = np.frombuffer(txn.get(b'csr_sorted_offsets'), dtype=np.int32).copy()
            self.fwd_tgt = np.frombuffer(txn.get(b'csr_sorted_targets'), dtype=np.int32).copy()
            self.fwd_wgt = np.frombuffer(txn.get(b'csr_sorted_weights'), dtype=np.float32).copy()

            # Full edges (all 55M+)
            self.full_off = np.frombuffer(txn.get(b'csr_full_offsets'), dtype=np.int32).copy()
            self.full_tgt = np.frombuffer(txn.get(b'csr_full_targets'), dtype=np.int32).copy()
            self.full_wgt = np.frombuffer(txn.get(b'csr_full_weights'), dtype=np.float32).copy()

            # PMI edges (semantic associations)
            self.pmi_off = np.frombuffer(txn.get(b'csr_pmi_offsets'), dtype=np.int32).copy()
            self.pmi_tgt = np.frombuffer(txn.get(b'csr_pmi_targets'), dtype=np.int32).copy()
            self.pmi_wgt = np.frombuffer(txn.get(b'csr_pmi_weights'), dtype=np.float32).copy()

            # Trigrams
            self.trigrams = self._load_trigrams(txn)

        # In-degree (computed from full edges)
        self.in_degree = np.bincount(
            self.full_tgt[self.full_tgt < self.vocab_size],
            minlength=self.vocab_size
        ).astype(np.int32)

    def _load_trigrams(self, txn):
        """Load trigram index: (prev, cur) -> {next: count}."""
        tri_data = txn.get(b'trigrams_v2')
        if tri_data is None:
            return {}
        n_entries = struct.unpack_from('<I', tri_data, 0)[0]
        index = {}
        offset = 4
        for _ in range(n_entries):
            key = struct.unpack_from('<Q', tri_data, offset)[0]
            offset += 8
            n_tgt = struct.unpack_from('<I', tri_data, offset)[0]
            offset += 4
            targets = {}
            for _ in range(n_tgt):
                next_idx = struct.unpack_from('<I', tri_data, offset)[0]
                count = struct.unpack_from('<I', tri_data, offset + 4)[0]
                offset += 8
                targets[next_idx] = count
            index[key] = targets
        return index

    def get_forward_edges(self, idx, top_k=50):
        """Top-K forward edges by weight. Returns (targets, weights)."""
        s, e = int(self.fwd_off[idx]), int(self.fwd_off[idx + 1])
        if s == e:
            return np.array([], np.int32), np.array([], np.float32)
        tgt, wgt = self.fwd_tgt[s:e], self.fwd_wgt[s:e]
        if len(tgt) > top_k:
            top = np.argpartition(wgt, -top_k)[-top_k:]
            tgt, wgt = tgt[top], wgt[top]
        return tgt, wgt

    def get_pmi_neighbors(self, idx):
        """PMI semantic neighbors. Returns (targets, weights)."""
        s, e = int(self.pmi_off[idx]), int(self.pmi_off[idx + 1])
        if s == e:
            return np.array([], np.int32), np.array([], np.float32)
        return self.pmi_tgt[s:e], self.pmi_wgt[s:e]

    def trigram_score(self, prev_idx, cur_idx, next_idx):
        """Trigram multiplier for a candidate next token."""
        key = prev_idx * self.vocab_size + cur_idx
        targets = self.trigrams.get(key)
        if targets is None:
            return 1.0
        count = targets.get(next_idx, 0)
        if count > 0:
            return 1.0 + float(np.log1p(count))
        return 0.5

    def tokenize(self, text):
        return [self.word_to_idx[w] for w in text.split() if w in self.word_to_idx]

    def detokenize(self, indices):
        return ' '.join(self.idx_to_word[i] for i in indices)

    def close(self):
        self.env.close()


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — PMI Activation (Wernicke's Area)
# Defines the semantic field: WHAT to talk about.
# Frozen at T=0 — generated tokens never expand the goal.
# ═══════════════════════════════════════════════════════════════════

def activate(graph, prompt_indices, top_pct=0.20):
    """Build semantic field from prompt via 1-hop PMI."""
    all_pmi = {}
    for idx in prompt_indices:
        if graph.in_degree[idx] > 20000:
            continue
        pmi_tgt, pmi_wgt = graph.get_pmi_neighbors(idx)
        for i in range(len(pmi_tgt)):
            t, w = int(pmi_tgt[i]), float(pmi_wgt[i])
            if t not in all_pmi or w > all_pmi[t]:
                all_pmi[t] = w

    if not all_pmi:
        for idx in prompt_indices:
            pmi_tgt, pmi_wgt = graph.get_pmi_neighbors(idx)
            for i in range(len(pmi_tgt)):
                t, w = int(pmi_tgt[i]), float(pmi_wgt[i])
                if t not in all_pmi or w > all_pmi[t]:
                    all_pmi[t] = w

    if not all_pmi:
        return set(prompt_indices), {}

    weights = sorted(all_pmi.values(), reverse=True)
    threshold = weights[int(len(weights) * top_pct)] if len(weights) > 5 else 0
    activated = set(prompt_indices)
    for t, w in all_pmi.items():
        if w >= threshold:
            activated.add(t)

    return activated, all_pmi


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Target Selection (PMI ∩ Reachability)
# ═══════════════════════════════════════════════════════════════════

def find_targets(graph, prompt_indices, context_indices, activated, pmi_scores):
    """Find content targets: activated AND reachable in 2-3 hops."""
    context_set = set(context_indices)
    last = context_indices[-1]

    reachable = {}
    tgt1, wgt1 = graph.get_forward_edges(last, top_k=50)
    for i in range(len(tgt1)):
        t = int(tgt1[i])
        if t not in context_set:
            reachable[t] = (1, float(wgt1[i]))

    for t1 in list(reachable.keys()):
        tgt2, wgt2 = graph.get_forward_edges(t1, top_k=30)
        for i in range(len(tgt2)):
            t = int(tgt2[i])
            if t not in context_set and t not in reachable:
                reachable[t] = (2, float(wgt2[i]))

    hop2 = [t for t, (d, _) in reachable.items() if d == 2]
    for t2 in hop2[:100]:
        tgt3, wgt3 = graph.get_forward_edges(t2, top_k=20)
        for i in range(len(tgt3)):
            t = int(tgt3[i])
            if t not in context_set and t not in reachable:
                reachable[t] = (3, float(wgt3[i]))

    targets = []
    for t in activated:
        if t in context_set or t not in reachable:
            continue
        if graph.in_degree[t] > 15000:
            continue
        hop_dist, _ = reachable[t]
        pmi_score = pmi_scores.get(t, 0)
        score = pmi_score * (4 - hop_dist)
        targets.append((t, score))

    targets.sort(key=lambda x: -x[1])
    return targets


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Walk (Broca's Area)
# ═══════════════════════════════════════════════════════════════════

def walk_to_target(graph, start, target, target_pmi, visited,
                   prev_token=None, max_steps=6, top_k=50):
    """Walk from start toward target using full graph topology.

    Score = (normalized_weight x trigram_multiplier) + (proximity x target_PMI)
    """
    ts = int(graph.full_off[target])
    te = int(graph.full_off[target + 1])
    target_out = set()
    for i in range(ts, min(te, ts + 500)):
        target_out.add(int(graph.full_tgt[i]))

    path = []
    current = start

    for step in range(max_steps):
        s = int(graph.fwd_off[current])
        e = int(graph.fwd_off[current + 1])
        if s == e:
            break

        tgt = graph.fwd_tgt[s:e]
        wgt = graph.fwd_wgt[s:e]
        if len(tgt) > top_k:
            top_idx = np.argpartition(wgt, -top_k)[-top_k:]
            tgt, wgt = tgt[top_idx], wgt[top_idx]

        # Direct hit?
        for i in range(len(tgt)):
            if int(tgt[i]) == target:
                path.append(target)
                return path

        max_w = float(max(wgt)) if len(wgt) > 0 else 1.0
        log_max = max(float(np.log1p(max_w)), 1e-6)

        best_t, best_score = -1, -999.0
        for i in range(len(tgt)):
            t = int(tgt[i])
            w = float(wgt[i])
            norm_w = float(np.log1p(w)) / log_max

            tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0

            proximity = 0.0
            ns = int(graph.fwd_off[t])
            ne = int(graph.fwd_off[t + 1])
            for j in range(ns, min(ne, ns + 200)):
                if int(graph.fwd_tgt[j]) == target:
                    proximity = 3.0
                    break
            if proximity == 0:
                n_tgts = set()
                for j in range(ns, min(ne, ns + 100)):
                    n_tgts.add(int(graph.fwd_tgt[j]))
                overlap = len(n_tgts & target_out)
                if overlap > 0:
                    proximity = min(overlap * 0.3, 2.0)

            score = (norm_w * tri_mult) + (proximity * target_pmi)

            visits = visited.get(t, 0)
            if visits >= 3:
                score -= 10.0
            elif visits > 0:
                score -= 0.3 * visits

            if score > best_score:
                best_score = score
                best_t = t

        if best_t < 0 or best_score < -5:
            break

        path.append(best_t)
        prev_token = current
        current = best_t

    return path


# ═══════════════════════════════════════════════════════════════════
# Generate
# ═══════════════════════════════════════════════════════════════════

def generate(graph, prompt_text, max_tokens=20, max_chains=15, verbose=False):
    """Generate text from a prompt.

    Activate -> walk -> deplete -> re-target -> walk -> halt.
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'targets_reached_words': []}

    activated, pmi_scores = activate(graph, prompt_indices)

    generated = []
    visited = {t: 1 for t in prompt_indices}
    current = prompt_indices[-1]
    prev_token = prompt_indices[-2] if len(prompt_indices) > 1 else None
    targets_reached = []
    depleted = set()

    for chain in range(max_chains):
        if len(generated) >= max_tokens:
            break

        context = list(prompt_indices) + generated
        targets = find_targets(graph, prompt_indices, context, activated, pmi_scores)
        targets = [(t, s) for t, s in targets if t not in depleted and t not in visited]

        if not targets:
            if verbose:
                print(f"  [halt: semantic field exhausted after {len(generated)} tokens]")
            break

        dest_idx, dest_score = targets[0]

        if verbose:
            print(f"  chain {chain}: target={graph.idx_to_word[dest_idx]} "
                  f"(PMI={dest_score:.2f}, {len(targets)} remaining)")

        path = walk_to_target(graph, current, dest_idx, dest_score, visited,
                              prev_token=prev_token, max_steps=6)

        reached = len(path) > 0 and path[-1] == dest_idx

        if not reached:
            depleted.add(dest_idx)
            if verbose:
                print(f"    missed (organic pruning)")
            continue

        for t in path:
            if len(generated) >= max_tokens:
                break
            generated.append(t)
            visited[t] = visited.get(t, 0) + 1
            prev_token = current
            current = t

        targets_reached.append(dest_idx)
        depleted.add(dest_idx)

        if verbose:
            print(f"    reached: {graph.detokenize(path)}")

    return {
        'text': graph.detokenize(prompt_indices + generated),
        'prompt': prompt_text,
        'generated_text': graph.detokenize(generated),
        'generated_tokens': generated,
        'n_generated': len(generated),
        'targets_reached': len(targets_reached),
        'targets_reached_words': [graph.idx_to_word[t] for t in targets_reached],
        'targets_depleted': len(depleted),
        'activated_size': len(activated),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS = [
    "The king",
    "She opened the door",
    "The army marched",
    "Scientists discovered",
    "The fire burned",
    "The river flows",
    "The ship sailed",
    "Dark clouds",
    "The music played softly",
    "The volcano erupted",
]


def main():
    parser = argparse.ArgumentParser(
        description="LLN - Zero-parameter language generation from graph topology")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to LMDB model directory (auto-downloads if not set)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt (or omit for demo)")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    parser.add_argument("--verbose", action="store_true",
                        help="Show target selection details")
    args = parser.parse_args()

    # Resolve model path
    if args.model:
        model_path = args.model
    else:
        model_path = download_model()

    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    load_time = time.time() - t0
    print(f"done ({load_time:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,} | "
          f"PMI: {len(graph.pmi_tgt):,} | Trigrams: {len(graph.trigrams):,}")
    print()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for prompt in prompts:
        t0 = time.time()
        result = generate(graph, prompt, max_tokens=args.max_tokens, verbose=args.verbose)
        gen_time = time.time() - t0

        print(f"  \"{prompt}\"")
        print(f"  -> {result['generated_text']}")
        print(f"     [{result['n_generated']} tokens, {result['targets_reached']} targets, "
              f"{gen_time:.3f}s]")
        print()

    graph.close()


if __name__ == "__main__":
    main()
