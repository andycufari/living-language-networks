#!/usr/bin/env python3
"""LLN — Living Language Network Generator

Zero-parameter language generation from graph topology.
Profile-matching walker: scores candidates by distance from measured
sentence anatomy, not by maximizing edge weight.

Usage:
    python generate.py --prompt "The fire burned"
    python generate.py --prompt "Scientists discovered" --max-tokens 20
    python generate.py --prompt "Dark clouds" --verbose
    python generate.py --interactive

The model is automatically downloaded from HuggingFace on first run.

Requirements: numpy, lmdb, huggingface_hub
    pip install numpy lmdb huggingface_hub
"""
import numpy as np
import lmdb
import json
import struct
import math
import time
import argparse
import os

# HuggingFace model config
HF_REPO = "cufa64/lln_v16_blend_100k"
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

        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="data.mdb",
            local_dir=model_dir,
            repo_type="model",
        )
        print(f"  Downloaded to {model_dir}")

        lock_path = os.path.join(model_dir, "lock.mdb")
        if os.path.exists(lock_path):
            os.remove(lock_path)

        return model_dir
    except ImportError:
        print("  huggingface_hub not installed. Install with: pip install huggingface_hub")
        print("  Or download manually from https://huggingface.co/cufa64/lln_v16_blend_100k")
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
        s, e = int(self.fwd_off[idx]), int(self.fwd_off[idx + 1])
        if s == e:
            return np.array([], np.int32), np.array([], np.float32)
        tgt = self.fwd_tgt[s:e]
        wgt = self.fwd_wgt[s:e]
        if len(tgt) > top_k:
            top = np.argpartition(wgt, -top_k)[-top_k:]
            tgt, wgt = tgt[top], wgt[top]
        return tgt, wgt

    def get_pmi_neighbors(self, idx):
        s, e = int(self.pmi_off[idx]), int(self.pmi_off[idx + 1])
        if s == e:
            return np.array([], np.int32), np.array([], np.float32)
        return self.pmi_tgt[s:e], self.pmi_wgt[s:e]

    def trigram_score(self, prev_idx, cur_idx, next_idx):
        key = prev_idx * self.vocab_size + cur_idx
        targets = self.trigrams.get(key)
        if targets is None:
            return 1.0
        count = targets.get(next_idx, 0)
        if count > 0:
            return 1.0 + float(np.log1p(count))
        return 0.3

    def tokenize(self, text):
        return [self.word_to_idx[w] for w in text.split() if w in self.word_to_idx]

    def detokenize(self, indices):
        return ' '.join(self.idx_to_word[i] for i in indices)

    def out_degree(self, idx):
        return int(self.fwd_off[idx + 1]) - int(self.fwd_off[idx])

    def close(self):
        self.env.close()


# ═══════════════════════════════════════════════════════════════════
# Sentence Profile — measured from 545 real corpus sentences
# ═══════════════════════════════════════════════════════════════════
#
# Each entry: (normalized_position, target_rank, target_pr, target_fwd_weight)
#
# Measured by .notes/sentence_anatomy.py on 545 sentences sampled from
# the v16 training corpus (fineweb-edu + gutenberg + openwebtext).
# The profile is corpus-invariant: all three sources produce profiles
# within 20% of each other on every feature.
#
# Real sentences have a "wave" shape: rank oscillates 105-140, forward
# weight stays in the 700K-1.5M band, pr_ratio stays 0.03-0.05.
# This is the topological signature of English.

SENTENCE_PROFILE = [
    (0.00, 334, 0.05, 433000),
    (0.10, 129, 0.03, 1119000),
    (0.20, 120, 0.03, 1019000),
    (0.30, 134, 0.03, 1026000),
    (0.40, 138, 0.04, 1472000),
    (0.50, 114, 0.04, 1229000),
    (0.60, 132, 0.03, 1079000),
    (0.70, 105, 0.04, 687000),
    (0.80, 136, 0.03, 1384000),
    (0.90, 105, 0.03, 808000),
]


def _profile_at(norm_pos):
    """Interpolate the sentence profile at a normalized position [0, 1]."""
    if norm_pos <= SENTENCE_PROFILE[0][0]:
        return SENTENCE_PROFILE[0][1:]
    if norm_pos >= SENTENCE_PROFILE[-1][0]:
        return SENTENCE_PROFILE[-1][1:]
    for i in range(len(SENTENCE_PROFILE) - 1):
        p0, r0, pr0, w0 = SENTENCE_PROFILE[i]
        p1, r1, pr1, w1 = SENTENCE_PROFILE[i + 1]
        if p0 <= norm_pos <= p1:
            t = (norm_pos - p0) / (p1 - p0)
            return (r0 + t * (r1 - r0), pr0 + t * (pr1 - pr0), w0 + t * (w1 - w0))
    return SENTENCE_PROFILE[-1][1:]


def _build_rank_table(graph):
    """Build rank[idx] = position in frequency-sorted vocab (0 = most common)."""
    ranked = np.argsort(-graph.in_degree[:graph.vocab_size])
    rank = np.empty(graph.vocab_size, dtype=np.int32)
    for r in range(graph.vocab_size):
        rank[int(ranked[r])] = r
    return rank


def _compute_weight_scale(graph):
    """Compute the model's edge weight scale for profile normalization.

    The sentence profile was measured on v16 (mean fwd_weight ~893).
    Other models have different scales. This returns a multiplier so
    that profile forward weights are expressed in model-native units.

    Returns: scale factor to multiply profile fwd_weight targets by.
    """
    # Sample forward edge weights to find the model's mean
    sample = graph.fwd_wgt[:min(len(graph.fwd_wgt), 2000000)]
    sample = sample[sample > 0]
    model_mean = float(np.mean(sample))
    # Profile was measured on v16 where mean sorted fwd_weight ≈ 2082
    PROFILE_MEAN = 2082.0
    scale = model_mean / PROFILE_MEAN
    # Snap to 1.0 if within 5% — avoids float drift on the reference model
    if 0.95 <= scale <= 1.05:
        return 1.0
    return scale


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — PMI Activation (Wernicke's Area)
# ═══════════════════════════════════════════════════════════════════

def activate(graph, prompt_indices, top_pct=0.20):
    """Build semantic field from prompt via 1-hop PMI.

    Frequency-penalized: raw PMI favors ultra-rare words (proper nouns,
    technical terms). We multiply by log1p(in_degree) so common words
    that are ALSO semantically close get prioritized.

    Capital penalty: capitalized tokens that aren't sentence starters
    are often title fragments (Ages, Horse, Elf, Jedi). Penalize by 0.3x.
    """
    prompt_all_proper = all(
        graph.idx_to_word[i][0].isupper() for i in prompt_indices
        if graph.idx_to_word[i][0].isalpha()
    )

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

    adjusted = {}
    for t, raw_pmi in all_pmi.items():
        freq_mult = float(np.log1p(graph.in_degree[t]))
        score = raw_pmi * freq_mult
        word = graph.idx_to_word[t]
        if not prompt_all_proper and word[0:1].isupper() and word[0:1].isalpha():
            score *= 0.3
        adjusted[t] = score

    weights = sorted(adjusted.values(), reverse=True)
    threshold = weights[int(len(weights) * top_pct)] if len(weights) > 5 else 0
    activated = set(prompt_indices)
    for t, score in adjusted.items():
        if score >= threshold:
            activated.add(t)

    return activated, adjusted


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Target Selection (PMI + Reachability + Flow)
# ═══════════════════════════════════════════════════════════════════

def find_targets(graph, prompt_indices, context_indices, activated, pmi_scores,
                 tokens_remaining=20):
    """Find content targets: activated AND reachable in 2-3 hops.

    Flow-aware: uses out-degree / in-degree ratio as a fast proxy for
    local push/receive mass. Sinks are penalized early in generation
    and only allowed as targets in the final stretch.
    """
    context_set = set(context_indices)
    last = context_indices[-1]

    anchor_nodes = set(prompt_indices) | {last}
    reachable = {}
    for anchor in anchor_nodes:
        tgt1, wgt1 = graph.get_forward_edges(anchor, top_k=50)
        for i in range(len(tgt1)):
            t = int(tgt1[i])
            if t not in context_set:
                if t not in reachable or reachable[t][0] > 1:
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

        out_deg = graph.out_degree(t)
        in_deg = max(1, int(graph.in_degree[t]))
        pr_ratio = out_deg / in_deg

        if pr_ratio < 0.4:
            if tokens_remaining > 5:
                score *= 0.2
        elif pr_ratio >= 0.9:
            score *= 1.5

        targets.append((t, score, pr_ratio))

    targets.sort(key=lambda x: -x[1])
    return targets


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Walk (Broca's Area) — Profile-Matching Beam Search
# ═══════════════════════════════════════════════════════════════════

def walk_to_target(graph, start, target, target_pmi, visited, rank_table,
                   prev_token=None, max_steps=8, top_k=50, beam_width=5,
                   full_path_length=0, projected_length=15, weight_scale=1.0):
    """Profile-matching beam search walk from start toward target.

    At each step, candidates are scored by how close their topological
    features (rank, pr_ratio, forward weight) are to the measured
    sentence profile at their normalized position. This produces output
    whose topological shape matches real English sentences.

    The profile was measured from 545 corpus sentences and is
    corpus-invariant across fineweb-edu, gutenberg, and openwebtext.
    """
    # Precompute target forward neighbors for topical bonus
    ts = int(graph.full_off[target])
    te = int(graph.full_off[target + 1])
    target_out = set()
    for i in range(ts, min(te, ts + 500)):
        target_out.add(int(graph.full_tgt[i]))

    beam = [(start, prev_token, [], 0.0)]

    for step in range(max_steps):
        candidates = []

        for cur, prev, path, cum_score in beam:
            tgt, wgt = graph.get_forward_edges(cur, top_k=top_k)
            if len(tgt) == 0:
                continue

            # Direct hit — return immediately
            for i in range(len(tgt)):
                if int(tgt[i]) == target:
                    return path + [target]

            path_set = set(path)

            # Position in the sentence for profile interpolation
            norm_pos = min(1.0, max(0.0,
                (full_path_length + len(path) + 1) / max(projected_length, 1)))
            p_rank, p_pr, p_fwd_w = _profile_at(norm_pos)
            target_rank = p_rank
            target_pr = p_pr
            target_fwd_w = p_fwd_w * weight_scale  # scale to model's edge weight range

            for i in range(len(tgt)):
                t = int(tgt[i])
                w = float(wgt[i])

                if t in path_set:
                    continue

                # Candidate topology
                cand_rank = int(rank_table[t])
                cand_in_deg = int(graph.in_degree[t])
                cand_out_deg = graph.out_degree(t)
                cand_pr = cand_out_deg / max(cand_in_deg, 1)

                # Profile distance (lower = closer to real sentence shape)
                # Weights: rank 0.5, pr 20.0, fwd 1.5 (tuned via .notes/tune_manual.py)
                rank_dist = abs(math.log1p(cand_rank) - math.log1p(target_rank)) * 0.5
                pr_dist = abs(cand_pr - target_pr) * 20.0
                fwd_dist = abs(math.log1p(w) - math.log1p(target_fwd_w)) * 1.5
                distance = rank_dist + pr_dist + fwd_dist

                # Topical bonus: does this candidate reach the target?
                topical = 0.0
                ns = int(graph.fwd_off[t])
                ne = int(graph.fwd_off[t + 1])
                direct_hit = False
                for j in range(ns, min(ne, ns + 200)):
                    if int(graph.fwd_tgt[j]) == target:
                        topical = 2.0
                        direct_hit = True
                        break
                if not direct_hit:
                    n_tgts = set()
                    for j in range(ns, min(ne, ns + 100)):
                        n_tgts.add(int(graph.fwd_tgt[j]))
                    overlap = len(n_tgts & target_out)
                    if overlap > 0:
                        topical = min(overlap * 0.2, 1.5)

                topical *= math.log1p(max(target_pmi, 0.1))

                # Score: minimize distance, add topical pull (2.5 tuned)
                score = -distance + (2.5 * topical)

                visits = visited.get(t, 0)
                if visits >= 3:
                    score -= 10.0
                elif visits > 0:
                    score -= 0.5 * visits

                new_path = path + [t]
                new_cum = cum_score + score
                candidates.append((t, cur, new_path, new_cum))

        if not candidates:
            break

        candidates.sort(key=lambda c: -c[3] / len(c[2]))
        beam = candidates[:beam_width]

    return []


# ═══════════════════════════════════════════════════════════════════
# Generate
# ═══════════════════════════════════════════════════════════════════

def generate(graph, prompt_text, max_tokens=20, max_chains=15, verbose=False,
             rank_table=None, projected_length=15, weight_scale=None):
    """Generate text from a prompt.

    Activate -> find targets -> profile-matching walk -> deplete -> repeat.
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '',
                'n_generated': 0, 'targets_reached': 0,
                'targets_reached_words': []}

    if rank_table is None:
        rank_table = _build_rank_table(graph)
    if weight_scale is None:
        weight_scale = _compute_weight_scale(graph)

    content_words = [i for i in prompt_indices if graph.in_degree[i] <= 20000]
    top_pct = 0.40 if len(content_words) <= 3 else 0.20
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=top_pct)

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
        tokens_remaining = max_tokens - len(generated)
        targets = find_targets(graph, prompt_indices, context, activated, pmi_scores,
                               tokens_remaining=tokens_remaining)
        targets = [(t, s, pr) for t, s, pr in targets if t not in depleted and t not in visited]

        if not targets:
            if verbose:
                print(f"  [halt: semantic field exhausted after {len(generated)} tokens]")
            break

        fatigue = 1.0 / (1.0 + 0.15 * chain)
        targets = [(t, s * fatigue, pr) for t, s, pr in targets]
        targets.sort(key=lambda x: -x[1])

        dest_idx, dest_score, dest_pr = targets[0]

        if verbose:
            if dest_pr < 0.4:
                role = "SINK"
            elif 0.9 <= dest_pr <= 1.1:
                role = "THROUGHPUT"
            elif dest_pr > 3.0:
                role = "SOURCE"
            else:
                role = "NEUTRAL"
            print(f"  chain {chain}: target={graph.idx_to_word[dest_idx]} "
                  f"(PMI={dest_score:.2f}, PR={dest_pr:.2f} [{role}], {len(targets)} remaining)")

        path = walk_to_target(graph, current, dest_idx, dest_score, visited,
                              rank_table, prev_token=prev_token,
                              full_path_length=len(prompt_indices) + len(generated),
                              projected_length=projected_length,
                              weight_scale=weight_scale)

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
    parser.add_argument("--prompt", type=str, nargs='+', default=None,
                        help="One or more prompts (or omit for demo)")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    parser.add_argument("--verbose", action="store_true",
                        help="Show target selection details")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: load model once, prompt repeatedly")
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        model_path = download_model()

    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    rank_table = _build_rank_table(graph)
    weight_scale = _compute_weight_scale(graph)
    load_time = time.time() - t0
    print(f"done ({load_time:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,} | "
          f"PMI: {len(graph.pmi_tgt):,} | Trigrams: {len(graph.trigrams):,}")
    print()

    if args.interactive:
        print("Interactive mode. Type a prompt, or 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                if prompt.lower() in ("quit", "exit", "q"):
                    print("Bye.")
                break
            t0 = time.time()
            result = generate(graph, prompt, max_tokens=args.max_tokens,
                              verbose=args.verbose, rank_table=rank_table,
                              weight_scale=weight_scale)
            gen_time = time.time() - t0
            print(f"  -> {result['generated_text']}")
            print(f"     [{result['n_generated']} tokens, "
                  f"{result['targets_reached']} hits, {gen_time:.3f}s]")
            print()
    else:
        prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

        for prompt in prompts:
            t0 = time.time()
            result = generate(graph, prompt, max_tokens=args.max_tokens,
                              verbose=args.verbose, rank_table=rank_table,
                              weight_scale=weight_scale)
            gen_time = time.time() - t0

            print(f"  \"{prompt}\"")
            print(f"  -> {result['generated_text']}")
            print(f"     [{result['n_generated']} tokens, "
                  f"{result['targets_reached']} hits, {gen_time:.3f}s]")
            print()

    graph.close()


if __name__ == "__main__":
    main()
