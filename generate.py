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

        # Download data.mdb
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="data.mdb",
            local_dir=model_dir,
            repo_type="model",
        )
        print(f"  Downloaded to {model_dir}")

        # Remove stale lock file if present — LMDB regenerates it
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
        """Top-K forward edges by weight from the frozen CSR arrays."""
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
        """PMI semantic neighbors from the frozen CSR arrays."""
        s, e = int(self.pmi_off[idx]), int(self.pmi_off[idx + 1])
        if s == e:
            return np.array([], np.int32), np.array([], np.float32)
        return self.pmi_tgt[s:e], self.pmi_wgt[s:e]

    def trigram_score(self, prev_idx, cur_idx, next_idx):
        """Trigram multiplier from the base graph.

        If the (prev, cur) pair exists in the trigram index but next_idx
        was never observed after it, that's a grammar violation: hard 0.1x.
        If no trigram data exists for this pair, stay neutral (1.0).
        """
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
        """Out-degree from the frozen graph."""
        return int(self.fwd_off[idx + 1]) - int(self.fwd_off[idx])

    def close(self):
        self.env.close()


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — PMI Activation (Wernicke's Area)
# Defines the semantic field: WHAT to talk about.
# Frozen at T=0 — generated tokens never expand the goal.
# ═══════════════════════════════════════════════════════════════════

def activate(graph, prompt_indices, top_pct=0.20):
    """Build semantic field from prompt via 1-hop PMI.

    Frequency-penalized: raw PMI favors ultra-rare words (proper nouns,
    technical terms). We multiply by log1p(in_degree) so common words
    that are ALSO semantically close get prioritized.

    Capital penalty: capitalized tokens that aren't sentence starters
    (Ages, Horse, Elf, Jedi) are often title fragments, not content.
    Penalize by 0.3x unless the prompt is all proper nouns.
    """
    # Detect if prompt is all proper nouns (e.g. "New York")
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

    # Adjust scores: frequency multiplier + capital penalty
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

    # Return adjusted scores (not raw PMI) so find_targets ranks correctly
    return activated, adjusted


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Target Selection (PMI ∩ Reachability)
# ═══════════════════════════════════════════════════════════════════

def find_targets(graph, prompt_indices, context_indices, activated, pmi_scores,
                 tokens_remaining=20):
    """Find content targets: activated AND reachable in 2-3 hops.

    Flow-aware: uses out-degree / in-degree ratio as a fast proxy for
    local push/receive mass. Sinks (low ratio) are penalized early in
    generation and only allowed as targets in the final stretch.
    """
    context_set = set(context_indices)
    last = context_indices[-1]

    # Dual-anchored: explore from current position AND prompt nodes
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

        # Flow-aware routing: out_degree / in_degree as fast mass proxy
        out_deg = graph.out_degree(t)
        in_deg = max(1, int(graph.in_degree[t]))
        pr_ratio = out_deg / in_deg

        if pr_ratio < 0.4:
            # Sink — penalize unless we're in the final stretch
            if tokens_remaining > 5:
                score *= 0.2
        elif pr_ratio >= 0.9:
            # Throughput or source — boost
            score *= 1.5

        targets.append((t, score, pr_ratio))

    targets.sort(key=lambda x: -x[1])
    return targets


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Walk (Broca's Area)
# ═══════════════════════════════════════════════════════════════════

def walk_to_target(graph, start, target, target_pmi, visited,
                   prev_token=None, max_steps=8, top_k=50, beam_width=5):
    """Beam search walk from start toward target.

    Maintains beam_width competing paths. Each step expands all paths,
    scores candidates using global absolute log-mass (not local normalization),
    and keeps the top beam_width by average score per step.

    Returns the first path that hits the target, or [] if none do.
    """
    # Precompute target's forward neighbors for proximity scoring
    ts = int(graph.full_off[target])
    te = int(graph.full_off[target + 1])
    target_out = set()
    for i in range(ts, min(te, ts + 500)):
        target_out.add(int(graph.full_tgt[i]))

    # Each beam entry: (current_token, prev_token, path_tokens, cumulative_score)
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

            for i in range(len(tgt)):
                t = int(tgt[i])
                w = float(wgt[i])

                # Strict path loop trap: never step on your own footprints
                if t in path_set:
                    continue

                # Global absolute log-mass (no local normalization)
                norm_w = float(np.log1p(w)) * 0.1

                tri_mult = graph.trigram_score(prev, cur, t) if prev is not None else 1.0

                # Proximity: can t reach target in 1 hop? Neighborhood overlap?
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

                # Log-compress target_pmi so pull doesn't drown grammar
                # Without log: grammar ~3, pull ~75. With log: grammar ~3, pull ~10.
                pull_strength = float(np.log1p(target_pmi))
                step_score = (norm_w * tri_mult) + (proximity * pull_strength)

                visits = visited.get(t, 0)
                if visits >= 3:
                    step_score -= 10.0
                elif visits > 0:
                    step_score -= 0.3 * visits

                new_cum = cum_score + step_score
                new_path = path + [t]
                candidates.append((t, cur, new_path, new_cum))

        if not candidates:
            break

        # Prune: keep top beam_width by average score per step
        # Average prevents long paths from winning purely by accumulation
        candidates.sort(key=lambda c: -c[3] / len(c[2]))
        beam = candidates[:beam_width]

    return []


# ═══════════════════════════════════════════════════════════════════
# Grammar-First Walker with Semantic Fencing
# ═══════════════════════════════════════════════════════════════════

def generate_fenced(graph, prompt_text, max_tokens=20, verbose=False):
    """Grammar-first generation with semantic fencing.

    Instead of walking TOWARD semantic targets (pathfinding),
    walks GRAMMATICALLY within a semantic territory (fenced roaming).

    The semantic field is a fence, not a destination. The walker follows
    the strongest trigram-weighted paths within the territory. When it
    lands on an activated content word, it depletes it. When no grammatical
    path stays inside the fence, it halts.
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'n_generated': 0,
                'content_hits': 0, 'content_words': [], 'activated_size': 0}

    # Phase 1: Activate semantic field (unchanged)
    content_words = [i for i in prompt_indices if graph.in_degree[i] <= 20000]
    top_pct = 0.40 if len(content_words) <= 3 else 0.20
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=top_pct)

    # Build the FENCE: activated words + function words + punctuation
    # Function words (in_degree > 5000) are grammatical glue, always allowed
    FUNCTION_THRESHOLD = 5000
    fence = set(activated)
    function_words = set()
    for i in range(graph.vocab_size):
        if graph.in_degree[i] > FUNCTION_THRESHOLD:
            fence.add(i)
            function_words.add(i)
        word = graph.idx_to_word[i]
        if word in {'.', ',', ';', ':', '!', '?', '(', ')', '"', '-', "'"}:
            fence.add(i)

    if verbose:
        n_content = len(activated) - len([i for i in activated if i in function_words])
        print(f"  Fence: {len(fence)} total ({n_content} content + "
              f"{len(function_words)} function words)")

    # Phase 2: Grammar-first walk
    generated = []
    current = prompt_indices[-1]
    prev_token = prompt_indices[-2] if len(prompt_indices) > 1 else None
    visited_count = {t: 1 for t in prompt_indices}
    depleted = set()
    content_hits = []
    stall_count = 0
    MAX_STALL = 3

    for step in range(max_tokens):
        tgt, wgt = graph.get_forward_edges(current, top_k=100)
        if len(tgt) == 0:
            if verbose:
                print(f"  [halt: no forward edges from '{graph.idx_to_word[current]}']")
            break

        # Score candidates: grammar only, filtered by fence
        candidates = []
        for i in range(len(tgt)):
            t = int(tgt[i])
            w = float(wgt[i])

            if t not in fence:
                continue
            if visited_count.get(t, 0) >= 3:
                continue

            norm_w = float(np.log1p(w))
            tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0
            score = norm_w * tri_mult

            # Visit penalty
            v = visited_count.get(t, 0)
            if v > 0:
                score *= 0.5 ** v

            # Content boost: prefer on-topic content over function word chains
            is_content = (t in activated and t not in function_words and t not in depleted)
            if is_content:
                pmi = pmi_scores.get(t, 0)
                score *= (1.0 + float(np.log1p(pmi)) * 0.3)

            candidates.append((t, score))

        if not candidates:
            stall_count += 1
            if stall_count >= MAX_STALL:
                if verbose:
                    print(f"  [halt: {MAX_STALL} consecutive stalls, fence exhausted]")
                break
            # Recovery: allow any grammatical step (temporary fence breach)
            for i in range(len(tgt)):
                t = int(tgt[i])
                w = float(wgt[i])
                if visited_count.get(t, 0) < 3:
                    norm_w = float(np.log1p(w))
                    tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0
                    candidates.append((t, norm_w * tri_mult * 0.3))
            if not candidates:
                if verbose:
                    print(f"  [halt: dead end at '{graph.idx_to_word[current]}']")
                break
        else:
            stall_count = 0

        candidates.sort(key=lambda x: -x[1])
        chosen, chosen_score = candidates[0]

        generated.append(chosen)
        visited_count[chosen] = visited_count.get(chosen, 0) + 1
        prev_token = current
        current = chosen

        is_content = (chosen in activated and
                      chosen not in function_words and
                      chosen not in depleted)
        if is_content:
            depleted.add(chosen)
            content_hits.append(chosen)

        if verbose:
            word = graph.idx_to_word[chosen]
            marker = " * CONTENT" if is_content else ""
            if stall_count > 0:
                marker += " [breach]"
            print(f"  step {step:2d}: {word:15s} (score={chosen_score:.2f}){marker}")

    return {
        'text': graph.detokenize(prompt_indices + generated),
        'prompt': prompt_text,
        'generated_text': graph.detokenize(generated),
        'n_generated': len(generated),
        'content_hits': len(content_hits),
        'content_words': [graph.idx_to_word[t] for t in content_hits],
        'activated_size': len(activated),
        'fence_size': len(fence),
    }


# ═══════════════════════════════════════════════════════════════════
# Target-Driven Generate (original architecture)
# ═══════════════════════════════════════════════════════════════════

def generate(graph, prompt_text, max_tokens=20, max_chains=15, verbose=False):
    """Generate text from a prompt.

    Activate -> walk -> deplete -> re-target -> walk -> halt.
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'targets_reached_words': []}

    # Widen activation for short prompts (≤3 content words)
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

        # Synaptic fatigue: later chains require stronger PMI to fire.
        # Hyperbolic decay — halves at chain ~7, never reaches zero.
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
                              prev_token=prev_token)

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
    parser.add_argument("--fenced", action="store_true",
                        help="Grammar-first walker with semantic fencing (experimental)")
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

    gen_fn = generate_fenced if args.fenced else generate

    if args.interactive:
        mode = "fenced" if args.fenced else "target-driven"
        print(f"Interactive mode ({mode}). Type a prompt, or 'quit' to exit.\n")
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
            result = gen_fn(graph, prompt, max_tokens=args.max_tokens, verbose=args.verbose)
            gen_time = time.time() - t0
            print(f"  -> {result['generated_text']}")
            hits = result.get('content_hits', result.get('targets_reached', 0))
            print(f"     [{result['n_generated']} tokens, {hits} hits, {gen_time:.3f}s]")
            if args.fenced and result.get('content_words'):
                print(f"     content: {', '.join(result['content_words'])}")
            print()
    else:
        prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

        for prompt in prompts:
            t0 = time.time()
            result = gen_fn(graph, prompt, max_tokens=args.max_tokens, verbose=args.verbose)
            gen_time = time.time() - t0

            print(f"  \"{prompt}\"")
            print(f"  -> {result['generated_text']}")
            hits = result.get('content_hits', result.get('targets_reached', 0))
            print(f"     [{result['n_generated']} tokens, {hits} hits, {gen_time:.3f}s]")
            if args.fenced and result.get('content_words'):
                print(f"     content: {', '.join(result['content_words'])}")
            print()

    graph.close()


if __name__ == "__main__":
    main()
