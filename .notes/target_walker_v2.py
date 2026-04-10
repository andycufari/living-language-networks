"""
Target Walker v2 — Refined Path Memory
=======================================

Refinements over target_walker_with_memory.py:

1. CONTENT-ONLY ON BOTH SIDES:
   - Path side: only content tokens from the window contribute to memory.
     Function words and punctuation are filtered out (they connect to
     everything — that's noise, not memory).
   - Candidate side: memory only nudges CONTENT candidates. Function-word
     candidates decide by grammar alone (raw weight * trigram). This
     preserves grammar primacy at function-word positions.

2. MULTIPLICATIVE NUDGE:
   Instead of `step_score += memory_alpha * density`,
   we use       `step_score *= (1 + memory_alpha * density)` for content
   candidates only. This means memory amplifies the candidate's existing
   grammar score rather than competing with it on the same scale.

3. ALL FIVE COEFFICIENTS EXPOSED:
   grammar_weight, proximity_direct, proximity_overlap, pull_strength,
   memory_alpha. Default values match the shipped walker so we can A/B
   by tweaking one at a time.

Definition of "content token":
   not in top-150 by in_degree AND not punctuation
   (Model-agnostic by rank, not absolute in_degree threshold.)

Usage:
    python .notes/target_walker_v2.py --prompt "The fire burned"
    python .notes/target_walker_v2.py --memory-alpha 1.0 --verbose
    python .notes/target_walker_v2.py --grammar-weight 0.2 --pull-strength 0.5
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np

from generate import LLNGraph, activate, find_targets, download_model
from vocab_rank import VocabRank


PUNCTUATION = {'.', ',', ';', ':', '!', '?', '(', ')', '"', '-', "'"}


# NOTE: we intentionally do NOT fork activate() here.
#
# The first PMI activation operates on the STATIC full graph where the
# in_degree distribution is known and fixed. The `in_degree > 20000`
# filter in the shipped activate() is principled: it's measuring against
# global topology to skip hubs whose neighborhoods are too broad to be
# topical. Changing it to rank-based causes downstream drift because the
# cutoffs shift slightly and target discovery changes.
#
# The rank-based fix belongs DOWNSTREAM of activation, inside find_targets
# and inside density/memory calculation — places that operate on the
# activated subnetwork, where global in_degree no longer reflects local
# behavior.
#
# We use the shipped `activate()` unchanged in generate_v2.


def find_targets_v2(graph, prompt_indices, context_indices, activated, pmi_scores,
                     tokens_remaining=20, sink_mode='reward', vocab_rank=None):
    """find_targets re-calibrated for v16 topology + rank-based filter.

    v16 topology finding (TOPOLOGY_DEBUG_V16.md): in a blended 3-corpus
    graph, content words became global sinks (fire=0.43, door=0.22,
    extinguisher=0.11). The shipped find_targets penalizes sinks — which
    means it penalizes ALL content targets in v16, leaving only generic
    throughput words.

    RANK-BASED FILTER: replaces `in_degree > 15000` with tier check.
    In v16, `discovered` had in_degree 15990 (just over 15000) but rank 628
    (clearly content). The absolute threshold was silently filtering
    "discovered" as a target, breaking "Scientists discovered" prompts.
    Rank-based filter keeps semantic intent without corpus-size drift.

    sink_mode options:
      'penalty' — shipped behavior (sink = bad, throughput = good)
      'neutral' — ignore pr_ratio entirely, rank by PMI only
      'reward'  — INVERT: sinks are topical attractors, reward them.
                  Used because in v16 content targets ARE sinks.
    """
    context_set = set(context_indices)
    last = context_indices[-1]

    # Dual-anchored reachability (unchanged)
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
        # Rank-based target filter (was: in_degree > 15000).
        #
        # Uses is_noise (top n_semi_function). Calibration shows this
        # is prompt-dependent: default prompts prefer higher cutoff
        # (fewer semi-common words as targets), weather prompts prefer
        # lower cutoff (more targets available in sparse fields).
        # Fixed at n_semi_function=500 for now; adaptive version in Phase 11.
        if vocab_rank is not None and vocab_rank.is_noise(t):
            continue
        hop_dist, _ = reachable[t]
        pmi_score = pmi_scores.get(t, 0)
        score = pmi_score * (4 - hop_dist)

        out_deg = graph.out_degree(t)
        in_deg = max(1, int(graph.in_degree[t]))
        pr_ratio = out_deg / in_deg

        if sink_mode == 'penalty':
            # Shipped behavior
            if pr_ratio < 0.4:
                if tokens_remaining > 5:
                    score *= 0.2
            elif pr_ratio >= 0.9:
                score *= 1.5
        elif sink_mode == 'reward':
            # v16 inversion: sinks are topical attractors
            # Deeper sink = stronger attractor = better target
            if pr_ratio < 0.4:
                score *= 1.5  # reward content sinks
            elif pr_ratio >= 2.0:
                # Too source-like = sentence starters, not targets
                score *= 0.5
        # 'neutral' = no adjustment, pure PMI ranking

        targets.append((t, score, pr_ratio))

    targets.sort(key=lambda x: -x[1])
    return targets


def build_content_mask(graph, vocab_rank=None, top_n_function=150):
    """Return set of token indices that are 'noise' for memory density.

    DEPRECATED: prefer passing a VocabRank and using vocab_rank.noise_set.
    This function is kept for backward compat with older call sites.
    """
    if vocab_rank is not None:
        # Union the function+semi tiers with punctuation tokens
        punct_ids = set()
        for i in range(graph.vocab_size):
            if graph.idx_to_word[i] in PUNCTUATION:
                punct_ids.add(i)
        return vocab_rank.noise_set | punct_ids
    # Fallback: old behavior
    in_deg_sorted = np.argsort(-graph.in_degree[:graph.vocab_size])
    true_function = set(int(i) for i in in_deg_sorted[:top_n_function])
    punct_ids = set()
    for i in range(graph.vocab_size):
        if graph.idx_to_word[i] in PUNCTUATION:
            punct_ids.add(i)
    return true_function | punct_ids


def walk_to_target_v2(graph, start, target, target_pmi, visited,
                       prev_token=None, max_steps=8, top_k=50,
                       beam_width=5,
                       full_path_window=None, noise_set=None,
                       grammar_weight=0.1,
                       proximity_direct=3.0,
                       proximity_overlap=0.3,
                       pull_strength=1.0,
                       memory_alpha=1.0):
    """Beam search with refined path memory.

    Scoring:
      grammar = grammar_weight * log1p(w) * tri_mult
      prox    = proximity_direct * direct_hit
              + proximity_overlap * overlap
      pull    = pull_strength * log1p(target_pmi) * (direct_hit or overlap>0)
      base    = grammar + prox + pull

      if candidate is content AND memory_alpha > 0:
          density = content_connections / max(content_window_size, 1)
          score = base * (1 + memory_alpha * density)
      else:
          score = base
    """
    # Target forward neighbors for proximity
    ts = int(graph.full_off[target])
    te = int(graph.full_off[target + 1])
    target_out = set()
    for i in range(ts, min(te, ts + 500)):
        target_out.add(int(graph.full_tgt[i]))

    # Build window forward-edge sets — CONTENT TOKENS ONLY
    # (path side of memory filter)
    window_forward = {}
    if full_path_window and noise_set is not None:
        for wp in set(full_path_window):
            if wp in noise_set:
                continue  # skip function words & punctuation from window
            s = int(graph.full_off[wp])
            e = int(graph.full_off[wp + 1])
            window_forward[wp] = set(int(graph.full_tgt[j]) for j in range(s, e))
    content_window_size = max(len(window_forward), 1)

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

                if t in path_set:
                    continue

                # Grammar
                norm_w = float(np.log1p(w)) * grammar_weight
                tri_mult = graph.trigram_score(prev, cur, t) if prev is not None else 1.0
                grammar = norm_w * tri_mult

                # Proximity
                proximity = 0.0
                ns = int(graph.fwd_off[t])
                ne = int(graph.fwd_off[t + 1])
                direct = False
                for j in range(ns, min(ne, ns + 200)):
                    if int(graph.fwd_tgt[j]) == target:
                        proximity = proximity_direct
                        direct = True
                        break
                if not direct:
                    n_tgts = set()
                    for j in range(ns, min(ne, ns + 100)):
                        n_tgts.add(int(graph.fwd_tgt[j]))
                    overlap = len(n_tgts & target_out)
                    if overlap > 0:
                        proximity = min(overlap * proximity_overlap, 2.0)

                # Pull
                pull = proximity * (pull_strength * float(np.log1p(target_pmi)))

                base = grammar + pull  # proximity is already baked into pull

                # Content-only memory nudge (multiplicative)
                is_content_candidate = (noise_set is not None and
                                        t not in noise_set)
                if is_content_candidate and memory_alpha > 0 and window_forward:
                    connections = 0
                    for wp, wp_fwd in window_forward.items():
                        if t in wp_fwd:
                            connections += 1
                    density = connections / content_window_size
                    if density > 0:
                        base = base * (1 + memory_alpha * density)

                step_score = base

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

        candidates.sort(key=lambda c: -c[3] / len(c[2]))
        beam = candidates[:beam_width]

    return []


def generate_v2(graph, prompt_text, max_tokens=20, max_chains=15, verbose=False,
                 grammar_weight=0.1, proximity_direct=3.0,
                 proximity_overlap=0.3, pull_strength=1.0,
                 memory_alpha=1.0, window_size=8,
                 sink_mode='penalty',
                 n_function=150, n_semi_function=500,
                 vocab_rank=None):
    """Generate with tunable coefficients, refined memory, rank-based filters.

    vocab_rank: optional pre-built VocabRank. If None, builds one from graph
    with n_function / n_semi_function cutoffs. Pass a pre-built one when
    calling generate_v2 many times to avoid rebuilding rank tables.
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'n_generated': 0,
                'targets_reached': 0}

    if vocab_rank is None:
        vocab_rank = VocabRank(graph, n_function=n_function,
                                n_semi_function=n_semi_function)

    # Count prompt content words (same semantic as shipped — <= 20000 in_degree).
    # NOTE: this specific threshold is preserved because it affects top_pct,
    # the activation widening for short prompts. Changing it drifts the
    # activation field. The rank fix is applied DOWNSTREAM in find_targets
    # and density calculation.
    content_words = [i for i in prompt_indices if graph.in_degree[i] <= 20000]
    top_pct = 0.40 if len(content_words) <= 3 else 0.20
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=top_pct)

    # Noise set for density = function + semi-function + punctuation
    noise_set = build_content_mask(graph, vocab_rank=vocab_rank)

    generated = []
    visited = {t: 1 for t in prompt_indices}
    depleted = set()
    targets_reached = []

    current = prompt_indices[-1]
    prev_token = prompt_indices[-2] if len(prompt_indices) > 1 else None
    chain = 0

    if verbose:
        print(f"  Activated: {len(activated)} | "
              f"grammar_w={grammar_weight} prox_d={proximity_direct} "
              f"prox_o={proximity_overlap} pull={pull_strength} "
              f"mem_a={memory_alpha} win={window_size}")

    while len(generated) < max_tokens and chain < max_chains:
        chain += 1
        tokens_remaining = max_tokens - len(generated)
        context = list(prompt_indices) + generated
        if sink_mode == 'penalty':
            # Shipped find_targets uses in_degree > 15000 absolute filter.
            # Leave as-is — this branch is for A/B against shipped behavior.
            targets = find_targets(graph, prompt_indices, context, activated, pmi_scores,
                                   tokens_remaining=tokens_remaining)
        else:
            # v2 find_targets uses rank-based filter via vocab_rank.
            targets = find_targets_v2(graph, prompt_indices, context, activated, pmi_scores,
                                       tokens_remaining=tokens_remaining,
                                       sink_mode=sink_mode,
                                       vocab_rank=vocab_rank)
        targets = [(t, s, pr) for t, s, pr in targets if t not in depleted and t not in visited]

        if not targets:
            if verbose:
                print(f"  [halt: field exhausted after {len(generated)} tokens]")
            break

        fatigue = 1.0 / (1.0 + 0.15 * chain)
        targets = [(t, s * fatigue, pr) for t, s, pr in targets]
        targets.sort(key=lambda x: -x[1])

        dest_idx, dest_score, dest_pr = targets[0]

        if verbose:
            role = ("SINK" if dest_pr < 0.4 else
                    "THROUGHPUT" if 0.9 <= dest_pr <= 1.1 else
                    "SOURCE" if dest_pr > 3.0 else "NEUTRAL")
            print(f"  chain {chain}: target={graph.idx_to_word[dest_idx]} "
                  f"(PMI={dest_score:.2f}, PR={dest_pr:.2f} [{role}], "
                  f"{len(targets)} remaining)")

        full_path = prompt_indices + generated
        full_path_window = full_path[-window_size:]

        path = walk_to_target_v2(
            graph, current, dest_idx, dest_score, visited,
            prev_token=prev_token,
            full_path_window=full_path_window,
            noise_set=noise_set,
            grammar_weight=grammar_weight,
            proximity_direct=proximity_direct,
            proximity_overlap=proximity_overlap,
            pull_strength=pull_strength,
            memory_alpha=memory_alpha,
        )

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
        'n_generated': len(generated),
        'targets_reached': len(targets_reached),
    }


DEFAULT_PROMPTS = [
    "The fire burned",
    "The king",
    "She opened the door",
    "The army marched",
    "Dark clouds",
    "The river flows",
    "Scientists discovered",
    "The old man walked",
]


def main():
    parser = argparse.ArgumentParser(description="LLN Target Walker v2 (tunable)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, nargs='+', default=None)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--grammar-weight", type=float, default=0.1)
    parser.add_argument("--proximity-direct", type=float, default=3.0)
    parser.add_argument("--proximity-overlap", type=float, default=0.3)
    parser.add_argument("--pull-strength", type=float, default=1.0)
    parser.add_argument("--memory-alpha", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--n-function", type=int, default=150,
                        help="Top-N by in_degree = function words")
    parser.add_argument("--n-semi-function", type=int, default=500,
                        help="Top-N cutoff for semi-function tier")
    parser.add_argument("--sink-mode", type=str, default='penalty',
                        choices=['penalty', 'neutral', 'reward'])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,}\n")

    # Build VocabRank once, reuse across prompts
    vr = VocabRank(graph, n_function=args.n_function,
                    n_semi_function=args.n_semi_function)

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    for prompt in prompts:
        t0 = time.time()
        r = generate_v2(graph, prompt,
                         max_tokens=args.max_tokens,
                         grammar_weight=args.grammar_weight,
                         proximity_direct=args.proximity_direct,
                         proximity_overlap=args.proximity_overlap,
                         pull_strength=args.pull_strength,
                         memory_alpha=args.memory_alpha,
                         window_size=args.window_size,
                         sink_mode=args.sink_mode,
                         vocab_rank=vr,
                         verbose=args.verbose)
        elapsed = time.time() - t0
        print(f'  "{prompt}"')
        print(f"  -> {r['generated_text']}")
        print(f"     [{r['n_generated']} tok, {r['targets_reached']} hits, {elapsed:.3f}s]")
        print()

    graph.close()


if __name__ == "__main__":
    main()
