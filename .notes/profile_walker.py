"""
Profile Walker (v3) — ride the wave

Instead of maximizing grammar + proximity + pull + memory, this walker
minimizes distance from the measured sentence profile at each position.

Sentence anatomy (from 545 real corpus sentences, sentence_anatomy.py):

  position    rank    pr_ratio   fwd_weight
  ----------- ------- ---------- -----------
  0.00        334     0.05       433K      (sentence start, source-like)
  0.10        129     0.03       1119K     (first connector slot)
  0.20        120     0.03       1019K
  0.30        134     0.03       1026K
  0.40        138     0.04       1472K
  0.50        114     0.04       1229K     (mid-sentence peak)
  0.60        132     0.03       1079K
  0.70        105     0.04       687K
  0.80        136     0.03       1384K
  0.90        105     0.03       808K      (near end)

Key properties:
  - Rank stays in 105-140 band throughout (except position 0)
  - Forward weight stays in 700K-1.5M band (NOT maximum, centered)
  - Pr_ratio stays in 0.03-0.05 (moderate sinks, not deep sinks)
  - Trigram coverage climbs from 0% to ~70% (not 100%)

Walker objective:
  score(candidate) = topical_bonus - distance_from_profile

Where distance is log-space for rank and forward weight (since these
span orders of magnitude), and linear for pr_ratio.

Architecture preserves v1's proven two-phase structure:
  1. find_targets (unchanged) — picks content targets
  2. walk_to_target (rewritten) — profile-matching beam score

Uses rank-based VocabRank for filtering. Keeps shipped activate().
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import time
import numpy as np

from generate import LLNGraph, activate, find_targets, download_model
from vocab_rank import VocabRank


# Measured sentence profile from .notes/SENTENCE_ANATOMY.md (545 sentences)
# Each entry: (norm_position, target_rank, target_pr, target_fwd_weight)
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


def profile_at(norm_pos):
    """Interpolate profile at arbitrary normalized position [0, 1]."""
    if norm_pos <= SENTENCE_PROFILE[0][0]:
        return SENTENCE_PROFILE[0][1:]
    if norm_pos >= SENTENCE_PROFILE[-1][0]:
        return SENTENCE_PROFILE[-1][1:]
    for i in range(len(SENTENCE_PROFILE) - 1):
        p0, r0, pr0, w0 = SENTENCE_PROFILE[i]
        p1, r1, pr1, w1 = SENTENCE_PROFILE[i + 1]
        if p0 <= norm_pos <= p1:
            t = (norm_pos - p0) / (p1 - p0)
            return (
                r0 + t * (r1 - r0),
                pr0 + t * (pr1 - pr0),
                w0 + t * (w1 - w0),
            )
    return SENTENCE_PROFILE[-1][1:]


def profile_distance(candidate_rank, candidate_pr, candidate_fwd_w,
                      target_rank, target_pr, target_fwd_w,
                      weight_rank=1.0, weight_pr=20.0, weight_fwd=1.0):
    """Distance from candidate topology to target profile.

    Lower is better. Uses log-space for rank and forward weight
    (they span orders of magnitude) and linear space for pr_ratio.
    """
    # Rank distance (log-space)
    rank_dist = abs(math.log1p(candidate_rank) - math.log1p(target_rank))

    # Pr distance (linear, scaled up because values are small)
    pr_dist = abs(candidate_pr - target_pr) * weight_pr

    # Forward weight distance (log-space)
    fwd_dist = abs(math.log1p(candidate_fwd_w) - math.log1p(target_fwd_w))

    return (weight_rank * rank_dist) + pr_dist + (weight_fwd * fwd_dist)


def walk_to_target_profile(graph, start, target, target_pmi, visited,
                            vocab_rank,
                            prev_token=None, max_steps=8, top_k=50,
                            beam_width=5,
                            full_path_length=0, projected_length=15,
                            topical_weight=2.0,
                            weight_rank=1.0, weight_pr=20.0, weight_fwd=1.0,
                            verbose=False):
    """Profile-matching beam search walk from start toward target.

    For each candidate, compute the expected profile at its position
    (normalized by projected sentence length) and score by negative
    distance to that profile plus a topical bonus.
    """
    # Precompute target neighbors for topical bonus (proximity to target)
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

            # Compute position for scoring: where would this candidate sit
            # in the normalized sentence? Use full_path_length + step + 1,
            # divided by projected_length.
            norm_pos = min(1.0, max(0.0,
                (full_path_length + len(path) + 1) / max(projected_length, 1)))
            target_rank, target_pr, target_fwd_w = profile_at(norm_pos)

            for i in range(len(tgt)):
                t = int(tgt[i])
                w = float(wgt[i])  # forward weight from cur to t

                if t in path_set:
                    continue

                # Skip true function words as targets? No — profile walk
                # should pick them NATURALLY at their correct positions.
                # Don't filter.

                # Compute candidate topology
                cand_rank = vocab_rank.get_rank(t)
                cand_in_deg = int(graph.in_degree[t])
                cand_out_deg = int(graph.out_degree(t))
                cand_pr = cand_out_deg / max(cand_in_deg, 1)
                cand_fwd_w = w  # the edge weight cur→t IS the fwd weight

                # Distance from profile (lower = better)
                dist = profile_distance(
                    cand_rank, cand_pr, cand_fwd_w,
                    target_rank, target_pr, target_fwd_w,
                    weight_rank=weight_rank,
                    weight_pr=weight_pr,
                    weight_fwd=weight_fwd,
                )

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

                # Combined score: minimize distance, add topical pull
                # Negate distance so higher = better (maximize score)
                score = -dist + (topical_weight * topical)

                # Revisit penalty
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

        # Keep top beam_width by average score per step
        candidates.sort(key=lambda c: -c[3] / len(c[2]))
        beam = candidates[:beam_width]

    return []


def generate_v3(graph, prompt_text, max_tokens=20, max_chains=15,
                 verbose=False, vocab_rank=None,
                 topical_weight=2.0,
                 weight_rank=1.0, weight_pr=20.0, weight_fwd=1.0,
                 projected_length=15):
    """Generate using profile-matching walker.

    Architecture: same two-phase as main (find_targets → walk_to_target).
    Only the beam score inside walk_to_target is replaced with
    profile-matching.
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'n_generated': 0,
                'targets_reached': 0}

    if vocab_rank is None:
        vocab_rank = VocabRank(graph)

    # Shipped activation (static graph, principled filter — leave alone)
    content_words = [i for i in prompt_indices if graph.in_degree[i] <= 20000]
    top_pct = 0.40 if len(content_words) <= 3 else 0.20
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=top_pct)

    generated = []
    visited = {t: 1 for t in prompt_indices}
    depleted = set()
    targets_reached = []

    current = prompt_indices[-1]
    prev_token = prompt_indices[-2] if len(prompt_indices) > 1 else None
    chain = 0

    if verbose:
        print(f"  Activated: {len(activated)} | topical_w={topical_weight} "
              f"rank_w={weight_rank} pr_w={weight_pr} fwd_w={weight_fwd} "
              f"proj_len={projected_length}")

    while len(generated) < max_tokens and chain < max_chains:
        chain += 1
        tokens_remaining = max_tokens - len(generated)
        context = list(prompt_indices) + generated
        # Use shipped find_targets for target selection (proven to work)
        targets = find_targets(graph, prompt_indices, context, activated,
                                pmi_scores, tokens_remaining=tokens_remaining)
        targets = [(t, s, pr) for t, s, pr in targets
                    if t not in depleted and t not in visited]

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

        path = walk_to_target_profile(
            graph, current, dest_idx, dest_score, visited, vocab_rank,
            prev_token=prev_token,
            full_path_length=len(prompt_indices) + len(generated),
            projected_length=projected_length,
            topical_weight=topical_weight,
            weight_rank=weight_rank,
            weight_pr=weight_pr,
            weight_fwd=weight_fwd,
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
    parser = argparse.ArgumentParser(description="LLN v3 profile walker")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, nargs='+', default=None)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--topical-weight", type=float, default=2.0)
    parser.add_argument("--weight-rank", type=float, default=1.0)
    parser.add_argument("--weight-pr", type=float, default=20.0)
    parser.add_argument("--weight-fwd", type=float, default=1.0)
    parser.add_argument("--projected-length", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,}\n")

    vr = VocabRank(graph)

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    for prompt in prompts:
        t0 = time.time()
        r = generate_v3(graph, prompt,
                         max_tokens=args.max_tokens,
                         vocab_rank=vr,
                         topical_weight=args.topical_weight,
                         weight_rank=args.weight_rank,
                         weight_pr=args.weight_pr,
                         weight_fwd=args.weight_fwd,
                         projected_length=args.projected_length,
                         verbose=args.verbose)
        elapsed = time.time() - t0
        print(f'  "{prompt}"')
        print(f"  -> {r['generated_text']}")
        print(f"     [{r['n_generated']} tok, {r['targets_reached']} hits, {elapsed:.3f}s]")
        print()

    graph.close()


if __name__ == "__main__":
    main()
