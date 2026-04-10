"""
Target Walker with Path Memory — isolated hypothesis test
==========================================================

Architecture (from the shipped generate.py — unchanged):
  1. Activate PMI field from prompt -> subnetwork
  2. Pick target from activated content words (flow-aware, depleting)
  3. Beam search walks current -> target
  4. Reach -> deplete -> next target
  5. Halt when semantic field exhausted

HYPOTHESIS being tested here, in isolation:
  Inside the beam search, when scoring a candidate next token,
  the score should consider how connected the candidate is to
  tokens ALREADY GENERATED in the full path (prompt + all prior
  walks), not just to the current sub-walk.

  Current shipped scoring:
    step_score = (log1p(w) * 0.1 * trigram) + (proximity * log1p(target_pmi))
    - Punishes loops inside this walk only (path_set = current walk path).
    - Does not know about tokens from previous walks.

  Proposed scoring:
    step_score = (log1p(w) * 0.1 * trigram)
               + (proximity * log1p(target_pmi))
               + (memory_alpha * path_memory)
    where:
      path_memory = count(full_path_window tokens connected to candidate) / window_size
      connection = candidate has a forward edge to OR from a full_path_window token
      full_path_window = last N tokens of (prompt + generated so far + current walk path)

This is a SECONDARY ranking signal. It doesn't replace anything. It
adds a topological-memory nudge that helps the beam prefer candidates
that are topologically connected to what the sentence has already said.

We compare A/B:
  A = shipped generate.py walker (ground truth)
  B = same, but with path_memory term in beam scoring

Run:
    python .notes/target_walker_with_memory.py --variant both
    python .notes/target_walker_with_memory.py --variant B --memory-alpha 2.0 --verbose
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np

from generate import (
    LLNGraph, activate, find_targets, download_model,
    walk_to_target as walk_to_target_shipped,
    generate as generate_shipped,
)


# ═══════════════════════════════════════════════════════════════════
# Experimental walk_to_target with path memory
# ═══════════════════════════════════════════════════════════════════

def walk_to_target_memory(graph, start, target, target_pmi, visited,
                           prev_token=None, max_steps=8, top_k=50,
                           beam_width=5,
                           full_path_window=None, memory_alpha=2.0,
                           content_only=True):
    """Beam search walk from start toward target, with path memory.

    Identical to the shipped walk_to_target except:
    - Takes full_path_window: list of tokens recently generated (last N
      tokens of the overall sentence, across all walks).
    - Adds memory_term = memory_alpha * density
      where density = connections_to_CONTENT_window_tokens / content_window_size

    content_only (default True): path memory only counts connections to
    CONTENT tokens in the window, not function words or punctuation.
    Rationale: every candidate trivially connects to "the", ",", ".", etc.
    That's not memory, it's noise. Genuine topical memory is connection
    to the content words that define what the sentence is about.
    """
    # Precompute target forward neighbors for proximity (unchanged)
    ts = int(graph.full_off[target])
    te = int(graph.full_off[target + 1])
    target_out = set()
    for i in range(ts, min(te, ts + 500)):
        target_out.add(int(graph.full_tgt[i]))

    # Precompute the full-path-window edge cache.
    # Only CONTENT tokens from the window contribute to memory — function
    # words and punctuation are noise (they connect to everything).
    # True function words: top ~150 tokens by in_degree (measured empirically).
    PUNCTUATION = {'.', ',', ';', ':', '!', '?', '(', ')', '"', '-', "'"}
    in_deg_sorted = np.argsort(-graph.in_degree[:graph.vocab_size])
    true_function = set(int(i) for i in in_deg_sorted[:150])

    window_forward = {}  # content token -> set of forward-edge targets
    if full_path_window:
        for wp in set(full_path_window):
            if content_only:
                if wp in true_function:
                    continue
                if graph.idx_to_word[wp] in PUNCTUATION:
                    continue
            s = int(graph.full_off[wp])
            e = int(graph.full_off[wp + 1])
            window_forward[wp] = set(int(graph.full_tgt[j]) for j in range(s, e))
    window_size = len(window_forward) if window_forward else 1

    beam = [(start, prev_token, [], 0.0)]

    for step in range(max_steps):
        candidates = []

        for cur, prev, path, cum_score in beam:
            tgt, wgt = graph.get_forward_edges(cur, top_k=top_k)
            if len(tgt) == 0:
                continue

            # Direct hit — return immediately (same as shipped)
            for i in range(len(tgt)):
                if int(tgt[i]) == target:
                    return path + [target]

            path_set = set(path)

            for i in range(len(tgt)):
                t = int(tgt[i])
                w = float(wgt[i])

                if t in path_set:
                    continue

                norm_w = float(np.log1p(w)) * 0.1
                tri_mult = graph.trigram_score(prev, cur, t) if prev is not None else 1.0

                # Proximity (unchanged)
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

                pull_strength = float(np.log1p(target_pmi))
                step_score = (norm_w * tri_mult) + (proximity * pull_strength)

                # === THE EXPERIMENT: path memory term ===
                # How many window tokens are connected to this candidate?
                # Bidirectional: window_tok -> candidate OR candidate -> window_tok.
                if window_forward:
                    connections = 0
                    # window -> candidate
                    for wp, wp_fwd in window_forward.items():
                        if t in wp_fwd:
                            connections += 1
                    # candidate -> window (only check if not already counted via window->cand)
                    t_ns = int(graph.full_off[t])
                    t_ne = int(graph.full_off[t + 1])
                    cand_fwd_set = set(int(graph.full_tgt[j]) for j in range(t_ns, t_ne))
                    for wp in window_forward:
                        if wp in cand_fwd_set and t not in window_forward[wp]:
                            connections += 1
                    density = connections / window_size
                    step_score += memory_alpha * density
                # === END experiment ===

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


# ═══════════════════════════════════════════════════════════════════
# Generate loop — copy of shipped generate() but calls our walk
# ═══════════════════════════════════════════════════════════════════

def generate_with_memory(graph, prompt_text, max_tokens=20, verbose=False,
                          memory_alpha=2.0, window_size=8):
    """Same as shipped generate() but walks use path memory."""
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'n_generated': 0,
                'targets_reached': 0}

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
        print(f"  Activated: {len(activated)} | memory_alpha={memory_alpha} "
              f"| window_size={window_size}")

    while len(generated) < max_tokens:
        chain += 1
        tokens_remaining = max_tokens - len(generated)
        context = list(prompt_indices) + generated
        targets = find_targets(graph, prompt_indices, context, activated, pmi_scores,
                               tokens_remaining=tokens_remaining)
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

        # Build full path window: prompt + everything generated so far, last N tokens
        full_path = prompt_indices + generated
        full_path_window = full_path[-window_size:]

        path = walk_to_target_memory(
            graph, current, dest_idx, dest_score, visited,
            prev_token=prev_token,
            full_path_window=full_path_window,
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
    parser = argparse.ArgumentParser(description="LLN Target Walker with Path Memory")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, nargs='+', default=None)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--variant", choices=['A', 'B', 'both'], default='both',
                        help="A=shipped walker, B=shipped walker + path memory")
    parser.add_argument("--memory-alpha", type=float, default=2.0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,}\n")

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    variants = ['A', 'B'] if args.variant == 'both' else [args.variant]

    for prompt in prompts:
        print(f'  "{prompt}"')
        for v in variants:
            t0 = time.time()
            if v == 'A':
                r = generate_shipped(graph, prompt, max_tokens=args.max_tokens,
                                      verbose=args.verbose)
            else:
                r = generate_with_memory(graph, prompt,
                                          max_tokens=args.max_tokens,
                                          memory_alpha=args.memory_alpha,
                                          window_size=args.window_size,
                                          verbose=args.verbose)
            elapsed = time.time() - t0
            label = "A(shipped)" if v == 'A' else f"B(mem a={args.memory_alpha} w={args.window_size})"
            print(f"  [{label}] -> {r['generated_text']}")
            print(f"     [{r['n_generated']} tok, {r.get('targets_reached', 0)} hits, {elapsed:.3f}s]")
        print()

    graph.close()


if __name__ == "__main__":
    main()
