"""
Minimal Path-Coherent Walker — Let the Topology Speak
======================================================

Hypothesis (from MANTRAS.md):
  1. WEIGHT IS GRAMMAR — edge weight already encodes grammatical flow.
     Do not invent a separate "grammar score". The weight IS the score.
  2. BALANCE IS THE FULL PATH — we score paths, not tokens.
  3. CONNECTIONS MUST INCLUDE MULTI-HOP — 1-hop-only rewards orbiting.
     With graph diameter ~5.17 (measured), 2-hop reach is minimum.
  5. SIMPLIFY ALWAYS WINS — strip every threshold, every overlay.
  9. EVERY CONSTRAINT IS SUSPECT — no FUNCTION_THRESHOLD, no fence,
     no noise_set, no PMI field, no content/function distinction.

Experiment:
  Run two variants on the same prompts and let the outputs speak:

  VARIANT A: pure weight
      score(candidate) = edge_weight
      (baseline — the raw topology)

  VARIANT B: weight * (1 + alpha * multi_hop_coherence)
      coherence = 1 if candidate has a 1-hop or 2-hop connection
                  back to any token in the last N path tokens, else 0
      (minimal memory — a binary nudge toward connected candidates)

Expected reading:
  - If A produces fluent-but-drifting word sequences and B produces the
    same fluency with tighter topical lock: coherence works, keep it.
  - If A and B produce the same thing: 1-hop neighbors of current already
    include almost all multi-hop-connected candidates (the graph is
    locally dense enough that coherence is redundant).
  - If B collapses earlier than A: the bonus is distorting flow. Reduce
    alpha or abandon.
  - If BOTH produce function-word salad: the topology won't yield content
    without training-layer intervention (e.g. transitive edges).

Usage:
    python .notes/path_coherent_walker.py --variant A
    python .notes/path_coherent_walker.py --variant B --alpha 0.5
    python .notes/path_coherent_walker.py --variant B --verbose --prompt "The fire burned"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from generate import LLNGraph, download_model


def multi_hop_connected(graph, candidate, window_path, window_neighbors):
    """Does candidate have a 1- or 2-hop connection to any window path token?

    window_neighbors: precomputed set of all 1-hop forward neighbors of
    tokens in window_path (i.e. anything reachable in 1 step from the
    last N path tokens). If candidate is in this set, it's 1-hop-reachable
    from the path. If any of candidate's own forward neighbors are in the
    window path itself, it's 1-hop-pointing at the path. For 2-hop, check
    if any of candidate's forward neighbors are in window_neighbors.

    Returns True if any connection exists (binary signal).
    """
    # candidate is 1-hop DOWNSTREAM of window (window_tok -> candidate)
    if candidate in window_neighbors:
        return True

    # candidate's forward neighbors include a window path token
    # (candidate -> window_tok, 1-hop UPSTREAM pointer)
    s = int(graph.full_off[candidate])
    e = int(graph.full_off[candidate + 1])
    window_set = set(window_path)
    for j in range(s, e):
        t = int(graph.full_tgt[j])
        if t in window_set:
            return True
        # 2-hop: candidate -> t -> window
        if t in window_neighbors:
            return True

    return False


def build_window_neighbors(graph, window_path):
    """Union of 1-hop forward neighbors of every token in the window."""
    neighbors = set()
    for p in window_path:
        s = int(graph.full_off[p])
        e = int(graph.full_off[p + 1])
        for j in range(s, e):
            neighbors.add(int(graph.full_tgt[j]))
    return neighbors


def generate_minimal(graph, prompt_text, max_tokens=20,
                      variant='A', alpha=0.5, window=6, top_k=40,
                      verbose=False):
    """Minimal walker. Weight is grammar. Optional multi-hop coherence nudge."""
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'generated_text': '', 'n_generated': 0}

    path = list(prompt_indices)
    generated = []
    visited = {t: 1 for t in path}

    if verbose:
        print(f"  variant={variant} alpha={alpha} window={window} top_k={top_k}")
        print(f"  prompt: {' '.join(graph.idx_to_word[i] for i in path)}")

    for step in range(max_tokens):
        current = path[-1]
        tgt, wgt = graph.get_forward_edges(current, top_k=top_k)
        if len(tgt) == 0:
            if verbose:
                print(f"  [halt: dead end]")
            break

        # Window = last N path tokens (including current)
        window_path = path[-window:]

        # For variant B: precompute 1-hop neighborhood of window
        window_neighbors = None
        if variant == 'B':
            window_neighbors = build_window_neighbors(graph, window_path)

        candidates = []
        for i in range(len(tgt)):
            t = int(tgt[i])
            w = float(wgt[i])

            # Loop control — the only "constraint". No threshold, just
            # "don't repeat the same word 3 times in a row".
            if visited.get(t, 0) >= 3:
                continue

            score = w  # MANTRA #1: weight IS the score
            coh = 0
            if variant == 'B':
                if multi_hop_connected(graph, t, window_path, window_neighbors):
                    coh = 1
                    score = w * (1 + alpha)

            candidates.append((t, score, w, coh))

        if not candidates:
            if verbose:
                print(f"  [halt: all candidates visited-capped]")
            break

        candidates.sort(key=lambda x: -x[1])
        chosen, chosen_score, raw_w, coh = candidates[0]

        if verbose:
            print(f"\n  step {step:2d} (current='{graph.idx_to_word[current]}'):")
            for j, c in enumerate(candidates[:5]):
                tag = "<-- chosen" if c[0] == chosen else ""
                coh_tag = " [CONN]" if c[3] else ""
                print(f"    {j+1}. {graph.idx_to_word[c[0]]:15s} "
                      f"w={c[2]:8.0f} score={c[1]:8.0f}{coh_tag} {tag}")

        generated.append(chosen)
        visited[chosen] = visited.get(chosen, 0) + 1
        path.append(chosen)

    gen_text = graph.detokenize(generated)
    return {
        'generated_text': gen_text,
        'n_generated': len(generated),
        'full_path': graph.detokenize(path),
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
    parser = argparse.ArgumentParser(description="LLN Minimal Walker (experiment)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, nargs='+', default=None)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--variant", choices=['A', 'B', 'both'], default='both',
                        help="A=pure weight, B=weight * (1+alpha*coherence), both=compare")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=40)
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
            r = generate_minimal(graph, prompt,
                                  max_tokens=args.max_tokens,
                                  variant=v, alpha=args.alpha,
                                  window=args.window, top_k=args.top_k,
                                  verbose=args.verbose)
            elapsed = time.time() - t0
            label = f"A(pure weight)" if v == 'A' else f"B(w*(1+{args.alpha}*coh))"
            print(f"  [{label}] -> {r['generated_text']}")
            print(f"     [{r['n_generated']} tok, {elapsed:.3f}s]")
        print()

    graph.close()


if __name__ == "__main__":
    main()
