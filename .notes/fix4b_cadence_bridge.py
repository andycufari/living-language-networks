#!/usr/bin/env python3
"""LLN Fix 4b: Cadence Walker with Bridge Search

When the cadence forces content and 1-hop fails, runs a mini beam search
(depth 3, width 3) that walks grammatically through the fence looking for
ANY undepleted content word. Returns the entire bridge path as a grammatical
phrase.

Usage:
    python fix4b_cadence_bridge.py --prompt "The fire burned" --verbose
    python fix4b_cadence_bridge.py --interactive --verbose --compare
    python fix4b_cadence_bridge.py --max-streak 2 --bridge-depth 4

Requires generate.py in the same directory.
"""
import numpy as np
import time
import argparse

from generate import LLNGraph, activate, download_model, DEFAULT_MODEL_DIR


FUNCTION_THRESHOLD = 5000
PUNCTUATION_TOKENS = {'.', ',', ';', ':', '!', '?', '(', ')', '"', '-', "'"}


def build_fence(graph, activated, pmi_scores):
    fence = set(activated)
    function_words = set()
    punctuation = set()

    for i in range(graph.vocab_size):
        word = graph.idx_to_word[i]
        if word in PUNCTUATION_TOKENS:
            fence.add(i)
            punctuation.add(i)
        elif graph.in_degree[i] > FUNCTION_THRESHOLD:
            fence.add(i)
            function_words.add(i)

    content_in_fence = activated - function_words - punctuation
    return fence, function_words, punctuation, content_in_fence


def bridge_to_content(graph, start, prev_token, fence, content_targets,
                      visited, max_depth=3, beam_width=3, top_k=80):
    """Mini beam search: find shortest grammatical path to any content word."""
    if not content_targets:
        return []

    beam = [(start, prev_token, [], 0.0)]

    for depth in range(max_depth):
        candidates = []

        for cur, prev, path, cum_score in beam:
            tgt, wgt = graph.get_forward_edges(cur, top_k=top_k)
            if len(tgt) == 0:
                continue

            for i in range(len(tgt)):
                t = int(tgt[i])
                w = float(wgt[i])

                if t not in fence:
                    continue
                if t in set(path) or visited.get(t, 0) >= 3:
                    continue

                norm_w = float(np.log1p(w))
                tri_mult = graph.trigram_score(prev, cur, t) if prev is not None else 1.0
                step_score = norm_w * tri_mult

                new_path = path + [t]
                new_cum = cum_score + step_score

                if t in content_targets:
                    return new_path

                candidates.append((t, cur, new_path, new_cum))

        if not candidates:
            break

        candidates.sort(key=lambda c: -c[3] / len(c[2]))
        beam = [(c[0], c[1], c[2], c[3]) for c in candidates[:beam_width]]

    return []


def generate_cadence(graph, prompt_text, max_tokens=20, max_streak=3,
                     bridge_depth=3, bridge_width=3, verbose=False):
    """Grammar-first walk with cadence-enforced content rhythm + bridge search."""
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'n_generated': 0,
                'content_hits': 0, 'content_ratio': '0%', 'content_words': []}

    content_in_prompt = [i for i in prompt_indices if graph.in_degree[i] <= 20000]
    top_pct = 0.40 if len(content_in_prompt) <= 3 else 0.20
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=top_pct)

    fence, function_words, punctuation, content_in_fence = build_fence(
        graph, activated, pmi_scores)

    if verbose:
        print(f"  Fence: {len(fence)} | Content: {len(content_in_fence)} | "
              f"Function: {len(function_words)} | Max streak: {max_streak}")

    generated = []
    current = prompt_indices[-1]
    prev_token = prompt_indices[-2] if len(prompt_indices) > 1 else None
    visited = {t: 1 for t in prompt_indices}
    depleted = set()
    content_hits = []
    func_streak = 0

    while len(generated) < max_tokens:
        must_content = (func_streak >= max_streak)

        if must_content:
            available_content = set()
            for t in content_in_fence - depleted:
                if visited.get(t, 0) < 2:
                    available_content.add(t)

            if available_content:
                # Try 1-hop
                tgt, wgt = graph.get_forward_edges(current, top_k=100)
                one_hop = []
                for i in range(len(tgt)):
                    t = int(tgt[i])
                    if t in available_content:
                        norm_w = float(np.log1p(float(wgt[i])))
                        tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0
                        pmi = pmi_scores.get(t, 0)
                        score = norm_w * tri_mult * (1.0 + np.log1p(pmi) * 0.5)
                        one_hop.append((t, score))

                if one_hop:
                    one_hop.sort(key=lambda x: -x[1])
                    chosen = one_hop[0][0]
                    generated.append(chosen)
                    visited[chosen] = visited.get(chosen, 0) + 1
                    prev_token = current
                    current = chosen
                    content_hits.append(chosen)
                    depleted.add(chosen)
                    func_streak = 0
                    if verbose:
                        word = graph.idx_to_word[chosen]
                        print(f"  step {len(generated)-1:2d}: {word:15s} "
                              f"(CONTENT 1-hop, pmi={pmi_scores.get(chosen,0):.1f})")
                    continue

                # 1-hop failed, bridge search
                bridge = bridge_to_content(
                    graph, current, prev_token, fence, available_content,
                    visited, max_depth=bridge_depth, beam_width=bridge_width)

                if bridge:
                    for t in bridge:
                        if len(generated) >= max_tokens:
                            break
                        generated.append(t)
                        visited[t] = visited.get(t, 0) + 1
                        prev_token = current
                        current = t

                        is_content = (t in content_in_fence and t not in depleted)
                        if is_content:
                            content_hits.append(t)
                            depleted.add(t)

                        if verbose:
                            word = graph.idx_to_word[t]
                            marker = "CONTENT" if is_content else "bridge"
                            print(f"  step {len(generated)-1:2d}: {word:15s} "
                                  f"({marker}, bridge path)")

                    func_streak = 0
                    continue

                if verbose:
                    print(f"  [bridge failed at '{graph.idx_to_word[current]}', "
                          f"{len(available_content)} unreachable]")

        # Normal grammar walk
        tgt, wgt = graph.get_forward_edges(current, top_k=100)
        if len(tgt) == 0:
            if verbose:
                print(f"  [halt: dead end at '{graph.idx_to_word[current]}']")
            break

        candidates = []
        for i in range(len(tgt)):
            t = int(tgt[i])
            w = float(wgt[i])
            if t not in fence:
                continue
            v = visited.get(t, 0)
            if v >= 3:
                continue

            norm_w = float(np.log1p(w))
            tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0
            score = norm_w * tri_mult
            if v > 0:
                score *= 0.5 ** v

            is_content = (t in content_in_fence and t not in depleted)
            if is_content:
                pmi = pmi_scores.get(t, 0)
                if pmi > 0:
                    score *= (1.0 + np.log1p(pmi) * 0.5)

            candidates.append((t, score, is_content))

        if not candidates:
            if verbose:
                print(f"  [halt: no fenced candidates at step {len(generated)}]")
            break

        candidates.sort(key=lambda x: -x[1])
        chosen, chosen_score, chosen_is_content = candidates[0]

        generated.append(chosen)
        visited[chosen] = visited.get(chosen, 0) + 1
        prev_token = current
        current = chosen

        if chosen_is_content:
            content_hits.append(chosen)
            depleted.add(chosen)
            func_streak = 0
        elif chosen in function_words or chosen in punctuation:
            func_streak += 1
        else:
            func_streak = max(0, func_streak - 1)

        if verbose:
            word = graph.idx_to_word[chosen]
            mode = "CONTENT" if chosen_is_content else "func" if chosen in function_words else "punct"
            print(f"  step {len(generated)-1:2d}: {word:15s} "
                  f"({mode:7s}, score={chosen_score:.2f}, streak={func_streak})")

    gen_text = graph.detokenize(generated)
    content_ratio = len(content_hits) / max(len(generated), 1)

    return {
        'text': graph.detokenize(prompt_indices + generated),
        'prompt': prompt_text,
        'generated_text': gen_text,
        'n_generated': len(generated),
        'content_hits': len(content_hits),
        'content_ratio': f"{content_ratio:.0%}",
        'content_words': [graph.idx_to_word[t] for t in content_hits],
        'activated_size': len(activated),
        'fence_size': len(fence),
    }


DEFAULT_PROMPTS = [
    "The fire burned",
    "The king",
    "She opened the door",
    "The army marched",
    "Scientists discovered",
    "The ship sailed",
    "Dark clouds",
    "The river flows",
    "The music played softly",
    "The volcano erupted",
    "the cat that the dog chased ran",
    "she didn't go to the store because it was closed",
]


def main():
    parser = argparse.ArgumentParser(
        description="LLN Fix 4b: Cadence Walker with Bridge Search")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--max-streak", type=int, default=3)
    parser.add_argument("--bridge-depth", type=int, default=3)
    parser.add_argument("--bridge-width", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,} | "
          f"PMI: {len(graph.pmi_tgt):,} | Trigrams: {len(graph.trigrams):,}")
    print()

    def run_prompt(prompt, verbose=False):
        t0 = time.time()
        result = generate_cadence(
            graph, prompt, max_tokens=args.max_tokens,
            max_streak=args.max_streak,
            bridge_depth=args.bridge_depth,
            bridge_width=args.bridge_width,
            verbose=verbose)
        gen_time = time.time() - t0

        print(f"  \"{prompt}\"")
        print(f"  -> {result['generated_text']}")
        print(f"     [{result['n_generated']} tok, "
              f"{result['content_hits']} content ({result['content_ratio']}): "
              f"{result['content_words']}, {gen_time:.3f}s]")

        if args.compare:
            from generate import generate as orig_gen
            t0 = time.time()
            orig = orig_gen(graph, prompt, max_tokens=args.max_tokens)
            print(f"  (orig) -> {orig['generated_text']}")
            print(f"     [{orig['n_generated']} tok, "
                  f"{orig['targets_reached']} targets, {time.time()-t0:.3f}s]")
        print()

    if args.interactive:
        print(f"Interactive mode (streak={args.max_streak}, "
              f"bridge={args.bridge_depth}x{args.bridge_width}). "
              f"'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
                break
            verbose = args.verbose
            if ' --v' in prompt:
                verbose = True
                prompt = prompt.replace(' --verbose', '').replace(' --v', '').strip()
            run_prompt(prompt, verbose=verbose)
    else:
        prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
        for prompt in prompts:
            run_prompt(prompt, verbose=args.verbose)

    graph.close()


if __name__ == "__main__":
    main()
