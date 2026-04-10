#!/usr/bin/env python3
"""LLN Fix 4: Cadence Walker — Enforced Content/Function Rhythm

Grammar-first generation with forced content cadence.
Instead of walking toward targets (pathfinding) or surfing function highways (Fix 3),
enforces the natural rhythm of English: func func CONTENT func func CONTENT.

Usage:
    python fix4_cadence.py --prompt "The fire burned" --verbose
    python fix4_cadence.py --interactive --verbose
    python fix4_cadence.py --interactive --verbose --compare
    python fix4_cadence.py --max-streak 2  (tighter cadence, more content)

Requires generate.py in the same directory.
"""
import numpy as np
import sys
import os
import time
import argparse

from generate import LLNGraph, activate, download_model, DEFAULT_MODEL_DIR


def build_fence(graph, activated, pmi_scores):
    """Build semantic fence: activated words + function words + punctuation."""
    FUNCTION_THRESHOLD = 5000

    fence = set(activated)
    function_words = set()
    punctuation = set()

    for i in range(graph.vocab_size):
        word = graph.idx_to_word[i]
        if word in {'.', ',', ';', ':', '!', '?', '(', ')', '"', '-', "'"}:
            fence.add(i)
            punctuation.add(i)
            continue
        if graph.in_degree[i] > FUNCTION_THRESHOLD:
            fence.add(i)
            function_words.add(i)

    content_in_fence = activated - function_words - punctuation
    return fence, function_words, punctuation, content_in_fence


def generate_cadence(graph, prompt_text, max_tokens=20, max_streak=3,
                     verbose=False):
    """Grammar-first generation with forced content cadence.

    Walk by grammar (trigram + edge weight) inside the semantic fence.
    After max_streak consecutive function words, FORCE a content word landing.
    Content word chosen by grammar x PMI (multiplicative).
    """
    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return {'text': prompt_text, 'generated_text': '', 'n_generated': 0,
                'content_hits': 0, 'content_words': []}

    content_words_in_prompt = [i for i in prompt_indices if graph.in_degree[i] <= 20000]
    top_pct = 0.40 if len(content_words_in_prompt) <= 3 else 0.20
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=top_pct)

    fence, function_words, punctuation, content_in_fence = build_fence(
        graph, activated, pmi_scores)

    if verbose:
        print(f"  Fence: {len(fence)} tokens | "
              f"Content: {len(content_in_fence)} | "
              f"Function: {len(function_words)} | "
              f"Cadence: max {max_streak} func before forced content")

    generated = []
    current = prompt_indices[-1]
    prev_token = prompt_indices[-2] if len(prompt_indices) > 1 else None
    visited = {t: 1 for t in prompt_indices}
    depleted = set()
    content_hits = []

    func_streak = 0
    force_attempts = 0
    MAX_FORCE_ATTEMPTS = 2

    for step in range(max_tokens):
        tgt, wgt = graph.get_forward_edges(current, top_k=100)
        if len(tgt) == 0:
            if verbose:
                print(f"  [halt: dead end at '{graph.idx_to_word[current]}']")
            break

        must_content = (func_streak >= max_streak and force_attempts < MAX_FORCE_ATTEMPTS)

        candidates = []
        for i in range(len(tgt)):
            t = int(tgt[i])
            w = float(wgt[i])

            if t not in fence:
                continue

            is_content = (t in content_in_fence and t not in depleted)

            if must_content and not is_content:
                continue

            v = visited.get(t, 0)
            if v >= 3:
                continue

            norm_w = float(np.log1p(w))
            tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0
            score = norm_w * tri_mult

            if v > 0:
                score *= 0.5 ** v

            if is_content:
                pmi = pmi_scores.get(t, 0)
                if pmi > 0:
                    score *= (1.0 + float(np.log1p(pmi)) * 0.5)

            candidates.append((t, score, is_content))

        if not candidates:
            if must_content:
                force_attempts += 1
                if verbose:
                    print(f"  step {step:2d}: [content force failed, "
                          f"attempt {force_attempts}/{MAX_FORCE_ATTEMPTS}]")
                for i in range(len(tgt)):
                    t = int(tgt[i])
                    w = float(wgt[i])
                    if t not in fence:
                        continue
                    if visited.get(t, 0) >= 3:
                        continue
                    norm_w = float(np.log1p(w))
                    tri_mult = graph.trigram_score(prev_token, current, t) if prev_token is not None else 1.0
                    score = norm_w * tri_mult
                    is_c = (t in content_in_fence and t not in depleted)
                    candidates.append((t, score, is_c))

            if not candidates:
                if verbose:
                    print(f"  [halt: no fenced candidates at step {step}]")
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
            force_attempts = 0
        elif chosen in function_words or chosen in punctuation:
            func_streak += 1
        else:
            func_streak = 0

        if verbose:
            word = graph.idx_to_word[chosen]
            mode = "CONTENT" if chosen_is_content else ("func" if chosen in function_words else "punct" if chosen in punctuation else "other")
            streak_info = f"streak={func_streak}" if not chosen_is_content else "streak=0"
            forced = " [FORCED]" if must_content and chosen_is_content else ""
            print(f"  step {step:2d}: {word:15s} ({mode:7s}, score={chosen_score:.2f}, "
                  f"{streak_info}){forced}")

    return {
        'text': graph.detokenize(prompt_indices + generated),
        'prompt': prompt_text,
        'generated_text': graph.detokenize(generated),
        'n_generated': len(generated),
        'content_hits': len(content_hits),
        'content_words': [graph.idx_to_word[t] for t in content_hits],
        'depleted': len(depleted),
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
        description="LLN Cadence Walker — grammar-first + forced content rhythm")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--max-streak", type=int, default=3,
                        help="Max function words before forcing content (default: 3)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--interactive", action="store_true",
                        help="Load once, accept prompts in a loop")
    parser.add_argument("--compare", action="store_true",
                        help="Run original walker side-by-side")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,} | "
          f"PMI: {len(graph.pmi_tgt):,} | Trigrams: {len(graph.trigrams):,}")
    print()

    def run_prompt(prompt):
        t0 = time.time()
        result = generate_cadence(graph, prompt, max_tokens=args.max_tokens,
                                  max_streak=args.max_streak, verbose=args.verbose)
        gen_time = time.time() - t0

        print(f"  \"{prompt}\"")
        print(f"  -> {result['generated_text']}")
        print(f"     [{result['n_generated']} tokens, "
              f"{result['content_hits']} content: {result['content_words']}, "
              f"{gen_time:.3f}s]")

        if args.compare:
            from generate import generate as original_generate
            t0 = time.time()
            orig = original_generate(graph, prompt, max_tokens=args.max_tokens)
            orig_time = time.time() - t0
            print(f"  (original) -> {orig['generated_text']}")
            print(f"     [{orig['n_generated']} tokens, "
                  f"{orig['targets_reached']} targets, {orig_time:.3f}s]")
        print()

    if args.interactive:
        print(f"Interactive mode (cadence={args.max_streak}). "
              f"Type a prompt, or 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
                if prompt.lower() in ('quit', 'exit', 'q'):
                    print("Bye.")
                break
            run_prompt(prompt)
    else:
        prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
        for prompt in prompts:
            run_prompt(prompt)

    graph.close()


if __name__ == "__main__":
    main()
