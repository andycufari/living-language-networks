#!/usr/bin/env python3
"""LLN — Living Language Network: Interactive Live Learning Mode

The model learns new facts in real-time and generates text that routes
through both long-term memory (the 32GB corpus graph) and short-term
memory (what you just taught it).

Usage:
    python living.py
    python living.py --model path/to/model.lmdb

Commands:
    LEARN: <text>       Absorb text into short-term memory
    GENERATE: <prompt>  Generate from prompt (using both memories)
    FORGET              Clear short-term memory
    MEMORY              Show short-term memory stats
    QUIT / EXIT         Exit

Requirements: numpy, lmdb, huggingface_hub
"""
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import LLNGraph, generate, download_model, DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="LLN Live Learning Mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to LMDB model directory")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    model_path = args.model or download_model()

    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,} | "
          f"PMI: {len(graph.pmi_tgt):,} | Trigrams: {len(graph.trigrams):,}")
    print()
    print("=" * 60)
    print("  LLN — Live Learning Mode")
    print("  LEARN: <text>       Teach the model something new")
    print("  GENERATE: <prompt>  Generate text from prompt")
    print("  FORGET              Clear short-term memory")
    print("  MEMORY              Show memory stats")
    print("  QUIT                Exit")
    print("=" * 60)
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        cmd = line.upper()

        if cmd in ("QUIT", "EXIT", "Q"):
            print("Bye.")
            break

        elif cmd == "FORGET":
            graph.forget()
            print("  Short-term memory cleared.")
            print()

        elif cmd == "MEMORY":
            stats = graph.memory_stats()
            print(f"  Short-term memory: {stats['fwd_edges']} forward edges, "
                  f"{stats['trigrams']} trigrams, {stats['pmi_links']} PMI links")
            print()

        elif line.upper().startswith("LEARN:"):
            text = line[6:].strip()
            if not text:
                print("  Nothing to learn. Usage: LEARN: The cat sat on the mat.")
                continue

            result = graph.learn(text)
            print(f"  Learned {result['learned']} bigrams")
            if result.get('oov'):
                print(f"  OOV (not in vocab): {', '.join(result['oov'])}")
            if result.get('content_words'):
                print(f"  Content words linked: {', '.join(result['content_words'])}")
            stats = graph.memory_stats()
            print(f"  Memory: {stats['fwd_edges']} fwd, {stats['trigrams']} tri, {stats['pmi_links']} pmi")
            print()

        elif line.upper().startswith("GENERATE:"):
            prompt = line[9:].strip()
            if not prompt:
                print("  No prompt. Usage: GENERATE: The fire burned")
                continue

            t0 = time.time()
            result = generate(graph, prompt, max_tokens=args.max_tokens, verbose=True)
            gen_time = time.time() - t0

            print(f"\n  \"{prompt}\"")
            print(f"  -> {result['generated_text']}")
            print(f"     [{result['n_generated']} tokens, {result['targets_reached']} targets, "
                  f"{gen_time:.3f}s]")
            print()

        else:
            # Treat bare text as GENERATE
            t0 = time.time()
            result = generate(graph, line, max_tokens=args.max_tokens, verbose=True)
            gen_time = time.time() - t0

            print(f"\n  \"{line}\"")
            print(f"  -> {result['generated_text']}")
            print(f"     [{result['n_generated']} tokens, {result['targets_reached']} targets, "
                  f"{gen_time:.3f}s]")
            print()

    graph.close()


if __name__ == "__main__":
    main()
