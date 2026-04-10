"""
Manual Coefficient Runner — read traces, form hypotheses, adjust

Not an automated tuner. A runner that loads the model once, runs a
named config across all default prompts, and optionally prints side-by-side
against the shipped walker. You iterate by editing CONFIGS below,
re-running, reading output, adjusting.

Usage:
    python .notes/tune_manual.py                     # run all configs in CONFIGS
    python .notes/tune_manual.py --config baseline   # run one named config
    python .notes/tune_manual.py --compare           # diff all configs vs 'baseline'
    python .notes/tune_manual.py --prompt "The fire burned"  # single prompt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from generate import LLNGraph, download_model, generate as generate_shipped
from target_walker_v2 import generate_v2
from vocab_rank import VocabRank


# ═══════════════════════════════════════════════════════════════════
# Named configurations — edit and re-run.
# Coefficients: grammar_weight, proximity_direct, proximity_overlap,
#               pull_strength, memory_alpha
# ═══════════════════════════════════════════════════════════════════

CONFIGS = {
    # Anchor: shipped walker exactly reproduced
    "shipped": {
        "walker": "shipped",
    },

    # Baseline v2: same coefficients as shipped, WITHOUT memory
    # Proves v2 framework reproduces shipped behavior when memory_alpha=0
    "v2_baseline": {
        "walker": "v2",
        "grammar_weight": 0.1,
        "proximity_direct": 3.0,
        "proximity_overlap": 0.3,
        "pull_strength": 1.0,
        "memory_alpha": 0.0,
    },

    # Experiment 2: Memory with content-only selectivity
    # Does content-only memory rescue sink-dominated prompts without
    # distorting working walks?
    "v2_memory_1": {
        "walker": "v2",
        "grammar_weight": 0.1,
        "proximity_direct": 3.0,
        "proximity_overlap": 0.3,
        "pull_strength": 1.0,
        "memory_alpha": 1.0,
    },

    # Experiment 3: Dial pull_strength DOWN
    # Reduce target pull, see if grammar becomes more audible
    "v2_pull_half": {
        "walker": "v2",
        "grammar_weight": 0.1,
        "proximity_direct": 3.0,
        "proximity_overlap": 0.3,
        "pull_strength": 0.5,
        "memory_alpha": 1.0,
    },

    # Experiment 4: Dial proximity_direct DOWN
    # Reduce the 3.0 direct-hit bonus, less shortcut behavior
    "v2_prox_half": {
        "walker": "v2",
        "grammar_weight": 0.1,
        "proximity_direct": 1.5,
        "proximity_overlap": 0.3,
        "pull_strength": 1.0,
        "memory_alpha": 1.0,
    },

    # Experiment 5: Grammar UP
    # Double the grammar coefficient. Does fluency improve?
    "v2_grammar_up": {
        "walker": "v2",
        "grammar_weight": 0.2,
        "proximity_direct": 3.0,
        "proximity_overlap": 0.3,
        "pull_strength": 1.0,
        "memory_alpha": 1.0,
    },

    # Experiment 6: All semantics dialed down, memory up
    # The "let grammar breathe" configuration
    "v2_grammar_first": {
        "walker": "v2",
        "grammar_weight": 0.3,
        "proximity_direct": 1.5,
        "proximity_overlap": 0.2,
        "pull_strength": 0.5,
        "memory_alpha": 1.5,
    },

    # Experiment 7: v2_memory_1 with gentler memory (alpha=0.5)
    # Does half the memory signal still rescue sinks without over-nudging?
    "v2_memory_half": {
        "walker": "v2",
        "grammar_weight": 0.1,
        "proximity_direct": 3.0,
        "proximity_overlap": 0.3,
        "pull_strength": 1.0,
        "memory_alpha": 0.5,
    },

    # Experiment 8: memory_1 + slightly more grammar weight
    # Best of both: memory rescue AND grammar primacy
    "v2_memory_grammar": {
        "walker": "v2",
        "grammar_weight": 0.15,
        "proximity_direct": 2.5,
        "proximity_overlap": 0.3,
        "pull_strength": 0.8,
        "memory_alpha": 1.0,
    },

    # Window size sweep — all use v2_memory_half coefficients (α=0.5)
    # Graph diameter measured at 5.17 hops, so window ~5-6 should match
    # the topological scale of coherence decay.
    "v2_memhalf_w4": {
        "walker": "v2",
        "grammar_weight": 0.1, "proximity_direct": 3.0,
        "proximity_overlap": 0.3, "pull_strength": 1.0, "memory_alpha": 0.5,
        "window_size": 4,
    },
    "v2_memhalf_w6": {
        "walker": "v2",
        "grammar_weight": 0.1, "proximity_direct": 3.0,
        "proximity_overlap": 0.3, "pull_strength": 1.0, "memory_alpha": 0.5,
        "window_size": 6,
    },
    "v2_memhalf_w12": {
        "walker": "v2",
        "grammar_weight": 0.1, "proximity_direct": 3.0,
        "proximity_overlap": 0.3, "pull_strength": 1.0, "memory_alpha": 0.5,
        "window_size": 12,
    },

    # Sink mode experiments — v16 topology fix for find_targets
    # Hypothesis: in v16, content words are sinks. Shipped penalizes them.
    # 'neutral' = ignore pr_ratio, rank by PMI only
    # 'reward'  = INVERT: reward content sinks as topical attractors
    "v2_memhalf_sink_neutral": {
        "walker": "v2",
        "grammar_weight": 0.1, "proximity_direct": 3.0,
        "proximity_overlap": 0.3, "pull_strength": 1.0, "memory_alpha": 0.5,
        "sink_mode": "neutral",
    },
    "v2_memhalf_sink_reward": {
        "walker": "v2",
        "grammar_weight": 0.1, "proximity_direct": 3.0,
        "proximity_overlap": 0.3, "pull_strength": 1.0, "memory_alpha": 0.5,
        "sink_mode": "reward",
    },

    # Literary gem config + sink_reward — can we preserve the gem while
    # fixing the other weather prompts that memory_grammar loses on?
    "v2_gem_sink_reward": {
        "walker": "v2",
        "grammar_weight": 0.15, "proximity_direct": 2.5,
        "proximity_overlap": 0.3, "pull_strength": 0.8, "memory_alpha": 1.0,
        "sink_mode": "reward",
    },
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

WEATHER_PROMPTS = [
    "Dark clouds",
    "Heavy rain",
    "The storm",
    "Lightning struck",
    "The wind howled",
    "Thunder rolled",
    "A cold front",
    "The sky turned",
]


def run_config(graph, config_name, config, prompts, max_tokens=20, verbose=False,
                vocab_rank=None):
    """Run a single config across all prompts. Returns list of results."""
    results = []
    for prompt in prompts:
        t0 = time.time()
        if config['walker'] == 'shipped':
            r = generate_shipped(graph, prompt, max_tokens=max_tokens, verbose=verbose)
        else:
            r = generate_v2(graph, prompt, max_tokens=max_tokens,
                             grammar_weight=config['grammar_weight'],
                             proximity_direct=config['proximity_direct'],
                             proximity_overlap=config['proximity_overlap'],
                             pull_strength=config['pull_strength'],
                             memory_alpha=config['memory_alpha'],
                             window_size=config.get('window_size', 8),
                             sink_mode=config.get('sink_mode', 'penalty'),
                             vocab_rank=vocab_rank,
                             verbose=verbose)
        elapsed = time.time() - t0
        results.append({
            'prompt': prompt,
            'text': r.get('generated_text', ''),
            'tokens': r.get('n_generated', 0),
            'hits': r.get('targets_reached', 0) if isinstance(r.get('targets_reached'), int)
                    else len(r.get('targets_reached', []) or []),
            'elapsed': elapsed,
        })
    return results


def print_results(config_name, results):
    print(f"\n=== {config_name} ===")
    for r in results:
        print(f'  "{r["prompt"]}"')
        print(f"  -> {r['text']}")
        print(f"     [{r['tokens']} tok, {r['hits']} hits, {r['elapsed']:.3f}s]")
    total_tok = sum(r['tokens'] for r in results)
    total_hit = sum(r['hits'] for r in results)
    total_time = sum(r['elapsed'] for r in results)
    print(f"\n  TOTALS: {total_tok} tokens, {total_hit} hits, {total_time:.1f}s")


def print_compare(baseline_results, other_results, baseline_name, other_name):
    print(f"\n=== COMPARE: {baseline_name} vs {other_name} ===")
    print(f'{"prompt":30s} {"baseline tok/hit":18s} {"new tok/hit":18s} delta')
    for b, o in zip(baseline_results, other_results):
        delta_tok = o['tokens'] - b['tokens']
        delta_hit = o['hits'] - b['hits']
        same = "=" if b['text'] == o['text'] else "!"
        print(f'{b["prompt"][:28]:30s} '
              f'{f"{b['tokens']}/{b['hits']}":18s} '
              f'{f"{o['tokens']}/{o['hits']}":18s} '
              f'tok:{delta_tok:+d} hit:{delta_hit:+d} {same}')


def main():
    parser = argparse.ArgumentParser(description="Manual coefficient runner")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Run single named config (otherwise runs all)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all non-shipped configs against 'shipped'")
    parser.add_argument("--prompt", type=str, nargs='+', default=None)
    parser.add_argument("--weather", action="store_true",
                        help="Use weather prompt set instead of defaults")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--n-function", type=int, default=150)
    parser.add_argument("--n-semi-function", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,}")

    vr = VocabRank(graph, n_function=args.n_function,
                    n_semi_function=args.n_semi_function)
    print(f"  VocabRank: n_function={args.n_function} "
          f"n_semi_function={args.n_semi_function}")

    if args.prompt:
        prompts = args.prompt
    elif args.weather:
        prompts = WEATHER_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    if args.config:
        if args.config not in CONFIGS:
            print(f"Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
            graph.close()
            return
        results = run_config(graph, args.config, CONFIGS[args.config], prompts,
                              max_tokens=args.max_tokens, verbose=args.verbose,
                              vocab_rank=vr)
        print_results(args.config, results)
    else:
        all_results = {}
        for name, config in CONFIGS.items():
            print(f"\nRunning: {name}...", flush=True)
            all_results[name] = run_config(graph, name, config, prompts,
                                            max_tokens=args.max_tokens,
                                            verbose=args.verbose,
                                            vocab_rank=vr)
            print_results(name, all_results[name])

        if args.compare:
            baseline = all_results.get('shipped')
            if baseline:
                for name, results in all_results.items():
                    if name == 'shipped':
                        continue
                    print_compare(baseline, results, 'shipped', name)

    graph.close()


if __name__ == "__main__":
    main()
