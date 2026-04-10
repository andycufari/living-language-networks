"""
LLN Graph Distance Diagnostic
==============================

Questions to answer:
  1. What's the hop distance between random token pairs?
  2. Is this a small-world graph? (most pairs within 3-5 hops?)
  3. Does path-connectivity density actually differentiate content from function words?
  4. What does the coherence landscape look like at each generation step?

Usage:
    python .notes/graph_distance_debug.py --quick
    python .notes/graph_distance_debug.py --prompt "The fire burned"
    python .notes/graph_distance_debug.py --distances-only
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import argparse
import random
from collections import deque

from generate import LLNGraph, activate, download_model


def bfs_distance(graph, source, target, max_depth=6, top_k=50):
    if source == target:
        return 0
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        tgt, _ = graph.get_forward_edges(node, top_k=top_k)
        for i in range(len(tgt)):
            t = int(tgt[i])
            if t == target:
                return depth + 1
            if t not in visited:
                visited.add(t)
                queue.append((t, depth + 1))
    return -1


def bfs_distance_set(graph, source, target_set, max_depth=4, top_k=50):
    found = {}
    remaining = set(target_set)
    if source in remaining:
        found[source] = 0
        remaining.discard(source)
    if not remaining:
        return found
    visited = {source}
    queue = deque([(source, 0)])
    while queue and remaining:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        tgt, _ = graph.get_forward_edges(node, top_k=top_k)
        for i in range(len(tgt)):
            t = int(tgt[i])
            if t in remaining:
                found[t] = depth + 1
                remaining.discard(t)
            if t not in visited:
                visited.add(t)
                queue.append((t, depth + 1))
    return found


def measure_random_distances(graph, n_pairs=200, max_depth=6, top_k=50):
    print(f"\n{'='*70}")
    print(f"  RANDOM PAIR DISTANCE DISTRIBUTION")
    print(f"  {n_pairs} pairs, max_depth={max_depth}, top_k={top_k}")
    print(f"{'='*70}")

    content_tokens = []
    for i in range(graph.vocab_size):
        if 10 < graph.in_degree[i] < 20000:
            if graph.out_degree(i) > 5:
                content_tokens.append(i)
    print(f"  Content token pool: {len(content_tokens)}")

    distances = []
    unreachable = 0
    t0 = time.time()
    for trial in range(n_pairs):
        a, b = random.sample(content_tokens, 2)
        d = bfs_distance(graph, a, b, max_depth=max_depth, top_k=top_k)
        if d == -1:
            unreachable += 1
        else:
            distances.append(d)
        if (trial + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... {trial+1}/{n_pairs} ({elapsed:.1f}s)")
    elapsed = time.time() - t0

    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"  Reachable: {len(distances)}/{n_pairs} ({100*len(distances)/n_pairs:.0f}%)")
    print(f"  Unreachable (>{max_depth} hops): {unreachable}/{n_pairs}")

    if distances:
        print(f"\n  Distance distribution:")
        for d in range(max_depth + 1):
            count = distances.count(d)
            bar = '#' * (count * 40 // max(len(distances), 1))
            pct = 100 * count / len(distances) if distances else 0
            print(f"    {d} hops: {count:4d} ({pct:5.1f}%) {bar}")
        print(f"\n  Mean: {np.mean(distances):.2f}")
        print(f"  Median: {np.median(distances):.1f}")
        print(f"  90th percentile: {np.percentile(distances, 90):.1f}")

    return distances


def measure_function_vs_content(graph, max_depth=4):
    print(f"\n{'='*70}")
    print(f"  FUNCTION vs CONTENT WORD CONNECTIVITY")
    print(f"{'='*70}")

    function_examples = []
    content_examples = []
    for i in range(graph.vocab_size):
        if graph.in_degree[i] > 10000 and len(function_examples) < 20:
            function_examples.append(i)
        elif 100 < graph.in_degree[i] < 5000 and graph.out_degree(i) > 10:
            if len(content_examples) < 50:
                content_examples.append(i)

    target_set = set(random.sample(content_examples, min(50, len(content_examples))))

    print(f"\n  Function words -> reachability to {len(target_set)} content targets (max {max_depth} hops):")
    for idx in function_examples[:10]:
        found = bfs_distance_set(graph, idx, target_set, max_depth=max_depth)
        word = graph.idx_to_word[idx]
        print(f"    {word:15s} (in_deg={graph.in_degree[idx]:6d}) -> "
              f"reaches {len(found)}/{len(target_set)} targets")

    print(f"\n  Content words -> reachability to {len(target_set)} content targets (max {max_depth} hops):")
    for idx in list(target_set)[:10]:
        found = bfs_distance_set(graph, idx, target_set, max_depth=max_depth)
        word = graph.idx_to_word[idx]
        print(f"    {word:15s} (in_deg={graph.in_degree[idx]:6d}) -> "
              f"reaches {len(found)}/{len(target_set)} targets")


def path_coherence_analysis(graph, prompt_text, max_steps=5):
    print(f"\n{'='*70}")
    print(f"  PATH COHERENCE ANALYSIS: \"{prompt_text}\"")
    print(f"{'='*70}")

    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        print("  [empty prompt]")
        return

    # Activate semantic field once
    activated, pmi_scores = activate(graph, prompt_indices, top_pct=0.20)
    print(f"  Activated field size: {len(activated)}")

    path = list(prompt_indices)
    current = path[-1]
    print(f"\n  Initial path: {' '.join(graph.idx_to_word[i] for i in path)}")

    for step in range(max_steps):
        tgt, wgt = graph.get_forward_edges(current, top_k=80)
        if len(tgt) == 0:
            print(f"  [dead end at step {step}]")
            break

        print(f"\n  --- Step {step} (current='{graph.idx_to_word[current]}') ---")
        print(f"  Path so far: {' '.join(graph.idx_to_word[i] for i in path)}")

        # Build path edge lookup (full edges) for all path tokens
        path_edges = {}
        for p in path:
            if p in path_edges:
                continue
            s = int(graph.full_off[p])
            e = int(graph.full_off[p + 1])
            edges = {}
            for j in range(s, e):
                edges[int(graph.full_tgt[j])] = float(graph.full_wgt[j])
            path_edges[p] = edges

        scored = []
        for i in range(min(len(tgt), 40)):
            t = int(tgt[i])
            w = float(wgt[i])
            word = graph.idx_to_word[t]

            connections = 0
            weight_sum = 0.0
            detail = []

            # For each path token, does it have a forward edge to t?
            for p, p_edges in path_edges.items():
                if t in p_edges:
                    connections += 1
                    weight_sum += p_edges[t]
                    detail.append(f"{graph.idx_to_word[p]}->{word}({p_edges[t]:.0f})")

            # Also check: does t have forward edges to path tokens? (bidirectional)
            t_s = int(graph.full_off[t])
            t_e = int(graph.full_off[t + 1])
            path_set = set(path)
            for j in range(t_s, min(t_e, t_s + 300)):
                pt = int(graph.full_tgt[j])
                if pt in path_set and pt not in path_edges.get(t, set()):
                    connections += 1
                    weight_sum += float(graph.full_wgt[j])
                    detail.append(f"{word}->{graph.idx_to_word[pt]}({float(graph.full_wgt[j]):.0f})")

            density = connections / max(len(path), 1)
            edge_score = float(np.log1p(w))
            in_field = t in activated

            scored.append({
                'word': word,
                'token': t,
                'edge_score': edge_score,
                'density': density,
                'connections': connections,
                'weight_sum': weight_sum,
                'in_field': in_field,
                'detail': detail,
            })

        # Sort by density
        scored.sort(key=lambda x: -x['density'])
        print(f"\n  Top 10 by PATH DENSITY (fraction of path connected):")
        for j, s in enumerate(scored[:10]):
            field_mark = "*" if s['in_field'] else " "
            conn_str = ', '.join(s['detail'][:3])
            if len(s['detail']) > 3:
                conn_str += f" +{len(s['detail'])-3}"
            print(f"    {j+1:2d}.{field_mark} {s['word']:15s} "
                  f"density={s['density']:.3f} ({s['connections']} conn) "
                  f"edge={s['edge_score']:.2f}  [{conn_str}]")

        scored_by_edge = sorted(scored, key=lambda x: -x['edge_score'])
        print(f"\n  Top 5 by EDGE WEIGHT (current approach):")
        for j, s in enumerate(scored_by_edge[:5]):
            field_mark = "*" if s['in_field'] else " "
            print(f"    {j+1:2d}.{field_mark} {s['word']:15s} "
                  f"edge={s['edge_score']:.2f} density={s['density']:.3f} "
                  f"({s['connections']} conn)")

        # Pick best in-field by density for next step
        in_field_scored = [s for s in scored if s['in_field']]
        if not in_field_scored:
            print(f"  [no in-field candidates — halt]")
            break
        best = max(in_field_scored, key=lambda x: x['density'] * 10 + x['edge_score'])
        path.append(best['token'])
        current = best['token']
        print(f"\n  -> Picked '{best['word']}' (in-field, density={best['density']:.3f})")

    print(f"\n  Final path: {' '.join(graph.idx_to_word[i] for i in path)}")


def connectivity_histogram(graph, prompt_text, sample_size=20000):
    print(f"\n{'='*70}")
    print(f"  CONNECTIVITY HISTOGRAM: \"{prompt_text}\"")
    print(f"{'='*70}")

    prompt_indices = graph.tokenize(prompt_text)
    if not prompt_indices:
        return

    path_set = set(prompt_indices)
    FUNC_THRESHOLD = 5000

    # Build forward-edge lookup for path tokens
    path_forward = {}
    for p in prompt_indices:
        s = int(graph.full_off[p])
        e = int(graph.full_off[p + 1])
        path_forward[p] = set(int(graph.full_tgt[j]) for j in range(s, e))

    content_conn = []
    function_conn = []

    for i in range(min(graph.vocab_size, sample_size)):
        if i in path_set:
            continue
        connections = 0
        for p, fwd in path_forward.items():
            if i in fwd:
                connections += 1
        # Also: i -> path
        s = int(graph.full_off[i])
        e = int(graph.full_off[i + 1])
        for j in range(s, min(e, s + 300)):
            if int(graph.full_tgt[j]) in path_set:
                connections += 1
                break  # count at most once per direction

        if graph.in_degree[i] > FUNC_THRESHOLD:
            function_conn.append(connections)
        elif graph.in_degree[i] > 10:
            content_conn.append(connections)

    print(f"\n  Sampled: {len(content_conn)} content, {len(function_conn)} function words")
    max_c = max(max(content_conn, default=0), max(function_conn, default=0))
    for c in range(max_c + 1):
        cc = content_conn.count(c)
        fc = function_conn.count(c)
        if cc == 0 and fc == 0:
            continue
        c_bar = '#' * min(cc * 30 // max(len(content_conn), 1), 40)
        f_bar = '.' * min(fc * 30 // max(len(function_conn), 1), 40)
        print(f"    {c} conn: content={cc:5d} {c_bar}")
        print(f"    {' '*8} func   ={fc:5d} {f_bar}")

    avg_content = np.mean(content_conn) if content_conn else 0
    avg_func = np.mean(function_conn) if function_conn else 0
    print(f"\n  Avg connections -- Content: {avg_content:.2f}, Function: {avg_func:.2f}")
    if avg_content > 0:
        print(f"  Function/Content ratio: {avg_func/avg_content:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="LLN Graph Distance Diagnostic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--distances-only", action="store_true")
    args = parser.parse_args()

    model_path = args.model if args.model else download_model()
    print(f"Loading model...", end=" ", flush=True)
    t0 = time.time()
    graph = LLNGraph(model_path)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"  Vocab: {graph.vocab_size:,} | Edges: {len(graph.full_tgt):,} | "
          f"PMI: {len(graph.pmi_tgt):,}")

    n_pairs = 50 if args.quick else 200
    measure_random_distances(graph, n_pairs=n_pairs)

    if args.distances_only:
        graph.close()
        return

    measure_function_vs_content(graph)

    prompts = [args.prompt] if args.prompt else [
        "The fire burned",
        "Dark clouds",
    ]
    for p in prompts:
        path_coherence_analysis(graph, p)
    for p in prompts[:1]:
        connectivity_histogram(graph, p)

    graph.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
