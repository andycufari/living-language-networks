"""
VocabRank — three-tier decanted vocabulary classification

Replaces absolute in_degree thresholds (`> 5000`, `> 15000`, `> 20000`)
with rank-based tiers. Model-agnostic: the same rank cutoffs mean the
same thing across v13, v16, and any future model.

Three tiers, top-down:

  FUNCTION     (rank 0..n_function-1)
    Classic glue: punctuation, articles, particles, copulas.
    ",", ".", "and", "of", "the", "is", "was", "to", "in", "that"
    Never count as content. Never activate from. Never target.

  SEMI-FUNCTION (rank n_function..n_semi_function-1)
    Common content that behaves function-like: "also", "often", "much",
    "really", "showed", "took". Grammatical glue with some meaning.
    Allowed through the fence. Not counted as content hits. Can be
    de-weighted in memory.

  CONTENT       (rank n_semi_function..)
    Real semantic tokens: "fire", "king", "brightly", "extinguisher".
    This is where generation targets live.

Default cutoffs (measured from v16 distribution in measure_vocab_ranks.py):

  n_function      = 150   (the natural knee in the in_degree curve)
  n_semi_function = 500   (end of semi-function zone; content below)

Both are exposed as constructor args so they can be swept.
"""
import numpy as np


class VocabRank:
    def __init__(self, graph, n_function=150, n_semi_function=500):
        """Build rank lookup for graph vocabulary.

        Args:
            graph: LLNGraph instance
            n_function: cutoff rank for "function" tier (default 150)
            n_semi_function: cutoff rank for "semi-function" tier (default 500)

        Two cutoffs, two uses:
          - n_function:      "true" function words (punctuation, articles,
                             particles). Used for .is_function. These are
                             skipped in activation and never targeted.
          - n_semi_function: end of the "noise" tier. Used for .is_noise
                             (density/memory calculation) AND for target
                             filtering. Top-N by in_degree are ineligible
                             as targets because they dilute topical ranking.

        Both are model-agnostic (rank-based) and tunable via CLI args.
        """
        self.n_function = n_function
        self.n_semi_function = n_semi_function
        self.vocab_size = graph.vocab_size

        # Sort descending by in_degree, build rank[idx] = position in sorted order
        in_deg = graph.in_degree[:graph.vocab_size]
        ranked = np.argsort(-in_deg)
        self.rank = np.empty(graph.vocab_size, dtype=np.int32)
        for r in range(graph.vocab_size):
            self.rank[int(ranked[r])] = r

        # Precompute sets for fast membership tests
        self.function_set = set(int(ranked[r]) for r in range(n_function))
        self.semi_function_set = set(int(ranked[r])
                                      for r in range(n_function, n_semi_function))
        # Noise set (function + semi) for density/memory/target filtering
        self.noise_set = self.function_set | self.semi_function_set
        # Content = everything not in function set.
        # NOTE: semi-function words are allowed in the "content" tier when used
        # as fence members (grammar glue). For memory/density and targets,
        # they're noise.

    def is_function(self, idx):
        """Top tier: pure function words and punctuation."""
        return int(idx) in self.function_set

    def is_semi_function(self, idx):
        """Middle tier: semi-function / common content words."""
        return int(idx) in self.semi_function_set

    def is_content(self, idx):
        """Content = not in function set. Semi-function counts as content
        for broad filters, but density/memory should use is_noise instead."""
        return int(idx) not in self.function_set

    def is_strict_content(self, idx):
        """Strict content = not function AND not semi-function.
        Use this for density calculation — these are the tokens whose
        connections carry topical information.
        """
        return int(idx) not in self.noise_set

    def is_noise(self, idx):
        """Noise for density/memory/target filtering = function + semi-function."""
        return int(idx) in self.noise_set

    def role(self, idx):
        """Return 'function', 'semi', or 'content' for an index."""
        r = int(self.rank[int(idx)])
        if r < self.n_function:
            return 'function'
        elif r < self.n_semi_function:
            return 'semi'
        else:
            return 'content'

    def get_rank(self, idx):
        return int(self.rank[int(idx)])
