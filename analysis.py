"""
Corpus profiling and pattern comparison.

Computes frequency profiles and identifies discriminative patterns between corpora.
"""

from collections import Counter
from parser import parse_text
from mining import count_patterns


def corpus_profile(texts: list[str], max_depth: int = 4, min_depth: int = 2, min_terminals: int = 2) -> dict[str, float]:
    """
    Compute a normalized frequency profile for a corpus.

    Frequencies are normalized by the number of sentences in the corpus,
    giving patterns-per-sentence rates.

    Args:
        texts: List of text strings (each may contain multiple sentences)
        max_depth: Maximum subtree depth for pattern extraction
        min_depth: Minimum subtree depth (default 2 excludes single POS tags)
        min_terminals: Minimum number of terminal (leaf) nodes in patterns

    Returns:
        Dict mapping patterns to their per-sentence frequencies
    """
    all_trees = []
    sentence_count = 0

    for text in texts:
        for tree in parse_text(text):
            all_trees.append(tree)
            sentence_count += 1

    if sentence_count == 0:
        return {}

    counts = count_patterns(all_trees, max_depth, min_depth, min_terminals)

    # Normalize by sentence count
    return {pattern: count / sentence_count for pattern, count in counts.items()}


def discriminative_patterns(
    profile1: dict[str, float],
    profile2: dict[str, float],
    top_k: int = 20
) -> list[tuple[str, float, float, float]]:
    """
    Find patterns that differ most between two corpus profiles.

    Returns patterns sorted by the absolute difference in frequency,
    which indicates patterns that are more characteristic of one corpus.

    Args:
        profile1: First corpus frequency profile
        profile2: Second corpus frequency profile
        top_k: Number of top discriminative patterns to return

    Returns:
        List of tuples (pattern, freq1, freq2, difference) sorted by |difference|
    """
    all_patterns = set(profile1.keys()) | set(profile2.keys())

    differences = []
    for pattern in all_patterns:
        freq1 = profile1.get(pattern, 0.0)
        freq2 = profile2.get(pattern, 0.0)
        diff = freq1 - freq2
        differences.append((pattern, freq1, freq2, diff))

    # Sort by absolute difference, descending
    differences.sort(key=lambda x: abs(x[3]), reverse=True)

    return differences[:top_k]


def filter_by_support(
    profile: dict[str, float],
    min_support: float
) -> dict[str, float]:
    """
    Remove patterns below a minimum support threshold.

    Args:
        profile: Frequency profile
        min_support: Minimum frequency to keep a pattern

    Returns:
        Filtered profile containing only patterns >= min_support
    """
    return {p: f for p, f in profile.items() if f >= min_support}


def top_patterns(profile: dict[str, float], top_k: int = 20) -> list[tuple[str, float]]:
    """
    Get the most frequent patterns from a profile.

    Args:
        profile: Frequency profile
        top_k: Number of top patterns to return

    Returns:
        List of (pattern, frequency) tuples sorted by frequency descending
    """
    sorted_patterns = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    return sorted_patterns[:top_k]


def compare_corpora(
    texts1: list[str],
    texts2: list[str],
    max_depth: int = 4,
    min_depth: int = 2,
    min_terminals: int = 2,
    min_support: float = 0.0,
    top_k: int = 20
) -> dict:
    """
    Full comparison of two corpora.

    Computes profiles for both corpora and finds discriminative patterns.

    Args:
        texts1: First corpus texts
        texts2: Second corpus texts
        max_depth: Maximum subtree depth
        min_depth: Minimum subtree depth (default 2 excludes single POS tags)
        min_terminals: Minimum number of terminal (leaf) nodes in patterns
        min_support: Minimum frequency threshold
        top_k: Number of top patterns to return

    Returns:
        Dict with keys:
            - profile1: First corpus profile
            - profile2: Second corpus profile
            - top_patterns1: Most frequent in corpus 1
            - top_patterns2: Most frequent in corpus 2
            - discriminative: Patterns that differ most between corpora
    """
    profile1 = corpus_profile(texts1, max_depth, min_depth, min_terminals)
    profile2 = corpus_profile(texts2, max_depth, min_depth, min_terminals)

    if min_support > 0:
        profile1 = filter_by_support(profile1, min_support)
        profile2 = filter_by_support(profile2, min_support)

    return {
        "profile1": profile1,
        "profile2": profile2,
        "top_patterns1": top_patterns(profile1, top_k),
        "top_patterns2": top_patterns(profile2, top_k),
        "discriminative": discriminative_patterns(profile1, profile2, top_k)
    }
