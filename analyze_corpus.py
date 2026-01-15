"""
Analyze a large corpus for syntactic motifs.

Usage:
    python analyze_corpus.py corpus_christie.json --limit 100
    python analyze_corpus.py corpus_christie.json --min-nodes 4 --score pmi
"""

import json
import argparse
import re
import math
from collections import Counter, defaultdict
from parser import parse_text, get_nlp
from mining import extract_patterns, extract_patterns_with_examples


def analyze_corpus(texts: list[str], max_depth: int = 4, min_terminals: int = 2,
                   verbose: bool = True, collect_examples: int = 2):
    """
    Extract and count syntactic motifs from a corpus.

    Args:
        texts: List of text passages
        max_depth: Maximum subtree depth
        min_terminals: Minimum terminal (leaf) nodes per pattern
        verbose: Print progress
        collect_examples: Number of examples to collect per pattern

    Returns:
        Tuple of (counts Counter, total_sentences, examples dict)
    """
    counts = Counter()
    examples = defaultdict(list)  # pattern -> list of (highlighted_words, sentence)
    total_sentences = 0

    nlp = get_nlp()

    for i, text in enumerate(texts):
        if verbose and i % 10 == 0:
            print(f"\rProcessing text {i+1}/{len(texts)}...", end="", flush=True)

        # Normalize whitespace
        text = ' '.join(text.split())

        try:
            doc = nlp(text)
            for sent in doc.sents:
                total_sentences += 1
                sent_text = sent.text.strip()
                from nltk import Tree
                tree = Tree.fromstring(sent._.parse_string)

                for pattern, highlighted, sentence in extract_patterns_with_examples(
                    tree, sent_text, max_depth=max_depth, min_terminals=min_terminals
                ):
                    counts[pattern] += 1
                    # Collect examples (limit per pattern)
                    if len(examples[pattern]) < collect_examples:
                        # Avoid duplicate examples
                        if not any(highlighted == ex[0] for ex in examples[pattern]):
                            examples[pattern].append((highlighted, sentence))

        except Exception as e:
            if verbose:
                print(f"\n  Warning: Failed to parse text {i}: {e}")
            continue

    if verbose:
        print(f"\rProcessed {len(texts)} texts, {total_sentences} sentences")

    return counts, total_sentences, dict(examples)


def extract_labels(pattern: str) -> list[str]:
    """Extract all node labels from a pattern string like '(NP (DT) (NN))'."""
    import re
    # Find all labels: word characters after opening paren
    return re.findall(r'\(([A-Z$]+[A-Z0-9$-]*)', pattern)


def compute_pmi_scores(counts: Counter, total_sentences: int, min_count: int = 5) -> dict[str, float]:
    """
    Compute PMI (Pointwise Mutual Information) scores for patterns.

    PMI measures how much more likely a pattern is to occur than expected
    if its component labels were independent.

    PMI(pattern) = log2(P(pattern) / P_expected)
    where P_expected = product of individual label probabilities

    Args:
        counts: Counter of pattern frequencies
        total_sentences: Total number of sentences
        min_count: Minimum count for a pattern to be scored

    Returns:
        Dict mapping patterns to PMI scores
    """
    # Count individual label frequencies across all patterns
    label_counts = Counter()
    total_label_occurrences = 0

    for pattern, count in counts.items():
        labels = extract_labels(pattern)
        for label in labels:
            label_counts[label] += count
            total_label_occurrences += count

    # Compute label probabilities
    label_probs = {label: c / total_label_occurrences for label, c in label_counts.items()}

    # Compute PMI for each pattern
    pmi_scores = {}
    total_patterns = sum(counts.values())

    for pattern, count in counts.items():
        if count < min_count:
            continue

        labels = extract_labels(pattern)
        if not labels:
            continue

        # Observed probability
        p_observed = count / total_patterns

        # Expected probability (product of independent label probs)
        p_expected = 1.0
        for label in labels:
            p_expected *= label_probs.get(label, 1e-10)

        # PMI = log2(observed / expected)
        if p_expected > 0 and p_observed > 0:
            pmi = math.log2(p_observed / p_expected)
            pmi_scores[pattern] = pmi

    return pmi_scores


def compute_npmi_scores(counts: Counter, total_sentences: int, min_count: int = 5) -> dict[str, float]:
    """
    Compute Normalized PMI scores (range -1 to 1).

    NPMI = PMI / -log2(P(pattern))

    This normalizes PMI to account for pattern frequency.
    """
    pmi_scores = compute_pmi_scores(counts, total_sentences, min_count)
    total_patterns = sum(counts.values())

    npmi_scores = {}
    for pattern, pmi in pmi_scores.items():
        p_pattern = counts[pattern] / total_patterns
        if p_pattern > 0:
            # Normalize by -log2(p) which is the self-information
            h_pattern = -math.log2(p_pattern)
            if h_pattern > 0:
                npmi_scores[pattern] = pmi / h_pattern

    return npmi_scores


def highlight_in_sentence(sentence: str, words: str) -> str:
    """Highlight the motif words in the sentence using **bold** markers."""
    # Escape regex special chars in words
    word_list = words.split()
    if not word_list:
        return sentence

    # Build pattern to find the sequence of words
    # Allow for punctuation attached to words
    pattern_parts = []
    for w in word_list:
        escaped = re.escape(w)
        pattern_parts.append(escaped)

    # Join with flexible whitespace/punctuation
    pattern = r'\b' + r'[\s,;:\'"]*'.join(pattern_parts) + r'\b'

    def replacer(match):
        return f"**{match.group(0)}**"

    result = re.sub(pattern, replacer, sentence, count=1, flags=re.IGNORECASE)
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze corpus for syntactic motifs")
    parser.add_argument("corpus", help="Path to corpus JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of texts")
    parser.add_argument("--max-depth", type=int, default=4, help="Max subtree depth")
    parser.add_argument("--min-terminals", type=int, default=2, help="Min terminal (leaf) nodes per pattern")
    parser.add_argument("--min-count", type=int, default=5, help="Min occurrences (patterns below this are excluded)")
    parser.add_argument("--top", type=int, default=30, help="Show top N patterns")
    parser.add_argument("--score", choices=["freq", "pmi", "npmi"], default="freq",
                        help="Scoring method: freq (frequency), pmi (pointwise mutual information), npmi (normalized PMI)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    args = parser.parse_args()

    # Load corpus
    with open(args.corpus) as f:
        texts = json.load(f)

    if args.limit:
        texts = texts[:args.limit]

    print(f"Corpus: {args.corpus}")
    print(f"Texts: {len(texts)}")
    print(f"Settings: max_depth={args.max_depth}, min_terminals={args.min_terminals}, min_count={args.min_count}, score={args.score}")
    print()

    # Analyze
    counts, total_sentences, examples = analyze_corpus(
        texts,
        max_depth=args.max_depth,
        min_terminals=args.min_terminals,
        collect_examples=2
    )

    # Filter by min_count
    filtered_counts = {p: c for p, c in counts.items() if c >= args.min_count}

    # Compute scores based on method
    if args.score == "pmi":
        scores = compute_pmi_scores(counts, total_sentences, min_count=args.min_count)
        score_label = "PMI"
    elif args.score == "npmi":
        scores = compute_npmi_scores(counts, total_sentences, min_count=args.min_count)
        score_label = "NPMI"
    else:
        # Frequency (per-sentence), filtered by min_count
        scores = {p: c / total_sentences for p, c in filtered_counts.items()}
        score_label = "freq"

    # Sort by score
    sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Display top patterns with examples
    from mining import count_terminal_nodes
    print(f"\nTop {args.top} syntactic motifs (by {score_label}):")
    print("=" * 80)

    for pattern, score in sorted_patterns[:args.top]:
        count = counts[pattern]
        freq = count / total_sentences
        n_terminals = count_terminal_nodes(pattern)

        if args.score in ("pmi", "npmi"):
            print(f"\n{score:+.3f} {score_label}  ({count:5d}x, {n_terminals}T, freq={freq:.4f})  {pattern}")
        else:
            print(f"\n{score:.4f}  ({count:5d}x, {n_terminals}T)  {pattern}")

        # Show examples
        if pattern in examples:
            for i, (words, sentence) in enumerate(examples[pattern][:2]):
                highlighted = highlight_in_sentence(sentence, words)
                # Truncate long sentences
                if len(highlighted) > 100:
                    highlighted = highlighted[:100] + "..."
                print(f"    ex{i+1}: {highlighted}")

    print(f"\n{'=' * 80}")
    print(f"Total unique patterns: {len(counts)}")
    print(f"Total patterns scored: {len(scores)}")
    print(f"Total sentences: {total_sentences}")

    # Save results
    if args.output:
        from mining import count_terminal_nodes
        results = {
            "corpus": args.corpus,
            "texts": len(texts),
            "sentences": total_sentences,
            "settings": {
                "max_depth": args.max_depth,
                "min_terminals": args.min_terminals,
                "min_count": args.min_count,
                "score_method": args.score,
            },
            "patterns": [
                {
                    "pattern": p,
                    "count": counts[p],
                    "terminals": count_terminal_nodes(p),
                    "frequency": round(counts[p] / total_sentences, 6),
                    "score": round(score, 6),
                    "score_type": args.score,
                    "examples": [
                        {"words": w, "sentence": s}
                        for w, s in examples.get(p, [])[:2]
                    ]
                }
                for p, score in sorted_patterns
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
