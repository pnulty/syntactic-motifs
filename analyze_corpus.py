"""
Analyze a pre-parsed corpus for syntactic motifs.

Usage:
    python analyze_corpus.py gutenberg_parsed.jsonl --limit 1000
    python analyze_corpus.py gutenberg_parsed.jsonl --author Dickens --min-terminals 3
    python analyze_corpus.py gutenberg_parsed.jsonl --author Dickens --score pmi --top 50
"""

import json
import argparse
import re
import math
import random
from collections import Counter, defaultdict
from nltk import Tree
from mining import extract_patterns_with_examples, count_terminal_nodes


def load_parsed_corpus(path: str, limit: int = None, author: str = None,
                       title: str = None, source: str = None) -> list[dict]:
    """
    Load pre-parsed sentences from JSONL file.

    Args:
        path: Path to JSONL file
        limit: Maximum sentences to load
        author: Filter by author name (case-insensitive substring)
        title: Filter by title (case-insensitive substring)
        source: Filter by source (e.g., "human", "llm")

    Returns:
        List of dicts with 'sentence', 'parse', and metadata
    """
    sentences = []
    with open(path) as f:
        for line in f:
            if limit and len(sentences) >= limit:
                break

            record = json.loads(line)

            # Apply filters
            if author and author.lower() not in record.get('author', '').lower():
                continue
            if title and title.lower() not in record.get('title', '').lower():
                continue
            if source and record.get('source') != source:
                continue

            sentences.append(record)

    return sentences


def load_balanced_corpus(path: str, max_per_author: int = 5000,
                         exclude_author: str = None, seed: int = 42) -> list[dict]:
    """
    Load corpus with balanced author representation.

    Args:
        path: Path to JSONL file
        max_per_author: Maximum sentences per author
        exclude_author: Author to exclude (for computing baseline)
        seed: Random seed for reproducible sampling

    Returns:
        List of dicts with balanced author representation
    """
    # Group by author
    by_author = defaultdict(list)
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            author = record.get('author', 'Unknown')
            if exclude_author and exclude_author.lower() in author.lower():
                continue
            by_author[author].append(record)

    # Sample up to max_per_author from each
    random.seed(seed)
    balanced = []
    for author, sentences in by_author.items():
        if len(sentences) <= max_per_author:
            balanced.extend(sentences)
        else:
            balanced.extend(random.sample(sentences, max_per_author))

    random.shuffle(balanced)
    return balanced


def analyze_parsed_corpus(sentences: list[dict], max_depth: int = 4,
                          min_terminals: int = 2, verbose: bool = True,
                          collect_examples: int = 2):
    """
    Extract and count syntactic motifs from pre-parsed sentences.

    Args:
        sentences: List of dicts with 'sentence' and 'parse' keys
        max_depth: Maximum subtree depth
        min_terminals: Minimum terminal (leaf) nodes per pattern
        verbose: Print progress
        collect_examples: Number of examples to collect per pattern

    Returns:
        Tuple of (counts Counter, total_sentences, examples dict)
    """
    counts = Counter()
    examples = defaultdict(list)
    total_sentences = 0
    errors = 0

    for i, record in enumerate(sentences):
        if verbose and i % 5000 == 0:
            print(f"\rProcessing sentence {i+1}/{len(sentences)}...", end="", flush=True)

        try:
            tree = Tree.fromstring(record['parse'])
            sent_text = record['sentence']
            total_sentences += 1

            for pattern, highlighted, sentence in extract_patterns_with_examples(
                tree, sent_text, max_depth=max_depth, min_terminals=min_terminals
            ):
                counts[pattern] += 1
                if len(examples[pattern]) < collect_examples:
                    if not any(highlighted == ex[0] for ex in examples[pattern]):
                        examples[pattern].append((highlighted, sentence))

        except Exception as e:
            errors += 1
            if verbose and errors <= 3:
                print(f"\n  Warning: Failed to process sentence {i}: {e}")
            continue

    if verbose:
        print(f"\rProcessed {total_sentences} sentences ({errors} errors)")

    return counts, total_sentences, dict(examples)


def extract_labels(pattern: str) -> list[str]:
    """Extract all node labels from a pattern string."""
    return re.findall(r'\(([A-Z$]+[A-Z0-9$-]*)', pattern)


def compute_pmi_vs_baseline(author_counts: Counter, author_total: int,
                            baseline_counts: Counter, baseline_total: int,
                            min_count: int = 5) -> dict[str, float]:
    """
    Compute PMI comparing author frequencies to corpus baseline.

    PMI(pattern, author) = log2(freq_author / freq_baseline)

    Positive = author overuses this pattern relative to baseline
    Negative = author underuses this pattern
    Zero = matches baseline

    Args:
        author_counts: Pattern counts for the author
        author_total: Total sentences for author
        baseline_counts: Pattern counts for balanced corpus
        baseline_total: Total sentences in baseline
        min_count: Minimum count in author corpus to include

    Returns:
        Dict mapping patterns to PMI scores
    """
    pmi_scores = {}

    for pattern, count in author_counts.items():
        if count < min_count:
            continue

        # Author frequency (per sentence)
        freq_author = count / author_total

        # Baseline frequency (per sentence)
        baseline_count = baseline_counts.get(pattern, 0)
        if baseline_count == 0:
            # Pattern doesn't appear in baseline - very distinctive!
            # Use a small smoothed value
            freq_baseline = 0.5 / baseline_total
        else:
            freq_baseline = baseline_count / baseline_total

        pmi = math.log2(freq_author / freq_baseline)
        pmi_scores[pattern] = pmi

    return pmi_scores


def highlight_in_sentence(sentence: str, words: str) -> str:
    """Highlight the motif words in the sentence using **bold** markers."""
    word_list = words.split()
    if not word_list:
        return sentence

    pattern_parts = [re.escape(w) for w in word_list]
    pattern = r'\b' + r'[\s,;:\'"]*'.join(pattern_parts) + r'\b'

    def replacer(match):
        return f"**{match.group(0)}**"

    return re.sub(pattern, replacer, sentence, count=1, flags=re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-parsed corpus for syntactic motifs")
    parser.add_argument("corpus", help="Path to pre-parsed JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sentences")
    parser.add_argument("--author", type=str, default=None, help="Filter by author")
    parser.add_argument("--title", type=str, default=None, help="Filter by title")
    parser.add_argument("--source", type=str, default=None, help="Filter by source (e.g., human, llm)")
    parser.add_argument("--max-depth", type=int, default=4, help="Max subtree depth")
    parser.add_argument("--min-terminals", type=int, default=2, help="Min terminal nodes per pattern")
    parser.add_argument("--min-count", type=int, default=5, help="Min occurrences to include")
    parser.add_argument("--top", type=int, default=30, help="Show top N patterns")
    parser.add_argument("--score", choices=["freq", "pmi"], default="freq",
                        help="Scoring: freq (raw frequency) or pmi (vs corpus baseline, requires --author)")
    parser.add_argument("--max-per-author", type=int, default=5000,
                        help="Max sentences per author in baseline (for balancing)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    args = parser.parse_args()

    # PMI mode requires an author filter
    if args.score == "pmi" and not args.author:
        print("Error: --score pmi requires --author to compare against corpus baseline")
        return

    # Load author's corpus
    print(f"Loading {args.corpus}...")
    sentences = load_parsed_corpus(
        args.corpus,
        limit=args.limit,
        author=args.author,
        title=args.title,
        source=args.source
    )

    if not sentences:
        print("No sentences matched the filters.")
        return

    # Show what we loaded
    authors = set(s.get('author', 'Unknown') for s in sentences)
    titles = set(s.get('title', 'Unknown') for s in sentences)
    print(f"Loaded {len(sentences)} sentences")
    print(f"Authors: {', '.join(sorted(authors)[:5])}{'...' if len(authors) > 5 else ''}")
    print(f"Titles: {len(titles)} works")
    print(f"Settings: max_depth={args.max_depth}, min_terminals={args.min_terminals}, min_count={args.min_count}")
    print()

    # Analyze author corpus
    counts, total_sentences, examples = analyze_parsed_corpus(
        sentences,
        max_depth=args.max_depth,
        min_terminals=args.min_terminals,
        collect_examples=2
    )

    # Filter by min_count
    filtered_counts = {p: c for p, c in counts.items() if c >= args.min_count}

    # Compute scores
    if args.score == "pmi":
        # Load balanced baseline (excluding target author)
        print(f"\nLoading balanced baseline (max {args.max_per_author}/author, excluding {args.author})...")
        baseline_sentences = load_balanced_corpus(
            args.corpus,
            max_per_author=args.max_per_author,
            exclude_author=args.author
        )
        print(f"Baseline: {len(baseline_sentences)} sentences")

        # Analyze baseline
        print("Analyzing baseline...")
        baseline_counts, baseline_total, _ = analyze_parsed_corpus(
            baseline_sentences,
            max_depth=args.max_depth,
            min_terminals=args.min_terminals,
            collect_examples=0,
            verbose=True
        )

        scores = compute_pmi_vs_baseline(
            counts, total_sentences,
            baseline_counts, baseline_total,
            min_count=args.min_count
        )
        score_label = "PMI"
    else:
        scores = {p: c / total_sentences for p, c in filtered_counts.items()}
        score_label = "freq"

    sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Display results
    if args.score == "pmi":
        # Show both overused (top) and underused (bottom) patterns
        print(f"\nTop {args.top} OVERUSED patterns (positive PMI = author uses more than baseline):")
        print("=" * 80)

        for pattern, score in sorted_patterns[:args.top]:
            count = counts[pattern]
            freq = count / total_sentences
            baseline_freq = baseline_counts.get(pattern, 0) / baseline_total if baseline_counts.get(pattern, 0) > 0 else 0
            n_terminals = count_terminal_nodes(pattern)
            ratio = freq / baseline_freq if baseline_freq > 0 else float('inf')

            print(f"\n{score:+.3f} PMI  ({count:5d}x, {n_terminals}T, author={freq:.4f}, baseline={baseline_freq:.4f}, {ratio:.1f}x)  {pattern}")

            if pattern in examples:
                for i, (words, sentence) in enumerate(examples[pattern][:2]):
                    highlighted = highlight_in_sentence(sentence, words)
                    if len(highlighted) > 100:
                        highlighted = highlighted[:100] + "..."
                    print(f"    ex{i+1}: {highlighted}")

        print(f"\n\nTop {args.top} UNDERUSED patterns (negative PMI = author uses less than baseline):")
        print("=" * 80)

        for pattern, score in sorted_patterns[-args.top:][::-1]:
            count = counts[pattern]
            freq = count / total_sentences
            baseline_freq = baseline_counts.get(pattern, 0) / baseline_total if baseline_counts.get(pattern, 0) > 0 else 0
            n_terminals = count_terminal_nodes(pattern)
            ratio = freq / baseline_freq if baseline_freq > 0 else float('inf')

            print(f"\n{score:+.3f} PMI  ({count:5d}x, {n_terminals}T, author={freq:.4f}, baseline={baseline_freq:.4f}, {ratio:.1f}x)  {pattern}")

            if pattern in examples:
                for i, (words, sentence) in enumerate(examples[pattern][:2]):
                    highlighted = highlight_in_sentence(sentence, words)
                    if len(highlighted) > 100:
                        highlighted = highlighted[:100] + "..."
                    print(f"    ex{i+1}: {highlighted}")
    else:
        print(f"\nTop {args.top} syntactic motifs (by {score_label}):")
        print("=" * 80)

        for pattern, score in sorted_patterns[:args.top]:
            count = counts[pattern]
            freq = count / total_sentences
            n_terminals = count_terminal_nodes(pattern)

            print(f"\n{score:.4f}  ({count:5d}x, {n_terminals}T)  {pattern}")

            if pattern in examples:
                for i, (words, sentence) in enumerate(examples[pattern][:2]):
                    highlighted = highlight_in_sentence(sentence, words)
                    if len(highlighted) > 100:
                        highlighted = highlighted[:100] + "..."
                    print(f"    ex{i+1}: {highlighted}")

    print(f"\n{'=' * 80}")
    print(f"Total unique patterns: {len(counts)}")
    print(f"Patterns meeting min_count: {len(filtered_counts)}")
    print(f"Total sentences: {total_sentences}")

    # Save results
    if args.output:
        results = {
            "corpus": args.corpus,
            "filters": {
                "author": args.author,
                "title": args.title,
                "source": args.source,
                "limit": args.limit,
            },
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
