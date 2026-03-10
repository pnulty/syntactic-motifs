"""
Generate pre-computed author motif data for the web explorer.

Computes PMI scores for each author vs. balanced corpus baseline,
then saves results to a single JSON file.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from nltk import Tree
from mining import extract_patterns_with_examples, count_terminal_nodes
from analyze_corpus import (
    load_parsed_corpus,
    load_balanced_corpus,
    analyze_parsed_corpus,
    compute_pmi_vs_baseline,
)


def collect_examples(sentences: list[dict], max_depth: int = 5,
                     min_terminals: int = 3, max_examples: int = 25) -> dict[str, list]:
    """Collect more examples per pattern for the explorer."""
    examples = defaultdict(list)

    for record in sentences:
        try:
            tree = Tree.fromstring(record['parse'])
            sent_text = record['sentence']
            title = record.get('title', 'Unknown')

            for pattern, highlighted, sentence in extract_patterns_with_examples(
                tree, sent_text, max_depth=max_depth, min_terminals=min_terminals
            ):
                if len(examples[pattern]) < max_examples:
                    examples[pattern].append({
                        'words': highlighted,
                        'sentence': sentence,
                        'title': title
                    })
        except Exception:
            continue

    return dict(examples)


def generate_author_data(corpus_path: str, author: str, baseline_counts: Counter,
                         baseline_total: int, max_depth: int = 5,
                         min_terminals: int = 3, min_count: int = 5,
                         top_n: int = 50, max_examples: int = 10000) -> dict:
    """Generate explorer data for a single author."""

    # Load author's sentences
    sentences = load_parsed_corpus(corpus_path, author=author)
    if not sentences:
        return None

    # Get titles
    titles = list(set(s.get('title', 'Unknown') for s in sentences))

    # Analyze author corpus
    counts, total_sentences, _ = analyze_parsed_corpus(
        sentences,
        max_depth=max_depth,
        min_terminals=min_terminals,
        collect_examples=0,
        verbose=False
    )

    # Compute PMI vs baseline
    pmi_scores = compute_pmi_vs_baseline(
        counts, total_sentences,
        baseline_counts, baseline_total,
        min_count=min_count
    )

    # Sort by PMI (overused first)
    sorted_patterns = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)

    # Get top overused and underused
    top_overused = sorted_patterns[:top_n]
    top_underused = sorted_patterns[-top_n:][::-1]

    # Collect examples for these patterns
    patterns_needed = set(p for p, _ in top_overused) | set(p for p, _ in top_underused)

    print(f"  Collecting examples for {len(patterns_needed)} patterns...")
    examples = collect_examples(
        sentences,
        max_depth=max_depth,
        min_terminals=min_terminals,
        max_examples=max_examples
    )

    # Build pattern data
    def build_pattern_entry(pattern, pmi):
        author_freq = counts[pattern] / total_sentences
        baseline_freq = baseline_counts.get(pattern, 0) / baseline_total
        ratio = author_freq / baseline_freq if baseline_freq > 0 else float('inf')

        return {
            'pattern': pattern,
            'pmi': round(pmi, 3),
            'count': counts[pattern],
            'author_freq': round(author_freq, 5),
            'baseline_freq': round(baseline_freq, 5),
            'ratio': round(ratio, 2) if ratio != float('inf') else None,
            'terminals': count_terminal_nodes(pattern),
            'examples': examples.get(pattern, [])
        }

    return {
        'author': author,
        'titles': titles,
        'sentences': total_sentences,
        'overused': [build_pattern_entry(p, pmi) for p, pmi in top_overused],
        'underused': [build_pattern_entry(p, pmi) for p, pmi in top_underused]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate explorer data")
    parser.add_argument("corpus", help="Path to parsed JSONL")
    parser.add_argument("--output", "-o", default="explorer_data.json", help="Output JSON file")
    parser.add_argument("--max-per-author", type=int, default=5000, help="Max sentences per author in baseline")
    parser.add_argument("--min-terminals", type=int, default=3, help="Min terminal nodes")
    parser.add_argument("--min-count", type=int, default=5, help="Min pattern count")
    parser.add_argument("--top", type=int, default=500, help="Top N patterns per direction")
    parser.add_argument("--max-examples", type=int, default=10000, help="Max examples per pattern")
    args = parser.parse_args()

    # Get all authors
    print("Scanning for authors...")
    author_counts = Counter()
    with open(args.corpus) as f:
        for line in f:
            record = json.loads(line)
            author_counts[record.get('author', 'Unknown')] += 1

    authors = sorted(author_counts.keys())
    print(f"Found {len(authors)} authors: {', '.join(authors)}")

    # Compute balanced baseline (excluding no one initially, we'll recompute per-author)
    # Actually, for efficiency, compute one baseline excluding nothing, then adjust
    # But the cleaner approach is per-author baseline. Let's do a shared baseline
    # that's "all authors balanced" - the PMI will still be meaningful.

    print(f"\nComputing shared balanced baseline (max {args.max_per_author}/author)...")
    baseline_sentences = load_balanced_corpus(
        args.corpus,
        max_per_author=args.max_per_author,
        exclude_author=None  # Include all for shared baseline
    )
    print(f"Baseline: {len(baseline_sentences)} sentences")

    print("Analyzing baseline...")
    baseline_counts, baseline_total, _ = analyze_parsed_corpus(
        baseline_sentences,
        max_depth=5,
        min_terminals=args.min_terminals,
        collect_examples=0,
        verbose=True
    )

    # Generate data for each author
    explorer_data = {
        'authors': [],
        'baseline_sentences': baseline_total,
        'settings': {
            'min_terminals': args.min_terminals,
            'min_count': args.min_count,
            'max_per_author': args.max_per_author
        }
    }

    for author in authors:
        print(f"\nProcessing {author} ({author_counts[author]} sentences)...")
        author_data = generate_author_data(
            args.corpus,
            author,
            baseline_counts,
            baseline_total,
            min_terminals=args.min_terminals,
            min_count=args.min_count,
            top_n=args.top,
            max_examples=args.max_examples
        )
        if author_data:
            explorer_data['authors'].append(author_data)

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(explorer_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
