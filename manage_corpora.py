"""
Load and manage text corpora for syntactic motif analysis.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass

from parser import get_nlp


GUTENBERG_DIR = Path("/home/paul/style-tests/data/human-gutenberg-sample")


@dataclass
class TextEntry:
    """A text with metadata."""
    title: str
    author: str
    year: int
    text: str
    filename: str

    def __repr__(self):
        return f"TextEntry({self.title!r}, {self.author}, {self.year}, {len(self.text):,} chars)"


def parse_filename(filename: str) -> tuple[str, str, int]:
    """
    Parse a filename like 'Alices_Adventures-Carroll-1865.txt'.

    Returns:
        Tuple of (title, author, year)
    """
    stem = Path(filename).stem  # Remove .txt
    parts = stem.rsplit("-", 2)  # Split from right to handle titles with hyphens

    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    title = parts[0].replace("_", " ")
    author = parts[1]
    year = int(parts[2])

    return title, author, year


def load_gutenberg_texts(directory: Path = GUTENBERG_DIR) -> list[TextEntry]:
    """
    Load all texts from the Gutenberg sample directory.

    Args:
        directory: Path to directory containing .txt files

    Returns:
        List of TextEntry objects with metadata
    """
    entries = []

    for filepath in sorted(directory.glob("*.txt")):
        title, author, year = parse_filename(filepath.name)
        text = filepath.read_text(encoding="utf-8")

        entries.append(TextEntry(
            title=title,
            author=author,
            year=year,
            text=text,
            filename=filepath.name
        ))

    return entries


def list_corpus(entries: list[TextEntry]):
    """Print a summary of the corpus."""
    print(f"Corpus: {len(entries)} texts\n")

    # Group by author
    by_author = {}
    for e in entries:
        by_author.setdefault(e.author, []).append(e)

    for author in sorted(by_author.keys()):
        texts = by_author[author]
        total_chars = sum(len(t.text) for t in texts)
        print(f"{author}: {len(texts)} texts, {total_chars:,} chars")
        for t in sorted(texts, key=lambda x: x.year):
            print(f"  - {t.title} ({t.year}) - {len(t.text):,} chars")

    print(f"\nTotal: {sum(len(e.text) for e in entries):,} characters")


def save_corpus_json(entries: list[TextEntry], output_path: str):
    """Save corpus to JSON file (texts only, as list of strings)."""
    texts = [e.text for e in entries]
    with open(output_path, 'w') as f:
        json.dump(texts, f, indent=2)
    print(f"Saved {len(texts)} texts to {output_path}")


def save_corpus_jsonl(entries: list[TextEntry], output_path: str):
    """Save corpus to JSONL file with metadata."""
    with open(output_path, 'w') as f:
        for e in entries:
            record = {
                "title": e.title,
                "author": e.author,
                "year": e.year,
                "text": e.text,
                "filename": e.filename
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(entries)} texts to {output_path}")


def parse_and_save_corpus(entries: list[TextEntry], output_path: str, source: str = "human"):
    """
    Parse all texts and save constituency parse trees to a JSONL file.

    Args:
        entries: List of TextEntry objects to parse
        output_path: Path to output JSONL file
        source: Source label ("human" or "llm")
    """
    nlp = get_nlp()
    total_sentences = 0
    failed_sentences = 0

    with open(output_path, 'w') as f:
        for i, entry in enumerate(entries):
            # Truncate to first 2M characters and normalize whitespace
            text = ' '.join(entry.text[:2_000_000].split())

            try:
                doc = nlp(text)
            except Exception as e:
                print(f"  Error parsing text {entry.title}: {e}", file=sys.stderr)
                continue

            for sent in doc.sents:
                try:
                    parse_string = sent._.parse_string
                    record = {
                        "source": source,
                        "author": entry.author,
                        "title": entry.title,
                        "year": entry.year,
                        "sentence": sent.text,
                        "parse": parse_string
                    }
                    f.write(json.dumps(record) + "\n")
                    total_sentences += 1
                except Exception as e:
                    failed_sentences += 1
                    print(f"  Failed to parse sentence in {entry.title}: {e}", file=sys.stderr)

            # Progress report every 10 texts
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(entries)} texts ({total_sentences} sentences)")

    print(f"\nDone! Saved {total_sentences} sentences to {output_path}")
    if failed_sentences > 0:
        print(f"  ({failed_sentences} sentences failed to parse)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage text corpora for syntactic motif analysis")
    parser.add_argument("--parse", metavar="OUTPUT", help="Parse corpus and save to OUTPUT.jsonl")
    parser.add_argument("--limit", type=int, help="Limit to N texts (for testing)")
    parser.add_argument("--source", default="human", help="Source label: 'human' or 'llm' (default: human)")
    args = parser.parse_args()

    print("Loading Gutenberg corpus...")
    entries = load_gutenberg_texts()

    if args.limit:
        entries = entries[:args.limit]
        print(f"Limited to {len(entries)} texts")

    if args.parse:
        output_path = args.parse if args.parse.endswith('.jsonl') else f"{args.parse}.jsonl"
        print(f"Parsing {len(entries)} texts...")
        parse_and_save_corpus(entries, output_path, source=args.source)
    else:
        list_corpus(entries)
