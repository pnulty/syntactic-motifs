"""
Parse human and LLM texts into constituency trees for motif analysis.

Reads text files from human/ and llm/ folders, parses with Benepar,
and outputs JSONL with source labels for explorer comparison.
"""

import argparse
import json
import random
from pathlib import Path
from parser import parse_text


def get_text_files(folder: Path) -> list[Path]:
    """Get all .txt files in a folder, sorted by name."""
    return sorted(folder.glob("*.txt"))


def parse_file(filepath: Path, source: str) -> list[dict]:
    """Parse a single text file into sentence records."""
    records = []

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        # Truncate very long texts (same as existing corpus parser)
        if len(text) > 2_000_000:
            text = text[:2_000_000]

        for tree in parse_text(text):
            sentence = " ".join(tree.leaves())
            parse_str = str(tree)
            records.append({
                "source": source,
                "filename": filepath.name,
                "sentence": sentence,
                "parse": parse_str,
                # Use source as "author" for compatibility with generate_explorer_data.py
                "author": source,
                "title": filepath.stem
            })
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")

    return records


def main():
    argparser = argparse.ArgumentParser(
        description="Parse human and LLM texts for syntactic motif analysis"
    )
    argparser.add_argument(
        "--human-dir",
        type=Path,
        default=Path("/home/paul/style-tests/data/human"),
        help="Directory containing human-written texts"
    )
    argparser.add_argument(
        "--llm-dir",
        type=Path,
        default=Path("/home/paul/style-tests/data/llm"),
        help="Directory containing LLM-generated texts"
    )
    argparser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("human_llm_parsed.jsonl"),
        help="Output JSONL file"
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LLM file sampling"
    )
    args = argparser.parse_args()

    # Get file lists
    human_files = get_text_files(args.human_dir)
    llm_files = get_text_files(args.llm_dir)

    print(f"Found {len(human_files)} human files")
    print(f"Found {len(llm_files)} LLM files")

    # Undersample LLM files to match human count
    random.seed(args.seed)
    n_human = len(human_files)
    if len(llm_files) > n_human:
        llm_files = random.sample(llm_files, n_human)
        print(f"Sampled {len(llm_files)} LLM files to match human count")

    # Parse all files
    all_records = []

    print("\nParsing human texts...")
    for i, filepath in enumerate(human_files, 1):
        print(f"  [{i}/{len(human_files)}] {filepath.name}")
        records = parse_file(filepath, "human")
        all_records.extend(records)
        print(f"    -> {len(records)} sentences")

    print("\nParsing LLM texts...")
    for i, filepath in enumerate(llm_files, 1):
        print(f"  [{i}/{len(llm_files)}] {filepath.name}")
        records = parse_file(filepath, "llm")
        all_records.extend(records)
        print(f"    -> {len(records)} sentences")

    # Write output
    print(f"\nWriting {len(all_records)} records to {args.output}...")
    with open(args.output, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # Summary
    human_count = sum(1 for r in all_records if r["source"] == "human")
    llm_count = sum(1 for r in all_records if r["source"] == "llm")
    print(f"\nDone! Summary:")
    print(f"  Human sentences: {human_count}")
    print(f"  LLM sentences: {llm_count}")
    print(f"  Total: {len(all_records)}")


if __name__ == "__main__":
    main()
