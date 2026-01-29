# Syntactic Motifs Project

## Overview
This project analyzes syntactic patterns (motifs) in text using constituency parsing. It extracts induced subtrees from parse trees and computes PMI scores to find distinctive patterns for different authors or text sources.

## Live Explorers
- **Gutenberg Authors**: https://pnulty.github.io/syntactic-motifs/explorer.html
- **Human vs LLM**: https://pnulty.github.io/syntactic-motifs/explorer_human_llm.html

## Key Files

### Core Analysis
- `parser.py` - Benepar/spaCy constituency parsing
- `mining.py` - Subtree extraction, pattern canonicalization, terminal extraction
- `analyze_corpus.py` - PMI computation, corpus loading utilities

### Data Generation
- `parse_human_llm.py` - Parses human/LLM texts from `/home/paul/style-tests/data/{human,llm}/`
- `generate_explorer_data.py` - Generates explorer JSON from parsed JSONL

### Explorers
- `explorer.html` + `explorer_data.json` - Gutenberg authors (23 authors)
- `explorer_human_llm.html` + `explorer_human_llm.json` - Human vs LLM comparison

### Data Files (gitignored)
- `gutenberg_parsed.jsonl` - Parsed Gutenberg corpus
- `human_llm_parsed.jsonl` - Parsed human/LLM texts (28 files each, LLM undersampled)

## Running

```bash
# Parse human/LLM texts (uses 28 human files, samples 28 from 252 LLM files)
uv run python parse_human_llm.py --output human_llm_parsed.jsonl

# Generate explorer data
uv run python generate_explorer_data.py human_llm_parsed.jsonl --output explorer_human_llm.json
uv run python generate_explorer_data.py gutenberg_parsed.jsonl --output explorer_data.json
```

## Recent Changes (Jan 2025)
- Added human vs LLM explorer for comparing syntactic patterns
- Fixed `get_terminals_at_depth` in mining.py to return ALL terminals under collapsed subtrees (was only returning first terminal, causing incorrect highlighting)
- Hosted explorers on GitHub Pages

## Potential Future Work
- Add more LLM models for comparison (currently just one source)
- Statistical significance testing for PMI differences
- Clustering of similar patterns
- Time-series analysis (how patterns change across an author's works)
- Export functionality in explorer (CSV of patterns)
- Side-by-side comparison view in explorer (two authors at once)
