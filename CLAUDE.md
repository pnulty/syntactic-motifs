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
- `explainer.html` - "What is a Syntactic Motif?" explanatory page with SVG tree diagrams

### Data Files (gitignored)
- `gutenberg_parsed.jsonl` - Parsed Gutenberg corpus
- `human_llm_parsed.jsonl` - Parsed human/LLM texts (28 files each, LLM undersampled)

### Tests
- `test_mining.py` - Unit tests for mining.py and analyze_corpus.py (46 tests)

## Running

```bash
# Parse human/LLM texts (uses 28 human files, samples 28 from 252 LLM files)
uv run python parse_human_llm.py --output human_llm_parsed.jsonl

# Generate explorer data
uv run python generate_explorer_data.py human_llm_parsed.jsonl --output explorer_human_llm.json
uv run python generate_explorer_data.py gutenberg_parsed.jsonl --output explorer_data.json

# Run tests
uv run python -m pytest test_mining.py -v
```

## Recent Changes (Mar 2025)
- Added SVG tree diagrams to both explorer pages (renders when a pattern is selected)
- Added example search box in explorer examples panel (filters as you type)
- Increased display limits: 200 patterns single view, 100 both view
- Increased max_examples from 10 to 100, --top from 50 to 200
- Increased max_depth from 4 to 5 to capture deeper patterns (e.g. "not just X, it was Y")
- Rewrote explainer.html to use "It was not just a challenge, it was a revelation" as primary example
- Tree renderer code (parseBracket, layoutTree, renderSVG etc.) shared across explainer and both explorers

## Recent Changes (Feb 2025)
- Added unit tests for mining and analysis modules (test_mining.py)

## Recent Changes (Jan 2025)
- Added human vs LLM explorer for comparing syntactic patterns
- Fixed `get_terminals_at_depth` in mining.py to return ALL terminals under collapsed subtrees (was only returning first terminal, causing incorrect highlighting)
- Hosted explorers on GitHub Pages

## Work in Progress (uncommitted)
- `build_classification_data.py` - New script for building classification datasets
- `classification_data/` - Output directory for classification data
- JSON data files need regenerating with max_depth=5

## Environment Notes
- Uses `en_core_web_sm` spaCy model (not `en_core_web_md`)
- benepar 0.2.0 requires `transformers<5` (T5Tokenizer API breaking change in v5)
- Opening HTML files via `file://` breaks `fetch()` for JSON data; use `python3 -m http.server` instead
- GitHub Pages serves the explorers publicly (no CORS issue there)

## Potential Future Work
- Add more LLM models for comparison (currently just one source)
- Statistical significance testing for PMI differences
- Clustering of similar patterns
- Time-series analysis (how patterns change across an author's works)
- Export functionality in explorer (CSV of patterns)
- Side-by-side comparison view in explorer (two authors at once)
- Sentence → motifs search (would require porting subtree extraction to JS or adding a backend)
