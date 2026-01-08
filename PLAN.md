# Syntactic Motifs Prototype Plan

## Goal
Create a Python prototype for extracting and mining syntactic motifs (subtree patterns) from constituency parse trees, with the aim of finding patterns that differentiate human vs LLM-generated text.

## Approach
**Exploratory mining** - Rather than defining motifs upfront, mine all subtree patterns from a corpus, count frequencies, and identify discriminative patterns. Later, map interesting patterns to literary-stylistic categories.

## Prototype Components

### 1. Core Pipeline (`parser.py`)
- **Parsing**: spaCy + benepar for constituency parsing
- **Tree → Graph**: Convert NLTK Tree → NetworkX DiGraph with node labels (POS/phrase tags)

### 2. Subtree Mining (`mining.py`)
Extract subtree patterns from parse trees:
- **Induced subtrees**: Connected subgraphs where all children of included nodes are included
- **Bounded depth**: Limit subtree depth (2-4) to control combinatorial explosion
- **Canonicalization**: Convert subtrees to canonical string form for counting (e.g., `(S (NP) (VP))`)
- Use recursive enumeration or adapt TreeMiner-style approach

Key functions:
- `extract_subtrees(tree, max_depth=3)` → list of subtree patterns
- `canonicalize(subtree)` → string representation
- `count_patterns(corpus)` → {pattern: count} dict

### 3. Analysis (`analysis.py`)
- `corpus_profile(texts)` → {pattern: frequency} normalized by sentence count
- `discriminative_patterns(profile1, profile2)` → patterns that differ most between corpora
- `filter_by_support(patterns, min_support)` → remove rare patterns

### 4. Demo (`demo.py`)
- Parse sample human + LLM texts
- Mine subtree patterns from each
- Report most frequent patterns
- Report most discriminative patterns between the two sources
- Output as JSON dict

## Subtree Representation
Parse trees like:
```
(S (NP (DT the) (NN cat)) (VP (VBD sat)))
```
Yield abstract patterns (ignoring terminals):
```
(S (NP (DT) (NN)) (VP (VBD)))  # depth 3
(S (NP) (VP))                   # depth 2
(NP (DT) (NN))                  # depth 2
```

## Dependencies
```
spacy>=3.0
benepar
networkx
nltk
```

## Files to Create
1. `parser.py` - Constituency parsing, tree→graph conversion
2. `mining.py` - Subtree extraction and counting
3. `analysis.py` - Profile computation, pattern comparison
4. `demo.py` - Example: compare human vs LLM text
5. `requirements.txt`

## Implementation Notes
- **Induced subtrees only** (all children of a node are either included or excluded)
- Start with depth 2-3 subtrees (parent + children/grandchildren labels)
- Canonical form: parenthesized string like `(S (NP) (VP))`
- For efficiency, use recursive generator rather than building all NetworkX subgraphs
- Include placeholder sample texts for demo (human-like vs LLM-like examples)

## Decisions Made
- Constituency parsing only (benepar) - phrase structure trees better for hierarchical motifs
- Exploratory mining approach - find patterns first, interpret later
- Induced subtrees only - simpler to implement and interpret
- JSON/dict output format
- Placeholder sample texts included for testing

---
*To continue: run Claude Code in `/home/paul/cc1` and say "implement the plan in PLAN.md"*
