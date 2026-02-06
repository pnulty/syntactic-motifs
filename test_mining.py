"""
Tests for the syntactic motifs extraction and analysis modules.
"""

import pytest
from collections import Counter
from nltk import Tree

from mining import (
    canonicalize,
    count_terminal_nodes,
    extract_induced_subtrees,
    extract_patterns,
    get_terminals,
    get_terminals_at_depth,
    count_patterns,
    extract_patterns_with_examples,
)
from analyze_corpus import (
    extract_labels,
    highlight_in_sentence,
    compute_pmi_vs_baseline,
)


# --- Test fixtures ---

@pytest.fixture
def simple_np_tree():
    """A simple NP tree: (NP (DT the) (NN cat))"""
    return Tree.fromstring("(NP (DT the) (NN cat))")


@pytest.fixture
def simple_sentence_tree():
    """A simple sentence: (S (NP (DT The) (NN cat)) (VP (VBZ sits)))"""
    return Tree.fromstring("(S (NP (DT The) (NN cat)) (VP (VBZ sits)))")


@pytest.fixture
def complex_sentence_tree():
    """A more complex sentence with PP:
    (S (NP (DT The) (JJ black) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP (DT the) (NN mat)))))
    """
    return Tree.fromstring(
        "(S (NP (DT The) (JJ black) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP (DT the) (NN mat)))))"
    )


# --- canonicalize tests ---

class TestCanonicalize:
    def test_leaf_node(self):
        """Leaf nodes (no children) are represented as (LABEL)."""
        tree = Tree("NP", [])
        assert canonicalize(tree) == "(NP)"

    def test_simple_np(self, simple_np_tree):
        """(NP (DT the) (NN cat)) with terminals is canonicalized."""
        # The children are also Trees with terminal children
        result = canonicalize(simple_np_tree)
        assert result == "(NP (DT the) (NN cat))"

    def test_abstracted_pattern(self):
        """A pattern without terminals (abstracted)."""
        tree = Tree("NP", [Tree("DT", []), Tree("NN", [])])
        assert canonicalize(tree) == "(NP (DT) (NN))"

    def test_nested_structure(self):
        """Test nested structure canonicalization."""
        tree = Tree("S", [
            Tree("NP", [Tree("DT", []), Tree("NN", [])]),
            Tree("VP", [Tree("VB", [])])
        ])
        assert canonicalize(tree) == "(S (NP (DT) (NN)) (VP (VB)))"

    def test_non_tree_input(self):
        """Non-tree input returns string representation."""
        assert canonicalize("word") == "word"


# --- count_terminal_nodes tests ---

class TestCountTerminalNodes:
    def test_simple_pattern(self):
        """(NP (DT) (NN)) has 2 terminals."""
        assert count_terminal_nodes("(NP (DT) (NN))") == 2

    def test_nested_pattern(self):
        """(PP (IN) (NP (DT) (NN))) has 3 terminals."""
        assert count_terminal_nodes("(PP (IN) (NP (DT) (NN)))") == 3

    def test_single_terminal(self):
        """(NP (NN)) has 1 terminal."""
        assert count_terminal_nodes("(NP (NN))") == 1

    def test_deep_nesting(self):
        """Complex pattern with multiple levels."""
        pattern = "(S (NP (DT) (NN)) (VP (VBZ) (PP (IN) (NP))))"
        # Terminals: DT, NN, VBZ, IN, NP (as a leaf)
        assert count_terminal_nodes(pattern) == 5

    def test_pattern_with_numbers(self):
        """POS tags can have numbers like CD, NN1."""
        assert count_terminal_nodes("(NP (CD) (NN))") == 2

    def test_pattern_with_dollar(self):
        """POS tags can have $ like $."""
        assert count_terminal_nodes("(NP ($) (CD))") == 2


# --- extract_induced_subtrees tests ---

class TestExtractInducedSubtrees:
    def test_depth_2_extraction(self, simple_np_tree):
        """Depth 2 extracts node + immediate children."""
        subtrees = list(extract_induced_subtrees(simple_np_tree, max_depth=2, min_depth=2))
        patterns = [canonicalize(st) for st in subtrees]
        assert "(NP (DT) (NN))" in patterns

    def test_depth_1_gives_leaf_patterns(self, simple_np_tree):
        """Depth 1 would give just the node label as a leaf."""
        subtrees = list(extract_induced_subtrees(simple_np_tree, max_depth=1, min_depth=1))
        patterns = [canonicalize(st) for st in subtrees]
        # At depth 1, we get: (NP), (DT), (NN)
        assert "(NP)" in patterns
        assert "(DT)" in patterns
        assert "(NN)" in patterns

    def test_min_depth_filtering(self, simple_sentence_tree):
        """min_depth=2 means we get patterns that include at least one level of children."""
        subtrees = list(extract_induced_subtrees(simple_sentence_tree, max_depth=2, min_depth=2))
        patterns = [canonicalize(st) for st in subtrees]
        # We should get patterns with children like (NP (DT) (NN))
        patterns_with_children = [p for p in patterns if p.count("(") > 1]
        assert len(patterns_with_children) > 0
        assert "(NP (DT) (NN))" in patterns
        # Note: Single-node patterns like (DT) still appear because POS nodes
        # have only terminal children, making them leaves at depth 2

    def test_max_depth_limiting(self, complex_sentence_tree):
        """max_depth limits how deep the extraction goes."""
        # At depth 2, we shouldn't see grandchildren
        subtrees = list(extract_induced_subtrees(complex_sentence_tree, max_depth=2, min_depth=2))
        for st in subtrees:
            # Check no subtree has depth > 2
            assert st.height() <= 2

    def test_non_tree_returns_empty(self):
        """Non-tree input yields nothing."""
        subtrees = list(extract_induced_subtrees("not a tree", max_depth=3, min_depth=2))
        assert subtrees == []


# --- extract_patterns tests ---

class TestExtractPatterns:
    def test_min_terminals_filtering(self, simple_sentence_tree):
        """min_terminals=2 filters out single-terminal patterns."""
        patterns = list(extract_patterns(simple_sentence_tree, max_depth=3, min_depth=2, min_terminals=2))
        for p in patterns:
            assert count_terminal_nodes(p) >= 2

    def test_returns_canonical_strings(self, simple_np_tree):
        """Patterns are returned as canonical strings."""
        patterns = list(extract_patterns(simple_np_tree, max_depth=2, min_depth=2, min_terminals=2))
        assert all(isinstance(p, str) for p in patterns)
        assert all(p.startswith("(") for p in patterns)

    def test_depth_range(self, simple_sentence_tree):
        """Patterns from min_depth to max_depth are extracted."""
        patterns_d2 = set(extract_patterns(simple_sentence_tree, max_depth=2, min_depth=2, min_terminals=1))
        patterns_d3 = set(extract_patterns(simple_sentence_tree, max_depth=3, min_depth=2, min_terminals=1))
        # Depth 3 should include all depth 2 patterns plus more
        assert patterns_d2.issubset(patterns_d3)


# --- get_terminals tests ---

class TestGetTerminals:
    def test_simple_tree(self, simple_np_tree):
        """Get terminals from simple NP."""
        terminals = get_terminals(simple_np_tree)
        assert terminals == ["the", "cat"]

    def test_complex_tree(self, complex_sentence_tree):
        """Get terminals from complex sentence."""
        terminals = get_terminals(complex_sentence_tree)
        assert terminals == ["The", "black", "cat", "sits", "on", "the", "mat"]

    def test_single_terminal(self):
        """A terminal string returns itself in a list."""
        assert get_terminals("word") == ["word"]


# --- get_terminals_at_depth tests ---

class TestGetTerminalsAtDepth:
    def test_depth_1_gets_all(self, simple_np_tree):
        """At depth 1, the whole subtree is collapsed, so get all terminals."""
        terminals = get_terminals_at_depth(simple_np_tree, 1)
        assert terminals == ["the", "cat"]

    def test_depth_2_gets_children_terminals(self, simple_sentence_tree):
        """At depth 2, get terminals from immediate children."""
        # (S (NP (DT The) (NN cat)) (VP (VBZ sits)))
        # At depth 2 from S: NP and VP are leaves in pattern, so get all their terminals
        terminals = get_terminals_at_depth(simple_sentence_tree, 2)
        assert terminals == ["The", "cat", "sits"]

    def test_non_tree_returns_string(self):
        """A non-tree (terminal word) returns itself."""
        assert get_terminals_at_depth("word", 1) == ["word"]

    def test_depth_3_goes_deeper(self, complex_sentence_tree):
        """At depth 3, includes grandchildren."""
        # (S (NP (DT The) (JJ black) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP ...))))
        # At depth 3 from S:
        # - NP children (DT, JJ, NN) are at depth 2 from children, so their terminals are included
        # - VP children (VBZ, PP) similarly
        terminals = get_terminals_at_depth(complex_sentence_tree, 3)
        # At depth 3, PP is a leaf in the pattern so we get all its terminals
        assert "The" in terminals
        assert "sits" in terminals


# --- count_patterns tests ---

class TestCountPatterns:
    def test_counts_across_trees(self, simple_np_tree, simple_sentence_tree):
        """Count patterns across multiple trees."""
        # The NP pattern should appear in both
        counts = count_patterns([simple_np_tree, simple_sentence_tree], max_depth=2, min_depth=2, min_terminals=2)
        assert "(NP (DT) (NN))" in counts
        # It appears once in each tree
        assert counts["(NP (DT) (NN))"] == 2

    def test_empty_list(self):
        """Empty tree list returns empty counter."""
        counts = count_patterns([], max_depth=3, min_depth=2, min_terminals=2)
        assert counts == Counter()

    def test_respects_min_terminals(self, simple_sentence_tree):
        """min_terminals filter is applied."""
        counts_2 = count_patterns([simple_sentence_tree], max_depth=3, min_depth=2, min_terminals=2)
        counts_1 = count_patterns([simple_sentence_tree], max_depth=3, min_depth=2, min_terminals=1)
        # With lower threshold, we should have more patterns
        assert len(counts_1) >= len(counts_2)


# --- extract_patterns_with_examples tests ---

class TestExtractPatternsWithExamples:
    def test_returns_pattern_words_sentence(self, simple_sentence_tree):
        """Returns tuples of (pattern, words, sentence)."""
        sentence = "The cat sits"
        results = list(extract_patterns_with_examples(
            simple_sentence_tree, sentence, max_depth=2, min_depth=2, min_terminals=2
        ))
        assert len(results) > 0
        for pattern, words, sent in results:
            assert isinstance(pattern, str)
            assert isinstance(words, str)
            assert sent == sentence

    def test_words_match_terminals(self, simple_np_tree):
        """The words correspond to the terminals covered by the pattern."""
        sentence = "the cat"
        results = list(extract_patterns_with_examples(
            simple_np_tree, sentence, max_depth=2, min_depth=2, min_terminals=2
        ))
        # (NP (DT) (NN)) should have words "the cat"
        np_results = [r for r in results if r[0] == "(NP (DT) (NN))"]
        assert len(np_results) == 1
        assert np_results[0][1] == "the cat"


# --- extract_labels tests ---

class TestExtractLabels:
    def test_simple_pattern(self):
        """Extract labels from simple pattern."""
        labels = extract_labels("(NP (DT) (NN))")
        assert labels == ["NP", "DT", "NN"]

    def test_nested_pattern(self):
        """Extract labels from nested pattern."""
        labels = extract_labels("(S (NP (DT) (NN)) (VP (VBZ)))")
        assert labels == ["S", "NP", "DT", "NN", "VP", "VBZ"]

    def test_pattern_with_hyphen(self):
        """Labels can contain hyphens like ADVP-TMP."""
        labels = extract_labels("(S (ADVP-TMP) (NP))")
        assert "ADVP-TMP" in labels


# --- highlight_in_sentence tests ---

class TestHighlightInSentence:
    def test_simple_highlight(self):
        """Highlights matching words with **bold**."""
        result = highlight_in_sentence("The cat sat on the mat", "cat sat")
        assert "**cat sat**" in result

    def test_case_insensitive(self):
        """Matching is case insensitive."""
        result = highlight_in_sentence("The Cat sat on the mat", "cat sat")
        assert "**" in result

    def test_empty_words(self):
        """Empty words returns original sentence."""
        result = highlight_in_sentence("The cat sat", "")
        assert result == "The cat sat"

    def test_words_with_punctuation(self):
        """Handles punctuation between words."""
        result = highlight_in_sentence("The cat, it seems, sat", "cat it seems")
        # Should match even with punctuation
        assert "**" in result

    def test_only_highlights_once(self):
        """Only the first occurrence is highlighted."""
        result = highlight_in_sentence("cat cat cat", "cat")
        assert result.count("**") == 2  # One opening, one closing


# --- compute_pmi_vs_baseline tests ---

class TestComputePmiVsBaseline:
    def test_basic_pmi(self):
        """Test basic PMI computation."""
        author_counts = Counter({"(NP (DT) (NN))": 100})
        baseline_counts = Counter({"(NP (DT) (NN))": 50})

        pmi = compute_pmi_vs_baseline(
            author_counts, author_total=100,
            baseline_counts=baseline_counts, baseline_total=100,
            min_count=5
        )

        # Author freq = 100/100 = 1.0
        # Baseline freq = 50/100 = 0.5
        # PMI = log2(1.0 / 0.5) = log2(2) = 1.0
        assert "(NP (DT) (NN))" in pmi
        assert abs(pmi["(NP (DT) (NN))"] - 1.0) < 0.001

    def test_underused_pattern(self):
        """Underused patterns have negative PMI."""
        author_counts = Counter({"(NP (DT) (NN))": 25})
        baseline_counts = Counter({"(NP (DT) (NN))": 100})

        pmi = compute_pmi_vs_baseline(
            author_counts, author_total=100,
            baseline_counts=baseline_counts, baseline_total=100,
            min_count=5
        )

        # Author freq = 25/100 = 0.25
        # Baseline freq = 100/100 = 1.0
        # PMI = log2(0.25 / 1.0) = log2(0.25) = -2.0
        assert pmi["(NP (DT) (NN))"] < 0

    def test_min_count_filtering(self):
        """Patterns below min_count are excluded."""
        author_counts = Counter({"(NP (DT) (NN))": 3})
        baseline_counts = Counter({"(NP (DT) (NN))": 100})

        pmi = compute_pmi_vs_baseline(
            author_counts, author_total=100,
            baseline_counts=baseline_counts, baseline_total=100,
            min_count=5
        )

        assert "(NP (DT) (NN))" not in pmi

    def test_pattern_not_in_baseline(self):
        """Patterns not in baseline get smoothed value."""
        author_counts = Counter({"(UNIQUE (PATTERN))": 10})
        baseline_counts = Counter()

        pmi = compute_pmi_vs_baseline(
            author_counts, author_total=100,
            baseline_counts=baseline_counts, baseline_total=100,
            min_count=5
        )

        # Should have a high positive PMI (author-distinctive pattern)
        assert "(UNIQUE (PATTERN))" in pmi
        assert pmi["(UNIQUE (PATTERN))"] > 0

    def test_equal_frequencies(self):
        """Equal frequencies give PMI of 0."""
        author_counts = Counter({"(NP (DT) (NN))": 50})
        baseline_counts = Counter({"(NP (DT) (NN))": 50})

        pmi = compute_pmi_vs_baseline(
            author_counts, author_total=100,
            baseline_counts=baseline_counts, baseline_total=100,
            min_count=5
        )

        assert abs(pmi["(NP (DT) (NN))"]) < 0.001


# --- Integration-style tests ---

class TestIntegration:
    def test_full_pipeline_with_trees(self, complex_sentence_tree):
        """Test the full extraction pipeline."""
        sentence = "The black cat sits on the mat"

        # Extract patterns with examples
        results = list(extract_patterns_with_examples(
            complex_sentence_tree, sentence,
            max_depth=3, min_depth=2, min_terminals=2
        ))

        # Should have multiple patterns
        assert len(results) > 0

        # Each result should have valid structure
        for pattern, words, sent in results:
            assert pattern.startswith("(")
            assert count_terminal_nodes(pattern) >= 2
            assert len(words) > 0
            assert sent == sentence

    def test_pattern_counting_consistency(self):
        """Pattern counts are consistent across methods."""
        tree = Tree.fromstring("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))")

        # Extract patterns one way
        patterns = list(extract_patterns(tree, max_depth=3, min_depth=2, min_terminals=2))

        # Count patterns another way
        counts = count_patterns([tree], max_depth=3, min_depth=2, min_terminals=2)

        # Should match
        assert len(patterns) == sum(counts.values())
