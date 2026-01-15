"""
Constituency parsing and tree-to-graph conversion.

Uses spaCy + benepar for parsing and converts NLTK Trees to NetworkX DiGraphs.
"""

import spacy
import benepar
import networkx as nx
from nltk import Tree
from typing import Iterator


# Global NLP pipeline (lazy loaded)
_nlp = None


def get_nlp():
    """Get or create the spaCy + benepar pipeline."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    return _nlp


def parse_text(text: str) -> Iterator[Tree]:
    """
    Parse text into constituency trees.

    Args:
        text: Input text (can contain multiple sentences)

    Yields:
        NLTK Tree objects, one per sentence
    """
    nlp = get_nlp()
    # Normalize whitespace: benepar's retokenization fails when newlines
    # appear as separate tokens (spaCy and T5 tokenizer handle them differently)
    text = ' '.join(text.split())
    doc = nlp(text)
    for sent in doc.sents:
        yield Tree.fromstring(sent._.parse_string)


def parse_sentence(text: str) -> Tree:
    """
    Parse a single sentence into a constituency tree.

    Args:
        text: Input sentence

    Returns:
        NLTK Tree object
    """
    return next(parse_text(text))


def tree_to_digraph(tree: Tree) -> nx.DiGraph:
    """
    Convert an NLTK Tree to a NetworkX DiGraph.

    Each node has attributes:
        - label: The POS/phrase tag (e.g., 'NP', 'VP', 'DT')
        - is_terminal: True for leaf nodes (words), False for non-terminals
        - text: For terminals only, the actual word

    Args:
        tree: NLTK Tree object

    Returns:
        NetworkX DiGraph with nodes labeled by their syntactic tags
    """
    graph = nx.DiGraph()
    _add_subtree_to_graph(tree, graph, parent=None, node_counter=[0])
    return graph


def _add_subtree_to_graph(tree, graph: nx.DiGraph, parent: int | None, node_counter: list) -> int:
    """
    Recursively add tree nodes to the graph.

    Args:
        tree: NLTK Tree or string (terminal)
        graph: NetworkX graph to add to
        parent: Parent node ID or None for root
        node_counter: Mutable counter for generating unique node IDs

    Returns:
        The ID of the current node
    """
    current_id = node_counter[0]
    node_counter[0] += 1

    if isinstance(tree, Tree):
        # Non-terminal node
        graph.add_node(current_id, label=tree.label(), is_terminal=False)
        if parent is not None:
            graph.add_edge(parent, current_id)

        for child in tree:
            _add_subtree_to_graph(child, graph, current_id, node_counter)
    else:
        # Terminal node (word)
        graph.add_node(current_id, label=str(tree), is_terminal=True, text=str(tree))
        if parent is not None:
            graph.add_edge(parent, current_id)

    return current_id


def get_nonterminal_subgraph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Return a subgraph containing only non-terminal nodes.

    Useful for pattern mining where we only care about structural patterns,
    not the actual words.

    Args:
        graph: Full parse tree graph

    Returns:
        Subgraph with only non-terminal (phrase/POS) nodes
    """
    nonterminal_nodes = [n for n, d in graph.nodes(data=True) if not d.get('is_terminal', False)]
    return graph.subgraph(nonterminal_nodes).copy()
