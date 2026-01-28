"""
Demo: Compare syntactic motifs between different authors and text types.

This script demonstrates the syntactic motifs pipeline by:
1. Parsing texts from different sources (Gutenberg authors, LLM-generated)
2. Mining subtree patterns from each
3. Reporting most frequent patterns and discriminative patterns
"""

import json
from analysis import compare_corpora, top_patterns, corpus_profile

# HP Lovecraft - The Call of Cthulhu (Project Gutenberg)
LOVECRAFT_TEXTS = [
    """I was expected to go over his papers with some thoroughness; and for
that purpose moved his entire set of files and boxes to my quarters in
Boston. Much of the material which I correlated will be later published
by the American Archeological Society, but there was one box which I
found exceedingly puzzling, and which I felt much averse from showing
to other eyes.""",

    """It had been locked, and I did not find the key till
it occurred to me to examine the personal ring which the professor
carried always in his pocket. Then, indeed, I succeeded in opening it,
but when I did so seemed only to be confronted by a greater and more
closely locked barrier.""",

    """For what could be the meaning of the queer clay
bas-relief and the disjointed jottings, ramblings, and cuttings which I
found? Had my uncle, in his latter years, become credulous of the most
superficial impostures?""",

    """The bas-relief was a rough rectangle less than an inch thick and about
five by six inches in area; obviously of modern origin. Its designs,
however, were far from modern in atmosphere and suggestion; for,
although the vagaries of cubism and futurism are many and wild, they do
not often reproduce that cryptic regularity which lurks in prehistoric
writing.""",

    """Above these apparent hieroglyphics was a figure of evidently pictorial
intent, though its impressionistic execution forbade a very clear
idea of its nature. It seemed to be a sort of monster, or symbol
representing a monster, of a form which only a diseased fancy could
conceive.""",

    """If I say that my somewhat extravagant imagination yielded
simultaneous pictures of an octopus, a dragon, and a human caricature,
I shall not be unfaithful to the spirit of the thing. A pulpy,
tentacled head surmounted a grotesque and scaly body with rudimentary
wings; but it was the general outline of the whole which made it most
shockingly frightful.""",

    """The first half of the principal manuscript told a very peculiar
tale. It appears that in March of 1925, a thin, dark young man of
neurotic and excited aspect had called upon the professor bearing
the singular clay bas-relief, which was then exceedingly damp and
fresh.""",

    """Wilcox was a precocious youth of known genius
but great eccentricity, and had from childhood excited attention
through the strange stories and odd dreams he was in the habit of
relating. He called himself psychically hypersensitive, but the staid
folk of the ancient commercial city dismissed him as merely queer.""",

    """On the occasion of the visit, ran the professor's manuscript, the
sculptor abruptly asked for the benefit of his host's archeological
knowledge in identifying the hieroglyphics on the bas-relief. He
spoke in a dreamy, stilted manner which suggested pose and alienated
sympathy.""",

    """There had been a slight earthquake tremor the night before, the most considerable
felt in New England for some years; and Wilcox's imagination had been
keenly affected. Upon retiring, he had had an unprecedented dream of
great Cyclopean cities of Titan blocks and sky-flung monoliths, all
dripping with green ooze and sinister with latent horror.""",
]

# Agatha Christie - The Mysterious Affair at Styles (Project Gutenberg)
CHRISTIE_TEXTS = [
    """Thus it came about that, three days later, I descended from the train
at an absurd little station, with no apparent reason for existence,
perched up in the midst of green fields and country lanes. John
Cavendish was waiting on the platform, and piloted me out to the car.""",

    """The village was situated about two miles from the little station,
and Styles Court lay a mile the other side of it. It was a still, warm
day in early July. As one looked out over the flat Essex country, lying
so green and peaceful under the afternoon sun, it seemed almost
impossible to believe that a great war was running its appointed course.""",

    """I felt I had suddenly strayed into another world. As we turned in at
the lodge gates, John said that he was afraid I would find it very quiet
down here. I told him that was just what I wanted.""",

    """It is pleasant enough if you want to lead the idle life. I drill
with the volunteers twice a week, and lend a hand at the farms. My wife
works regularly on the land. She is up at five every morning to milk,
and keeps at it steadily until lunchtime.""",

    """Miss Howard shook hands with a hearty, almost painful, grip. I had an
impression of very blue eyes in a sunburnt face. She was a pleasant
looking woman of about forty, with a deep voice, almost manly in its
stentorian tones, and had a large sensible square body.""",

    """Weeds grow like house afire. She said she could not keep even with them.
She would press me in. I told her I should be only too delighted to make
myself useful. She warned me not to say it, as I would regret it later.""",

    """She led the way round the house to where tea was spread under the shade
of a large sycamore. A figure rose from one of the basket chairs, and
came a few steps to meet us.""",

    """I shall never forget my first sight of Mary Cavendish. Her tall,
slender form was outlined against the bright light. The vivid sense of
slumbering fire seemed to find expression only in those wonderful
tawny eyes of hers, remarkable eyes, different from any others.""",

    """The intense power of stillness she possessed nevertheless conveyed the
impression of a wild untamed spirit in an exquisitely civilised body.
All these things are burnt into my memory. I shall never forget them.""",

    """She greeted me with a few words of pleasant welcome in a low clear
voice, and I sank into a basket chair feeling distinctly glad that I
had accepted the invitation. She gave me some tea, and her few quiet
remarks heightened my first impression of her as a fascinating woman.""",
]

# Sample LLM-like texts (formal, verbose, hedged language)
LLM_TEXTS = [
    "The old house exhibited signs of deterioration that were consistent with prolonged neglect.",
    "It is important to note that the individual proceeded with the action despite receiving advice to the contrary.",
    "Before discussing the implications of the budget reductions, it would be beneficial to consume a caffeinated beverage.",
    "The subject initiated a rapid movement in response to the perceived threat of pursuit.",
    "The workspace demonstrated a significant level of disorganization, which could potentially impact productivity.",
    "The individual displayed a facial expression that could be interpreted as insincere before departing.",
    "The feline positioned itself upon the floor covering, which represents a common behavioral pattern.",
    "The meteorological conditions have been characterized by precipitation for a period of three consecutive days.",
    "The individual was located at the drinking establishment, where he was consuming an alcoholic beverage in isolation.",
    "It is understandable that feelings of apprehension are being experienced by multiple parties in this situation.",
]


def format_discriminative(patterns: list[tuple[str, float, float, float]], label1: str, label2: str) -> list[dict]:
    """Format discriminative patterns for JSON output."""
    return [
        {
            "pattern": p[0],
            f"freq_{label1}": round(p[1], 4),
            f"freq_{label2}": round(p[2], 4),
            "difference": round(p[3], 4),
            "favors": label1 if p[3] > 0 else label2
        }
        for p in patterns
    ]


def format_top_patterns(patterns: list[tuple[str, float]]) -> list[dict]:
    """Format top patterns for JSON output."""
    return [
        {"pattern": p[0], "frequency": round(p[1], 4)}
        for p in patterns
    ]


def print_comparison(results: dict, label1: str, label2: str, top_k: int = 15):
    """Print comparison results in a readable format."""
    print(f"\nTop {top_k} patterns in {label1.upper()} texts:")
    print("-" * 50)
    for pattern, freq in results["top_patterns1"][:top_k]:
        print(f"  {freq:.3f}  {pattern}")

    print(f"\nTop {top_k} patterns in {label2.upper()} texts:")
    print("-" * 50)
    for pattern, freq in results["top_patterns2"][:top_k]:
        print(f"  {freq:.3f}  {pattern}")

    print(f"\nMost discriminative patterns ({label1} vs {label2}):")
    print("-" * 50)
    for pattern, freq1, freq2, diff in results["discriminative"][:top_k]:
        direction = label1.upper() if diff > 0 else label2.upper()
        print(f"  {abs(diff):+.3f} ({direction:10})  {pattern}")
        print(f"          {label1}={freq1:.3f}, {label2}={freq2:.3f}")


def main():
    print("Syntactic Motifs Demo")
    print("=" * 60)
    print(f"Lovecraft texts: {len(LOVECRAFT_TEXTS)} passages")
    print(f"Christie texts: {len(CHRISTIE_TEXTS)} passages")
    print(f"LLM texts: {len(LLM_TEXTS)} samples")
    print()

    # Compare Lovecraft vs Christie
    print("=" * 60)
    print("COMPARISON 1: Lovecraft vs Christie")
    print("=" * 60)
    print("Parsing and extracting patterns (max_depth=4, min_nodes>=3)...")

    results1 = compare_corpora(
        LOVECRAFT_TEXTS,
        CHRISTIE_TEXTS,
        max_depth=4,
        min_depth=2,
        min_terminals=3,
        top_k=15
    )
    print_comparison(results1, "lovecraft", "christie")

    # Compare Christie vs LLM
    print("\n" + "=" * 60)
    print("COMPARISON 2: Christie (human) vs LLM")
    print("=" * 60)

    results2 = compare_corpora(
        CHRISTIE_TEXTS,
        LLM_TEXTS,
        max_depth=4,
        min_depth=2,
        min_terminals=3,
        top_k=15
    )
    print_comparison(results2, "christie", "llm")

    # JSON output
    output = {
        "comparisons": [
            {
                "name": "Lovecraft vs Christie",
                "corpus1": {"name": "lovecraft", "sample_count": len(LOVECRAFT_TEXTS)},
                "corpus2": {"name": "christie", "sample_count": len(CHRISTIE_TEXTS)},
                "top_patterns_1": format_top_patterns(results1["top_patterns1"]),
                "top_patterns_2": format_top_patterns(results1["top_patterns2"]),
                "discriminative": format_discriminative(results1["discriminative"], "lovecraft", "christie"),
            },
            {
                "name": "Christie vs LLM",
                "corpus1": {"name": "christie", "sample_count": len(CHRISTIE_TEXTS)},
                "corpus2": {"name": "llm", "sample_count": len(LLM_TEXTS)},
                "top_patterns_1": format_top_patterns(results2["top_patterns1"]),
                "top_patterns_2": format_top_patterns(results2["top_patterns2"]),
                "discriminative": format_discriminative(results2["discriminative"], "christie", "llm"),
            },
        ],
        "settings": {
            "max_depth": 4,
            "min_depth": 2,
            "min_terminals": 3,
        }
    }

    print("\n" + "=" * 60)
    print("JSON Output:")
    print(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    main()
