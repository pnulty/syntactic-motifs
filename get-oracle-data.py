"""
Download and prepare text corpora for syntactic motif analysis.
"""

import json
from datasets import load_dataset


def get_christie_texts(n_samples: int = 500) -> list[str]:
    """Load Agatha Christie texts from HuggingFace."""
    ds = load_dataset("realdanielbyrne/AgathaChristieText")
    texts = [row['text'] for row in ds['train']]
    # Take a sample, preferring medium-length texts
    texts = [t for t in texts if 500 < len(t) < 5000]
    return texts[:n_samples]


def get_lovecraft_texts(n_samples: int = 500) -> list[str]:
    """Load H.P. Lovecraft texts from HuggingFace."""
    ds = load_dataset("TristanBehrens/lovecraftcorpus")
    texts = [row['text'] for row in ds['train']]
    # Take a sample, preferring medium-length texts
    texts = [t for t in texts if 500 < len(t) < 5000]
    return texts[:n_samples]


def save_corpus(texts: list[str], filename: str):
    """Save corpus to JSON file."""
    with open(filename, 'w') as f:
        json.dump(texts, f, indent=2)
    print(f"Saved {len(texts)} texts to {filename}")


def print_stats(texts: list[str], name: str):
    """Print corpus statistics."""
    print(f"\n{name} corpus stats:")
    print(f"  Texts: {len(texts)}")
    print(f"  Total chars: {sum(len(t) for t in texts):,}")
    print(f"  Avg chars/text: {sum(len(t) for t in texts) // len(texts):,}")


if __name__ == "__main__":
    print("Downloading Agatha Christie corpus...")
    christie = get_christie_texts(500)
    save_corpus(christie, "corpus_christie.json")
    print_stats(christie, "Christie")

    print("\nDownloading H.P. Lovecraft corpus...")
    lovecraft = get_lovecraft_texts(500)
    save_corpus(lovecraft, "corpus_lovecraft.json")
    print_stats(lovecraft, "Lovecraft")
