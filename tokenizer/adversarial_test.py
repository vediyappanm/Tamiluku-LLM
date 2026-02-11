"""
adversarial_test.py — AMB Adversarial Sanity Check
===================================================
Tests the current implementation against the user's specific edge cases.
"""

import sys
from pathlib import Path

# Fix path to include current dir
sys.path.append(str(Path(__file__).parent))

from tamil_unicode import TamilDeepNormalizer
from akshara import AksharaSegmenter
from morpheme import MorphemeSegmenter
from tokenizers import Tokenizer

def _ensure_utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_print(line: str):
    try:
        print(line)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()

def run_adversarial_test():
    _ensure_utf8_stdout()
    norm = TamilDeepNormalizer()
    ak_seg = AksharaSegmenter()
    m_seg = MorphemeSegmenter()
    
    # Load the sample AMB tokenizer if it exists
    tokenizer_path = Path("models/amb_tokenizer/tokenizer.json")
    tokenizer = None
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    edge_cases = [
        "சென்னையிலிருந்துதான்",  # Triple suffix stacking
        "பார்த்திருக்கமாட்டார்கள்",  # Complex verb inflection
        "COVID-19க்கான",          # Code-mixing + case marker
        "😊நன்றி❤️",              # Emoji handling
    ]

    safe_print(f"{'Text':<25} | {'Morpheme Split':<35}")
    safe_print("-" * 65)

    for text in edge_cases:
        # Pre-process
        cleaned = norm.normalize(text)
        # Morpheme split
        words = cleaned.split()
        morphemes = [m_seg.segment_word(w) for w in words]
        morpheme_str = " | ".join(morphemes)

        safe_print(f"{text:<25} | {morpheme_str:<35}")

        if tokenizer:
            encoded = tokenizer.encode(cleaned)
            decoded_pieces = [tokenizer.decode([i]) for i in encoded.ids]
            safe_print(f"  > AMB Tokens: {decoded_pieces}")

if __name__ == "__main__":
    run_adversarial_test()
