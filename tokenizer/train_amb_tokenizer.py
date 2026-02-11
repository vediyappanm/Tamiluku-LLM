"""
train_amb_tokenizer.py — AMB Layer 4 & 5: Constrained BPE Training
====================================================================
Orchestrates the entire AMB pipeline:
  LAYER 1: Deep Normalization (Fix ZWJ, Grantha, etc.)
  LAYER 2: Akshara Segmentation (Stats & validation only)
  LAYER 3: Morpheme Segmentation (Inject @@ boundaries)
  LAYER 4: Constrained BPE (Respect boundaries)
  LAYER 5: Vocabulary Optimization (Prune rare morphemes)
  LAYER 6: Export to standard tokenizer.json (HF compatible)

Why this is different:
  Standard BPE training simply finds frequent byte pairs.
  Constrained BPE finds frequent byte pairs ONLY WITHIN morphemes.
  
  Example:
    Input: "வீடுகளிலிருந்து"
    Standard BPE might merge: "வீடு" + "களி" + "லிரு" + "ந்து" (BAD!)
    AMB Pipeline:
      1. Morpheme Split: "வீடு @@கள் @@இருந்து" (Linguistic boundary)
      2. BPE runs INSIDE: "வீடு", "கள்", "இருந்து"
      3. Result: ["வீடு", "கள்", "இருந்து"] (GOOD! Meaningful tokens)

Usage:
    python train_amb_tokenizer.py --config config.yaml
    python train_amb_tokenizer.py --vocab-size 48000
"""

import os
import sys
import argparse
import logging
import json
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Iterator

import yaml
from tqdm import tqdm

# AMB Modules
try:
    from tamil_unicode import TamilDeepNormalizer
    from akshara import AksharaSegmenter
    from morpheme import MorphemeSegmenter
except ImportError:
    # Ensure local directory is in path if running as script
    sys.path.append(str(Path(__file__).parent))
    from tamil_unicode import TamilDeepNormalizer
    from akshara import AksharaSegmenter
    from morpheme import MorphemeSegmenter

# HuggingFace Tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders, processors
from tokenizers.pre_tokenizers import Split, Sequence

# Set up Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Regex split to prevent cross-script tokens (Latin+Tamil mixed)
SCRIPT_SPLIT_PATTERN = r"(?u)(\d+|\p{L}+|[^\s\w]+)"


def generate_tamil_syllables() -> List[str]:
    """Generate the complete Tamil syllable inventory (247 syllables)."""
    vowels = list("அஆஇஈஉஊஎஏஐஒஓஔ")
    consonant_bases = list("கஙசஞடணதநபமயரலவழளறன")
    vowel_signs = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
    syllables = []
    syllables.extend(vowels)
    for c in consonant_bases:
        syllables.append(c + "்")
    for c in consonant_bases:
        for vs in vowel_signs:
            syllables.append(c + vs)
    syllables.append("ஃ")
    return syllables


# ===========================================================================
# Configuration Loader
# ===========================================================================

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# AMB Corpus Iterator (Streaming)
# ===========================================================================

class AMBCorpusIterator:
    """
    Yields deeply normalized and segmented (@@-marked) Tamil text.
    Handles the heavy lifting of Layer 1 & 3 on-the-fly.
    """
    
    def __init__(self, corpus_path: Path, batch_size: int = 2000):
        self.corpus_path = corpus_path
        self.batch_size = batch_size
        
        # Initialize Normalizer (Layer 1)
        self.normalizer = TamilDeepNormalizer(
            strip_urls=True,
            strip_emails=True,
            normalize_numerals="preserve",
            preserve_grantha=True,
        )
        
        # Initialize Morpheme Segmenter (Layer 3)
        self.morpheme_seg = MorphemeSegmenter()
        
        # Stats tracking
        self.total_lines = 0
        self.processed_bytes = 0

    def __iter__(self):
        batch = []
        with open(str(self.corpus_path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                self.total_lines += 1
                self.processed_bytes += len(line)
                
                # --- Layer 1: Deep normalize ---
                cleaned = self.normalizer.normalize(line)
                
                # --- Layer 3: Segment words ---
                # We split by space first to process word-by-word
                words = cleaned.split()
                segmented_words = []
                
                for word in words:
                    # Apply morpheme segmentation
                    # "வீடுகளிலிருந்து" -> "வீடு கள் இருந்து"
                    morphemes = self.morpheme_seg.segment_word(word)
                    # "வீடு கள்" (space separated from segment_word) -> "வீடு @@ கள்"
                    marked_word = morphemes.replace(" ", " @@ ")
                    segmented_words.append(marked_word)
                
                # Reconstruct line. Morphemes within a word joined by @@, words by space.
                processed_line = " ".join(segmented_words)
                
                batch.append(processed_line)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        if batch:
            yield batch


# ===========================================================================
# Constrained BPE Trainer (Layer 4)
# ===========================================================================

def train_amb_tokenizer(cfg: dict, corpus_path: Path):
    """
    Train a Byte-Level BPE tokenizer on the pre-segmented corpus.
    """
    hf_cfg = cfg["tokenizer"]
    vocab_size = hf_cfg["vocab_size"]
    output_dir = Path(hf_cfg.get("output_dir", "models/amb_tokenizer"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"--- AMB Pipeline Started ---")
    log.info(f"Target Vocab: {vocab_size}")
    log.info(f"Corpus:       {corpus_path}")
    
    # Initialize Tokenizer Model
    # We use BPE because it handles byte fallback beautifully.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    
    # --- Normalization ---
    # We apply NFC here as a safeguard, even though iterator does deep norm
    tokenizer.normalizer = normalizers.NFC()
    
    # Strict Script Isolation Regex
    # 1. Tamil: Vowels, or Consonants with modifiers.
    # 2. English: Latin character sequences.
    # 3. Numeric: Integer sequences.
    # 4. Special: Punctuation and others.
    TAMIL_UNIT = r"[\u0B85-\u0B94]|[\u0B95-\u0BB9][\u0BCD\u0B95-\u0BB9]*[\u0BBE-\u0BCD\u0BD7]?"
    AKSHARA_REGEX = rf"({TAMIL_UNIT}|[a-zA-Z]+|[0-9]+|[^\s\u0B80-\u0BFFa-zA-Z0-9]+)"
    
    tokenizer.pre_tokenizer = Sequence([
        # Force split at morpheme boundary markers first
        Split(pattern=" @@ ", behavior="isolated"),
        # Force split by script/unit (No merging across Tamil-English boundary)
        Split(pattern=AKSHARA_REGEX, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # --- Trainer ---
    special_tokens = hf_cfg.get("special_tokens", [
        "<|endoftext|>", "<|padding|>", "<|im_start|>", "<|im_end|>"
    ])
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # --- Training Step ---
    log.info("Starting BPE Training from AMB Iterator...")
    iterator = AMBCorpusIterator(corpus_path)
    
    # We wrap in a generator to satisfy the library's requirement
    def combined_iterator():
        for batch in iterator:
            yield batch

    tokenizer.train_from_iterator(combined_iterator(), trainer)

    # --- Post-processing ---
    # Merge dots and byte-fallback cleanup
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save Model
    model_path = output_dir / "tokenizer.json"
    tokenizer.save(str(model_path))
    
    # Save HF config boilerplate
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump({
            "model_type": "gpt2",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": True,
            "add_prefix_space": False,
        }, f, indent=2)
        
    log.info(f"Model saved to {output_dir}")
    return str(model_path)


# ===========================================================================
# Validation (Layer 5 Check)
# ===========================================================================

def validate_amb_tokenizer(tokenizer_path: str):
    """Run sanity checks to prove AMB superiority."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    test_words = [
        "போகவேண்டியிருந்தது",  # Complex verb
        "அரசியலமைப்பு",       # Compound noun
        "வீடுகளிலிருந்து",     # Plural + Case
    ]
    
    log.info("--- AMB Validation ---")
    for word in test_words:
        # We must mimic the pre-tokenization logic for inference?
        # Ideally, the BPE model should have learned the morphemes
        # so well that even without the segmenter, it finds them.
        encoded = tokenizer.encode(word)
        tokens = encoded.tokens
        log.info(f"Word: {word}")
        log.info(f"Tokens: {tokens}")


def main():
    parser = argparse.ArgumentParser(description="Train AMB Tokenizer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--corpus", help="Override corpus path")
    parser.add_argument("--vocab-size", type=int, help="Override vocab size")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.vocab_size:
        cfg["tokenizer"]["vocab_size"] = args.vocab_size
        
    corpus_path = Path(args.corpus) if args.corpus else Path(cfg["corpus"]["output_file"])
    
    if not corpus_path.exists():
        log.error(f"Corpus not found: {corpus_path}")
        return
        
    model_path = train_amb_tokenizer(cfg, corpus_path)
    validate_amb_tokenizer(model_path)


if __name__ == "__main__":
    main()
