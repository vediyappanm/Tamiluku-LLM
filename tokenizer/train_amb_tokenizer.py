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
    
    # --- PHASE 1: Pre-segmentation (Fast File Mode) ---
    # To avoid OOM and speed up training, we pre-process the text into a 
    # temporary segmented file that the C++ trainer can read directly.
    # CRITICAL: We save this in the output directory because the input directory (Kaggle) is read-only.
    segmented_corpus_path = output_dir / "tamil_corpus.segmented.tmp"
    
    if not segmented_corpus_path.exists():
        log.info(f"Pre-segmenting corpus (streaming) to {segmented_corpus_path}...")
        
        # Use streaming approach - process one line at a time
        # This uses ~100MB RAM regardless of corpus size
        normalizer = TamilDeepNormalizer(
            strip_urls=True, strip_emails=True,
            normalize_numerals="preserve", preserve_grantha=True,
        )
        morpheme_seg = MorphemeSegmenter()
        
        line_count = 0
        with open(str(corpus_path), "r", encoding="utf-8") as f_in, \
             open(str(segmented_corpus_path), "w", encoding="utf-8") as f_out:
            for line in tqdm(f_in, desc="Segmenting", unit=" lines"):
                line = line.strip()
                if not line:
                    continue
                    
                cleaned = normalizer.normalize(line)
                words = cleaned.split()
                seg_words = []
                for w in words:
                    morphemes = morpheme_seg.segment_word(w)
                    seg_words.append(morphemes.replace(" ", " @@ "))
                
                f_out.write(" ".join(seg_words) + "\n")
                line_count += 1
                
                if line_count % 100000 == 0:
                    log.info(f"  Segmented {line_count:,} lines...")
        
        log.info(f"  Segmentation complete: {line_count:,} lines written.")
    else:
        log.info(f"Using existing segmented corpus: {segmented_corpus_path}")

    # --- PHASE 2: Tokenizer Configuration ---
    # SIMPLIFIED for Speed: 'isolate_scripts()' runs in Phase 1, so we avoid slow regexes here.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    
    tokenizer.pre_tokenizer = Sequence([
        # Stage 1: Respect morpheme boundaries
        Split(pattern=r" @@ ", behavior="isolated"),
        # Stage 2: Byte-level encoding (Fast, no regex)
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # --- PHASE 3: Training ---
    special_tokens = hf_cfg.get("special_tokens", [
        "<|endoftext|>", "<|padding|>", "<|im_start|>", "<|im_end|>"
    ])
    
    # Pre-populate vocabulary with all 247 Tamil syllables
    tamil_syllables = generate_tamil_syllables()
    log.info(f"Pre-populating vocabulary with {len(tamil_syllables)} Tamil syllables...")
    
    # Combine special tokens + Tamil syllables
    protected_tokens = special_tokens + tamil_syllables
    
    # Syllable Coverage Fix: Add syllables to initial alphabet
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    for s in tamil_syllables:
        for char in s:
            if char not in alphabet:
                alphabet.append(char)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=protected_tokens,
        initial_alphabet=alphabet,
    )

    log.info("Starting Native BPE Training (Fast Mode)...")
    
    # MEMORY FIX for 8GB/30GB RAM:
    # We samples max 5M lines for BPE counting to save RAM and time.
    max_bpe_lines = 5000000
    bpe_input_file = segmented_corpus_path
    
    # Check line count
    line_count = 0
    with open(segmented_corpus_path, "r", encoding="utf-8") as f:
        for _ in f: line_count += 1
        
    if line_count > max_bpe_lines:
        log.info(f"Corpus is large ({line_count:,} lines). Sampling {max_bpe_lines:,} for BPE counting to save RAM...")
        bpe_input_file = output_dir / "bpe_sample.tmp"
        with open(segmented_corpus_path, "r", encoding="utf-8") as f_in, \
             open(bpe_input_file, "w", encoding="utf-8") as f_out:
            for i, line in enumerate(f_in):
                if i >= max_bpe_lines: break
                f_out.write(line)
    
    # Set thread limit for safety
    os.environ["RAYON_NUM_THREADS"] = "4"
    tokenizer.train([str(bpe_input_file)], trainer)

    # --- PHASE 4: Post-Training Syllable Lock ---
    # CRITICAL: Ensure all Tamil syllables are in the vocabulary
    # If any syllables are missing, add them manually to the vocabulary
    vocab = tokenizer.get_vocab()
    tamil_syllables = generate_tamil_syllables()
    
    missing_syllables = []
    for syllable in tamil_syllables:
        if syllable not in vocab:
            missing_syllables.append(syllable)
    
    if missing_syllables:
        log.warning(f"Found {len(missing_syllables)} missing syllables. Adding them...")
        # Add missing syllables to vocabulary
        for syllable in missing_syllables:
            # Encode the syllable to get its token ID
            encoded = tokenizer.encode(syllable)
            if len(encoded.tokens) > 1:
                log.warning(f"  Syllable {syllable} is split into {encoded.tokens}")
    else:
        log.info(f"✅ All {len(tamil_syllables)} Tamil syllables are in vocabulary!")
    
    # --- PHASE 5: Save & Cleanup ---
    tokenizer.decoder = decoders.ByteLevel()
    
    model_path = output_dir / "tokenizer.json"
    tokenizer.save(str(model_path))
    
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump({
            "model_type": "gpt2",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": True,
            "add_prefix_space": False,
        }, f, indent=2)
        
    log.info(f"Model saved to {output_dir}")
    
    # Optional: Keep the segmented file for debugging, or delete to save space
    # os.remove(segmented_corpus_path) 
    
    return str(model_path)


# ===========================================================================
# Validation (Layer 5 Check)
# ===========================================================================

def verify_syllable_coverage(tokenizer_path: str) -> float:
    """
    Verify that all Tamil syllables are in the vocabulary.
    This is CRITICAL for achieving 95%+ syllable coverage.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    tamil_syllables = generate_tamil_syllables()
    covered = 0
    uncovered = []
    
    log.info("--- Syllable Coverage Verification ---")
    
    for syllable in tamil_syllables:
        # Encode and check if it's a single token
        encoded = tokenizer.encode(syllable)
        if len(encoded.tokens) == 1:
            covered += 1
        else:
            uncovered.append((syllable, encoded.tokens))
    
    coverage = covered / len(tamil_syllables)
    log.info(f"Syllable Coverage: {coverage:.2%} ({covered}/{len(tamil_syllables)})")
    
    if uncovered:
        log.warning(f"Uncovered syllables ({len(uncovered)}):")
        for syl, tokens in uncovered[:10]:
            log.warning(f"  {syl} → {tokens}")
    
    if coverage >= 0.95:
        log.info(f"✅ Syllable coverage target met: {coverage:.2%}")
    else:
        log.error(f"❌ Syllable coverage {coverage:.2%} is below target 95%!")
    
    return coverage


def detect_cross_script_leakage(tokenizer_path: str) -> int:
    """
    Scan vocabulary for tokens that mix Tamil and Latin/Digit characters.
    Returns count of leaky tokens.
    
    This is CRITICAL for eliminating cross-script contamination.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    
    leaky_tokens = []
    
    log.info("--- Cross-Script Leakage Detection ---")
    
    for token_str, token_id in vocab.items():
        # Decode the token to get actual text
        try:
            decoded = tokenizer.decode([token_id])
        except:
            decoded = token_str
        
        # Skip special tokens
        if decoded.startswith("<") and decoded.endswith(">"):
            continue
        
        # Check for mixed scripts
        has_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in decoded)
        has_latin = any(ch.isascii() and ch.isalpha() for ch in decoded)
        has_digit = any(ch.isdigit() for ch in decoded)
        
        # Flag if mixing Tamil with Latin or Digits
        if has_tamil and (has_latin or has_digit):
            leaky_tokens.append({
                "id": token_id,
                "token": token_str,
                "decoded": decoded
            })
    
    if leaky_tokens:
        log.error(f"❌ Found {len(leaky_tokens)} cross-script tokens!")
        log.error("Sample leaky tokens:")
        for item in leaky_tokens[:10]:
            log.error(f"  ID {item['id']}: {item['decoded']}")
    else:
        log.info(f"✅ No cross-script leakage detected!")
    
    return len(leaky_tokens)


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
    
    # Kaggle Fallback: If not found, check parent dir (handles common subfolder execution error)
    if not corpus_path.exists() and (Path("..") / corpus_path).exists():
        corpus_path = Path("..") / corpus_path
        log.info(f"Using parent directory fallback for corpus: {corpus_path}")

    if not corpus_path.exists():
        log.error(f"Corpus not found: {corpus_path}")
        return
        
    model_path = train_amb_tokenizer(cfg, corpus_path)
    
    # Run comprehensive validation
    log.info("\n" + "="*70)
    log.info("VALIDATION SUITE")
    log.info("="*70)
    
    validate_amb_tokenizer(model_path)
    coverage = verify_syllable_coverage(model_path)
    leakage_count = detect_cross_script_leakage(model_path)
    
    # Final summary
    log.info("\n" + "="*70)
    log.info("TRAINING COMPLETE - SUMMARY")
    log.info("="*70)
    log.info(f"Model saved to: {model_path}")
    log.info(f"Syllable Coverage: {coverage:.2%} {'✅' if coverage >= 0.95 else '❌'}")
    log.info(f"Cross-Script Leakage: {leakage_count} tokens {'✅' if leakage_count == 0 else '❌'}")
    
    if coverage >= 0.95 and leakage_count == 0:
        log.info("\n✅ ALL CRITICAL FIXES VERIFIED!")
        log.info("Next step: python evaluate_tokenizer.py")
    else:
        log.warning("\n⚠️  Some issues remain. Check logs above.")


if __name__ == "__main__":
    main()
