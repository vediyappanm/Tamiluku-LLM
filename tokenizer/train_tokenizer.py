"""
train_tokenizer.py - Production-Grade Tamil BPE Tokenizer Training
====================================================================
Trains a byte-level BPE tokenizer using the HuggingFace `tokenizers`
library - the same algorithm used by GPT-4, LLaMA-3, and Mistral.

Key design choices:
  - Byte-level BPE (like GPT-4) - no unknown tokens ever
  - Tamil-aware pre-tokenization regex - prevents cross-word merges
  - 48K vocab target - covers all 247 Tamil syllables + morpheme combos
  - NFC normalization built into tokenizer pipeline
  - Native HuggingFace format - no conversion needed for transformers

Two modes:
  1. "huggingface" (default) - HuggingFace tokenizers byte-level BPE
  2. "sentencepiece" - SentencePiece BPE (legacy, for compatibility)

Usage:
    python train_tokenizer.py [--config config.yaml]
    python train_tokenizer.py --engine amb            # AMB Layered Architecture
    python train_tokenizer.py --engine sentencepiece  # Legacy mode
    python train_tokenizer.py --vocab-size 48000
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tamil Pre-tokenization Patterns
# ---------------------------------------------------------------------------

# GPT-4 style regex adapted for Tamil language
# This regex splits text into meaningful chunks BEFORE BPE runs:
#   1. Tamil words (consonant clusters + vowel signs + pulli sequences)
#   2. English/Latin words (with contractions)
#   3. Numbers (integers and decimals)
#   4. Individual whitespace characters
#   5. Other punctuation/symbols
TAMIL_PRETOK_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""              # English contractions
    r"""|[\u0B80-\u0BFF]+"""                      # Tamil Unicode block (words)
    r"""|[a-zA-Z]+"""                              # Latin words
    r"""|[0-9]+"""                                 # Numbers
    r"""| ?[^\s\u0B80-\u0BFFa-zA-Z0-9]+"""        # Punctuation / symbols
    r"""|\s+(?!\S)"""                              # Trailing whitespace
    r"""|\s"""                                     # Individual whitespace
)

# Regex split to prevent cross-script tokens (Latin+Tamil mixed)
SCRIPT_SPLIT_PATTERN = r"(?u)(\d+|\p{L}+|[^\s\w]+)"


# ---------------------------------------------------------------------------
# Corpus Validation
# ---------------------------------------------------------------------------

def validate_corpus(corpus_path: Path) -> dict:
    """Quick validation of the training corpus."""
    if not corpus_path.exists():
        log.error(f"Corpus file not found: {corpus_path}")
        log.error("Run 'python collect_corpus.py' and 'python normalize.py' first.")
        sys.exit(1)

    size_bytes = corpus_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)

    # Sample first 10K lines for quick stats
    line_count = 0
    char_count = 0
    tamil_char_count = 0
    sample_lines = 10000

    with open(str(corpus_path), "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count <= sample_lines:
                char_count += len(line)
                tamil_char_count += sum(1 for ch in line if 0x0B80 <= ord(ch) <= 0x0BFF)

    tamil_ratio = tamil_char_count / max(char_count, 1)

    stats = {
        "size_mb": round(size_mb, 1),
        "size_gb": round(size_gb, 2),
        "lines": line_count,
        "tamil_ratio_sample": round(tamil_ratio, 4),
    }

    log.info(f"Corpus validation:")
    log.info(f"  Size: {size_gb:.2f} GB ({size_mb:.0f} MB)")
    log.info(f"  Lines: {line_count:,}")
    log.info(f"  Tamil character ratio (sample): {tamil_ratio:.2%}")

    if size_mb < 100:
        log.warning(
            f"Corpus is only {size_mb:.1f} MB. For production quality (48K vocab), "
            "recommended minimum is 5 GB. Results may have poor coverage."
        )
    elif size_mb < 1000:
        log.warning(
            f"Corpus is {size_mb:.0f} MB. Acceptable for initial training. "
            "For GPT-4-class quality, target 10+ GB."
        )
    else:
        log.info(f"  Corpus size is sufficient for production-grade training.")

    if tamil_ratio < 0.5:
        log.warning(f"Tamil character ratio is {tamil_ratio:.2%}. Check normalization.")

    return stats


def corpus_iterator(corpus_path: Path, batch_size: int = 1000) -> Iterator[List[str]]:
    """Yield batches of lines from the corpus for tokenizer training."""
    batch = []
    with open(str(corpus_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Engine 1: HuggingFace tokenizers (Byte-Level BPE) -- RECOMMENDED
# ---------------------------------------------------------------------------

def train_huggingface_bpe(cfg: dict, corpus_path: Path) -> str:
    """
    Train a byte-level BPE tokenizer using the HuggingFace tokenizers library.
    Same algorithm as GPT-4, LLaMA-3, Mistral.
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders, processors
        from tokenizers.pre_tokenizers import Split
        import regex
    except ImportError:
        log.error("Install tokenizers and regex: pip install tokenizers regex")
        sys.exit(1)

    hf_cfg = cfg["tokenizer"]
    vocab_size = hf_cfg["vocab_size"]
    output_dir = Path(hf_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Training HuggingFace Byte-Level BPE tokenizer...")
    log.info(f"  Vocab size:         {vocab_size:,}")
    log.info(f"  Pre-tokenization:   Tamil-aware regex")
    log.info(f"  Algorithm:          Byte-Level BPE (GPT-4 style)")
    log.info(f"  Output:             {output_dir}")

    # --- Build tokenizer ---
    tokenizer = Tokenizer(models.BPE())

    # Normalizer: NFC for Tamil canonical forms
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer: Tamil-aware regex splitting
    # This is the key innovation - it prevents BPE from merging across
    # word boundaries, which creates garbage tokens
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=SCRIPT_SPLIT_PATTERN,
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.Split(
            pattern=TAMIL_PRETOK_PATTERN,
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # --- Special tokens ---
    special_tokens = hf_cfg.get("special_tokens", [
        "<|endoftext|>",
        "<|padding|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
    ])

    # --- Trainer ---
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=hf_cfg.get("min_frequency", 2),
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # --- Train ---
    log.info("Starting BPE training (this may take hours for large corpora)...")

    # Use iterator for memory efficiency on large files
    tokenizer.train_from_iterator(
        corpus_iterator(corpus_path, batch_size=hf_cfg.get("batch_size", 1000)),
        trainer=trainer,
        length=None,  # Unknown length for streaming
    )

    log.info(f"Training complete. Vocab size: {tokenizer.get_vocab_size():,}")

    # --- Save ---
    tokenizer_path = str(output_dir / "tokenizer.json")
    tokenizer.save(tokenizer_path)
    log.info(f"Saved tokenizer to {tokenizer_path}")

    # Also save in HuggingFace format for easy loading with transformers
    _save_hf_tokenizer_config(output_dir, special_tokens, vocab_size)

    return tokenizer_path


def _save_hf_tokenizer_config(output_dir: Path, special_tokens: list, vocab_size: int):
    """Save tokenizer_config.json for HuggingFace transformers compatibility."""
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "gpt2",
        "bos_token": special_tokens[0] if special_tokens else "<|endoftext|>",
        "eos_token": special_tokens[0] if special_tokens else "<|endoftext|>",
        "unk_token": None,
        "pad_token": special_tokens[1] if len(special_tokens) > 1 else "<|padding|>",
        "vocab_size": vocab_size,
        "model_max_length": 8192,
        "clean_up_tokenization_spaces": False,
    }

    config_path = output_dir / "tokenizer_config.json"
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Saved tokenizer config to {config_path}")


# ---------------------------------------------------------------------------
# Engine 2: SentencePiece BPE (Legacy)
# ---------------------------------------------------------------------------

def train_sentencepiece_bpe(cfg: dict, corpus_path: Path) -> str:
    """Train SentencePiece BPE tokenizer (legacy mode)."""
    try:
        import sentencepiece as spm
    except ImportError:
        log.error("Install sentencepiece: pip install sentencepiece")
        sys.exit(1)

    sp_cfg = cfg["sentencepiece"]
    model_prefix = sp_cfg["model_prefix"]
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Build special token strings
    user_symbols = sp_cfg.get("user_defined_symbols", [])
    user_symbols = [s.replace("{{", "<|").replace("}}", "|>") for s in user_symbols]
    user_symbols_str = ",".join(user_symbols) if user_symbols else ""

    control_symbols = sp_cfg.get("control_symbols", [])
    control_symbols = [s.replace("{{", "<|").replace("}}", "|>") for s in control_symbols]
    control_symbols_str = ",".join(control_symbols) if control_symbols else ""

    log.info(f"Training SentencePiece BPE tokenizer (legacy mode)...")
    log.info(f"  Vocab size:           {sp_cfg['vocab_size']:,}")
    log.info(f"  Character coverage:   {sp_cfg['character_coverage']}")
    log.info(f"  Byte fallback:        {sp_cfg['byte_fallback']}")
    log.info(f"  Split by script:      {sp_cfg['split_by_unicode_script']}")

    train_args = {
        "input": str(corpus_path),
        "model_prefix": model_prefix,
        "model_type": sp_cfg["model_type"],
        "vocab_size": sp_cfg["vocab_size"],
        "character_coverage": sp_cfg["character_coverage"],
        "byte_fallback": sp_cfg["byte_fallback"],
        "split_by_unicode_script": sp_cfg["split_by_unicode_script"],
        "split_by_whitespace": sp_cfg["split_by_whitespace"],
        "normalization_rule_name": sp_cfg["normalization_rule_name"],
        "num_threads": sp_cfg["num_threads"],
        "input_sentence_size": sp_cfg["input_sentence_size"],
        "max_sentence_length": sp_cfg["max_sentence_length"],
        "shuffle_input_sentence": sp_cfg["shuffle_input_sentence"],
        "seed_sentencepiece_size": sp_cfg["seed_sentencepiece_size"],
    }

    if user_symbols_str:
        train_args["user_defined_symbols"] = user_symbols_str
    if control_symbols_str:
        train_args["control_symbols"] = control_symbols_str

    spm.SentencePieceTrainer.train(**{
        k: str(v) if isinstance(v, bool) else v
        for k, v in train_args.items()
    })

    model_file = f"{model_prefix}.model"
    if not os.path.exists(model_file):
        log.error("Training failed - model file not created.")
        sys.exit(1)

    log.info(f"SentencePiece training complete: {model_file}")
    return model_file


# ---------------------------------------------------------------------------
# Sanity Checks & Analysis
# ---------------------------------------------------------------------------

def sanity_check_hf(tokenizer_path: str):
    """Run sanity checks on a HuggingFace tokenizer."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)

    log.info(f"--- Sanity Check (HuggingFace BPE) ---")
    log.info(f"Vocab size: {tokenizer.get_vocab_size():,}")

    test_sentences = [
        "தமிழ் ஒரு அழகான மொழி.",
        "போகவேண்டியிருந்தது என்னால் முடியவில்லை.",
        "இந்திய அரசியலமைப்புச் சட்டம் அனைவருக்கும் சமத்துவத்தை உறுதிசெய்கிறது.",
        "2024-ல் தமிழ்நாட்டின் மக்கள்தொகை ௭.௫ கோடி.",
        "Machine learning model-ஐ train பண்ணனும்.",
        "Hello world, this is a test.",
        "கற்றுக்கொள்ள வேண்டிய பாடங்கள் நிறைய இருக்கின்றன.",
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது.",
    ]

    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        tokens = encoded.tokens
        ids = encoded.ids
        decoded = tokenizer.decode(ids)

        n_words = len(sentence.split())
        n_tokens = len(tokens)
        fertility = n_tokens / max(n_words, 1)

        log.info(f"")
        log.info(f"  Input:     {sentence}")
        log.info(f"  Tokens:    {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        log.info(f"  Words: {n_words}, Tokens: {n_tokens}, Fertility: {fertility:.2f}")

        if decoded.strip() == sentence.strip():
            log.info(f"  Roundtrip: OK")
        else:
            log.warning(f"  Roundtrip: MISMATCH")
            log.warning(f"    Expected: {repr(sentence)}")
            log.warning(f"    Got:      {repr(decoded)}")


def sanity_check_sp(model_path: str):
    """Run sanity checks on a SentencePiece tokenizer."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    log.info(f"--- Sanity Check (SentencePiece) ---")
    log.info(f"Vocab size: {sp.get_piece_size():,}")

    test_sentences = [
        "தமிழ் ஒரு அழகான மொழி.",
        "போகவேண்டியிருந்தது என்னால் முடியவில்லை.",
        "இந்திய அரசியலமைப்புச் சட்டம் அனைவருக்கும் சமத்துவத்தை உறுதிசெய்கிறது.",
        "Machine learning model-ஐ train பண்ணனும்.",
        "Hello world, this is a test.",
    ]

    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)
        decoded = sp.decode(ids)

        n_words = len(sentence.split())
        n_tokens = len(tokens)
        fertility = n_tokens / max(n_words, 1)

        log.info(f"")
        log.info(f"  Input:     {sentence}")
        log.info(f"  Tokens:    {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        log.info(f"  Words: {n_words}, Tokens: {n_tokens}, Fertility: {fertility:.2f}")
        log.info(f"  Roundtrip: {'OK' if decoded == sentence else 'MISMATCH'}")


def analyze_vocabulary_hf(tokenizer_path: str) -> dict:
    """Analyze vocabulary composition of a HuggingFace tokenizer."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    total = len(vocab)

    tamil_tokens = 0
    english_tokens = 0
    mixed_tokens = 0
    special_tokens = 0
    number_tokens = 0
    punct_tokens = 0
    byte_tokens = 0

    for token, token_id in vocab.items():
        # Decode bytes to get actual text
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            decoded = token

        if token.startswith("<") and token.endswith(">"):
            special_tokens += 1
            continue

        decoded = decoded.strip()
        if not decoded:
            continue

        # Check the actual characters in the decoded token
        has_tamil = any(0x0B80 <= ord(ch) <= 0x0BFF for ch in decoded)
        has_latin = any(ch.isascii() and ch.isalpha() for ch in decoded)
        has_digit = any(ch.isdigit() for ch in decoded)

        if has_tamil and has_latin:
            mixed_tokens += 1
        elif has_tamil:
            tamil_tokens += 1
        elif has_latin:
            english_tokens += 1
        elif has_digit:
            number_tokens += 1
        else:
            punct_tokens += 1

    stats = {
        "total_vocab": total,
        "tamil_tokens": tamil_tokens,
        "english_tokens": english_tokens,
        "mixed_tokens": mixed_tokens,
        "special_tokens": special_tokens,
        "number_tokens": number_tokens,
        "punctuation_tokens": punct_tokens,
        "tamil_ratio": round(tamil_tokens / max(total, 1), 4),
    }

    log.info(f"--- Vocabulary Analysis ---")
    log.info(f"  Total:         {total:,}")
    log.info(f"  Tamil:         {tamil_tokens:,} ({tamil_tokens/max(total,1):.1%})")
    log.info(f"  English:       {english_tokens:,} ({english_tokens/max(total,1):.1%})")
    log.info(f"  Mixed:         {mixed_tokens:,} ({mixed_tokens/max(total,1):.1%})")
    log.info(f"  Numbers:       {number_tokens:,}")
    log.info(f"  Punctuation:   {punct_tokens:,}")
    log.info(f"  Special:       {special_tokens:,}")

    if mixed_tokens > 0:
        log.warning(
            f"  Found {mixed_tokens} cross-script tokens. "
            "Check pre-tokenization pattern."
        )

    report_path = Path("reports") / "vocab_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Saved to {report_path}")

    return stats


def analyze_vocabulary_sp(model_path: str) -> dict:
    """Analyze SentencePiece vocabulary composition."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    total = sp.get_piece_size()
    tamil_tokens = 0
    english_tokens = 0
    byte_tokens = 0
    special_tokens = 0
    mixed_tokens = 0

    for i in range(total):
        piece = sp.id_to_piece(i)
        if piece.startswith("<") and piece.endswith(">"):
            special_tokens += 1
        elif piece.startswith("<0x"):
            byte_tokens += 1
        else:
            has_tamil = any(0x0B80 <= ord(ch) <= 0x0BFF for ch in piece)
            has_latin = any(ch.isascii() and ch.isalpha() for ch in piece)
            if has_tamil and has_latin:
                mixed_tokens += 1
            elif has_tamil:
                tamil_tokens += 1
            elif has_latin:
                english_tokens += 1

    stats = {
        "total_vocab": total,
        "tamil_tokens": tamil_tokens,
        "english_tokens": english_tokens,
        "byte_fallback_tokens": byte_tokens,
        "special_tokens": special_tokens,
        "cross_script_tokens": mixed_tokens,
        "tamil_ratio": round(tamil_tokens / max(total, 1), 4),
    }

    log.info(f"--- Vocabulary Analysis (SentencePiece) ---")
    log.info(f"  Total:         {total:,}")
    log.info(f"  Tamil:         {tamil_tokens:,} ({tamil_tokens/max(total,1):.1%})")
    log.info(f"  English:       {english_tokens:,} ({english_tokens/max(total,1):.1%})")
    log.info(f"  Byte fallback: {byte_tokens:,}")
    log.info(f"  Special:       {special_tokens:,}")
    log.info(f"  Cross-script:  {mixed_tokens:,}")

    report_path = Path("reports") / "vocab_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train production-grade Tamil BPE tokenizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--engine", default=None, choices=["amb", "huggingface", "sentencepiece"],
        help="Training engine (default: from config)"
    )
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only run analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_path = Path(cfg["corpus"]["output_file"])
    engine = args.engine or cfg.get("tokenizer", {}).get("engine", "huggingface")

    # Override vocab size if specified
    if args.vocab_size:
        if engine == "huggingface":
            cfg.setdefault("tokenizer", {})["vocab_size"] = args.vocab_size
        else:
            cfg["sentencepiece"]["vocab_size"] = args.vocab_size

    if not args.skip_train:
        corpus_stats = validate_corpus(corpus_path)

        if engine == "amb":
            from train_amb_tokenizer import train_amb_tokenizer
            model_path = train_amb_tokenizer(cfg, corpus_path)
            sanity_check_hf(model_path)
            analyze_vocabulary_hf(model_path)
        elif engine == "huggingface":
            model_path = train_huggingface_bpe(cfg, corpus_path)
            sanity_check_hf(model_path)
            analyze_vocabulary_hf(model_path)
        else:
            model_path = train_sentencepiece_bpe(cfg, corpus_path)
            sanity_check_sp(model_path)
            analyze_vocabulary_sp(model_path)

        log.info(f"\nTraining engine: {engine}")
        log.info(f"Model saved. Next step: python evaluate_tokenizer.py")
    else:
        # Analysis only
        if engine in ("amb", "huggingface"):
            output_dir = cfg.get("tokenizer", {}).get("output_dir", "models/tamil_tokenizer")
            model_path = str(Path(output_dir) / "tokenizer.json")
            if os.path.exists(model_path):
                sanity_check_hf(model_path)
                analyze_vocabulary_hf(model_path)
            else:
                log.error(f"Tokenizer not found: {model_path}")
        else:
            model_prefix = cfg["sentencepiece"]["model_prefix"]
            model_path = f"{model_prefix}.model"
            if os.path.exists(model_path):
                sanity_check_sp(model_path)
                analyze_vocabulary_sp(model_path)
            else:
                log.error(f"Model not found: {model_path}")


if __name__ == "__main__":
    main()
