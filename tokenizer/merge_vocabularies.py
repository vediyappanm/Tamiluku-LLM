"""
merge_vocabularies.py - Merge Tamil Tokenizer into Base LLM Tokenizer
=======================================================================
Merges a custom-trained Tamil tokenizer (HuggingFace or SentencePiece)
into an existing LLM's tokenizer vocabulary.

Supports two merge paths:
  1. HuggingFace BPE -> Base LLM (recommended, native format)
  2. SentencePiece BPE -> Base LLM (legacy)

Algorithm:
  1. Load base tokenizer from HuggingFace Hub
  2. Load Tamil tokenizer (HF or SP format)
  3. Extract Tamil-only tokens (no cross-script, no duplicates)
  4. Add new tokens to base tokenizer
  5. Save merged tokenizer in HuggingFace format

Usage:
    python merge_vocabularies.py [--config config.yaml]
    python merge_vocabularies.py --base-model meta-llama/Llama-3-8B
    python merge_vocabularies.py --engine sentencepiece  # Legacy mode
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Set, List, Dict

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


def is_tamil_token(token: str) -> bool:
    return any(0x0B80 <= ord(ch) <= 0x0BFF for ch in token)


def is_cross_script(token: str) -> bool:
    """Check if a token mixes Tamil and Latin scripts."""
    clean = token.lstrip("▁").lstrip("Ġ")
    if not clean:
        return False
    has_tamil = any(0x0B80 <= ord(ch) <= 0x0BFF for ch in clean)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in clean)
    return has_tamil and has_latin


# ---------------------------------------------------------------------------
# Extract tokens from HuggingFace tokenizer
# ---------------------------------------------------------------------------

def extract_tamil_tokens_hf(tokenizer_path: str) -> List[str]:
    """Extract Tamil tokens from a HuggingFace tokenizer.json file."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()

    tamil_tokens = []
    for token, token_id in vocab.items():
        # Skip special tokens
        if token.startswith("<") and token.endswith(">"):
            continue
        if token.startswith("<|") and token.endswith("|>"):
            continue

        # Check for Tamil content (accounting for byte-level BPE encoding)
        # In byte-level BPE, Tamil UTF-8 bytes get mapped to single-byte chars
        # We need to try decoding to check
        try:
            decoded = tokenizer.decode([token_id])
            if is_tamil_token(decoded) and not is_cross_script(decoded):
                tamil_tokens.append(token)
        except Exception:
            if is_tamil_token(token) and not is_cross_script(token):
                tamil_tokens.append(token)

    log.info(f"Extracted {len(tamil_tokens):,} Tamil tokens from HF tokenizer")
    return tamil_tokens


# ---------------------------------------------------------------------------
# Extract tokens from SentencePiece model
# ---------------------------------------------------------------------------

def extract_tamil_tokens_sp(sp_model_path: str, skip_byte: bool = True) -> List[str]:
    """Extract Tamil tokens from a SentencePiece model."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    tamil_tokens = []
    total = sp.get_piece_size()

    for i in range(total):
        piece = sp.id_to_piece(i)

        # Skip byte-fallback tokens
        if skip_byte and piece.startswith("<0x") and piece.endswith(">"):
            continue

        # Skip special tokens
        if (piece.startswith("<") and piece.endswith(">")) or \
           (piece.startswith("<|") and piece.endswith("|>")):
            continue

        clean = piece.lstrip("▁")
        if clean and is_tamil_token(clean) and not is_cross_script(clean):
            tamil_tokens.append(piece)

    log.info(f"Extracted {len(tamil_tokens):,} Tamil tokens from SentencePiece model")
    return tamil_tokens


# ---------------------------------------------------------------------------
# Merge into base tokenizer
# ---------------------------------------------------------------------------

def merge_into_base_tokenizer(
    base_model_id: str,
    tamil_tokens: List[str],
    output_dir: str,
    skip_duplicates: bool = True,
) -> Dict:
    """Add Tamil tokens to a base HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.error("Install transformers: pip install transformers")
        sys.exit(1)

    log.info(f"Loading base tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    original_size = len(tokenizer)
    log.info(f"Base vocabulary size: {original_size:,}")

    # Get existing vocabulary
    existing_vocab = set(tokenizer.get_vocab().keys())

    # Filter tokens
    new_tokens = []
    skipped_duplicate = 0
    skipped_cross_script = 0

    for token in tamil_tokens:
        # Check duplicates with various prefix conventions
        variants = [token, token.lstrip("▁"), "▁" + token.lstrip("▁")]
        # Also check Ġ prefix (GPT-2/LLaMA style)
        variants.extend([token.lstrip("Ġ"), "Ġ" + token.lstrip("Ġ")])

        if skip_duplicates and any(v in existing_vocab for v in variants):
            skipped_duplicate += 1
            continue

        # Final cross-script check
        clean = token.lstrip("▁").lstrip("Ġ")
        if is_cross_script(clean):
            skipped_cross_script += 1
            continue

        new_tokens.append(token)

    log.info(f"New tokens to add: {len(new_tokens):,}")
    log.info(f"  Skipped (duplicate):     {skipped_duplicate:,}")
    log.info(f"  Skipped (cross-script):  {skipped_cross_script:,}")

    # Add tokens
    num_added = tokenizer.add_tokens(new_tokens)
    final_size = len(tokenizer)

    log.info(f"Added {num_added:,} tokens. New vocabulary size: {final_size:,}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    log.info(f"Saved merged tokenizer to {output_path}")

    stats = {
        "base_model": base_model_id,
        "original_vocab_size": original_size,
        "tamil_tokens_extracted": len(tamil_tokens),
        "new_tokens_added": num_added,
        "skipped_duplicate": skipped_duplicate,
        "skipped_cross_script": skipped_cross_script,
        "final_vocab_size": final_size,
        "output_dir": str(output_path),
    }

    return stats


def verify_merged_tokenizer(output_dir: str):
    """Quick verification of the merged tokenizer."""
    from transformers import AutoTokenizer

    log.info("Verifying merged tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

    test_cases = [
        "தமிழ் ஒரு அழகான மொழி.",
        "இந்திய அரசியலமைப்புச் சட்டம்",
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது.",
        "Machine learning model-ஐ train பண்ணனும்.",
        "Hello world",
    ]

    all_ok = True
    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        words = text.split()
        fertility = len(tokens) / max(len(words), 1)

        roundtrip_ok = decoded.strip() == text.strip()
        if not roundtrip_ok:
            all_ok = False

        log.info(f"  Input:     {text}")
        log.info(f"  Tokens:    {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        log.info(f"  Fertility: {fertility:.2f}")
        log.info(f"  Roundtrip: {'OK' if roundtrip_ok else 'MISMATCH'}")
        log.info(f"")

    if all_ok:
        log.info("All verification tests passed!")
    else:
        log.warning("Some roundtrip tests failed. Check merged tokenizer.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge Tamil tokenizer into base LLM tokenizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--engine", default=None, choices=["huggingface", "sentencepiece"])
    parser.add_argument("--tamil-model", default=None, help="Tamil tokenizer path")
    parser.add_argument("--base-model", default=None, help="Base HF model ID")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    merge_cfg = cfg["merge"]
    engine = args.engine or cfg.get("tokenizer", {}).get("engine", "huggingface")

    base_model_id = args.base_model or merge_cfg["base_model"]
    output_dir = args.output or merge_cfg["output_dir"]

    # Extract Tamil tokens based on engine
    if engine == "huggingface":
        tok_output_dir = cfg.get("tokenizer", {}).get("output_dir", "models/tamil_tokenizer")
        tamil_model_path = args.tamil_model or str(Path(tok_output_dir) / "tokenizer.json")

        if not os.path.exists(tamil_model_path):
            log.error(f"Tamil tokenizer not found: {tamil_model_path}")
            log.error("Run 'python train_tokenizer.py' first.")
            sys.exit(1)

        tamil_tokens = extract_tamil_tokens_hf(tamil_model_path)
    else:
        tamil_model_path = args.tamil_model or f"{cfg['sentencepiece']['model_prefix']}.model"

        if not os.path.exists(tamil_model_path):
            log.error(f"Tamil model not found: {tamil_model_path}")
            log.error("Run 'python train_tokenizer.py --engine sentencepiece' first.")
            sys.exit(1)

        tamil_tokens = extract_tamil_tokens_sp(
            tamil_model_path,
            skip_byte=merge_cfg.get("skip_byte_fallback_tokens", True),
        )

    # Merge
    stats = merge_into_base_tokenizer(
        base_model_id=base_model_id,
        tamil_tokens=tamil_tokens,
        output_dir=output_dir,
        skip_duplicates=merge_cfg.get("skip_duplicates", True),
    )

    # Verify
    verify_merged_tokenizer(output_dir)

    # Save stats
    report_path = Path("reports") / "merge_stats.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Merge stats saved to {report_path}")
    log.info(f"\nNext step: python resize_embeddings.py")


if __name__ == "__main__":
    main()
