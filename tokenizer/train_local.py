# train_local.py - Bulletproof AMB Training for 8GB RAM
# ===========================================================
# Processes 700MB+ corpus safely on low-RAM machines.
#
# Memory Strategy:
#   Phase 1 (Segment): Streams line-by-line, writes to disk. RAM ~200MB
#   Phase 2 (Train):   C++/Rust engine reads from disk.     RAM ~2-3GB
#   Total peak:        ~3GB (safe on 8GB Windows machine)
#
# Usage:
#   python tokenizer/train_local.py
#   python tokenizer/train_local.py --corpus tamil_corpus.txt --max-mb 700 --vocab-size 64000

import os
import sys
import gc
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Special tokens (constructed to avoid markup issues)
_ST = ["endoftext", "padding", "im_start", "im_end",
       "begin_of_text", "end_of_text"]
SPECIAL_TOKENS = ["<|" + t + "|>" for t in _ST]


# ===================================================================
# Phase 1: Streaming Segmentation (RAM: ~200MB)
# ===================================================================
def phase1_segment(corpus_path, output_path, max_mb):
    """
    Stream corpus line-by-line, apply morpheme segmentation,
    write directly to disk. Never holds more than 1 line in RAM.
    """
    # Import heavy modules only when needed
    from tamil_unicode import TamilDeepNormalizer
    from morpheme import MorphemeSegmenter

    norm = TamilDeepNormalizer(
        strip_urls=True, strip_emails=True,
        normalize_numerals="preserve", preserve_grantha=True,
    )
    mseg = MorphemeSegmenter()

    max_bytes = max_mb * 1024 * 1024
    written = 0
    count = 0

    log.info(f"[Phase 1] Segmenting {max_mb} MB from {corpus_path.name} ...")
    log.info(f"  Output: {output_path}")

    from tqdm import tqdm

    with open(str(corpus_path), "r", encoding="utf-8") as fin, \
         open(str(output_path), "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Segmenting", unit=" lines"):
            if written >= max_bytes:
                break
            line = line.strip()
            if not line:
                continue

            # Layer 1: Deep Normalize
            cleaned = norm.normalize(line)

            # Layer 3: Morpheme segmentation with boundary markers
            words = cleaned.split()
            seg = []
            for w in words:
                m = mseg.segment_word(w)
                seg.append(m.replace(" ", " @@ "))

            out = " ".join(seg) + "\n"
            fout.write(out)
            written += len(out.encode("utf-8"))
            count += 1

            if count % 100000 == 0:
                log.info(f"  {count:,} lines, {written / 1048576:.1f} MB ...")

    final_mb = written / (1024 * 1024)
    log.info(f"  [Phase 1 DONE] {count:,} lines, {final_mb:.1f} MB")

    # FREE all segmentation objects before Phase 2
    del norm, mseg
    gc.collect()
    log.info(f"  Memory freed for Phase 2")

    return count


# ===================================================================
# Phase 2: BPE Training (RAM: ~2-3GB via C++/Rust engine)
# ===================================================================
def phase2_train(segmented_path, output_dir, vocab_size):
    """
    Train BPE using the HuggingFace tokenizers C++/Rust engine.
    Reads directly from the segmented file on disk.
    """
    from tokenizers import (
        Tokenizer, models, trainers,
        pre_tokenizers, normalizers, decoders,
    )
    from tokenizers.pre_tokenizers import Split, Sequence

    output_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()

    # Script Isolator: prevents Tamil+English byte mixing
    # This regex ensures Tamil characters stay grouped together,
    # English stays together, numbers stay together.
    # The BPE can merge WITHIN these groups but NEVER across.
    ISOLATOR = r"[\u0B80-\u0BFF]+|[a-zA-Z]+|[0-9]+|[^\s\u0B80-\u0BFFa-zA-Z0-9]+"

    tok.pre_tokenizer = Sequence([
        # 1. Respect morpheme boundaries from Layer 3
        Split(pattern=" @@ ", behavior="isolated"),
        # 2. Respect script boundaries (Tamil vs English vs Numbers)
        Split(pattern=ISOLATOR, behavior="isolated"),
        # 3. Byte-level encoding (handles any character, like GPT-4)
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # For a 750MB+ corpus, min_frequency=2 creates too many rare pairs.
    # Increasing to 5 significantly speeds up training and reduces RAM usage
    # without any loss in production quality.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=5,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    log.info(f"[Phase 2] Training {vocab_size:,} vocab BPE on {segmented_path.name} ...")
    log.info(f"  RAM optimization: min_frequency=5")
    
    # Final RAM cleanup check before C++ engine takes over
    gc.collect()

    # Native file training - C++ reads file directly
    tok.train([str(segmented_path)], trainer)

    tok.decoder = decoders.ByteLevel()

    # Save
    model_path = output_dir / "tokenizer.json"
    tok.save(str(model_path))

    cfg = {
        "model_type": "gpt2",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": vocab_size,
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log.info(f"  [Phase 2 DONE] Saved to {output_dir}")
    return str(model_path)


# ===================================================================
# Phase 3: Sanity Check
# ===================================================================
def phase3_check(path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    log.info(f"[Phase 3] Sanity Check")
    log.info(f"  Vocab size: {tok.get_vocab_size():,}")

    tests = [
        "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb4\u0b95\u0bbe\u0ba9 \u0bae\u0bca\u0bb4\u0bbf.",
        "\u0bb5\u0bc0\u0b9f\u0bc1\u0b95\u0bb3\u0bbf\u0bb2\u0bbf\u0bb0\u0bc1\u0ba8\u0bcd\u0ba4\u0bc1 \u0bb5\u0ba8\u0bcd\u0ba4\u0bbe\u0ba9\u0bcd.",
        "\u0b87\u0ba8\u0bcd\u0ba4\u0bbf\u0baf \u0b85\u0bb0\u0b9a\u0bbf\u0baf\u0bb2\u0bae\u0bc8\u0baa\u0bcd\u0baa\u0bc1\u0b9a\u0bcd \u0b9a\u0b9f\u0bcd\u0b9f\u0bae\u0bcd.",
        "Machine learning model train \u0baa\u0ba3\u0bcd\u0ba3\u0ba9\u0bc1\u0bae\u0bcd.",
        "\u0b95\u0bb1\u0bcd\u0bb1\u0bc1\u0b95\u0bcd\u0b95\u0bca\u0bb3\u0bcd\u0bb3 \u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bbf\u0baf \u0baa\u0bbe\u0b9f\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0ba8\u0bbf\u0bb1\u0bc8\u0baf \u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bbf\u0ba9\u0bcd\u0bb1\u0ba9.",
    ]
    for s in tests:
        enc = tok.encode(s)
        nw = max(len(s.split()), 1)
        dec = tok.decode(enc.ids)
        ok = "OK" if dec.strip() == s.strip() else "MISMATCH"
        fert = len(enc.tokens) / nw
        log.info(f"  [{ok}] \"{s[:40]}...\" -> {len(enc.tokens)} tokens, fertility {fert:.2f}")

    log.info(f"  [Phase 3 DONE]")


# ===================================================================
# Main
# ===================================================================
def main():
    p = argparse.ArgumentParser(description="Bulletproof AMB Training")
    p.add_argument("--corpus", default="tamil_corpus.txt",
                   help="Path to cleaned Tamil corpus")
    p.add_argument("--max-mb", type=int, default=700,
                   help="Max MB to segment (default 700)")
    p.add_argument("--vocab-size", type=int, default=64000,
                   help="Vocabulary size (default 64000)")
    p.add_argument("--output-dir", default="tokenizer/models/amb_tokenizer",
                   help="Output directory for trained model")
    p.add_argument("--force", action="store_true",
                   help="Force re-segmentation even if cached file exists")
    args = p.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        # Try relative to tokenizer dir
        alt = Path("tokenizer") / args.corpus
        if alt.exists():
            corpus = alt
        else:
            log.error(f"Corpus not found: {corpus}")
            log.error(f"  Also tried: {alt}")
            sys.exit(1)

    # Kaggle/Local Fix: Always save the segmented file in the current working 
    # directory, NOT in the potentially read-only input folder.
    seg = Path(corpus.name).with_suffix(f".seg{args.max_mb}mb.txt")

    log.info("=" * 60)
    log.info("BULLETPROOF AMB TOKENIZER TRAINING")
    log.info("=" * 60)
    log.info(f"  Corpus:     {corpus} ({corpus.stat().st_size / 1048576:.1f} MB)")
    log.info(f"  Subset:     {args.max_mb} MB")
    log.info(f"  Vocab:      {args.vocab_size:,}")
    log.info(f"  Output:     {args.output_dir}")
    log.info("=" * 60)

    # Phase 1: Segment
    if args.force and seg.exists():
        seg.unlink()
        log.info(f"  Deleted old segmented file: {seg}")

    if not seg.exists():
        phase1_segment(corpus, seg, args.max_mb)
    else:
        log.info(f"  Reusing cached: {seg} ({seg.stat().st_size / 1048576:.1f} MB)")

    # Force garbage collection before heavy Phase 2
    gc.collect()

    # Phase 2: Train
    model_path = phase2_train(seg, Path(args.output_dir), args.vocab_size)

    # Phase 3: Verify
    phase3_check(model_path)

    log.info("")
    log.info("=" * 60)
    log.info("TRAINING COMPLETE!")
    log.info("=" * 60)
    log.info(f"  Model: {model_path}")
    log.info(f"  Next:  python tokenizer/evaluate_tokenizer.py --model {model_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
