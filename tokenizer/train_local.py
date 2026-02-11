# train_local.py - World-Class AMB Training (30GB RAM Optimized)
# =================================================================
# Validated for Kaggle P100/T4 instances (30GB RAM).
# Implements streaming, memory monitoring, and aggressive GC.
#
# Usage:
#   python tokenizer/train_local.py --corpus tamil_corpus.txt --max-mb 750 --vocab-size 64000

import os
import sys
import gc
import time
import json
import logging
import argparse
import shutil
from pathlib import Path

# Try importing psutil for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Special tokens
_ST = ["endoftext", "padding", "im_start", "im_end"]
SPECIAL_TOKENS = ["<|" + t + "|>" for t in _ST]


class MemoryMonitor:
    """Active memory guardian."""
    def __init__(self, threshold_gb=5.0):
        self.threshold_gb = threshold_gb
        
    def check(self, stage=""):
        if not psutil: return
        
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        used_gb = (mem.total - mem.available) / (1024**3)
        
        log.info(f"[{stage}] RAM: {used_gb:.1f}/{total_gb:.1f} GB (Free: {available_gb:.1f} GB)")
        
        if available_gb < self.threshold_gb:
            log.warning(f"⚠️ LOW MEMORY (<{self.threshold_gb}GB). Forcing GC...")
            gc.collect()
            time.sleep(1)
            
            # Recheck
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            log.info(f"  -> Reclaimed. New Free: {available_gb:.1f} GB")

monitor = MemoryMonitor(threshold_gb=5.0)


# ===================================================================
# Phase 1: Streaming Segmentation (CPU Heavy, RAM Light)
# ===================================================================
def phase1_segment(corpus_path, output_path, max_mb):
    """
    Stream corpus line-by-line. Never loads file into RAM.
    """
    from tamil_unicode import TamilDeepNormalizer
    from morpheme import MorphemeSegmenter

    monitor.check("Start Phase 1")
    
    norm = TamilDeepNormalizer(
        strip_urls=True, strip_emails=True,
        normalize_numerals="preserve", preserve_grantha=True,
    )
    mseg = MorphemeSegmenter()

    max_bytes = max_mb * 1024 * 1024
    written = 0
    count = 0

    log.info(f"filesize: {corpus_path.stat().st_size / (1024**2):.1f} MB")
    log.info(f"[Phase 1] Segmenting {max_mb} MB ...")

    from tqdm import tqdm
    start_time = time.time()

    with open(str(corpus_path), "r", encoding="utf-8") as fin, \
         open(str(output_path), "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Segmenting", unit=" lines"):
            if written >= max_bytes:
                break
            line = line.strip()
            if not line:
                continue

            # Layer 1 & 3
            cleaned = norm.normalize(line)
            words = cleaned.split()
            seg = [mseg.segment_word(w).replace(" ", " @@ ") for w in words]
            
            out = " ".join(seg) + "\n"
            fout.write(out)
            written += len(out.encode("utf-8"))
            count += 1

            if count % 100000 == 0:
                monitor.check(f"Seg {count//1000}k")

    elapsed = time.time() - start_time
    log.info(f"[Phase 1 DONE] {count:,} lines in {elapsed/60:.1f} min")
    
    # Cleanup heavy objects
    del norm, mseg
    gc.collect()
    monitor.check("End Phase 1")
    return count


# ===================================================================
# Phase 2: BPE Training (RAM Heavy)
# ===================================================================
def phase2_train(segmented_path, output_dir, vocab_size):
    """
    Uses optimized settings for 30GB RAM.
    """
    from tokenizers import (
        Tokenizer, models, trainers,
        pre_tokenizers, normalizers, decoders,
    )
    from tokenizers.pre_tokenizers import Split, Sequence

    monitor.check("Start Phase 2")
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()

    # Script Isolator
    ISOLATOR = r"[\u0B80-\u0BFF]+|[a-zA-Z]+|[0-9]+|[^\s\u0B80-\u0BFFa-zA-Z0-9]+"

    tok.pre_tokenizer = Sequence([
        Split(pattern=" @@ ", behavior="isolated"),
        Split(pattern=ISOLATOR, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # Limit threads to avoid thread contention stall
    os.environ["RAYON_NUM_THREADS"] = "8"  # Optimized for Kaggle 4-core runtimes (2 threads/core)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=5,  # Crucial for RAM optimization
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=32, # Prevent ultra-long garbage tokens
    )

    log.info(f"[Phase 2] Training {vocab_size:,} vocab on {segmented_path.name}")
    log.info(f"  Settings: min_freq=5, threads=8, max_len=32")
    
    gc.collect() # Final purge
    
    # Native training (Rust)
    tok.train([str(segmented_path)], trainer)

    tok.decoder = decoders.ByteLevel()
    
    # Save
    model_path = output_dir / "tokenizer.json"
    tok.save(str(model_path))
    
    # Config
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump({
            "model_type": "gpt2",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "vocab_size": vocab_size,
        }, f, indent=2)

    monitor.check("End Phase 2")
    return str(model_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="tamil_corpus.txt")
    p.add_argument("--max-mb", type=int, default=750)
    p.add_argument("--vocab-size", type=int, default=64000)
    p.add_argument("--output-dir", default="tokenizer/models/amb_tokenizer")
    args = p.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        log.error(f"Corpus not found: {corpus}")
        sys.exit(1)

    # Auto-detect RAM to scale parameters if needed
    if psutil:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        log.info(f"Detected System RAM: {total_gb:.1f} GB")
        
        if total_gb < 20 and args.max_mb > 500:
            log.warning(f"⚠️ <20GB RAM detected. Downgrading max-mb to 400MB for safety.")
            args.max_mb = 400
            args.vocab_size = 50257

    # Phase 1: Segment
    seg_file = Path(f"tamil_corpus.seg{args.max_mb}mb.txt")
    
    if seg_file.exists():
         log.info(f"Reusing existing segment file: {seg_file}")
    else:
         phase1_segment(corpus, seg_file, args.max_mb)

    # Phase 2: Train
    phase2_train(seg_file, Path(args.output_dir), args.vocab_size)
    
    log.info("="*60)
    log.info(f"✅ SUCCESS! Production tokenizer ready.")
    log.info("="*60)

if __name__ == "__main__":
    main()
