# train_production.py - The Architecture-Aware AMB Training Pipeline
# ====================================================================
# A consolidated, memory-safe pipeline that trains BOTH:
# 1. AMB Tokenizer (Morpheme-Aware)
# 2. Baseline BPE Tokenizer (Standard GPT-2 style)
# AND benchmarks them against each other automatically.
#
# Usage:
#   python tokenizer/train_production.py --corpus tamil_corpus.txt --vocab-size 48000
#
# Features:
# - Auto-detects RAM (Safety First)
# - Streams data (No OOM)
# - Dual-Training (Scientific comparison)

import os
import sys
import gc
import time
import json
import logging
import argparse
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
            log.warning(f"⚠️ LOW MEMORY. Forcing GC...")
            gc.collect()

monitor = MemoryMonitor(threshold_gb=3.0)


# ===================================================================
# Phase 1: Streaming Segmentation (CPU Heavy)
# ===================================================================
def phase1_segment(corpus_path, output_path, max_mb):
    """Streams data line-by-line using AMB logic."""
    from tamil_unicode import TamilDeepNormalizer
    from morpheme import MorphemeSegmenter

    monitor.check("Start Seg")
    
    norm = TamilDeepNormalizer(
        strip_urls=True, strip_emails=True,
        normalize_numerals="preserve", preserve_grantha=True,
    )
    mseg = MorphemeSegmenter()

    max_bytes = max_mb * 1024 * 1024
    written = 0
    count = 0

    log.info(f"[Phase 1] Segmenting {max_mb} MB ...")
    from tqdm import tqdm
    
    with open(str(corpus_path), "r", encoding="utf-8") as fin, \
         open(str(output_path), "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Segmenting", unit=" lines"):
            if written >= max_bytes: break
            line = line.strip()
            if not line: continue

            # AMB Logic: Normalize -> Split -> Segment with @@
            cleaned = norm.normalize(line)
            words = cleaned.split()
            seg = [mseg.segment_word(w).replace(" ", " @@ ") for w in words]
            
            out = " ".join(seg) + "\n"
            fout.write(out)
            written += len(out.encode("utf-8"))
            count += 1
            
            if count % 100000 == 0: monitor.check(f"Seg {count//1000}k")

    log.info(f"[Phase 1 DONE] {count:,} lines segmented.")
    del norm, mseg
    gc.collect()
    return count

# ===================================================================
# Phase 2: Dual Trainer (RAM Heavy)
# ===================================================================
def train_model(corpus_path, output_dir, vocab_size, model_name, is_amb):
    """Generic trainer for both AMB and Baseline."""
    from tokenizers import (
        Tokenizer, models, trainers,
        pre_tokenizers, normalizers, decoders,
    )
    from tokenizers.pre_tokenizers import Split, Sequence

    log.info(f"--- Training {model_name.upper()} (Vocab: {vocab_size}) ---")
    monitor.check(f"Start {model_name}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()

    if is_amb:
        # AMB: Respects @@ boundaries and Scripts
        SCRIPT_ISOLATOR = r"[\u0B80-\u0BFF]+|[a-zA-Z]+|[0-9]+|[^\s\u0B80-\u0BFFa-zA-Z0-9]+"
        tok.pre_tokenizer = Sequence([
            Split(pattern=" @@ ", behavior="isolated"),
            Split(pattern=SCRIPT_ISOLATOR, behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ])
    else:
        # Baseline: Pure ByteLevel (GPT-2 style)
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=5,  # RAM Optimization
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=32,
    )

    # Use 4 threads to prevent lock-ups
    if "RAYON_NUM_THREADS" not in os.environ:
        os.environ["RAYON_NUM_THREADS"] = "4"

    tok.train([str(corpus_path)], trainer)
    tok.decoder = decoders.ByteLevel()
    
    save_path = output_dir / f"{model_name}.json"
    tok.save(str(save_path))
    
    with open(output_dir / f"{model_name}_config.json", "w") as f:
        json.dump({
            "model_type": "gpt2",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "vocab_size": vocab_size,
        }, f, indent=2)

    monitor.check(f"End {model_name}")
    return str(save_path)

# ===================================================================
# Phase 3: Benchmarking
# ===================================================================
def benchmark(tokenizer_path, test_file, sample_lines=2000):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    word_count = 0
    token_count = 0
    bytes_count = 0
    
    with open(test_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_lines: break
            line = line.strip()
            if not line: continue
            words = line.split()
            word_count += len(words)
            bytes_count += len(line.encode("utf-8"))
            encoded = tok.encode(line)
            token_count += len(encoded.tokens)
            
    return {
        "fertility": token_count / word_count if word_count else 0,
        "compression": bytes_count / token_count if token_count else 0
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="tamil_corpus.txt")
    p.add_argument("--limit-mb", type=int, default=0, help="0 = Auto-detect based on RAM")
    p.add_argument("--vocab-size", type=int, default=48000)
    args = p.parse_args()

    # 1. Auto-Scaling Logic (Safety First)
    if args.limit_mb == 0:
        if psutil:
            total_gb = psutil.virtual_memory().total / (1024**3)
            if total_gb > 28: args.limit_mb = 750   # 30GB Node
            elif total_gb > 14: args.limit_mb = 350 # 16GB Node
            else: args.limit_mb = 150              # 8GB Local
        else:
            args.limit_mb = 150 # Conservative Fallback

    log.info(f"🚀 Starting Production Training (Limit: {args.limit_mb} MB)")
    
    corpus = Path(args.corpus)
    if not corpus.exists():
        log.error("Corpus not found.")
        sys.exit(1)

    # 2. Segment for AMB
    seg_file = Path(f"corpus_amb_{args.limit_mb}mb.txt")
    if not seg_file.exists():
        phase1_segment(corpus, seg_file, args.limit_mb)
    else:
        log.info(f"Reusing segmented file: {seg_file}")

    # 3. Create Baseline Raw File (subset)
    raw_sub = Path(f"corpus_raw_{args.limit_mb}mb.txt")
    if not raw_sub.exists():
        log.info(f"Creating raw subset: {raw_sub}")
        with open(corpus, "r", encoding="utf-8") as fin, open(raw_sub, "w", encoding="utf-8") as fout:
            written = 0
            limit = args.limit_mb * 1024 * 1024
            for line in fin:
                if written >= limit: break
                fout.write(line)
                written += len(line.encode("utf-8"))

    # 4. Train Dual Models
    out_dir = Path("tokenizer/models/production")
    
    # AMB (Trained on Segmented text)
    amb_path = train_model(seg_file, out_dir, args.vocab_size, "tokenizer_amb", is_amb=True)
    
    # Baseline (Trained on Raw text)
    base_path = train_model(raw_sub, out_dir, args.vocab_size, "tokenizer_baseline", is_amb=False)

    # 5. Benchmark
    log.info("--- 📊 FINAL BENCHMARK ---")
    res_amb = benchmark(amb_path, str(raw_sub))
    res_base = benchmark(base_path, str(raw_sub))
    
    print(f"\n{'Metric':<20} | {'Baseline':<10} | {'AMB (Yours)':<10} | {'Delta'}")
    print("-" * 60)
    f_b, f_a = res_base['fertility'], res_amb['fertility']
    print(f"{'Fertility':<20} | {f_b:<10.3f} | {f_a:<10.3f} | {f_a-f_b:.3f}")
    c_b, c_a = res_base['compression'], res_amb['compression']
    print(f"{'Compression':<20} | {c_b:<10.1f} | {c_a:<10.1f} | {c_a-c_b:.1f}")
    print("-" * 60)
    print("Interpretation: Lower Fertility is better. Higher Compression is better.")
    print("If AMB fertility is slightly higher, it means it's splitting linguistically correctly")
    print("instead of aggressively merging phrases.")

if __name__ == "__main__":
    main()
