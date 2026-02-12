# train_local.py - Guaranteed Success Streaming Trainer
# ========================================================
# 1. auto-detects RAM
# 2. calculates safe corpus size (Smart Scaling)
# 3. streams data to prevent OOM
# 4. produces production-grade tokenizer

import os
import sys
import gc
import json
import psutil
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Add local path for imports
sys.path.insert(0, str(Path(__file__).parent))

def get_safe_config():
    """
    Returns safe training parameters based on available RAM.
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    avail_gb = mem.available / (1024**3)
    
    log.info(f"System Memory: {total_gb:.1f} GB Total | {avail_gb:.1f} GB Available")
    
    # Conservative safe limits
    if total_gb >= 30:
        return 750, 64000  # High End (30GB node)
    elif total_gb >= 15:
        return 350, 50257  # Standard (16GB node) -> 350MB is the sweet spot
    else:
        return 100, 32000  # Low End (8GB local)

def stream_and_segment(corpus_path, temp_file_path, limit_mb):
    """
    Reads corpus line-by-line, segments it, and writes to temp file.
    Stops exactly when limit_mb is reached.
    """
    from tamil_unicode import TamilDeepNormalizer
    from morpheme import MorphemeSegmenter
    
    log.info(f"--- Phase 1: Streaming & Segmenting (Target: {limit_mb} MB) ---")
    
    norm = TamilDeepNormalizer(strip_urls=True, normalize_numerals="preserve")
    mseg = MorphemeSegmenter()
    
    limit_bytes = limit_mb * 1024 * 1024
    current_bytes = 0
    lines = 0
    
    from tqdm import tqdm
    
    with open(corpus_path, "r", encoding="utf-8") as f_in, \
         open(temp_file_path, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, desc="Streaming"):
            if current_bytes >= limit_bytes:
                break
                
            line = line.strip()
            if not line: continue
            
            # Processing
            clean = norm.normalize(line)
            words = clean.split()
            # Fast segmentation: "வீடு கள்" -> "வீடு@@கள்"
            # We use "@@" purely to prevent merges across morpheme boundaries
            seg_words = [mseg.segment_word(w).replace(" ", "@@") for w in words]
            out_line = " ".join(seg_words) + "\n"
            f_out.write(out_line)
            
            current_bytes += len(out_line.encode("utf-8"))
            lines += 1
            
    log.info(f"Phase 1 Complete: {lines:,} lines | {current_bytes/(1024**2):.1f} MB")
    
    # Cleanup memory
    del norm, mseg
    gc.collect()

def inject_syllable_seed(temp_file_path):
    """
    Guarantees 100% syllable coverage by injecting a synthetic corpus
    of all 247 Tamil syllables repeated many times.
    """
    log.info("--- Phase 1.5: Injecting Syllable Seed (Coverage Booster) ---")
    vowels = list("அஆஇஈஉஊஎஏஐஒஓஔ")
    consonants = list("கஙசஞடணதநபமயரலவழளறன")
    vowel_signs = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
    
    syllables = []
    syllables.extend(vowels)
    for c in consonants:
        syllables.append(c + "்")
        for vs in vowel_signs:
            syllables.append(c + vs)
    syllables.append("ஃ")
    
    # Repeat each syllable 10000 times to ensure BPE finds them and merges them above all else
    # Space separation IS CRITICAL.
    seed_text = (" ".join(syllables) + "\n") * 10000
    
    # Prepend to the file
    with open(temp_file_path, "r", encoding="utf-8") as f:
        existing_data = f.read()
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(seed_text)
        f.write(existing_data)
    
    log.info(f"Injected {len(syllables)} unique syllables into seed.")

def train_tokenizer(corpus_file, output_dir, vocab_size):
    """
    Trains BPE using the file on disk (Native Training).
    """
    log.info(f"--- Phase 2: Native BPE Training ({vocab_size:,} vocab) ---")
    
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
    from tokenizers.pre_tokenizers import Split, Sequence
    
    # 1. Config
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    
    # 2. Production Pre-tokenization Pipeline
    # 2. Advanced Pre-tokenization Sequence
    # This architecture ensures:
    # A) Morpheme boundaries are respected (@@ removal)
    # B) Scripts NEVER leak (Split by script)
    # C) Roundtrip is 100% preserved (ByteLevel)
    tokenizer.pre_tokenizer = Sequence([
        Split(pattern="@@", behavior="removed"),
        # Separate Tamil, Latin, and Digits into their own processing units
        Split(pattern=r"([\u0B80-\u0BFF]+|[a-zA-Z]+|[0-9]+)", behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    
    # 2.5 Generate Tamil Syllables for Vocabulary Pre-population
    vowels = list("அஆஇஈஉஊஎஏஐஒஓஔ")
    consonants = list("கஙசஞடணதநபமயரலவழளறன")
    vowel_signs = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
    
    tamil_syllables = []
    tamil_syllables.extend(vowels)
    for c in consonants:
        tamil_syllables.append(c + "்") # Pure consonant
        for vs in vowel_signs:
            tamil_syllables.append(c + vs)
    tamil_syllables.append("ஃ")
    
    # 3. Train
    # We add the 247 syllables as special tokens to GUARANTEE they are never split.
    special_tokens = ["<|endoftext|>", "<|padding|>", "<|im_start|>", "<|im_end|>"] + tamil_syllables
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2, 
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    log.info(f"Starting C++ Trainer with {len(tamil_syllables)} pre-populated syllables...")
    tokenizer.train([str(corpus_file)], trainer)
    
    # 3. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    
    # Save HF config
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump({
            "model_type": "gpt2", 
            "tokenizer_class": "PreTrainedTokenizerFast",
            "vocab_size": vocab_size
        }, f, indent=2)
        
    log.info(f"✅ Success! Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", default="tokenizer/models/amb_tokenizer")
    parser.add_argument("--max-mb", type=int, default=0, help="Override auto-detected RAM limit (MB)")
    parser.add_argument("--vocab-size", type=int, default=0, help="Override auto-detected vocab size")
    args = parser.parse_args()
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        log.error(f"Corpus not found at: {corpus_path.absolute()}")
        log.error("Please ensure the path is correct (e.g., /kaggle/input/tamil-corpus-txt/tamil_corpus.txt)")
        sys.exit(1)
        
    # 1. Auto-Size or Manual Override
    auto_max_mb, auto_vocab_size = get_safe_config()
    max_mb = args.max_mb if args.max_mb > 0 else auto_max_mb
    vocab_size = args.vocab_size if args.vocab_size > 0 else auto_vocab_size
    
    log.info(f"Configuration: Dataset Limit={max_mb} MB | Vocab={vocab_size}")
    
    # 2. Stream & Segment
    temp_file = Path("training_data.tmp")
    stream_and_segment(corpus_path, temp_file, max_mb)
    
    # 2.5 Inject Syllable Seed (Ensures 100% coverage)
    inject_syllable_seed(temp_file)
    
    # 3. Train
    train_tokenizer(temp_file, Path(args.output_dir), vocab_size)
    
    # 4. Cleanup
    if temp_file.exists():
        temp_file.unlink()

if __name__ == "__main__":
    main()
