"""
local_8gb_train.py - Memory-Optimized Local Training
====================================================
Designed to run on 8GB RAM. 
Uses a 100MB sample of the corpus for speed and safety.
"""

import os
import sys
from pathlib import Path

def create_local_sample(source_file, target_file, size_mb=100):
    print(f"Creating {size_mb}MB sample from {source_file}...")
    target_bytes = size_mb * 1024 * 1024
    
    with open(source_file, 'r', encoding='utf-8') as f_in, \
         open(target_file, 'w', encoding='utf-8') as f_out:
        current_bytes = 0
        for line in f_in:
            f_out.write(line)
            current_bytes += len(line.encode('utf-8'))
            if current_bytes >= target_bytes:
                break
    
    print(f"✅ Sample created: {target_file}")

def main():
    root = Path("c:/Users/ELCOT/OneDrive/Documents/Tamiluku-LLM")
    source_corpus = root / "tamil_corpus.txt"
    local_data_dir = root / "data/local_test"
    local_data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_corpus = local_data_dir / "sample_100mb.txt"
    
    # Check if source exists
    if not source_corpus.exists():
        print(f"❌ Source corpus not found at {source_corpus}")
        print("Please ensure tamil_corpus.txt is in the root directory.")
        return

    # Create 100MB sample if not exists
    if not sample_corpus.exists():
        create_local_sample(source_corpus, sample_corpus)

    print("\n" + "="*70)
    print("STARTING 8GB MEMORY-OPTIMIZED TRAINING")
    print("="*70)
    print(f"  Vocab Size: 32,000")
    print(f"  RAM Target: < 4GB")
    print(f"  Syllable Shield: ENABLED")
    
    # Run training
    # We use a 32k vocab which is perfect for 8GB RAM and local testing
    train_cmd = (
        f"python tokenizer/train_amb_tokenizer.py "
        f"--corpus {sample_corpus} "
        f"--vocab-size 32000"
    )
    
    print(f"\nRunning command: {train_cmd}")
    os.system(train_cmd)
    
    print("\n" + "="*70)
    print("TRAINING FINISHED")
    print("="*70)
    print("\nNow running evaluation on the local model...")
    
    eval_cmd = "python tokenizer/evaluate_tokenizer.py --model tokenizer/models/amb_tokenizer/tokenizer.json --engine amb"
    os.system(eval_cmd)

if __name__ == "__main__":
    main()
