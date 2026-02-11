"""
pilot_train_and_val.py — 673MB Pilot Validation Run
===================================================
1. Extracts text from tawiki-latest.xml (673 MB)
2. Normalizes text
3. Trains AMB Tokenizer (Vocab 16,000)
4. Runs full evaluation
"""

import os
import subprocess
import yaml
from pathlib import Path

def run_step(cmd, desc):
    print(f"\n🚀 [Pilot Step] {desc}")
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {desc}: {e}")
        return False
    return True

def main():
    # 0. Config Setup
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Modify config for pilot
    cfg["tokenizer"]["vocab_size"] = 16000
    cfg["corpus"]["sources"]["wikipedia"]["enabled"] = True
    
    with open("config_pilot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # 1. Collection (Extract Wiki)
    xml_path = Path("data/raw/tawiki-latest.xml")
    if not xml_path.exists():
        print("❌ Error: tawiki-latest.xml not found in data/raw/")
        return

    wiki_txt = Path("data/raw/wikipedia_ta.txt")
    if wiki_txt.exists() and wiki_txt.stat().st_size > 100 * 1024 * 1024:
        print(f"⏭️ Skipping Extraction: {wiki_txt} already exists ({wiki_txt.stat().st_size/1e6:.1f} MB)")
    else:
        if not run_step("python collect_corpus.py --config config_pilot.yaml --sources wikipedia", "Extracting Wikipedia Articles"):
            return

    # 2. Normalization
    clean_txt = Path("data/cleaned/tamil_corpus.txt")
    if clean_txt.exists() and clean_txt.stat().st_size > 100 * 1024 * 1024:
        print(f"⏭️ Skipping Normalization: {clean_txt} already exists ({clean_txt.stat().st_size/1e6:.1f} MB)")
    else:
        if not run_step("python normalize.py --config config_pilot.yaml --workers 4", "Normalizing Corpus"):
            return

    # 3. Training
    if not run_step("python train_tokenizer.py --config config_pilot.yaml --engine amb", "Training AMB Tokenizer (16K Vocab)"):
        return

    # 4. Evaluation
    if not run_step("python evaluate_tokenizer.py --engine amb --config config_pilot.yaml", "Running Full Evaluation"):
        return

    print("\n✅ Pilot Phased Rollout (Phase 1) Completed!")
    print("Check reports/eval_report.json for results.")

if __name__ == "__main__":
    main()
