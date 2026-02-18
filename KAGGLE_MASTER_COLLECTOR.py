#!/usr/bin/env python3
"""
KAGGLE MASTER DATA COLLECTOR
============================
Optimized for Kaggle Notebooks (30GB RAM / High Bandwidth)

This script runs the Tamiluku-LLM data collection suite and
ensures outputs are saved to /kaggle/working/ for easy access.

Usage in Kaggle:
  !git pull
  !python KAGGLE_MASTER_COLLECTOR.py --gb 10
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_kaggle_env():
    print("🚀 Setting up Kaggle environment for Tamiluku-LLM...")
    
    # 1. Install dependencies
    req_path = Path("data_collection/requirements.txt")
    if req_path.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path)], check=True)
    else:
        print("❌ requirements.txt not found!")
        sys.exit(1)

    # 2. Setup output directory
    output_dir = Path("/kaggle/working/tamil_gold_corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def run_collection(output_dir, gb):
    print(f"📥 Starting collection for {gb}GB of Tamil data...")
    
    # Use the orchestrator
    orchestrator_path = Path("data_collection/orchestrator.py")
    
    cmd = [
        sys.executable,
        str(orchestrator_path),
        "--output-dir", str(output_dir),
        "--target-gb", str(gb),
        "--max-workers", "8" # Kaggle has 4-8 cores, we can use 8 threads for I/O
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Collection complete! File is at: {output_dir}/raw_tamil_gold.txt")
    except subprocess.CalledProcessError as e:
        print(f"❌ Collection failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, default=10.0, help="Target size in GB")
    args = parser.parse_args()

    # Detect if running on Kaggle
    is_kaggle = os.path.exists("/kaggle/working")
    
    if is_kaggle:
        target_dir = setup_kaggle_env()
        run_collection(target_dir, args.gb)
    else:
        print("⚠️ Not on Kaggle. Running locally in ./tamil_corpus")
        local_dir = Path("./tamil_corpus")
        local_dir.mkdir(exist_ok=True)
        run_collection(local_dir, args.gb)
