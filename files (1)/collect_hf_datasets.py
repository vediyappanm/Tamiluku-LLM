"""
Tamiluku-LLM: Large-Scale Tamil Dataset Collector (HuggingFace)
=================================================================
Domain 4: Colloquial & Social + Large Web Corpora

Sources (via HuggingFace datasets):
  1. CulturaX (uonlp/CulturaX) — 6.3T tokens, Tamil slice ~50GB+
     THE gold standard for multilingual LLM pretraining.
  2. IndicCorp V2 (ai4bharat/IndicCorpV2) — Curated Indian language corpus
     High-quality, deduplicated Tamil text from multiple domains.
  3. OSCAR (oscar-corpus/OSCAR-2301) — Common Crawl filtered
     Large-scale web text, needs careful filtering.
  4. Sangraha (ai4bharat/sangraha) — Latest AI4Bharat corpus
  5. mc4 Tamil — Google's mC4 Tamil split

Why this matters for AMB:
  These datasets capture MODERN AGGLUTINATION patterns —
  colloquial Tamil, social media style, and informal writing
  that standard dictionaries don't cover. Essential for the
  BPE vocabulary to learn real-world token distributions.

CRITICAL: These datasets are multi-terabyte.
  We STREAM them and extract only the Tamil portions.
  Target: ~6GB from these sources combined.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    clean_and_filter_text, save_clean_text,
    get_file_size_gb, logger
)


# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASET_CONFIGS = {
    "culturax": {
        "name": "CulturaX (Tamil)",
        "hf_path": "uonlp/CulturaX",
        "hf_config": "ta",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "target_gb": 3.0,
        "description": "Gold standard multilingual corpus. Tamil slice.",
        "trust_remote_code": True,
    },
    "indic_corp_v2": {
        "name": "IndicCorp V2 (Tamil)",
        "hf_path": "ai4bharat/IndicCorpV2",
        "hf_config": "ta",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "target_gb": 2.0,
        "description": "AI4Bharat curated Indian language corpus.",
        "trust_remote_code": True,
    },
    "oscar_tamil": {
        "name": "OSCAR 2301 (Tamil)",
        "hf_path": "oscar-corpus/OSCAR-2301",
        "hf_config": "ta",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "target_gb": 2.0,
        "description": "Common Crawl filtered for Tamil.",
        "trust_remote_code": True,
    },
    "sangraha": {
        "name": "Sangraha (Tamil)",
        "hf_path": "ai4bharat/sangraha",
        "hf_config": "verified/tam",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "target_gb": 1.5,
        "description": "AI4Bharat's latest verified corpus.",
        "trust_remote_code": True,
    },
    "mc4_tamil": {
        "name": "mC4 Tamil",
        "hf_path": "mc4",
        "hf_config": "ta",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "target_gb": 2.0,
        "description": "Google's mC4 multilingual corpus, Tamil split.",
        "trust_remote_code": True,
    },
}


# ============================================================
# STREAMING DATASET COLLECTOR
# ============================================================

class HFDatasetCollector:
    """
    Streams HuggingFace datasets and extracts clean Tamil text.

    Key design decisions:
    - STREAMING mode: Never loads full dataset into RAM
    - Target-based: Stops when reaching target GB
    - Checkpoint: Saves progress every 10K documents
    - Resumable: Checks existing file size before starting
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_dataset(self, config_key: str,
                        config: dict,
                        min_tamil_ratio: float = 0.60) -> Optional[Path]:
        """
        Stream a single HuggingFace dataset and extract Tamil text.
        """
        from datasets import load_dataset

        output_file = self.output_dir / f"{config_key}.txt"
        target_gb = config.get('target_gb', 2.0)

        # Check if we already have enough data
        current_gb = get_file_size_gb(output_file)
        if current_gb >= target_gb:
            logger.info(
                f"Skipping {config['name']}: already have "
                f"{current_gb:.2f} GB (target: {target_gb} GB)"
            )
            return output_file

        logger.info(f"Loading {config['name']} (streaming mode)...")
        logger.info(f"  HF Path: {config['hf_path']}")
        logger.info(f"  Config: {config.get('hf_config', 'default')}")
        logger.info(f"  Target: {target_gb} GB")

        try:
            # Load with streaming
            load_kwargs = {
                'path': config['hf_path'],
                'split': config.get('split', 'train'),
                'streaming': True,
            }

            if config.get('hf_config'):
                load_kwargs['name'] = config['hf_config']

            if config.get('trust_remote_code'):
                load_kwargs['trust_remote_code'] = True

            dataset = load_dataset(**load_kwargs)

            text_field = config.get('text_field', 'text')
            total_bytes = int(current_gb * (1024 ** 3))  # Resume from existing
            processed = 0
            skipped = 0
            checkpoint_interval = 10000

            with tqdm(desc=config['name'], unit=' docs',
                      postfix={'GB': f"{current_gb:.2f}/{target_gb}"}) as pbar:

                for doc in dataset:
                    # Check target
                    current_gb = total_bytes / (1024 ** 3)
                    if current_gb >= target_gb:
                        logger.info(f"Reached target {target_gb} GB for {config['name']}")
                        break

                    # Extract text
                    text = doc.get(text_field, '')
                    if not text or len(text) < 50:
                        skipped += 1
                        pbar.update(1)
                        continue

                    # Clean and filter
                    cleaned = clean_and_filter_text(text, min_tamil_ratio=min_tamil_ratio)

                    if len(cleaned) > 100:
                        bytes_written = save_clean_text(
                            cleaned + '\n\n',
                            output_file
                        )
                        total_bytes += bytes_written
                        processed += 1
                    else:
                        skipped += 1

                    pbar.update(1)
                    if processed % checkpoint_interval == 0:
                        current_gb = total_bytes / (1024 ** 3)
                        pbar.set_postfix({
                            'GB': f"{current_gb:.2f}/{target_gb}",
                            'kept': processed,
                            'skip': skipped,
                        })

            final_gb = total_bytes / (1024 ** 3)
            logger.info(
                f"{config['name']}: {processed} docs kept, "
                f"{skipped} skipped, {final_gb:.2f} GB total"
            )
            return output_file

        except Exception as e:
            logger.error(f"Failed to collect {config['name']}: {e}")
            logger.error(f"  This may require: huggingface-cli login")
            logger.error(f"  Some datasets need HF access approval.")
            return None

    def collect_all(self,
                    datasets: list[str] = None,
                    total_target_gb: float = 8.0) -> list[Path]:
        """
        Collect from all configured datasets.

        Args:
            datasets: List of config keys to collect (None = all)
            total_target_gb: Stop when total exceeds this
        """
        output_files = []
        total_collected_gb = 0.0

        configs_to_process = datasets or list(DATASET_CONFIGS.keys())

        for config_key in configs_to_process:
            if config_key not in DATASET_CONFIGS:
                logger.warning(f"Unknown dataset config: {config_key}")
                continue

            if total_collected_gb >= total_target_gb:
                logger.info(f"Total target {total_target_gb} GB reached. Stopping.")
                break

            config = DATASET_CONFIGS[config_key]

            logger.info("=" * 60)
            logger.info(f"DATASET: {config['name']}")
            logger.info(f"  {config['description']}")
            logger.info("=" * 60)

            output_file = self.collect_dataset(config_key, config)
            if output_file and output_file.exists():
                file_gb = get_file_size_gb(output_file)
                total_collected_gb += file_gb
                output_files.append(output_file)
                logger.info(f"  Running total: {total_collected_gb:.2f} GB")

        logger.info(f"\nTotal collected from HF datasets: {total_collected_gb:.2f} GB")
        return output_files


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def collect_hf_datasets(output_dir: Path,
                        total_target_gb: float = 8.0,
                        datasets: list[str] = None) -> list[Path]:
    """
    Entry point for HuggingFace dataset collection.

    Priority order (by quality/relevance):
    1. CulturaX — Highest quality, deduplicated
    2. IndicCorp V2 — Curated for Indian languages
    3. Sangraha — Latest AI4Bharat
    4. OSCAR — Large but needs more filtering
    5. mC4 — Fallback for volume
    """
    output_dir = Path(output_dir) / "hf_datasets"
    collector = HFDatasetCollector(output_dir)

    # Priority order
    priority_order = datasets or [
        "culturax",
        "indic_corp_v2",
        "sangraha",
        "oscar_tamil",
        "mc4_tamil",
    ]

    return collector.collect_all(
        datasets=priority_order,
        total_target_gb=total_target_gb,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Collect Tamil data from HuggingFace datasets"
    )
    parser.add_argument("--output-dir", type=str, default="./tamil_corpus")
    parser.add_argument("--target-gb", type=float, default=8.0,
                        help="Target corpus size in GB")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Specific datasets to collect")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Tamil Datasets:")
        print("=" * 60)
        for key, config in DATASET_CONFIGS.items():
            print(f"  {key:20s} | {config['name']}")
            print(f"  {'':20s} | Target: {config['target_gb']} GB")
            print(f"  {'':20s} | {config['description']}")
            print()
        sys.exit(0)

    files = collect_hf_datasets(
        Path(args.output_dir),
        total_target_gb=args.target_gb,
        datasets=args.datasets,
    )
    print(f"\nCollected files: {[str(f) for f in files]}")
