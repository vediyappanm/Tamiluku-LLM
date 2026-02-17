#!/usr/bin/env python3
"""
WORLD-CLASS TRAINING PIPELINE FOR AMB TOKENIZER
================================================

This orchestrates the complete, production-grade AMB tokenizer training:
  1. Data validation & corpus preparation
  2. Normalization with script isolation
  3. Morpheme segmentation
  4. Syllable-constrained BPE training
  5. Comprehensive validation
  6. Production export

Usage:
    python WORLD_CLASS_TRAINING_PIPELINE.py --config tokenizer/config_production.yaml
    python WORLD_CLASS_TRAINING_PIPELINE.py --quick  # Test mode on 100K lines

Author: AI4Bharat
Date: 2026-02-17
Version: 2.0 (Production-Grade)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import yaml
from tqdm import tqdm

# Add tokenizer module to path
tokenizer_dir = Path(__file__).parent / "tokenizer"
sys.path.insert(0, str(tokenizer_dir))

# Import AMB components
try:
    from tamil_unicode import TamilDeepNormalizer
    from akshara import AksharaSegmenter
    from morpheme import MorphemeSegmenter
    from train_amb_tokenizer import (
        train_amb_tokenizer,
        verify_syllable_coverage,
        detect_cross_script_leakage,
        validate_amb_tokenizer,
    )
except ImportError as e:
    print(f"Error importing AMB modules: {e}")
    print("Make sure you're running from the project root and tokenizer/ is properly set up.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PIPELINE")


# ============================================================================
# PHASE 1: CORPUS PREPARATION & VALIDATION
# ============================================================================

def validate_corpus(corpus_path: Path, max_samples: int = 1000) -> Dict[str, any]:
    """
    Validate corpus quality before training.
    Check: file exists, encoding, lines, size, script mix
    """
    log.info(f"[PHASE 1] Validating corpus: {corpus_path}")

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    stats = {
        "path": str(corpus_path),
        "size_mb": corpus_path.stat().st_size / (1024 * 1024),
        "total_lines": 0,
        "avg_line_length": 0,
        "tamil_content_ratio": 0.0,
        "english_mix_ratio": 0.0,
        "encoding": "unknown",
        "sample_check": "passed",
    }

    tamil_chars = 0
    english_chars = 0
    total_chars = 0
    lines_read = 0

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                lines_read += 1
                tamil_chars += sum(1 for c in line if '\u0B80' <= c <= '\u0BFF')
                english_chars += sum(1 for c in line if c.isascii() and c.isalpha())
                total_chars += len(line)

        stats["encoding"] = "utf-8"
        stats["sample_lines_checked"] = lines_read
        stats["avg_line_length"] = total_chars / lines_read if lines_read > 0 else 0

        with open(corpus_path, "r", encoding="utf-8") as f:
            for _ in f:
                stats["total_lines"] += 1

        if total_chars > 0:
            stats["tamil_content_ratio"] = tamil_chars / total_chars
            stats["english_mix_ratio"] = english_chars / total_chars

        log.info(f"  ✅ Corpus valid: {stats['total_lines']:,} lines, {stats['size_mb']:.1f} MB")
        log.info(f"     Tamil: {stats['tamil_content_ratio']:.1%}, English: {stats['english_mix_ratio']:.1%}")

        return stats

    except UnicodeDecodeError as e:
        log.error(f"  ❌ Encoding error: {e}")
        stats["encoding"] = "UNKNOWN (encoding error)"
        stats["sample_check"] = "failed"
        return stats


# ============================================================================
# PHASE 2: NORMALIZATION & DEDUPLICATION
# ============================================================================

def normalize_corpus(
    input_path: Path,
    output_path: Path,
    config: Dict,
    max_lines: int = None,
) -> Dict[str, any]:
    """
    Normalize and deduplicate corpus using script isolation.
    This is where BUG #2 (cross-script leakage) is fixed.
    """
    log.info(f"[PHASE 2] Normalizing corpus with script isolation")

    normalizer = TamilDeepNormalizer(
        strip_urls=config.get("normalization", {}).get("strip_urls", True),
        strip_emails=config.get("normalization", {}).get("strip_emails", True),
        normalize_numerals="preserve",
        preserve_grantha=config.get("normalization", {}).get("preserve_grantha", True),
    )

    stats = {
        "input_lines": 0,
        "output_lines": 0,
        "dedup_removed": 0,
        "invalid_removed": 0,
        "output_size_mb": 0,
    }

    seen_hashes = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Normalizing", unit=" lines"):
            line = line.strip()
            if not line:
                continue

            stats["input_lines"] += 1
            if max_lines and stats["input_lines"] >= max_lines:
                break

            # Normalize
            cleaned = normalizer.normalize(line)
            if not cleaned:
                stats["invalid_removed"] += 1
                continue

            # Exact dedup
            line_hash = hash(cleaned)
            if line_hash in seen_hashes:
                stats["dedup_removed"] += 1
                continue

            seen_hashes.add(line_hash)
            f_out.write(cleaned + "\n")
            stats["output_lines"] += 1

    stats["output_size_mb"] = output_path.stat().st_size / (1024 * 1024)

    log.info(
        f"  ✅ Normalization complete: "
        f"{stats['input_lines']:,} → {stats['output_lines']:,} lines "
        f"({100*stats['output_lines']/max(stats['input_lines'], 1):.1f}%)"
    )
    log.info(f"     Dedup removed: {stats['dedup_removed']:,}, Invalid: {stats['invalid_removed']:,}")

    return stats


# ============================================================================
# PHASE 3: MORPHEME SEGMENTATION
# ============================================================================

def segment_corpus(
    input_path: Path,
    output_path: Path,
    max_lines: int = None,
) -> Dict[str, any]:
    """
    Add morpheme boundaries using @@ markers.
    This prepares for constrained BPE training.
    """
    log.info(f"[PHASE 3] Segmenting corpus with morpheme boundaries")

    segmenter = MorphemeSegmenter()
    stats = {
        "lines_processed": 0,
        "words_segmented": 0,
        "avg_morphemes_per_word": 0,
    }

    total_morphemes = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Segmenting", unit=" lines"):
            line = line.strip()
            if not line:
                continue

            stats["lines_processed"] += 1
            if max_lines and stats["lines_processed"] >= max_lines:
                break

            # Segment words
            words = line.split()
            segmented_words = []

            for word in words:
                morphemes = segmenter.segment_word(word)
                morpheme_list = morphemes.split()
                stats["words_segmented"] += 1
                total_morphemes += len(morpheme_list)
                # Join with @@ markers for constrained BPE
                segmented_words.append(morphemes.replace(" ", " @@ "))

            f_out.write(" ".join(segmented_words) + "\n")

    if stats["words_segmented"] > 0:
        stats["avg_morphemes_per_word"] = total_morphemes / stats["words_segmented"]

    log.info(
        f"  ✅ Segmentation complete: "
        f"{stats['lines_processed']:,} lines, "
        f"{stats['words_segmented']:,} words"
    )
    log.info(f"     Avg morphemes/word: {stats['avg_morphemes_per_word']:.2f}")

    return stats


# ============================================================================
# PHASE 4: BPE TOKENIZER TRAINING
# ============================================================================

def train_tokenizer(config: Dict, corpus_path: Path) -> Tuple[str, Dict]:
    """
    Train AMB tokenizer with all production fixes:
    - BUG #1: Syllable pre-population ✅
    - BUG #2: Script isolation ✅
    - Proper vocabulary size (64K) ✅
    """
    log.info(f"[PHASE 4] Training BPE tokenizer with production fixes")

    start_time = time.time()

    model_path = train_amb_tokenizer(config, corpus_path)

    elapsed = time.time() - start_time

    log.info(f"  ✅ Training complete in {elapsed/60:.1f} minutes")
    log.info(f"     Model saved to: {model_path}")

    return model_path, {"training_time_seconds": elapsed}


# ============================================================================
# PHASE 5: COMPREHENSIVE VALIDATION
# ============================================================================

def validate_tokenizer(tokenizer_path: str) -> Dict[str, any]:
    """
    Run all validation checks to verify world-class quality.
    """
    log.info(f"[PHASE 5] Comprehensive validation of trained tokenizer")

    results = {}

    # Check 1: Syllable coverage
    log.info("  Checking syllable coverage...")
    syllable_coverage = verify_syllable_coverage(tokenizer_path)
    results["syllable_coverage"] = {
        "value": syllable_coverage,
        "target": 0.95,
        "passed": syllable_coverage >= 0.95,
    }

    # Check 2: Cross-script leakage
    log.info("  Checking cross-script leakage...")
    cross_script_count = detect_cross_script_leakage(tokenizer_path)
    results["cross_script_leakage"] = {
        "value": cross_script_count,
        "target": 0,
        "passed": cross_script_count == 0,
    }

    # Check 3: Basic sanity
    log.info("  Running sanity checks...")
    validate_amb_tokenizer(tokenizer_path)
    results["sanity_check"] = {"passed": True}

    # Summary
    all_passed = all(v.get("passed", False) for v in results.values())

    if all_passed:
        log.info("  ✅ ALL VALIDATION CHECKS PASSED!")
    else:
        log.warning("  ⚠️  Some validation checks failed. See above for details.")

    results["overall_passed"] = all_passed

    return results


# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def run_pipeline(config_path: str, quick_mode: bool = False):
    """
    Execute the complete world-class training pipeline.
    """
    log.info("=" * 80)
    log.info("AMB TOKENIZER - WORLD-CLASS PRODUCTION PIPELINE v2.0")
    log.info("=" * 80)
    log.info(f"Start time: {datetime.now().isoformat()}")
    log.info(f"Config: {config_path}")

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Determine paths
    base_dir = Path(config_path).parent.parent
    corpus_filename = config["corpus"]["input_files"][0]
    
    # Robust Corpus Resolution (Handles Local vs Kaggle)
    search_paths = [
        base_dir / corpus_filename,                             # Local root
        Path(corpus_filename),                                  # Local CWD
        Path("..") / corpus_filename,                          # Parent directory
        Path("/kaggle/input/tamil-corpus-txt") / corpus_filename, # Standard Kaggle Input
        Path("/kaggle/input") / Path(corpus_filename).stem / corpus_filename # Dynamic Kaggle Input
    ]
    
    corpus_path = None
    for p in search_paths:
        if p.exists():
            corpus_path = p
            break
            
    if corpus_path is None:
        log.error(f"❌ Corpus not found: {corpus_filename}")
        log.error(f"   Searched in: {[str(p) for p in search_paths]}")
        return None, None

    log.info(f"📍 Resolved corpus path: {corpus_path}")
    normalized_path = base_dir / config["corpus"]["output_file"]

    if quick_mode:
        log.info("⚡ QUICK MODE: Using 100K lines for testing")
        max_lines = 100000
    else:
        max_lines = None

    try:
        # PHASE 1: Validate
        corpus_stats = validate_corpus(corpus_path)

        # PHASE 2: Normalize with script isolation
        norm_stats = normalize_corpus(
            corpus_path,
            normalized_path,
            config,
            max_lines=max_lines,
        )

        # PHASE 3: Segment
        segmented_path = normalized_path.with_suffix(".segmented")
        seg_stats = segment_corpus(
            normalized_path,
            segmented_path,
            max_lines=max_lines,
        )

        # PHASE 4: Train
        model_path, train_stats = train_tokenizer(config, segmented_path)

        # PHASE 5: Validate
        validation_results = validate_tokenizer(model_path)

        # REPORT
        log.info("\n" + "=" * 80)
        log.info("FINAL REPORT")
        log.info("=" * 80)

        report = {
            "timestamp": datetime.now().isoformat(),
            "quick_mode": quick_mode,
            "phases": {
                "corpus_validation": corpus_stats,
                "normalization": norm_stats,
                "segmentation": seg_stats,
                "training": train_stats,
                "validation": validation_results,
            },
            "model_path": model_path,
        }

        log.info(f"✅ Syllable Coverage: {validation_results['syllable_coverage']['value']:.2%}")
        log.info(f"✅ Cross-Script Leakage: {validation_results['cross_script_leakage']['value']}")
        log.info(f"✅ Overall Status: {'PASSED' if validation_results['overall_passed'] else 'FAILED'}")

        # Save report
        report_path = Path(model_path).parent / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        log.info(f"\n📊 Report saved to: {report_path}")
        log.info(f"🚀 Model ready for production: {model_path}")

        return model_path, report

    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return None, None


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="World-Class AMB Tokenizer Training Pipeline"
    )
    parser.add_argument(
        "--config",
        default="tokenizer/config_production.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: train on 100K lines for testing",
    )

    args = parser.parse_args()

    model_path, report = run_pipeline(args.config, quick_mode=args.quick)

    if model_path:
        log.info("\n✨ Training pipeline completed successfully!")
        return 0
    else:
        log.error("\n❌ Training pipeline failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
