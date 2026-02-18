#!/usr/bin/env python3
"""
Tamiluku-LLM: Master Data Collection Orchestrator
====================================================

Single-command pipeline to collect 10GB of high-quality Tamil text
for training the AMB (Akshara-Morpheme-BPE) tokenizer.

Usage:
  # Full pipeline (all 5 domains)
  python orchestrator.py --output-dir ./tamil_corpus --target-gb 10

  # Skip specific domains
  python orchestrator.py --skip news legal

  # Only HuggingFace datasets (fastest, largest volume)
  python orchestrator.py --only hf

  # Dry run (show what would be collected)
  python orchestrator.py --dry-run

Architecture:
  ┌─────────────────────────────────────────────┐
  │           ORCHESTRATOR (this file)           │
  ├─────────┬─────────┬─────────┬───────┬───────┤
  │Classical│  News   │  Wiki   │  HF   │ Legal │
  │ ~500MB  │ ~1.5GB  │ ~800MB  │ ~6GB  │~500MB │
  ├─────────┴─────────┴─────────┴───────┴───────┤
  │              MERGE + DEDUP                   │
  │           raw_tamil_gold.txt                 │
  │              (~10 GB target)                 │
  └─────────────────────────────────────────────┘

Requirements:
  - Python 3.8+
  - ~30GB free disk space (raw + processed)
  - ~4GB RAM minimum (streaming mode)
  - Internet connection
  - HuggingFace account (for gated datasets): huggingface-cli login
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_file_size_gb, estimate_corpus_stats, logger


# ============================================================
# DOMAIN COLLECTORS
# ============================================================

DOMAINS = {
    "classical": {
        "name": "Classical & Literary Tamil",
        "module": "collectors.collect_classical",
        "function": "collect_classical",
        "est_size": "~500 MB",
        "description": "Project Madurai, Wikisource — pure Tamil roots",
        "priority": 1,
    },
    "wiki": {
        "name": "Tamil Wikipedia",
        "module": "collectors.collect_wiki",
        "function": "collect_wiki",
        "est_size": "~800 MB",
        "description": "Wikipedia dump — technical terms, compound words",
        "priority": 2,
    },
    "news": {
        "name": "News & Formal",
        "module": "collectors.collect_news",
        "function": "collect_news",
        "est_size": "~1.5 GB",
        "description": "BBC Tamil, Dinamalar, The Hindu — case markers",
        "priority": 3,
    },
    "legal": {
        "name": "Legal & Administrative",
        "module": "collectors.collect_legal",
        "function": "collect_legal",
        "est_size": "~500 MB",
        "description": "TN Gazette, Government Orders — formal register",
        "priority": 4,
    },
    "hf": {
        "name": "Large-Scale HF Datasets",
        "module": "collectors.collect_hf_datasets",
        "function": "collect_hf_datasets",
        "est_size": "~6 GB",
        "description": "CulturaX, IndicCorp, OSCAR — modern agglutination",
        "priority": 5,
    },
}


def run_domain_collector(domain_key: str,
                         output_dir: Path,
                         max_workers: int = 4) -> list[Path]:
    """
    Dynamically import and run a domain collector.
    """
    domain = DOMAINS[domain_key]
    module_name = domain['module']
    func_name = domain['function']

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info(f"║  DOMAIN: {domain['name']:50s}║")
    logger.info(f"║  Est. Size: {domain['est_size']:48s}║")
    logger.info(f"║  {domain['description']:58s}║")
    logger.info("╚" + "═" * 60 + "╝")

    try:
        # Dynamic import
        import importlib
        mod = importlib.import_module(module_name)
        collect_func = getattr(mod, func_name)

        # Run collector
        start_time = time.time()
        output_files = collect_func(output_dir, max_workers=max_workers)
        elapsed = time.time() - start_time

        logger.info(
            f"\n  ✓ {domain['name']} completed in {elapsed:.0f}s "
            f"({len(output_files)} files)"
        )

        return output_files

    except Exception as e:
        logger.error(f"\n  ✗ {domain['name']} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def orchestrate(output_dir: str = "./tamil_corpus",
                target_gb: float = 10.0,
                max_workers: int = 4,
                skip_domains: list = None,
                only_domains: list = None,
                dry_run: bool = False):
    """
    Main orchestration function.
    Runs all domain collectors and merges into raw_tamil_gold.txt.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_domains = skip_domains or []
    only_domains = only_domains or list(DOMAINS.keys())

    # Header
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║      TAMILUKU-LLM DATA COLLECTION PIPELINE               ║")
    logger.info("║      AMB Tokenizer Training Corpus Builder                ║")
    logger.info("╚" + "═" * 60 + "╝")
    logger.info(f"  Output Directory : {output_dir}")
    logger.info(f"  Target Size      : {target_gb} GB")
    logger.info(f"  Max Workers      : {max_workers}")
    logger.info(f"  Domains          : {', '.join(only_domains)}")
    logger.info(f"  Skip             : {', '.join(skip_domains) or 'none'}")
    logger.info(f"  Start Time       : {datetime.now().isoformat()}")
    logger.info("")

    if dry_run:
        logger.info("DRY RUN — showing plan only:\n")
        for key in sorted(DOMAINS, key=lambda k: DOMAINS[k]['priority']):
            if key in skip_domains:
                continue
            if key not in only_domains:
                continue
            d = DOMAINS[key]
            logger.info(f"  [{d['priority']}] {d['name']:30s} {d['est_size']:>10s}")
            logger.info(f"      {d['description']}")
            logger.info(f"      Module: {d['module']}.{d['function']}")
            logger.info("")
        return

    # Run collectors
    all_files = []
    pipeline_stats = {}

    # Sort by priority
    sorted_domains = sorted(
        only_domains,
        key=lambda k: DOMAINS.get(k, {}).get('priority', 99)
    )

    for domain_key in sorted_domains:
        if domain_key in skip_domains:
            logger.info(f"\n  ⊘ Skipping: {DOMAINS[domain_key]['name']}")
            continue

        if domain_key not in DOMAINS:
            logger.warning(f"  Unknown domain: {domain_key}")
            continue

        start = time.time()
        files = run_domain_collector(domain_key, output_dir, max_workers)
        elapsed = time.time() - start

        all_files.extend(files)
        pipeline_stats[domain_key] = {
            'files': len(files),
            'elapsed_seconds': elapsed,
            'file_paths': [str(f) for f in files],
        }

    # Merge phase
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║              MERGE & DEDUPLICATION PHASE                  ║")
    logger.info("╚" + "═" * 60 + "╝")

    from merge_gold import merge_corpus

    gold_output = output_dir / "raw_tamil_gold.txt"
    merge_stats = merge_corpus(
        input_dir=output_dir,
        output_file=gold_output,
        target_gb=target_gb,
    )

    # Final report
    final_size = get_file_size_gb(gold_output)

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║              PIPELINE COMPLETE                            ║")
    logger.info("╠" + "═" * 60 + "╣")
    logger.info(f"║  Output: {str(gold_output):50s}║")
    logger.info(f"║  Size:   {final_size:.2f} GB" +
                " " * (51 - len(f"{final_size:.2f} GB")) + "║")
    logger.info(f"║  Lines:  {merge_stats.get('total_lines_written', 0):,}" +
                " " * max(1, 51 - len(f"{merge_stats.get('total_lines_written', 0):,}")) + "║")
    logger.info("╚" + "═" * 60 + "╝")

    # Save pipeline report
    report = {
        'timestamp': datetime.now().isoformat(),
        'output_file': str(gold_output),
        'final_size_gb': final_size,
        'target_gb': target_gb,
        'domain_stats': pipeline_stats,
        'merge_stats': merge_stats,
    }

    report_path = output_dir / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Full report: {report_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tamiluku-LLM: Tamil Gold Corpus Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python orchestrator.py --output-dir ./tamil_corpus --target-gb 10

  # Quick start (HuggingFace datasets only — fastest)
  python orchestrator.py --only hf --target-gb 6

  # Classical + Wiki only
  python orchestrator.py --only classical wiki --target-gb 2

  # Skip slow scrapers
  python orchestrator.py --skip news legal

  # Dry run
  python orchestrator.py --dry-run

Available domains:
  classical  — Project Madurai, Wikisource (~500 MB)
  wiki       — Tamil Wikipedia dump (~800 MB)
  news       — BBC Tamil, Dinamalar, The Hindu (~1.5 GB)
  legal      — TN Government Gazette (~500 MB)
  hf         — CulturaX, IndicCorp, OSCAR (~6 GB)
        """
    )

    parser.add_argument("--output-dir", type=str, default="./tamil_corpus",
                        help="Output directory for collected data")
    parser.add_argument("--target-gb", type=float, default=10.0,
                        help="Target corpus size in GB (default: 10)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel download threads (default: 4)")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=list(DOMAINS.keys()),
                        help="Domains to skip")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=list(DOMAINS.keys()),
                        help="Only run these domains")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")

    args = parser.parse_args()

    orchestrate(
        output_dir=args.output_dir,
        target_gb=args.target_gb,
        max_workers=args.max_workers,
        skip_domains=args.skip,
        only_domains=args.only or list(DOMAINS.keys()),
        dry_run=args.dry_run,
    )
