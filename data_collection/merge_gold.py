"""
Tamiluku-LLM: Corpus Merger & Deduplicator
=============================================
Final stage: Merge all domain-specific text files into
a single `raw_tamil_gold.txt` with deduplication.

Process:
  1. Scan all .txt files in the corpus directory tree
  2. Merge in priority order (classical first, web last)
  3. Line-level deduplication using MD5 hashes
  4. Final quality filter pass
  5. Report corpus statistics
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Generator
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    normalize_tamil_text, is_quality_tamil_line,
    count_tamil_chars, LineDeduplicator,
    get_file_size_gb, estimate_corpus_stats, logger
)

# ============================================================
# MERGE CONFIGURATION
# ============================================================

# Priority order for merging (higher priority = included first)
# This ensures classical roots get priority in case of dedup collisions
DOMAIN_PRIORITY = [
    "classical",        # Pure roots, base vocabulary
    "wiki",             # Technical terms, compound words
    "legal",            # Formal register, unique suffixes
    "news",             # Case markers, formal agglutination
    "hf_datasets",      # Modern/colloquial, high volume
]


def discover_corpus_files(input_dir: Path) -> list[tuple[str, Path]]:
    """
    Discover all .txt corpus files, organized by domain priority.
    Returns: list of (domain_name, file_path) tuples in priority order.
    """
    files_by_domain = {}

    for txt_file in sorted(input_dir.rglob("*.txt")):
        # Determine domain from directory structure
        rel_path = txt_file.relative_to(input_dir)
        parts = rel_path.parts

        if len(parts) > 0:
            domain = parts[0]
        else:
            domain = "unknown"

        if domain not in files_by_domain:
            files_by_domain[domain] = []
        files_by_domain[domain].append(txt_file)

    # Sort by priority
    ordered_files = []
    for domain in DOMAIN_PRIORITY:
        if domain in files_by_domain:
            for f in files_by_domain[domain]:
                ordered_files.append((domain, f))
            del files_by_domain[domain]

    # Add any remaining domains not in priority list
    for domain, files in files_by_domain.items():
        for f in files:
            ordered_files.append((domain, f))

    return ordered_files


def stream_lines_from_file(filepath: Path) -> Generator[str, None, None]:
    """Stream lines from a file, handling encoding issues."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n\r')
                if line:
                    yield line
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")


def merge_corpus(input_dir: Path,
                 output_file: Path,
                 target_gb: float = 10.0,
                 min_tamil_ratio: float = 0.60) -> dict:
    """
    Merge all corpus files into a single gold file.

    Args:
        input_dir: Directory containing domain subdirectories
        output_file: Path for the merged output
        target_gb: Target file size in GB
        min_tamil_ratio: Minimum Tamil character ratio for quality filter

    Returns: Statistics dictionary
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  TAMILUKU-LLM: CORPUS MERGER & DEDUPLICATOR")
    logger.info("=" * 70)

    # Discover all corpus files
    corpus_files = discover_corpus_files(input_dir)
    logger.info(f"\nDiscovered {len(corpus_files)} corpus files:")
    for domain, filepath in corpus_files:
        size_mb = filepath.stat().st_size / (1024 ** 2) if filepath.exists() else 0
        logger.info(f"  [{domain:15s}] {filepath.name:40s} ({size_mb:.1f} MB)")

    # Initialize deduplicator
    dedup = LineDeduplicator()

    # Statistics
    stats = {
        'start_time': datetime.now().isoformat(),
        'input_files': len(corpus_files),
        'domain_stats': {},
        'total_lines_read': 0,
        'total_lines_written': 0,
        'total_lines_filtered': 0,
        'total_lines_deduplicated': 0,
        'total_bytes_written': 0,
    }

    # Merge
    with open(output_file, 'w', encoding='utf-8') as fout:
        for domain, filepath in corpus_files:
            if not filepath.exists():
                continue

            domain_lines_read = 0
            domain_lines_written = 0
            domain_bytes = 0

            logger.info(f"\nProcessing: [{domain}] {filepath.name}")

            for line in tqdm(
                stream_lines_from_file(filepath),
                desc=f"  {filepath.name}",
                unit=" lines"
            ):
                domain_lines_read += 1
                stats['total_lines_read'] += 1

                # Normalize
                line = normalize_tamil_text(line)
                if not line:
                    continue

                # Quality filter
                if not is_quality_tamil_line(line, min_tamil_ratio=min_tamil_ratio):
                    stats['total_lines_filtered'] += 1
                    continue

                # Deduplication
                if dedup.is_duplicate(line):
                    stats['total_lines_deduplicated'] += 1
                    continue

                # Write
                fout.write(line + '\n')
                domain_lines_written += 1
                domain_bytes += len(line.encode('utf-8')) + 1
                stats['total_lines_written'] += 1
                stats['total_bytes_written'] += domain_bytes

                # Check target size
                current_gb = stats['total_bytes_written'] / (1024 ** 3)
                if current_gb >= target_gb:
                    logger.info(f"\n  Reached target size: {current_gb:.2f} GB")
                    break

            # Domain stats
            stats['domain_stats'][domain] = stats['domain_stats'].get(domain, {
                'files': 0, 'lines_read': 0, 'lines_written': 0, 'bytes': 0
            })
            stats['domain_stats'][domain]['files'] += 1
            stats['domain_stats'][domain]['lines_read'] += domain_lines_read
            stats['domain_stats'][domain]['lines_written'] += domain_lines_written
            stats['domain_stats'][domain]['bytes'] += domain_bytes

            logger.info(
                f"  → {domain_lines_written}/{domain_lines_read} lines kept "
                f"({domain_bytes / (1024**2):.1f} MB)"
            )

            # Check overall target
            if stats['total_bytes_written'] / (1024 ** 3) >= target_gb:
                break

    # Final stats
    stats['end_time'] = datetime.now().isoformat()
    stats['output_size_gb'] = get_file_size_gb(output_file)
    stats['output_size_mb'] = stats['output_size_gb'] * 1024
    stats['dedup_ratio'] = (
        stats['total_lines_deduplicated'] /
        max(stats['total_lines_read'], 1)
    )
    stats['filter_ratio'] = (
        stats['total_lines_filtered'] /
        max(stats['total_lines_read'], 1)
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("  MERGE COMPLETE — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Size: {stats['output_size_gb']:.2f} GB ({stats['output_size_mb']:.0f} MB)")
    logger.info(f"  Lines: {stats['total_lines_written']:,}")
    logger.info(f"  Dedup rate: {stats['dedup_ratio']:.1%}")
    logger.info(f"  Filter rate: {stats['filter_ratio']:.1%}")
    logger.info("")
    logger.info("  Domain breakdown:")
    for domain, ds in stats['domain_stats'].items():
        pct = ds['bytes'] / max(stats['total_bytes_written'], 1) * 100
        logger.info(
            f"    {domain:15s}: {ds['lines_written']:>8,} lines "
            f"({ds['bytes'] / (1024**2):>7.1f} MB, {pct:>5.1f}%)"
        )
    logger.info("=" * 70)

    # Save stats
    stats_file = output_file.parent / "merge_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"  Stats saved: {stats_file}")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Merge Tamil corpus files into raw_tamil_gold.txt"
    )
    parser.add_argument("--input-dir", type=str, default="./tamil_corpus",
                        help="Directory containing domain subdirectories")
    parser.add_argument("--output", type=str, default="raw_tamil_gold.txt",
                        help="Output merged file path")
    parser.add_argument("--target-gb", type=float, default=10.0,
                        help="Target file size in GB")
    parser.add_argument("--min-tamil-ratio", type=float, default=0.60,
                        help="Minimum Tamil char ratio for quality filter")
    args = parser.parse_args()

    stats = merge_corpus(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output),
        target_gb=args.target_gb,
        min_tamil_ratio=args.min_tamil_ratio,
    )
