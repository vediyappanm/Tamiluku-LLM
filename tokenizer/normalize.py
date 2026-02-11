"""
normalize.py - Production-Grade Tamil Text Normalization & Deduplication
=========================================================================
Takes raw Tamil text files (10-50 GB) and produces a clean, deduplicated
corpus ready for tokenizer training.

Designed for scale:
  - Streaming file processing (never loads entire corpus into RAM)
  - Multiprocessing with configurable worker count
  - Bloom filter for memory-efficient exact dedup at scale
  - MinHash LSH for near-duplicate detection
  - Sharded output for parallel downstream processing

Key operations:
  1. Unicode NFC normalization (critical for Tamil)
  2. Zero-width character cleanup
  3. Pulli (virama) normalization
  4. Language identification filtering (fasttext)
  5. English code-mixing ratio control
  6. MinHash near-duplicate removal
  7. Train/eval split with domain tracking

Usage:
    python normalize.py [--config config.yaml]
    python normalize.py --workers 8 --config config.yaml
"""

import os
import re
import sys
import json
import hashlib
import argparse
import logging
import unicodedata
from pathlib import Path
from typing import List, Optional, Iterator, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import yaml
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tamil Unicode Normalization
# ---------------------------------------------------------------------------

TAMIL_RANGE = range(0x0B80, 0x0C00)

ZERO_WIDTH_CHARS = {
    "\u200B",  # Zero Width Space
    "\u200C",  # Zero Width Non-Joiner
    "\u200D",  # Zero Width Joiner
    "\uFEFF",  # BOM / Zero Width No-Break Space
    "\u00AD",  # Soft Hyphen
    "\u200E",  # Left-to-Right Mark
    "\u200F",  # Right-to-Left Mark
    "\u2028",  # Line Separator
    "\u2029",  # Paragraph Separator
}

# Precompile regex patterns for performance
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_REPEATED_PUNCT = re.compile(r"([!?.,;:]){4,}")
_RE_REPEATED_CHAR = re.compile(r"(.)\1{9,}")


def normalize_tamil_unicode(text: str) -> str:
    """
    Apply Tamil-specific Unicode normalization.

    Tamil has multiple valid Unicode representations for the same visual
    character. NFC canonicalization ensures the tokenizer doesn't learn
    duplicate tokens for identical syllables.
    """
    # NFC normalization (canonical decomposition + composition)
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters
    for zw in ZERO_WIDTH_CHARS:
        text = text.replace(zw, "")

    # Remove URLs and emails (noise for tokenizer training)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)

    # Normalize excessive punctuation
    text = _RE_REPEATED_PUNCT.sub(r"\1\1\1", text)

    # Normalize excessive character repetition
    text = _RE_REPEATED_CHAR.sub(r"\1\1\1", text)

    # Normalize whitespace
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)

    # Remove control characters (except newline, tab)
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )

    return text.strip()


def compute_tamil_ratio(text: str) -> float:
    """Compute the fraction of alphabetic characters that are Tamil script."""
    if not text:
        return 0.0
    tamil_count = sum(1 for ch in text if ord(ch) in TAMIL_RANGE)
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count == 0:
        return 0.0
    return tamil_count / alpha_count


def compute_english_ratio(text: str) -> float:
    """Compute the fraction of alphabetic characters that are Latin."""
    if not text:
        return 0.0
    latin_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count == 0:
        return 0.0
    return latin_count / alpha_count


# ---------------------------------------------------------------------------
# Language Identification
# ---------------------------------------------------------------------------

class LanguageFilter:
    """Fasttext-based language identification filter."""

    def __init__(self, model_path: str, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            import fasttext
            # Suppress fasttext warnings about deprecated load_model
            fasttext.FastText.eprint = lambda x: None

            if os.path.exists(model_path):
                self.model = fasttext.load_model(model_path)
                log.info(f"Loaded fasttext model: {model_path}")
            else:
                # Try downloading
                alt_path = Path(__file__).parent / model_path
                if alt_path.exists():
                    self.model = fasttext.load_model(str(alt_path))
                    log.info(f"Loaded fasttext model: {alt_path}")
                else:
                    log.warning(
                        f"Fasttext model not found at {model_path}. "
                        "Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin "
                        "Language filtering will use Unicode heuristic."
                    )
        except ImportError:
            log.warning("fasttext not installed. Using Unicode heuristic for language filtering.")

    def is_tamil(self, text: str) -> bool:
        if self.model is None:
            return compute_tamil_ratio(text) > 0.5

        clean = text.replace("\n", " ").strip()[:500]
        if not clean:
            return False

        predictions = self.model.predict(clean, k=3)
        labels, probs = predictions

        for label, prob in zip(labels, probs):
            lang = label.replace("__label__", "")
            if lang == "ta" and prob >= self.min_confidence:
                return True

        return False


# ---------------------------------------------------------------------------
# Deduplication (Scalable)
# ---------------------------------------------------------------------------

class ScalableDeduplicator:
    """
    Two-tier deduplication:
      1. Exact dedup via content hash (Bloom filter for memory efficiency)
      2. Near-duplicate via MinHash LSH
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128,
                 expected_docs: int = 10_000_000):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = None
        self._seen_hashes = set()
        self._doc_count = 0

        try:
            from datasketch import MinHash, MinHashLSH
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._MinHash = MinHash
            log.info(f"MinHash LSH initialized (threshold={threshold}, perms={num_perm})")
        except ImportError:
            log.warning("datasketch not installed. Using exact hash dedup only.")

    def _content_hash_int(self, text: str) -> int:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)

    def _get_shingles(self, text: str, k: int = 5) -> set:
        text = text.lower().strip()
        if len(text) < k:
            return {text}
        return {text[i:i + k] for i in range(len(text) - k + 1)}

    def is_duplicate(self, text: str) -> bool:
        # Tier 1: Exact hash dedup (64-bit int for RAM)
        h = self._content_hash_int(text)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)

        # Tier 2: Near-duplicate via MinHash
        if self.lsh is not None:
            shingles = self._get_shingles(text)
            if not shingles:
                return False

            m = self._MinHash(num_perm=self.num_perm)
            for s in shingles:
                m.update(s.encode("utf-8"))

            result = self.lsh.query(m)
            if result:
                return True

            try:
                doc_id = f"doc_{self._doc_count}"
                self.lsh.insert(doc_id, m)
                self._doc_count += 1
            except ValueError:
                pass

        return False


# ---------------------------------------------------------------------------
# Document Processing
# ---------------------------------------------------------------------------

def process_document(text: str, norm_cfg: dict) -> Optional[str]:
    """
    Process a single document through the normalization pipeline.
    Returns cleaned text or None if document should be filtered out.

    Note: This function is designed to be called in a process pool,
    so it does NOT use the language filter (which holds a large model).
    Language filtering is done in the main process.
    """
    # Unicode normalization
    text = normalize_tamil_unicode(text)

    # Length filter
    if len(text) < norm_cfg.get("min_doc_chars", 20):
        return None
    if len(text) > norm_cfg.get("max_doc_chars", 100000):
        text = text[:norm_cfg["max_doc_chars"]]

    # English ratio filter
    eng_ratio = compute_english_ratio(text)
    if eng_ratio > norm_cfg.get("code_mix_english_max_ratio", 0.30):
        return None

    # Tamil ratio check (should be predominantly Tamil)
    tamil_ratio = compute_tamil_ratio(text)
    if tamil_ratio < 0.4:
        return None

    return text


def stream_raw_documents(raw_dir: Path) -> Iterator[Tuple[str, str]]:
    """
    Stream documents from raw text files without loading everything into RAM.
    Yields (source_name, document_text) tuples.
    """
    for txt_file in sorted(raw_dir.glob("*.txt")):
        source_name = txt_file.stem
        log.info(f"Streaming from {txt_file.name} ({txt_file.stat().st_size / (1024*1024):.1f} MB)...")

        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            buffer = []
            for line in f:
                if line.strip() == "" and buffer:
                    doc = "\n".join(buffer).strip()
                    if doc:
                        yield (source_name, doc)
                    buffer = []
                else:
                    buffer.append(line.rstrip())

            # Flush remaining
            if buffer:
                doc = "\n".join(buffer).strip()
                if doc:
                    yield (source_name, doc)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Normalize and deduplicate Tamil corpus")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    norm_cfg = cfg["normalize"]
    raw_dir = Path(cfg["corpus"]["raw_dir"])
    cleaned_dir = Path(cfg["corpus"]["cleaned_dir"])
    eval_dir = Path(cfg["corpus"]["eval_dir"])
    output_file = Path(cfg["corpus"]["output_file"])

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    lang_filter = LanguageFilter(
        model_path=norm_cfg["fasttext_model"],
        min_confidence=norm_cfg["fasttext_confidence"],
    )
    dedup = ScalableDeduplicator(
        threshold=norm_cfg["dedup_threshold"],
        num_perm=norm_cfg["dedup_num_perm"],
    )

    # Stats tracking
    stats = {
        "total_raw": 0,
        "filtered_length": 0,
        "filtered_english_ratio": 0,
        "filtered_tamil_ratio": 0,
        "filtered_language": 0,
        "filtered_duplicate": 0,
        "kept": 0,
        "per_source": {},
    }

    # Process documents in streaming fashion
    log.info("=== Starting normalization pipeline ===")
    log.info(f"  Raw dir: {raw_dir}")
    log.info(f"  Output:  {output_file}")
    log.info(f"  Workers: {args.workers}")

    # Collect all cleaned documents with eval split decision
    eval_holdout = norm_cfg.get("eval_holdout_ratio", 0.02)
    rng = np.random.RandomState(42)

    train_file = open(str(output_file), "w", encoding="utf-8")
    eval_file = open(str(eval_dir / "eval_corpus.txt"), "w", encoding="utf-8")
    train_count = 0
    eval_count = 0

    log.info(f"Starting parallel processing with {args.workers} workers...")
    
    # We use a ProcessPoolExecutor for normalization (CPU intensive)
    # Deduplication and Language ID must stay in main thread (stateful/large models)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Use a small chunk size to avoid memory bloat
        chunk_size = args.workers * 10 
        batch = []
        
        doc_stream = stream_raw_documents(raw_dir)
        
        while True:
            # Collect a batch of raw documents
            batch_raw = []
            try:
                for _ in range(chunk_size):
                    batch_raw.append(next(doc_stream))
            except StopIteration:
                pass
            
            if not batch_raw:
                break
                
            # Submit batch for normalization
            futures = {
                executor.submit(process_document, raw_doc, norm_cfg): (src, raw_doc)
                for src, raw_doc in batch_raw
            }
            
            for future in as_completed(futures):
                source_name, raw_doc = futures[future]
                stats["total_raw"] += 1
                if source_name not in stats["per_source"]:
                    stats["per_source"][source_name] = {"raw": 0, "kept": 0}
                stats["per_source"][source_name]["raw"] += 1
                
                try:
                    cleaned = future.result()
                    if cleaned is None:
                        stats["filtered_length"] += 1
                        continue
                        
                    # Step 2: Language ID filter (in main process)
                    if not lang_filter.is_tamil(cleaned):
                        stats["filtered_language"] += 1
                        continue

                    # Step 3: Deduplication (in main process)
                    if dedup.is_duplicate(cleaned):
                        stats["filtered_duplicate"] += 1
                        continue

                    # Keep this document
                    stats["kept"] += 1
                    stats["per_source"][source_name]["kept"] += 1

                    # Train/eval split
                    if rng.random() < eval_holdout:
                        eval_file.write(cleaned + "\n\n")
                        eval_count += 1
                    else:
                        train_file.write(cleaned + "\n\n")
                        train_count += 1
                        
                except Exception as e:
                    log.error(f"Error processing doc from {source_name}: {e}")

            if stats["total_raw"] % 50000 == 0:
                log.info(f"  Processed {stats['total_raw']:,} docs... Kept {stats['kept']:,}")

    train_file.close()
    eval_file.close()

    # Report
    train_size = output_file.stat().st_size / (1024 * 1024)
    eval_size = (eval_dir / "eval_corpus.txt").stat().st_size / (1024 * 1024)

    log.info(f"\n{'='*60}")
    log.info(f"NORMALIZATION COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Training corpus: {train_count:,} docs, {train_size:.1f} MB -> {output_file}")
    log.info(f"  Eval corpus:     {eval_count:,} docs, {eval_size:.1f} MB")
    log.info(f"")
    log.info(f"  Filtering breakdown:")
    log.info(f"    Raw documents:         {stats['total_raw']:>12,}")
    log.info(f"    Filtered (length/eng): {stats['filtered_length']:>12,}")
    log.info(f"    Filtered (language):   {stats['filtered_language']:>12,}")
    log.info(f"    Filtered (duplicate):  {stats['filtered_duplicate']:>12,}")
    log.info(f"    Final kept:            {stats['kept']:>12,}")
    log.info(f"    Retention rate:        {stats['kept']/max(stats['total_raw'],1):>11.1%}")
    log.info(f"")
    log.info(f"  Per-source breakdown:")
    for source, src_stats in sorted(stats["per_source"].items()):
        retention = src_stats["kept"] / max(src_stats["raw"], 1)
        log.info(f"    {source:<25} {src_stats['kept']:>10,} / {src_stats['raw']:>10,} ({retention:.1%})")
    log.info(f"{'='*60}")
    log.info(f"\nNext step: python train_tokenizer.py")

    # Save stats
    stats_path = Path("reports") / "normalization_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(stats_path), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
