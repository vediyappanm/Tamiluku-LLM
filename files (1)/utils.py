"""
Tamiluku-LLM: Shared Utilities for Tamil Corpus Collection
============================================================
Core functions for:
  - Tamil script detection (Unicode block: U+0B80–U+0BFF)
  - Code-mix filtering (reject English-heavy lines)
  - UTF-8 NFC normalization
  - Parallel download manager
  - Text deduplication (MinHash)
"""

import re
import os
import unicodedata
import logging
from typing import Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from hashlib import md5

import requests
from tqdm import tqdm

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tamiluku_utils")

# ============================================================
# UNICODE CONSTANTS FOR TAMIL
# ============================================================
# Tamil Unicode block: U+0B80 to U+0BFF
TAMIL_RANGE = re.compile(r'[\u0B80-\u0BFF]')
# Latin/ASCII letters
LATIN_RANGE = re.compile(r'[A-Za-z]')
# Digits
DIGIT_RANGE = re.compile(r'[0-9]')
# Consecutive English words pattern (5+ words in a row)
CONSECUTIVE_ENGLISH = re.compile(r'(?:\b[A-Za-z]+\b[\s,;:]+){5,}')
# Common English stopwords that appear in code-mixed text
ENGLISH_WORD_PATTERN = re.compile(r'\b[A-Za-z]{2,}\b')


def normalize_tamil_text(text: str) -> str:
    """
    Apply NFC normalization and clean up Tamil text.
    - NFC normalization ensures consistent Unicode representation
    - Removes zero-width joiners/non-joiners (ZWJ/ZWNJ) unless needed
    - Normalizes whitespace
    """
    # NFC normalization (critical for Tamil - prevents decomposed forms)
    text = unicodedata.normalize("NFC", text)

    # Normalize various whitespace to single space
    text = re.sub(r'[\t\r]+', ' ', text)

    # Remove excessive newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove null bytes and control characters (except newline)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def count_tamil_chars(text: str) -> int:
    """Count Tamil Unicode characters in text."""
    return len(TAMIL_RANGE.findall(text))


def count_latin_chars(text: str) -> int:
    """Count Latin/ASCII characters in text."""
    return len(LATIN_RANGE.findall(text))


def is_quality_tamil_line(line: str,
                          min_tamil_ratio: float = 0.60,
                          max_english_word_ratio: float = 0.40,
                          min_line_length: int = 20,
                          max_line_length: int = 50000) -> bool:
    """
    Determine if a line is high-quality Tamil text.

    Rejection criteria:
    1. Too short (< min_line_length chars) — likely fragments
    2. Too long (> max_line_length) — likely data artifacts
    3. Tamil char ratio < min_tamil_ratio among all script chars
    4. Contains 5+ consecutive English words
    5. English word count > max_english_word_ratio of total words

    Returns True if the line passes all quality checks.
    """
    line = line.strip()

    # Length check
    if len(line) < min_line_length or len(line) > max_line_length:
        return False

    # Count script characters
    tamil_count = count_tamil_chars(line)
    latin_count = count_latin_chars(line)
    total_script = tamil_count + latin_count

    # Must have some Tamil content
    if tamil_count == 0:
        return False

    # Tamil ratio check (among script characters, ignoring digits/punctuation)
    if total_script > 0:
        tamil_ratio = tamil_count / total_script
        if tamil_ratio < min_tamil_ratio:
            return False

    # Consecutive English words check
    if CONSECUTIVE_ENGLISH.search(line):
        return False

    # English word ratio check
    words = line.split()
    if len(words) > 0:
        english_words = len(ENGLISH_WORD_PATTERN.findall(line))
        if english_words / len(words) > max_english_word_ratio:
            return False

    return True


def clean_and_filter_text(raw_text: str,
                          min_tamil_ratio: float = 0.60) -> str:
    """
    Process raw text: normalize, filter lines, return clean Tamil text.
    """
    normalized = normalize_tamil_text(raw_text)
    lines = normalized.split('\n')

    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            # Preserve paragraph boundaries
            if clean_lines and clean_lines[-1] != '':
                clean_lines.append('')
            continue

        if is_quality_tamil_line(line, min_tamil_ratio=min_tamil_ratio):
            clean_lines.append(line)

    return '\n'.join(clean_lines)


def save_clean_text(text: str, filepath: Path, mode: str = 'a') -> int:
    """
    Save cleaned text to file in UTF-8 encoding.
    Returns number of bytes written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')
    return len(text.encode('utf-8'))


def get_file_size_gb(filepath: Path) -> float:
    """Get file size in gigabytes."""
    if filepath.exists():
        return filepath.stat().st_size / (1024 ** 3)
    return 0.0


# ============================================================
# PARALLEL DOWNLOAD MANAGER
# ============================================================

class ParallelDownloader:
    """
    Thread-based parallel downloader with progress tracking.
    Designed for downloading multiple files/URLs concurrently.
    """

    def __init__(self, max_workers: int = 8, timeout: int = 30):
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TamilukuLLM-DataCollector/1.0 (Research; Tamil NLP)'
        })

    def download_url(self, url: str, output_path: Path) -> Optional[Path]:
        """Download a single URL to a file."""
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            total = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {url} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def download_batch(self, url_path_pairs: list[tuple[str, Path]]) -> list[Path]:
        """
        Download multiple URLs in parallel.
        Args: list of (url, output_path) tuples
        Returns: list of successfully downloaded file paths
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_url, url, path): (url, path)
                for url, path in url_path_pairs
            }

            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)

        logger.info(f"Downloaded {len(results)}/{len(url_path_pairs)} files successfully")
        return results

    def fetch_text(self, url: str) -> Optional[str]:
        """Fetch text content from a URL."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def fetch_batch_text(self, urls: list[str]) -> dict[str, str]:
        """Fetch text from multiple URLs in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch_text, url): url
                for url in urls
            }

            with tqdm(total=len(futures), desc="Fetching") as pbar:
                for future in as_completed(futures):
                    url = futures[future]
                    text = future.result()
                    if text:
                        results[url] = text
                    pbar.update(1)

        return results


# ============================================================
# DEDUPLICATION (LINE-LEVEL)
# ============================================================

class LineDeduplicator:
    """
    Fast line-level deduplication using MD5 hashes.
    For large-scale dedup, use MinHash LSH (datasketch).
    """

    def __init__(self):
        self.seen_hashes = set()
        self.total_lines = 0
        self.duplicate_lines = 0

    def is_duplicate(self, line: str) -> bool:
        """Check if a line is a duplicate."""
        self.total_lines += 1
        line_hash = md5(line.strip().encode('utf-8')).hexdigest()

        if line_hash in self.seen_hashes:
            self.duplicate_lines += 1
            return True

        self.seen_hashes.add(line_hash)
        return False

    def deduplicate_file(self, input_path: Path, output_path: Path) -> dict:
        """Deduplicate a text file line by line."""
        self.seen_hashes.clear()
        self.total_lines = 0
        self.duplicate_lines = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                if line.strip() and not self.is_duplicate(line):
                    fout.write(line)

        stats = {
            'total_lines': self.total_lines,
            'duplicate_lines': self.duplicate_lines,
            'unique_lines': self.total_lines - self.duplicate_lines,
            'dedup_ratio': self.duplicate_lines / max(self.total_lines, 1)
        }
        logger.info(f"Dedup: {stats['unique_lines']}/{stats['total_lines']} "
                     f"unique ({stats['dedup_ratio']:.1%} duplicates removed)")
        return stats


# ============================================================
# STREAMING TEXT PROCESSOR
# ============================================================

def stream_clean_lines(filepath: Path,
                       min_tamil_ratio: float = 0.60) -> Generator[str, None, None]:
    """
    Generator that streams cleaned Tamil lines from a file.
    Memory-efficient for large files.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = normalize_tamil_text(line.strip())
            if line and is_quality_tamil_line(line, min_tamil_ratio=min_tamil_ratio):
                yield line


def estimate_corpus_stats(filepath: Path) -> dict:
    """Quick stats for a corpus file."""
    if not filepath.exists():
        return {'exists': False}

    size_bytes = filepath.stat().st_size
    line_count = 0
    tamil_char_count = 0
    sample_lines = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line_count += 1
            tamil_char_count += count_tamil_chars(line)
            if i < 5:
                sample_lines.append(line.strip()[:100])

    return {
        'exists': True,
        'size_gb': size_bytes / (1024 ** 3),
        'size_mb': size_bytes / (1024 ** 2),
        'line_count': line_count,
        'tamil_char_count': tamil_char_count,
        'sample_lines': sample_lines
    }


if __name__ == "__main__":
    # Quick self-test
    test_lines = [
        "தமிழ் ஒரு அழகான மொழி",                      # Pure Tamil — PASS
        "This is a fully English sentence",              # Pure English — FAIL
        "தமிழ் and English mixed text here now yes",     # Code-mixed — FAIL
        "இன்றைய செய்திகளில் AI பற்றி பேசுவோம்",         # Light English — PASS
        "hi",                                            # Too short — FAIL
        "நம் நாட்டின் வரலாறு மிகவும் பழமையானது. சிந்து சமவெளி நாகரிகம் உலகின் மிகப் பழமையான நாகரிகங்களில் ஒன்று.",  # PASS
    ]

    print("=" * 60)
    print("Tamil Line Quality Filter — Self Test")
    print("=" * 60)

    for line in test_lines:
        result = is_quality_tamil_line(line)
        status = "✅ PASS" if result else "❌ FAIL"
        display = line[:60] + "..." if len(line) > 60 else line
        print(f"  {status} | {display}")

    print("\nNormalization test:")
    raw = "தமிழ்\t\tநாடு\r\n\n\n\nநல்ல   நாடு"
    clean = normalize_tamil_text(raw)
    print(f"  Raw:   {repr(raw[:50])}")
    print(f"  Clean: {repr(clean[:50])}")
