"""
Tamiluku-LLM: Tamil Wikipedia Dump Collector
===============================================
Domain 3: Technical & Wiki Content

Source: Tamil Wikipedia (ta.wikipedia.org)
  - Latest dump from dumps.wikimedia.org
  - ~150K+ articles covering diverse topics
  - Rich in compound words and technical vocabulary

Why this matters for AMB:
  Wikipedia contains COMPOUND WORDS (சொற்கோவை) and technical
  terminology that stress-test the BPE vocabulary.
  Also provides standardized formal Tamil prose.

Pipeline:
  1. Download latest tawiki-*-articles.xml.bz2
  2. Extract articles using mwparserfromhell (MediaWiki parser)
  3. Clean wikitext markup → plain Tamil text
  4. Apply code-mix filter
"""

import os
import re
import sys
import bz2
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Optional, Generator
from io import BytesIO

import requests
from tqdm import tqdm

try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False
    print("WARNING: mwparserfromhell not installed. Install: pip install mwparserfromhell")

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    normalize_tamil_text, clean_and_filter_text,
    save_clean_text, logger
)

# ============================================================
# WIKIPEDIA DUMP CONFIGURATION
# ============================================================

WIKI_DUMP_BASE = "https://dumps.wikimedia.org/tawiki/latest/"
DUMP_FILENAME = "tawiki-latest-pages-articles.xml.bz2"
DUMP_URL = f"{WIKI_DUMP_BASE}{DUMP_FILENAME}"

# Alternative: Use the HuggingFace mirror (often faster)
HF_WIKI_DATASET = "wikimedia/wikipedia"
HF_WIKI_CONFIG = "20231101.ta"  # Tamil Wikipedia snapshot


# ============================================================
# METHOD 1: HUGGINGFACE DATASETS (RECOMMENDED — FASTEST)
# ============================================================

def collect_wiki_from_huggingface(output_dir: Path,
                                  max_articles: int = 200000) -> Path:
    """
    Download Tamil Wikipedia from HuggingFace datasets.
    This is the FASTEST method — pre-processed, streaming-ready.

    Dataset: wikimedia/wikipedia (config: 20231101.ta)
    """
    from datasets import load_dataset

    output_file = output_dir / "wiki_hf.txt"
    logger.info(f"Loading Tamil Wikipedia from HuggingFace -> {output_file}")

    try:
        # Stream to avoid loading entire dataset into RAM
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ta",
            split="train",
            streaming=True,
        )

        total_bytes = 0
        processed = 0

        with tqdm(desc="Wiki (HuggingFace)", unit=" articles") as pbar:
            for i, article in enumerate(dataset):
                if i >= max_articles:
                    break

                text = article.get('text', '')
                if not text:
                    continue

                # Clean and filter
                cleaned = clean_and_filter_text(text, min_tamil_ratio=0.45)

                if len(cleaned) > 100:
                    total_bytes += save_clean_text(
                        cleaned + '\n\n',
                        output_file
                    )
                    processed += 1

                pbar.update(1)
                pbar.set_postfix({
                    'articles': processed,
                    'MB': f"{total_bytes / (1024**2):.1f}"
                })

        logger.info(
            f"Wiki (HF): {processed} articles, "
            f"{total_bytes / (1024**2):.1f} MB"
        )
        return output_file

    except Exception as e:
        logger.error(f"HuggingFace wiki download failed: {e}")
        logger.info("Falling back to direct dump download...")
        return collect_wiki_from_dump(output_dir)


# ============================================================
# METHOD 2: DIRECT DUMP DOWNLOAD + EXTRACTION
# ============================================================

def download_wiki_dump(output_dir: Path) -> Path:
    """Download the Tamil Wikipedia XML dump (bz2 compressed)."""
    dump_path = output_dir / DUMP_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)

    if dump_path.exists():
        logger.info(f"Dump already exists: {dump_path}")
        return dump_path

    logger.info(f"Downloading Tamil Wikipedia dump from {DUMP_URL}")
    logger.info("This may take 10-30 minutes depending on connection speed...")

    response = requests.get(DUMP_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dump_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc="Downloading Wiki Dump") as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Download complete: {dump_path} ({dump_path.stat().st_size / (1024**3):.2f} GB)")
    return dump_path


def parse_wiki_xml_streaming(dump_path: Path) -> Generator[dict, None, None]:
    """
    Stream-parse the Wikipedia XML dump.
    Memory-efficient: processes one article at a time.
    """
    logger.info(f"Stream-parsing XML from {dump_path}")

    # MediaWiki namespace
    ns = '{http://www.mediawiki.org/xml/export-0.10/}'

    with bz2.open(dump_path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))

        for event, elem in context:
            if elem.tag == f'{ns}page':
                # Extract page data
                title_elem = elem.find(f'{ns}title')
                text_elem = elem.find(f'.//{ns}text')
                ns_elem = elem.find(f'{ns}ns')

                # Only main namespace (ns=0)
                if ns_elem is not None and ns_elem.text == '0':
                    title = title_elem.text if title_elem is not None else ''
                    text = text_elem.text if text_elem is not None else ''

                    if text and not text.startswith('#REDIRECT'):
                        yield {
                            'title': title,
                            'text': text,
                        }

                # Free memory
                elem.clear()


def clean_wikitext(wikitext: str) -> str:
    """
    Convert MediaWiki markup to plain text.
    Uses mwparserfromhell for robust parsing.
    """
    if HAS_MWPARSER:
        try:
            parsed = mwparserfromhell.parse(wikitext)
            text = parsed.strip_code(
                normalize=True,
                collapse=True,
                keep_template_params=False,
            )
        except Exception:
            text = _regex_clean_wikitext(wikitext)
    else:
        text = _regex_clean_wikitext(wikitext)

    return normalize_tamil_text(text)


def _regex_clean_wikitext(text: str) -> str:
    """Fallback regex-based wikitext cleaner."""
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove categories [[Category:...]]
    text = re.sub(r'\[\[(?:Category|பகுப்பு):[^\]]*\]\]', '', text, flags=re.I)
    # Convert links [[text|display]] -> display, [[text]] -> text
    text = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove external links [http://... text]
    text = re.sub(r'\[https?://[^\]]*\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove wiki markup
    text = re.sub(r"'{2,}", '', text)  # Bold/italic
    text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)  # Headers
    # Remove file/image references
    text = re.sub(r'\[\[(?:File|Image|படிமம்):[^\]]*\]\]', '', text, flags=re.I)
    # Remove references <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)

    return text


def collect_wiki_from_dump(output_dir: Path,
                           max_articles: int = 200000) -> Path:
    """
    Download and extract Tamil Wikipedia from XML dump.
    """
    output_file = output_dir / "wiki_dump.txt"

    # Download dump
    dump_path = download_wiki_dump(output_dir)

    # Parse and extract
    total_bytes = 0
    processed = 0

    with tqdm(desc="Extracting Wiki Articles", unit=" articles") as pbar:
        for article in parse_wiki_xml_streaming(dump_path):
            if processed >= max_articles:
                break

            text = clean_wikitext(article['text'])
            cleaned = clean_and_filter_text(text, min_tamil_ratio=0.45)

            if len(cleaned) > 100:
                total_bytes += save_clean_text(
                    cleaned + '\n\n',
                    output_file
                )
                processed += 1

            pbar.update(1)
            pbar.set_postfix({
                'articles': processed,
                'MB': f"{total_bytes / (1024**2):.1f}"
            })

    logger.info(
        f"Wiki Dump: {processed} articles, "
        f"{total_bytes / (1024**2):.1f} MB"
    )
    return output_file


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def collect_wiki(output_dir: Path, method: str = "huggingface", max_workers: int = 4, **kwargs) -> list[Path]:
    """
    Collect Tamil Wikipedia content.

    Args:
        output_dir: Base output directory
        method: "huggingface" (recommended) or "dump" (direct download)
    """
    output_dir = Path(output_dir) / "wiki"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("COLLECTING: Tamil Wikipedia")
    logger.info("=" * 60)

    if method == "huggingface":
        output_file = collect_wiki_from_huggingface(output_dir)
    else:
        output_file = collect_wiki_from_dump(output_dir)

    return [output_file]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Tamil Wikipedia")
    parser.add_argument("--output-dir", type=str, default="./tamil_corpus")
    parser.add_argument("--method", choices=["huggingface", "dump"],
                        default="huggingface")
    args = parser.parse_args()

    files = collect_wiki(Path(args.output_dir), method=args.method)
    print(f"\nCollected files: {[str(f) for f in files]}")
