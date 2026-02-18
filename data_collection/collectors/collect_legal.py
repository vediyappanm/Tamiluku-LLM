"""
Tamiluku-LLM: Legal & Administrative Tamil Collector
======================================================
Domain 5: Government & Legal Text

Sources:
  1. Tamil Nadu Government Gazette (tngazette.in)
  2. TN Government Orders (tngo.in)
  3. Legislative Assembly proceedings
  4. India Code — Tamil translations of central acts
  5. Tamil Nadu State Planning Commission reports

Why this matters for AMB:
  Legal/administrative Tamil uses UNIQUE agglutination:
  - Long compound words (அரசாணை, சட்டமன்றம்)
  - Formal suffixes rarely seen in colloquial text
  - Standardized spelling (no colloquial shortcuts)
  Essential for vocabulary coverage of formal register.

Note: Government websites may have inconsistent structure.
      We use multiple extraction strategies.
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    normalize_tamil_text, clean_and_filter_text,
    save_clean_text, ParallelDownloader, logger
)


# ============================================================
# GOVERNMENT SOURCES
# ============================================================

GOVT_SOURCES = {
    "tn_gazette": {
        "name": "Tamil Nadu Government Gazette",
        "urls": [
            "https://www.stationeryprinting.tn.gov.in/gazette/gazette_list.php",
        ],
        "description": "Official state gazette with GOs in Tamil",
    },
    "tn_assembly": {
        "name": "TN Legislative Assembly",
        "urls": [
            "https://www.assembly.tn.gov.in/",
        ],
        "description": "Assembly proceedings and debates in Tamil",
    },
    "india_code_tamil": {
        "name": "India Code (Tamil translations)",
        "urls": [
            "https://www.indiacode.nic.in/",
        ],
        "description": "Central laws translated to Tamil",
    },
}


# ============================================================
# GOVERNMENT TEXT COLLECTOR
# ============================================================

class GovtTextCollector:
    """
    Collects legal and administrative Tamil text from government sources.

    Strategy:
    1. Crawl known government portals for Tamil content
    2. Download available PDF/HTML documents
    3. Extract text using trafilatura + fallback
    4. Apply quality filter (relaxed for formal text)
    """

    def __init__(self, output_dir: Path, max_workers: int = 4):
        self.output_dir = output_dir / "legal"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TamilukuLLM-Research/1.0 (Academic NLP Research)',
            'Accept-Language': 'ta,en;q=0.5',
        })

    def crawl_tn_gazette(self, max_pages: int = 500) -> Path:
        """
        Crawl Tamil Nadu Government Gazette for GO texts.
        """
        output_file = self.output_dir / "tn_gazette.txt"
        logger.info("Crawling TN Government Gazette...")

        # Known gazette URL patterns
        gazette_base = "https://www.stationeryprinting.tn.gov.in/gazette"
        total_bytes = 0
        processed = 0

        # Try to access gazette listing
        try:
            response = self.session.get(
                f"{gazette_base}/gazette_list.php",
                timeout=30
            )

            if response.ok:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find links to gazette issues
                links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '.php' in href or '.html' in href:
                        from urllib.parse import urljoin
                        full_url = urljoin(gazette_base + '/', href)
                        links.append(full_url)

                links = list(set(links))[:max_pages]
                logger.info(f"Found {len(links)} gazette pages")

                for url in tqdm(links, desc="TN Gazette"):
                    time.sleep(1.0)  # Rate limit
                    try:
                        page_response = self.session.get(url, timeout=30)
                        if page_response.ok:
                            if HAS_TRAFILATURA:
                                text = trafilatura.extract(
                                    page_response.text,
                                    target_language='ta'
                                )
                            else:
                                bs = BeautifulSoup(page_response.text, 'html.parser')
                                text = bs.get_text(separator='\n')

                            if text:
                                cleaned = clean_and_filter_text(
                                    text, min_tamil_ratio=0.40
                                )
                                if len(cleaned) > 100:
                                    total_bytes += save_clean_text(
                                        cleaned + '\n\n', output_file
                                    )
                                    processed += 1
                    except Exception:
                        continue

        except Exception as e:
            logger.warning(f"TN Gazette access issue: {e}")
            logger.info("Government sites may require VPN or direct access from India")

        logger.info(f"TN Gazette: {processed} pages, {total_bytes / (1024**2):.1f} MB")
        return output_file

    def collect_from_wikisource_legal(self) -> Path:
        """
        Collect legal/administrative texts from Tamil Wikisource.
        Categories: சட்டம், அரசாணை, etc.
        """
        output_file = self.output_dir / "wikisource_legal.txt"
        logger.info("Collecting legal texts from Tamil Wikisource...")

        API_URL = "https://ta.wikisource.org/w/api.php"

        # Search for legal/administrative content
        legal_categories = [
            'சட்டம்',           # Law
            'அரசாணை',          # Government Order
            'அரசியலமைப்பு',     # Constitution
        ]

        total_bytes = 0
        processed = 0

        for category in legal_categories:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'பகுப்பு:{category}',
                'cmlimit': 500,
                'format': 'json',
            }

            try:
                response = self.session.get(API_URL, params=params, timeout=30)
                data = response.json()
                members = data.get('query', {}).get('categorymembers', [])

                for member in tqdm(members, desc=f"Category: {category}"):
                    # Get page content
                    page_params = {
                        'action': 'query',
                        'titles': member['title'],
                        'prop': 'extracts',
                        'explaintext': True,
                        'format': 'json',
                    }

                    try:
                        page_resp = self.session.get(
                            API_URL, params=page_params, timeout=30
                        )
                        page_data = page_resp.json()
                        pages = page_data.get('query', {}).get('pages', {})

                        for pid, pdata in pages.items():
                            text = pdata.get('extract', '')
                            if text:
                                cleaned = clean_and_filter_text(
                                    text, min_tamil_ratio=0.40
                                )
                                if len(cleaned) > 100:
                                    total_bytes += save_clean_text(
                                        cleaned + '\n\n', output_file
                                    )
                                    processed += 1
                    except Exception:
                        continue

            except Exception as e:
                logger.warning(f"Category {category}: {e}")

        logger.info(f"Wikisource Legal: {processed} pages, {total_bytes / (1024**2):.1f} MB")
        return output_file

    def collect_from_hf_legal(self) -> Optional[Path]:
        """
        Try to collect Tamil legal text from HuggingFace datasets.
        """
        output_file = self.output_dir / "hf_legal.txt"
        logger.info("Checking HuggingFace for Tamil legal datasets...")

        try:
            from datasets import load_dataset

            # Try known legal datasets with Tamil content
            legal_datasets = [
                ("law-ai/InLegalNER", None),       # Indian Legal NER
                ("openslr/slr65", None),            # May have Tamil legal
            ]

            total_bytes = 0
            for dataset_name, config in legal_datasets:
                try:
                    logger.info(f"Trying: {dataset_name}")
                    ds = load_dataset(
                        dataset_name,
                        config,
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    )

                    for i, item in enumerate(ds):
                        if i > 50000:
                            break

                        text = item.get('text', '') or item.get('sentence', '')
                        if text:
                            cleaned = clean_and_filter_text(text, min_tamil_ratio=0.50)
                            if len(cleaned) > 50:
                                total_bytes += save_clean_text(
                                    cleaned + '\n', output_file
                                )

                except Exception as e:
                    logger.debug(f"Dataset {dataset_name} not available: {e}")
                    continue

            if total_bytes > 0:
                logger.info(f"HF Legal: {total_bytes / (1024**2):.1f} MB")
                return output_file

        except ImportError:
            logger.warning("datasets library not installed")

        return None


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def collect_legal(output_dir: Path, max_workers: int = 4) -> list[Path]:
    """Collect all legal/administrative Tamil texts."""
    collector = GovtTextCollector(output_dir, max_workers=max_workers)
    output_files = []

    logger.info("=" * 60)
    logger.info("COLLECTING: Legal & Administrative Tamil")
    logger.info("=" * 60)

    # 1. TN Government Gazette
    output_files.append(collector.crawl_tn_gazette())

    # 2. Wikisource legal texts
    output_files.append(collector.collect_from_wikisource_legal())

    # 3. HuggingFace legal datasets
    hf_legal = collector.collect_from_hf_legal()
    if hf_legal:
        output_files.append(hf_legal)

    return [f for f in output_files if f and f.exists()]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Tamil Legal/Admin Texts")
    parser.add_argument("--output-dir", type=str, default="./tamil_corpus")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    files = collect_legal(Path(args.output_dir), args.max_workers)
    print(f"\nCollected files: {[str(f) for f in files]}")
