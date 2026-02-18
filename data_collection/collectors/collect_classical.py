"""
Tamiluku-LLM: Classical & Literary Tamil Collector
====================================================
Domain 1: Classical Tamil Texts

Sources:
  1. Project Madurai (projectmadurai.org) — Largest free Tamil e-library
     - 500+ classical texts, Sangam literature, Thirukkural, Silappadikaram
     - Files in UTF-8 HTML/text format
  2. Tamil Wikisource (ta.wikisource.org) — Community-curated classical texts
  3. Tamil Virtual University archives

Why this matters for AMB:
  Classical texts contain "pure" Tamil roots without English borrowings.
  These are essential for Layer 3 (Morpheme Segmentation) to learn
  base root forms before agglutination patterns.
"""

import os
import re
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    normalize_tamil_text, clean_and_filter_text,
    save_clean_text, ParallelDownloader, logger
)

# ============================================================
# PROJECT MADURAI COLLECTOR
# ============================================================

class ProjectMaduraiCollector:
    """
    Collects classical Tamil texts from Project Madurai.

    Project Madurai (projectmadurai.org) hosts 500+ Tamil literary works
    in HTML format. We crawl the index pages, extract text links,
    and download/clean the content.

    Note: Be respectful — add delays between requests.
    """

    BASE_URL = "https://www.projectmadurai.org"
    INDEX_URL = "https://www.projectmadurai.org/pmworks.html"

    def __init__(self, output_dir: Path, max_workers: int = 4):
        self.output_dir = output_dir / "project_madurai"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = ParallelDownloader(max_workers=max_workers)
        self.collected_urls = []

    def discover_text_links(self) -> list[str]:
        """
        Crawl Project Madurai index to find all text file links.
        PM organizes texts by number (pm0001 - pm0600+).
        """
        logger.info("Discovering Project Madurai text links...")

        try:
            response = requests.get(
                self.INDEX_URL,
                timeout=30,
                headers={'User-Agent': 'TamilukuLLM-Research/1.0'}
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # PM text pages follow patterns like pm_etext/pmXXXX.html
                # IMPORTANT: Skip PDF links — we can only parse HTML/text
                if href.lower().endswith('.pdf'):
                    continue
                if re.search(r'pm\d{3,4}', href, re.IGNORECASE):
                    full_url = urljoin(self.BASE_URL, href)
                    # Only keep HTML/text URLs
                    if any(full_url.lower().endswith(ext) for ext in ['.html', '.htm', '.txt', '']):
                        links.append(full_url)

            # Deduplicate
            links = list(set(links))
            logger.info(f"Discovered {len(links)} text links from Project Madurai (PDFs excluded)")
            return links

        except Exception as e:
            logger.error(f"Failed to crawl PM index: {e}")
            # Fallback: generate known URL patterns
            return self._generate_fallback_urls()

    def _generate_fallback_urls(self) -> list[str]:
        """Generate URLs based on known PM numbering pattern."""
        urls = []
        for i in range(1, 601):
            # PM uses formats like pm0001.html, pm0123.html
            url = f"{self.BASE_URL}/pm_etext/utf8/pm{i:04d}.html"
            urls.append(url)
        logger.info(f"Generated {len(urls)} fallback PM URLs")
        return urls

    def extract_text_from_html(self, html: str) -> str:
        """Extract clean Tamil text from a PM HTML page."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove scripts, styles, navigation
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()

        # Get text content
        text = soup.get_text(separator='\n')

        # Clean up
        text = normalize_tamil_text(text)

        # Remove common PM boilerplate lines
        boilerplate_patterns = [
            r'Project Madurai',
            r'www\.projectmadurai\.org',
            r'This file was last',
            r'Etext preparation',
            r'Proof reading',
            r'©\s*\d{4}',
            r'This e-text',
        ]

        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip boilerplate
            if any(re.search(pat, line, re.IGNORECASE) for pat in boilerplate_patterns):
                continue
            clean_lines.append(line)

        return '\n'.join(clean_lines)

    def collect(self, max_texts: int = 600) -> Path:
        """
        Main collection method.
        Downloads and cleans all available PM texts.
        """
        output_file = self.output_dir / "project_madurai_combined.txt"
        logger.info(f"Starting Project Madurai collection -> {output_file}")

        urls = self.discover_text_links()[:max_texts]

        total_bytes = 0
        processed = 0

        # Use parallel fetching
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.downloader.fetch_text, url): url
                for url in urls
            }

            with tqdm(total=len(futures), desc="Project Madurai") as pbar:
                for future in as_completed(futures):
                    url = futures[future]
                    html = future.result()

                    if html:
                        text = self.extract_text_from_html(html)
                        # Apply Tamil quality filter (relaxed for classical texts)
                        cleaned = clean_and_filter_text(text, min_tamil_ratio=0.50)

                        if len(cleaned) > 100:  # At least 100 chars of content
                            total_bytes += save_clean_text(
                                cleaned + '\n\n',
                                output_file
                            )
                            processed += 1

                    pbar.update(1)

        logger.info(
            f"Project Madurai: {processed} texts, "
            f"{total_bytes / (1024**2):.1f} MB collected"
        )
        return output_file


# ============================================================
# TAMIL WIKISOURCE COLLECTOR
# ============================================================

class TamilWikisourceCollector:
    """
    Collects classical Tamil texts from Tamil Wikisource.

    Uses the MediaWiki API to enumerate and download pages.
    Wikisource contains curated classical works including:
    - Thirukkural, Sangam anthologies, Kambaramayanam
    - Naalayira Divya Prabandham, Thevaram
    """

    API_URL = "https://ta.wikisource.org/w/api.php"

    def __init__(self, output_dir: Path, max_workers: int = 4):
        self.output_dir = output_dir / "wikisource"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TamilukuLLM-Research/1.0 (Tamil NLP Research)'
        })

    def get_all_pages(self, namespace: int = 0, limit: int = 5000) -> list[str]:
        """Enumerate all pages in Tamil Wikisource main namespace."""
        pages = []
        params = {
            'action': 'query',
            'list': 'allpages',
            'apnamespace': namespace,
            'aplimit': 500,
            'format': 'json'
        }

        while len(pages) < limit:
            try:
                response = self.session.get(self.API_URL, params=params, timeout=30)
                data = response.json()

                batch = data.get('query', {}).get('allpages', [])
                pages.extend([p['title'] for p in batch])

                # Check for continuation
                if 'continue' in data:
                    params['apcontinue'] = data['continue']['apcontinue']
                else:
                    break

            except Exception as e:
                logger.error(f"Wikisource API error: {e}")
                break

        logger.info(f"Found {len(pages)} Wikisource pages")
        return pages[:limit]

    def get_page_text(self, title: str) -> Optional[str]:
        """Get plain text content of a Wikisource page via revisions API."""
        # Use revisions API instead of extracts (more reliable for Wikisource)
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main',
            'format': 'json',
            'formatversion': '2',
        }

        for attempt in range(3):
            try:
                response = self.session.get(self.API_URL, params=params, timeout=30)
                if response.status_code != 200:
                    time.sleep(2 ** attempt)
                    continue
                data = response.json()

                pages = data.get('query', {}).get('pages', [])
                for page_data in pages:
                    if page_data.get('missing'):
                        continue
                    revisions = page_data.get('revisions', [])
                    if revisions:
                        content = revisions[0].get('slots', {}).get('main', {}).get('content', '')
                        if content:
                            # Strip wiki markup (basic)
                            import mwparserfromhell
                            try:
                                parsed = mwparserfromhell.parse(content)
                                return parsed.strip_code(normalize=True, collapse=True)
                            except Exception:
                                # Fallback: regex cleanup
                                text = re.sub(r'\{\{[^}]*\}\}', '', content)
                                text = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', text)
                                text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
                                text = re.sub(r'<[^>]+>', '', text)
                                text = re.sub(r"'{2,}", '', text)
                                text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)
                                return text
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to get page '{title}': {e}")
                time.sleep(1)

        return None

    def collect(self, max_pages: int = 5000) -> Path:
        """Collect all Tamil Wikisource texts."""
        output_file = self.output_dir / "wikisource_combined.txt"
        logger.info(f"Starting Wikisource collection -> {output_file}")

        pages = self.get_all_pages(limit=max_pages)
        total_bytes = 0
        processed = 0

        # Process in batches of 20 to avoid API rate-limiting
        batch_size = 20
        with tqdm(total=len(pages), desc="Wikisource") as pbar:
            for i in range(0, len(pages), batch_size):
                batch = pages[i:i + batch_size]
                for title in batch:
                    text = self.get_page_text(title)
                    if text:
                        cleaned = clean_and_filter_text(text, min_tamil_ratio=0.50)
                        if len(cleaned) > 50:
                            total_bytes += save_clean_text(
                                cleaned + '\n\n',
                                output_file
                            )
                            processed += 1
                    pbar.update(1)
                    pbar.set_postfix({'texts': processed, 'MB': f"{total_bytes/(1024**2):.1f}"})
                # Rate-limit: small delay between batches
                time.sleep(0.5)

        logger.info(
            f"Wikisource: {processed} pages, "
            f"{total_bytes / (1024**2):.1f} MB collected"
        )
        return output_file


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def collect_classical(output_dir: Path, max_workers: int = 4) -> list[Path]:
    """
    Collect all classical/literary Tamil texts.
    Returns list of output file paths.
    """
    output_dir = Path(output_dir) / "classical"
    output_files = []

    # 1. Project Madurai
    logger.info("=" * 60)
    logger.info("PHASE 1: Project Madurai (Classical Tamil E-Library)")
    logger.info("=" * 60)
    pm = ProjectMaduraiCollector(output_dir, max_workers=max_workers)
    output_files.append(pm.collect())

    # 2. Tamil Wikisource
    logger.info("=" * 60)
    logger.info("PHASE 2: Tamil Wikisource (Curated Classical Texts)")
    logger.info("=" * 60)
    ws = TamilWikisourceCollector(output_dir, max_workers=max_workers)
    output_files.append(ws.collect())

    return output_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Classical Tamil Texts")
    parser.add_argument("--output-dir", type=str, default="./tamil_corpus")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    files = collect_classical(Path(args.output_dir), args.max_workers)
    print(f"\nCollected files: {[str(f) for f in files]}")
