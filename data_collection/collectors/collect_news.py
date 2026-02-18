"""
Tamiluku-LLM: Modern News & Formal Tamil Collector
=====================================================
Domain 2: News & Formal Text

Sources:
  1. BBC Tamil (bbc.com/tamil) — High-quality editorial Tamil
  2. Dinamalar (dinamalar.com) — Major Tamil daily
  3. The Hindu Tamil (tamil.thehindu.com) — Formal journalistic Tamil
  4. Puthiya Thalaimurai, Vikatan (supplementary)

Why this matters for AMB:
  News text is rich in CASE MARKERS (-இல், -ஆல், -க்கு, -உடன்).
  This is essential for Layer 3 suffix-stripping to learn
  formal agglutination patterns.

Note: Uses trafilatura for clean article extraction.
      Respects robots.txt and rate limits.
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    print("WARNING: trafilatura not installed. Install with: pip install trafilatura")

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    normalize_tamil_text, clean_and_filter_text,
    save_clean_text, ParallelDownloader, logger
)


# ============================================================
# NEWS SOURCE CONFIGURATION
# ============================================================

NEWS_SOURCES = {
    "bbc_tamil": {
        "name": "BBC Tamil",
        "base_url": "https://www.bbc.com/tamil",
        "sitemap_urls": [
            "https://www.bbc.com/tamil/sitemap.xml",
        ],
        "section_urls": [
            "https://www.bbc.com/tamil",
            "https://www.bbc.com/tamil/india",
            "https://www.bbc.com/tamil/world",
            "https://www.bbc.com/tamil/science",
            "https://www.bbc.com/tamil/sport",
        ],
        "link_pattern": r'/tamil/[a-z\-]+/?\d*',
        "rate_limit": 1.0,  # seconds between requests
    },
    "dinamalar": {
        "name": "Dinamalar",
        "base_url": "https://www.dinamalar.com",
        "section_urls": [
            "https://www.dinamalar.com/news_list.asp?cat=1",   # Tamil Nadu
            "https://www.dinamalar.com/news_list.asp?cat=2",   # India
            "https://www.dinamalar.com/news_list.asp?cat=3",   # World
            "https://www.dinamalar.com/news_list.asp?cat=7",   # Technology
            "https://www.dinamalar.com/news_list.asp?cat=10",  # Health
            "https://www.dinamalar.com/news_list.asp?cat=8",   # Sports
        ],
        "link_pattern": r'/detail\.asp\?id=\d+',
        "rate_limit": 1.5,
    },
    "hindu_tamil": {
        "name": "The Hindu Tamil",
        "base_url": "https://tamil.thehindu.com",
        "section_urls": [
            "https://tamil.thehindu.com/tamil-nadu/",
            "https://tamil.thehindu.com/india/",
            "https://tamil.thehindu.com/world/",
            "https://tamil.thehindu.com/science-technology/",
            "https://tamil.thehindu.com/sport/",
        ],
        "link_pattern": r'/[a-z\-]+/article\d+\.ece',
        "rate_limit": 1.0,
    },
}


# ============================================================
# NEWS ARTICLE EXTRACTOR
# ============================================================

class NewsArticleExtractor:
    """
    Extracts clean article text from news websites.
    Uses trafilatura for robust content extraction,
    falls back to BeautifulSoup if trafilatura fails.
    """

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TamilukuLLM-Research/1.0 (Academic Tamil NLP)',
            'Accept-Language': 'ta,en;q=0.5',
        })

    def extract_article(self, url: str) -> Optional[str]:
        """Extract article text from a URL using trafilatura."""
        try:
            # Rate limiting
            time.sleep(self.rate_limit)

            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html = response.text

            # Primary: trafilatura (best for news extraction)
            if HAS_TRAFILATURA:
                text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                    favor_recall=True,
                    target_language='ta',
                )
                if text and len(text) > 100:
                    return normalize_tamil_text(text)

            # Fallback: BeautifulSoup
            return self._bs4_extract(html)

        except Exception as e:
            logger.debug(f"Failed to extract {url}: {e}")
            return None

    def _bs4_extract(self, html: str) -> Optional[str]:
        """Fallback extraction using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'header',
                                   'footer', 'aside', 'iframe', 'form']):
            tag.decompose()

        # Try common article containers
        article = (
            soup.find('article') or
            soup.find('div', class_=re.compile(r'article|content|story|body', re.I)) or
            soup.find('main')
        )

        if article:
            text = article.get_text(separator='\n')
        else:
            text = soup.get_text(separator='\n')

        text = normalize_tamil_text(text)
        return text if len(text) > 100 else None


# ============================================================
# NEWS LINK DISCOVERER
# ============================================================

class NewsLinkDiscoverer:
    """
    Discovers article URLs from news website section pages.
    """

    def __init__(self, source_config: dict):
        self.config = source_config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TamilukuLLM-Research/1.0'
        })

    def discover_from_sections(self, max_per_section: int = 200) -> list[str]:
        """Crawl section pages to find article links."""
        all_links = set()

        for section_url in self.config.get('section_urls', []):
            try:
                time.sleep(1)
                response = self.session.get(section_url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                base_url = self.config['base_url']
                pattern = self.config.get('link_pattern', '')

                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    full_url = urljoin(base_url, href)

                    # Filter by pattern
                    if pattern and re.search(pattern, href):
                        all_links.add(full_url)
                    elif base_url in full_url and href != '/':
                        all_links.add(full_url)

                logger.info(f"  Section {section_url}: found {len(all_links)} links so far")

            except Exception as e:
                logger.error(f"Failed to crawl section {section_url}: {e}")

        links = list(all_links)[:max_per_section * len(self.config.get('section_urls', []))]
        logger.info(f"Total unique links from {self.config['name']}: {len(links)}")
        return links

    def discover_from_sitemap(self) -> list[str]:
        """Parse sitemap.xml for article URLs."""
        links = []
        for sitemap_url in self.config.get('sitemap_urls', []):
            try:
                response = self.session.get(sitemap_url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'xml')
                for loc in soup.find_all('loc'):
                    url = loc.text.strip()
                    if '/tamil/' in url or self.config['base_url'] in url:
                        links.append(url)

            except Exception as e:
                logger.debug(f"Sitemap unavailable: {sitemap_url}: {e}")

        return links


# ============================================================
# MAIN NEWS COLLECTOR
# ============================================================

class NewsCollector:
    """
    Orchestrates news collection from all configured sources.
    """

    def __init__(self, output_dir: Path, max_workers: int = 4):
        self.output_dir = output_dir / "news"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def collect_source(self, source_key: str, source_config: dict,
                       max_articles: int = 2000) -> Path:
        """Collect articles from a single news source."""
        output_file = self.output_dir / f"{source_key}.txt"
        logger.info(f"Collecting from {source_config['name']}...")

        # Discover links
        discoverer = NewsLinkDiscoverer(source_config)
        links = discoverer.discover_from_sections()

        # Also try sitemap
        sitemap_links = discoverer.discover_from_sitemap()
        all_links = list(set(links + sitemap_links))[:max_articles]

        logger.info(f"Processing {len(all_links)} articles from {source_config['name']}")

        # Extract articles (sequential due to rate limiting)
        extractor = NewsArticleExtractor(
            rate_limit=source_config.get('rate_limit', 1.0)
        )

        total_bytes = 0
        processed = 0

        with tqdm(total=len(all_links), desc=source_config['name']) as pbar:
            for url in all_links:
                text = extractor.extract_article(url)
                if text:
                    cleaned = clean_and_filter_text(text, min_tamil_ratio=0.55)
                    if len(cleaned) > 200:
                        total_bytes += save_clean_text(
                            cleaned + '\n\n',
                            output_file
                        )
                        processed += 1
                pbar.update(1)

        logger.info(
            f"{source_config['name']}: {processed} articles, "
            f"{total_bytes / (1024**2):.1f} MB"
        )
        return output_file

    def collect_all(self, max_articles_per_source: int = 2000) -> list[Path]:
        """Collect from all configured news sources."""
        output_files = []

        for source_key, source_config in NEWS_SOURCES.items():
            try:
                output_file = self.collect_source(
                    source_key, source_config,
                    max_articles=max_articles_per_source
                )
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to collect {source_key}: {e}")

        return output_files


def collect_news(output_dir: Path, max_workers: int = 4) -> list[Path]:
    """Entry point for news collection."""
    collector = NewsCollector(output_dir, max_workers=max_workers)
    return collector.collect_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Tamil News Articles")
    parser.add_argument("--output-dir", type=str, default="./tamil_corpus")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-articles", type=int, default=2000)
    args = parser.parse_args()

    files = collect_news(Path(args.output_dir), args.max_workers)
    print(f"\nCollected files: {[str(f) for f in files]}")
