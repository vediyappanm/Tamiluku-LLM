"""
collect_corpus.py - Production-Grade Tamil Corpus Collection Pipeline
======================================================================
Downloads and aggregates 20+ GB of Tamil text from multiple public sources:

  1. Tamil Wikipedia       (~600 MB)  - XML dump -> plain text
  2. CulturaX Tamil        (~5 GB)    - UCSB cleaned CommonCrawl + mC4
  3. OSCAR-2301 Tamil      (~5 GB)    - Web crawl via HuggingFace
  4. IndicCorp v2 Tamil    (~3 GB)    - AI4Bharat curated Indian corpus
  5. CC-100 Tamil          (~3 GB)    - Facebook's cleaned CommonCrawl
  6. Samanantar Tamil      (~500 MB)  - AI4Bharat parallel corpus (Tamil side)
  7. mC4 Tamil             (~5 GB)    - Google's multilingual C4

Usage:
    python collect_corpus.py [--config config.yaml]
    python collect_corpus.py --sources wikipedia,culturax,oscar
    python collect_corpus.py --max-docs 1000000
"""

import os
import sys
import bz2
import re
import argparse
import logging
from pathlib import Path
from typing import Generator, Optional

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict):
    for key in ("raw_dir", "cleaned_dir", "eval_dir"):
        Path(cfg["corpus"][key]).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)


def count_file_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count


def write_docs_streaming(txt_path: Path, docs_iter, desc: str, max_docs: int = 0):
    """Write documents from an iterator to a text file with progress tracking."""
    count = 0
    bytes_written = 0
    with open(str(txt_path), "w", encoding="utf-8") as out:
        for doc in tqdm(docs_iter, desc=desc):
            if max_docs > 0 and count >= max_docs:
                break
            text = doc.strip()
            if len(text) > 50:
                out.write(text + "\n\n")
                count += 1
                bytes_written += len(text.encode("utf-8")) + 2
    size_mb = bytes_written / (1024 * 1024)
    log.info(f"  {desc}: saved {count:,} documents ({size_mb:.1f} MB) to {txt_path}")
    return count


# ---------------------------------------------------------------------------
# Source 1: Tamil Wikipedia (Fixed extraction)
# ---------------------------------------------------------------------------

def _extract_wiki_text_iterparse(xml_path: str) -> Generator[str, None, None]:
    """
    Extract plain text from MediaWiki XML dump using iterparse.
    Handles multiple XML namespace versions and large files efficiently.
    """
    try:
        import mwparserfromhell
        use_mw = True
    except ImportError:
        log.warning("mwparserfromhell not installed. Using regex fallback.")
        use_mw = False

    from xml.etree import ElementTree as ET

    log.info(f"Parsing Wikipedia XML: {xml_path}")

    # Auto-detect namespace from the XML file
    namespace = ""
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ns_match = re.search(r'xmlns="(http://www\.mediawiki\.org/xml/export-[^"]+)"', line)
            if ns_match:
                namespace = f"{{{ns_match.group(1)}}}"
                log.info(f"  Detected XML namespace: {ns_match.group(1)}")
                break
            if line.strip().startswith("<page"):
                break

    text_tag = f"{namespace}text"
    page_tag = f"{namespace}page"
    title_tag = f"{namespace}title"
    ns_tag = f"{namespace}ns"
    count = 0

    try:
        context = ET.iterparse(xml_path, events=("end",))
        for event, elem in context:
            if elem.tag == page_tag:
                # Check namespace - only article pages (ns=0)
                ns_elem = elem.find(ns_tag)
                if ns_elem is not None and ns_elem.text != "0":
                    elem.clear()
                    continue

                text_elem = elem.find(f".//{text_tag}")
                if text_elem is not None and text_elem.text:
                    raw = text_elem.text

                    # Skip redirects
                    if raw.strip().upper().startswith("#REDIRECT") or raw.strip().startswith("#வழிமாற்று"):
                        elem.clear()
                        continue

                    try:
                        if use_mw:
                            import mwparserfromhell
                            wikicode = mwparserfromhell.parse(raw)
                            plain = wikicode.strip_code()
                        else:
                            plain = _regex_clean_wiki(raw)

                        # Additional cleanup
                        plain = re.sub(r"\[\[Category:.*?\]\]", "", plain, flags=re.IGNORECASE)
                        plain = re.sub(r"\[\[பகுப்பு:.*?\]\]", "", plain)
                        plain = re.sub(r"\{\{[^}]*\}\}", "", plain, flags=re.DOTALL)
                        plain = re.sub(r"\{\|.*?\|\}", "", plain, flags=re.DOTALL)
                        plain = re.sub(r"==+\s*", "\n", plain)
                        plain = re.sub(r"\s*==+", "\n", plain)
                        plain = re.sub(r"\n{3,}", "\n\n", plain)
                        plain = plain.strip()

                        if len(plain) > 100:
                            yield plain
                            count += 1
                            if count % 10000 == 0:
                                log.info(f"  Extracted {count:,} articles...")
                    except Exception:
                        pass

                elem.clear()

    except ET.ParseError as e:
        log.warning(f"XML iterparse failed at article {count}: {e}")
        log.info("Falling back to line-by-line extraction...")
        yield from _extract_wiki_text_linewise(xml_path, start_from=count)
        return

    log.info(f"Extracted {count:,} articles from Wikipedia")


_FILE_LINK_PREFIX_RE = re.compile(
    r"^\s*(?:file|image|media|\u0BAA\u0B9F\u0BBF\u0BAE\u0BAE\u0BCD)\s*:",
    re.IGNORECASE,
)


def _strip_wiki_file_links(text: str) -> str:
    """Remove file/image/media links like [[File:...|thumb|...]] with nesting support."""
    if "[[" not in text:
        return text

    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i + 2] == "[[":
            probe = text[i + 2:i + 64]
            if _FILE_LINK_PREFIX_RE.match(probe):
                depth = 1
                i += 2
                while i < n and depth > 0:
                    if text[i:i + 2] == "[[":
                        depth += 1
                        i += 2
                        continue
                    if text[i:i + 2] == "]]":
                        depth -= 1
                        i += 2
                        continue
                    i += 1
                continue
        out.append(text[i])
        i += 1

    return "".join(out)


def _regex_clean_wiki(raw: str) -> str:
    """Clean MediaWiki markup using regex (fallback when mwparserfromhell unavailable)."""
    text = raw
    # Remove templates
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}":
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    text = "".join(result)

    # Remove File/Image/Media links (prevents "thumb|..." leakage)
    text = _strip_wiki_file_links(text)

    # Wikilinks: [[target|display]] -> display, [[target]] -> target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    # External links: [url text] -> text
    text = re.sub(r'\[https?://[^\s\]]+ ([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]*\]', '', text)
    # HTML tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Bold/italic
    text = re.sub(r"'{2,5}", '', text)
    # Lists
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    return text


def _extract_wiki_text_linewise(xml_path: str, start_from: int = 0) -> Generator[str, None, None]:
    """Fallback line-by-line XML extraction for malformed dumps."""
    count = 0
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        content = []
        in_text = False
        for line in f:
            if "<text" in line:
                in_text = True
                # Extract content after the <text ...> tag
                match = re.search(r'<text[^>]*>(.*)', line, re.DOTALL)
                if match:
                    line = match.group(1)
                else:
                    continue

            if in_text:
                end_idx = line.find("</text>")
                if end_idx != -1:
                    content.append(line[:end_idx])
                    text_blob = "".join(content)

                    if not text_blob.strip().upper().startswith("#REDIRECT"):
                        text_blob = _regex_clean_wiki(text_blob)
                        text_blob = re.sub(r"\[\[Category:.*?\]\]", "", text_blob)
                        text_blob = re.sub(r"\[\[பகுப்பு:.*?\]\]", "", text_blob)
                        text_blob = re.sub(r"\n{3,}", "\n\n", text_blob)
                        text_blob = text_blob.strip()

                        if len(text_blob) > 100:
                            count += 1
                            if count > start_from:
                                yield text_blob

                    content = []
                    in_text = False
                else:
                    content.append(line)

    log.info(f"Line-by-line extraction: {count:,} articles")


def download_wikipedia(cfg: dict, max_docs: int = 0):
    """Download and extract Tamil Wikipedia dump."""
    import urllib.request

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    url = cfg["corpus"]["sources"]["wikipedia"]["url"]
    bz2_path = raw_dir / "tawiki-latest.xml.bz2"
    xml_path = raw_dir / "tawiki-latest.xml"
    txt_path = raw_dir / "wikipedia_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"Wikipedia text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    # Download
    if not bz2_path.exists():
        log.info(f"Downloading Tamil Wikipedia dump ({url})...")
        urllib.request.urlretrieve(url, str(bz2_path))
        log.info(f"Downloaded to {bz2_path}")

    # Decompress
    if not xml_path.exists():
        log.info("Decompressing bz2 (this may take several minutes)...")
        with bz2.open(str(bz2_path), "rb") as f_in, open(str(xml_path), "wb") as f_out:
            total_written = 0
            for chunk in iter(lambda: f_in.read(4 * 1024 * 1024), b""):
                f_out.write(chunk)
                total_written += len(chunk)
                if total_written % (100 * 1024 * 1024) == 0:
                    log.info(f"  Decompressed {total_written / (1024*1024):.0f} MB...")
        log.info(f"Decompressed to {xml_path} ({total_written / (1024*1024):.0f} MB)")

    # Extract text
    log.info("Extracting articles from Wikipedia XML...")
    docs_iter = _extract_wiki_text_iterparse(str(xml_path))
    write_docs_streaming(txt_path, docs_iter, "Wikipedia", max_docs)

    log.info(f"Wikipedia Tamil text saved to {txt_path}")


# ---------------------------------------------------------------------------
# Source 2: CulturaX (Cleaned CommonCrawl + mC4, best quality)
# ---------------------------------------------------------------------------

def download_culturax(cfg: dict, max_docs: int = 0):
    """Download Tamil split from CulturaX - highest quality web corpus."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    txt_path = raw_dir / "culturax_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"CulturaX text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    cx_cfg = cfg["corpus"]["sources"]["culturax"]
    log.info(f"Downloading CulturaX Tamil ({cx_cfg['dataset']})...")
    log.info("  This is a large dataset (~5 GB). Download may take a while.")

    try:
        ds = load_dataset(
            cx_cfg["dataset"],
            cx_cfg["language"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        def doc_iter():
            for sample in ds:
                text = sample.get("text", "")
                if text and len(text.strip()) > 50:
                    yield text.strip()

        write_docs_streaming(txt_path, doc_iter(), "CulturaX", max_docs)

    except Exception as e:
        log.warning(f"CulturaX download failed: {e}")
        log.info("  You may need: huggingface-cli login")
        log.info("  Or manually download and place in data/raw/culturax_ta.txt")


# ---------------------------------------------------------------------------
# Source 3: OSCAR-2301
# ---------------------------------------------------------------------------

def download_oscar(cfg: dict, max_docs: int = 0):
    """Download Tamil split from OSCAR via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    txt_path = raw_dir / "oscar_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"OSCAR text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    oscar_cfg = cfg["corpus"]["sources"]["oscar"]
    log.info(f"Downloading OSCAR Tamil ({oscar_cfg['dataset']})...")

    try:
        ds = load_dataset(
            oscar_cfg["dataset"],
            language=oscar_cfg["language"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        def doc_iter():
            for sample in ds:
                text = sample.get("text", "")
                if text and len(text.strip()) > 50:
                    yield text.strip()

        write_docs_streaming(txt_path, doc_iter(), "OSCAR", max_docs)

    except Exception as e:
        log.warning(f"OSCAR download failed (may need HF token): {e}")
        log.info("  Skipping. Place file manually in data/raw/oscar_ta.txt")


# ---------------------------------------------------------------------------
# Source 4: IndicCorp v2
# ---------------------------------------------------------------------------

def download_indiccorp(cfg: dict, max_docs: int = 0):
    """Download Tamil split from IndicCorp via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    txt_path = raw_dir / "indiccorp_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"IndicCorp text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    ic_cfg = cfg["corpus"]["sources"]["indiccorp"]
    log.info(f"Downloading IndicCorp Tamil ({ic_cfg['dataset']})...")

    try:
        ds = load_dataset(
            ic_cfg["dataset"],
            ic_cfg["language"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        def doc_iter():
            for sample in ds:
                text = sample.get("text", "")
                if text and len(text.strip()) > 50:
                    yield text.strip()

        write_docs_streaming(txt_path, doc_iter(), "IndicCorp", max_docs)

    except Exception as e:
        log.warning(f"IndicCorp download failed: {e}")
        log.info("  Skipping. Place file manually in data/raw/indiccorp_ta.txt")


# ---------------------------------------------------------------------------
# Source 5: CC-100 Tamil
# ---------------------------------------------------------------------------

def download_cc100(cfg: dict, max_docs: int = 0):
    """Download Tamil split from CC-100 (Facebook's cleaned CommonCrawl)."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    txt_path = raw_dir / "cc100_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"CC-100 text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    cc_cfg = cfg["corpus"]["sources"]["cc100"]
    log.info(f"Downloading CC-100 Tamil ({cc_cfg['dataset']})...")

    try:
        ds = load_dataset(
            cc_cfg["dataset"],
            lang=cc_cfg["language"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        def doc_iter():
            for sample in ds:
                text = sample.get("text", "")
                if text and len(text.strip()) > 50:
                    yield text.strip()

        write_docs_streaming(txt_path, doc_iter(), "CC-100", max_docs)

    except Exception as e:
        log.warning(f"CC-100 download failed: {e}")
        log.info("  Skipping. Place file manually in data/raw/cc100_ta.txt")


# ---------------------------------------------------------------------------
# Source 6: Samanantar (Parallel corpus - Tamil side only)
# ---------------------------------------------------------------------------

def download_samanantar(cfg: dict, max_docs: int = 0):
    """Download Tamil side from Samanantar parallel corpus."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    txt_path = raw_dir / "samanantar_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"Samanantar text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    sam_cfg = cfg["corpus"]["sources"]["samanantar"]
    log.info(f"Downloading Samanantar Tamil ({sam_cfg['dataset']})...")

    try:
        ds = load_dataset(
            sam_cfg["dataset"],
            sam_cfg["language"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        def doc_iter():
            for sample in ds:
                # Samanantar has tgt (Tamil) and src (English) fields
                text = sample.get("tgt", sample.get("text", ""))
                if text and len(text.strip()) > 20:
                    yield text.strip()

        write_docs_streaming(txt_path, doc_iter(), "Samanantar", max_docs)

    except Exception as e:
        log.warning(f"Samanantar download failed: {e}")
        log.info("  Skipping. Place file manually in data/raw/samanantar_ta.txt")


# ---------------------------------------------------------------------------
# Source 7: mC4 Tamil
# ---------------------------------------------------------------------------

def download_mc4(cfg: dict, max_docs: int = 0):
    """Download Tamil split from mC4 (Google's multilingual C4)."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    txt_path = raw_dir / "mc4_ta.txt"

    if txt_path.exists() and txt_path.stat().st_size > 1000:
        size_mb = txt_path.stat().st_size / (1024 * 1024)
        log.info(f"mC4 text already exists: {txt_path} ({size_mb:.1f} MB)")
        return

    mc4_cfg = cfg["corpus"]["sources"]["mc4"]
    log.info(f"Downloading mC4 Tamil ({mc4_cfg['dataset']})...")

    try:
        ds = load_dataset(
            mc4_cfg["dataset"],
            languages=[mc4_cfg["language"]],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        def doc_iter():
            for sample in ds:
                text = sample.get("text", "")
                if text and len(text.strip()) > 50:
                    yield text.strip()

        write_docs_streaming(txt_path, doc_iter(), "mC4", max_docs)

    except Exception as e:
        log.warning(f"mC4 download failed: {e}")
        log.info("  Skipping. Place file manually in data/raw/mc4_ta.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCE_HANDLERS = {
    "wikipedia": download_wikipedia,
    "culturax": download_culturax,
    "oscar": download_oscar,
    "indiccorp": download_indiccorp,
    "cc100": download_cc100,
    "samanantar": download_samanantar,
    "mc4": download_mc4,
}


def main():
    parser = argparse.ArgumentParser(description="Collect Tamil corpus from public sources")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--sources", default=None,
        help="Comma-separated source names to download (default: all enabled in config)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help="Max documents per source (0 = unlimited)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    # Determine which sources to download
    if args.sources:
        source_names = [s.strip() for s in args.sources.split(",")]
    else:
        source_names = [
            name for name, src_cfg in cfg["corpus"]["sources"].items()
            if src_cfg.get("enabled", False)
        ]

    log.info(f"=== Tamil Corpus Collection Pipeline ===")
    log.info(f"Sources to download: {', '.join(source_names)}")
    if args.max_docs > 0:
        log.info(f"Max documents per source: {args.max_docs:,}")

    # Download each source
    for name in source_names:
        handler = SOURCE_HANDLERS.get(name)
        if handler is None:
            log.warning(f"Unknown source: {name}. Available: {list(SOURCE_HANDLERS.keys())}")
            continue

        log.info(f"\n--- Downloading: {name} ---")
        try:
            handler(cfg, max_docs=args.max_docs)
        except Exception as e:
            log.error(f"{name} processing failed: {e}")
            log.info("Continuing with other sources...")

    # Summary
    raw_dir = Path(cfg["corpus"]["raw_dir"])
    log.info(f"\n{'='*60}")
    log.info(f"CORPUS COLLECTION SUMMARY")
    log.info(f"{'='*60}")

    total_bytes = 0
    total_docs = 0
    for txt_file in sorted(raw_dir.glob("*.txt")):
        size = txt_file.stat().st_size
        size_mb = size / (1024 * 1024)
        # Estimate docs by counting double-newlines
        doc_count = 0
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip() == "":
                    doc_count += 1
        doc_count = max(doc_count // 2, 1)  # Rough estimate
        total_bytes += size
        total_docs += doc_count
        log.info(f"  {txt_file.name:<30} {size_mb:>8.1f} MB  (~{doc_count:>10,} docs)")

    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)
    log.info(f"{'='*60}")
    log.info(f"  TOTAL: {total_gb:.2f} GB ({total_mb:.0f} MB), ~{total_docs:,} documents")
    log.info(f"{'='*60}")

    if total_mb < 500:
        log.warning(
            f"Corpus is only {total_mb:.0f} MB. For production-grade tokenization, "
            "target at least 10 GB. Enable more sources in config.yaml."
        )
    elif total_mb < 5000:
        log.info("Good corpus size for initial training. For GPT-4-class quality, target 20+ GB.")
    else:
        log.info("Excellent corpus size for production-grade tokenizer training!")

    log.info(f"\nNext step: python normalize.py")


if __name__ == "__main__":
    main()
