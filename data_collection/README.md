# Tamiluku-LLM: Tamil Gold Corpus Collection Pipeline

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  MASTER ORCHESTRATOR                     │
│              orchestrator.py (Entry Point)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Domain 1 │ │ Domain 2 │ │ Domain 3 │ │ Domain 4 │   │
│  │Classical │ │  News &  │ │ Wiki &   │ │Colloquial│   │
│  │Literary  │ │  Formal  │ │Technical │ │ & Social │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       │            │            │            │          │
│  ┌────▼────────────▼────────────▼────────────▼────┐     │
│  │         SHARED UTILITIES (utils.py)            │     │
│  │  • Tamil Script Detection (Unicode Ranges)     │     │
│  │  • Code-Mix Filter (English-Heavy Removal)     │     │
│  │  • UTF-8 Normalization (NFC)                   │     │
│  │  • Parallel Download Manager                   │     │
│  │  • Deduplication (MinHash)                     │     │
│  └────────────────────┬───────────────────────────┘     │
│                       │                                  │
│              ┌────────▼────────┐                         │
│              │  MERGER SCRIPT  │                         │
│              │  merge_gold.py  │                         │
│              │                 │                         │
│              │ raw_tamil_gold  │                         │
│              │     .txt        │                         │
│              └─────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python orchestrator.py --output-dir ./tamil_corpus --target-gb 10

# 3. Or run individual collectors
python collectors/collect_wiki.py --output-dir ./tamil_corpus/wiki
python collectors/collect_hf_datasets.py --output-dir ./tamil_corpus/hf
python collectors/collect_news.py --output-dir ./tamil_corpus/news
python collectors/collect_classical.py --output-dir ./tamil_corpus/classical

# 4. Merge all into gold file
python merge_gold.py --input-dir ./tamil_corpus --output raw_tamil_gold.txt
```

## Expected Corpus Composition

| Domain              | Source                        | Est. Size |
|---------------------|-------------------------------|-----------|
| Classical/Literary  | Project Madurai, Wikisource   | ~500MB    |
| News/Formal         | BBC Tamil, News sites         | ~1.5GB    |
| Wiki/Technical      | Tamil Wikipedia dump          | ~800MB    |
| Colloquial/Social   | CulturaX, IndicCorp, OSCAR   | ~6GB      |
| Legal/Admin         | TN Govt Gazettes              | ~500MB    |
| **Total Target**    |                               | **~10GB** |

## Code-Mix Filtering Logic

Lines are rejected if:
- Tamil character ratio < 60% of all script characters
- Contains > 5 consecutive English words
- Total English words > 40% of all words

This preserves natural Tamil text with occasional English terms
(common in technical/modern writing) while removing Tanglish-heavy content.
