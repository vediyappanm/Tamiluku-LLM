# train_local.py - Memory-Safe AMB Tokenizer Training for 8GB RAM
# Usage: python tokenizer/train_local.py
#        python tokenizer/train_local.py --max-mb 150
#        python tokenizer/train_local.py --vocab-size 32000

import os, sys, json, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
from tokenizers.pre_tokenizers import Split, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Build special tokens without angle brackets in source (avoids tool issues)
_ST = ["endoftext", "padding", "im_start", "im_end"]
SPECIAL_TOKENS = ["<|" + t + "|>" for t in _ST]


def create_safe_subset(corpus_path, output_path, max_mb=100):
    """Stream first N MB, apply morpheme segmentation. RAM ~300MB."""
    from tamil_unicode import TamilDeepNormalizer
    from morpheme import MorphemeSegmenter

    norm = TamilDeepNormalizer(
        strip_urls=True, strip_emails=True,
        normalize_numerals="preserve", preserve_grantha=True,
    )
    mseg = MorphemeSegmenter()

    max_bytes = max_mb * 1024 * 1024
    written = 0
    count = 0

    log.info(f"Creating {max_mb} MB segmented subset ...")

    with open(str(corpus_path), "r", encoding="utf-8") as fin, \
         open(str(output_path), "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Segmenting", unit=" lines"):
            if written >= max_bytes:
                break
            line = line.strip()
            if not line:
                continue

            cleaned = norm.normalize(line)
            words = cleaned.split()
            seg = []
            for w in words:
                m = mseg.segment_word(w)
                seg.append(m.replace(" ", " @@ "))

            out = " ".join(seg) + "\n"
            fout.write(out)
            written += len(out.encode("utf-8"))
            count += 1

            if count % 50000 == 0:
                log.info(f"  {count:,} lines, {written / 1048576:.1f} MB ...")

    log.info(f"  Subset: {count:,} lines, {written / 1048576:.1f} MB")
    return count


def train_bpe(seg_path, output_dir, vocab_size=48000):
    """Train BPE on pre-segmented file. RAM ~1-2GB for 100MB corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()

    ISOLATOR = r"[\u0B80-\u0BFF]+|[a-zA-Z]+|[0-9]+|[^\s\u0B80-\u0BFFa-zA-Z0-9]+"

    tok.pre_tokenizer = Sequence([
        Split(pattern=" @@ ", behavior="isolated"),
        Split(pattern=ISOLATOR, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    log.info(f"Training {vocab_size:,} vocab BPE on {seg_path} ...")
    tok.train([str(seg_path)], trainer)

    tok.decoder = decoders.ByteLevel()

    model_path = output_dir / "tokenizer.json"
    tok.save(str(model_path))

    cfg = {
        "model_type": "gpt2",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": vocab_size,
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log.info(f"Saved to {output_dir}")
    return str(model_path)


def sanity_check(path):
    tok = Tokenizer.from_file(path)
    log.info(f"Vocab size: {tok.get_vocab_size():,}")
    tests = [
        "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb4\u0b95\u0bbe\u0ba9 \u0bae\u0bca\u0bb4\u0bbf.",
        "\u0bb5\u0bc0\u0b9f\u0bc1\u0b95\u0bb3\u0bbf\u0bb2\u0bbf\u0bb0\u0bc1\u0ba8\u0bcd\u0ba4\u0bc1 \u0bb5\u0ba8\u0bcd\u0ba4\u0bbe\u0ba9\u0bcd.",
        "Machine learning model train \u0baa\u0ba3\u0bcd\u0ba3\u0ba9\u0bc1\u0bae\u0bcd.",
    ]
    for s in tests:
        enc = tok.encode(s)
        nw = max(len(s.split()), 1)
        dec = tok.decode(enc.ids)
        ok = "OK" if dec.strip() == s.strip() else "MISMATCH"
        log.info(f"  Input:     {s}")
        log.info(f"  Tokens:    {enc.tokens[:15]}")
        log.info(f"  Fertility: {len(enc.tokens)/nw:.2f}")
        log.info(f"  Roundtrip: {ok}")


def main():
    p = argparse.ArgumentParser(description="Memory-safe AMB training")
    p.add_argument("--corpus", default="tamil_corpus.txt")
    p.add_argument("--max-mb", type=int, default=100,
                   help="Max MB subset (default 100, safe for 8GB RAM)")
    p.add_argument("--vocab-size", type=int, default=48000)
    p.add_argument("--output-dir", default="tokenizer/models/amb_tokenizer")
    args = p.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        log.error(f"Corpus not found: {corpus}")
        sys.exit(1)

    seg = corpus.with_suffix(".local_seg.txt")

    if not seg.exists():
        create_safe_subset(corpus, seg, args.max_mb)
    else:
        log.info(f"Reusing {seg}")

    model = train_bpe(seg, Path(args.output_dir), args.vocab_size)
    sanity_check(model)
    log.info("DONE! Production tokenizer ready.")


if __name__ == "__main__":
    main()
