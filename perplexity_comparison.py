"""
perplexity_comparison.py
========================
Train two tiny BERT masked-LM models with different tokenizers and compare
perplexity and tokenization efficiency metrics.

Default behavior:
- AMB tokenizer: tokenizer/models/amb_tokenizer/tokenizer.json
- Baseline tokenizer: tokenizer/models/hf_tokenizer/tokenizer.json
- Train data: tokenizer/data/cleaned/tamil_corpus.txt
- Eval data: tokenizer/data/eval/eval_corpus.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def ensure_special_tokens(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    # Prefer existing padding token if present in vocab
    vocab = tokenizer.get_vocab()
    if tokenizer.pad_token is None and "<|padding|>" in vocab:
        tokenizer.pad_token = "<|padding|>"

    specials = {}
    if tokenizer.unk_token is None:
        specials["unk_token"] = "[UNK]"
    if tokenizer.pad_token is None:
        specials["pad_token"] = "[PAD]"
    if tokenizer.cls_token is None:
        specials["cls_token"] = "[CLS]"
    if tokenizer.sep_token is None:
        specials["sep_token"] = "[SEP]"
    if tokenizer.mask_token is None:
        specials["mask_token"] = "[MASK]"

    if specials:
        tokenizer.add_special_tokens(specials)

    return tokenizer


def build_tokenized_datasets(
    tokenizer: PreTrainedTokenizerFast,
    train_path: Path,
    eval_path: Path,
    block_size: int,
    max_train_lines: int | None,
    max_eval_lines: int | None,
):
    raw = load_dataset(
        "text",
        data_files={"train": str(train_path), "eval": str(eval_path)},
    )

    if max_train_lines is not None:
        raw["train"] = raw["train"].select(range(min(max_train_lines, len(raw["train"]))))
    if max_eval_lines is not None:
        raw["eval"] = raw["eval"].select(range(min(max_eval_lines, len(raw["eval"]))))

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = raw.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length < block_size:
            return {}
        total_length = (total_length // block_size) * block_size

        result = {}
        for k, t in concatenated.items():
            result[k] = [t[i : i + block_size] for i in range(0, total_length, block_size)]

        # Minimal fields for MLM
        n_chunks = total_length // block_size
        result["attention_mask"] = [[1] * block_size for _ in range(n_chunks)]
        result["token_type_ids"] = [[0] * block_size for _ in range(n_chunks)]
        return result

    grouped = tokenized.map(group_texts, batched=True)
    return grouped["train"], grouped["eval"]


def compute_token_stats(
    tokenizer: PreTrainedTokenizerFast,
    eval_path: Path,
    max_lines: int | None,
) -> Tuple[float, float]:
    total_words = 0
    total_tokens = 0
    total_bytes = 0

    with open(eval_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            total_words += len(line.split())
            total_tokens += len(tokenizer.encode(line, add_special_tokens=False))
            total_bytes += len(line.encode("utf-8"))

    avg_tokens_per_word = total_tokens / max(total_words, 1)
    bytes_per_token = total_bytes / max(total_tokens, 1)
    bits_per_token = bytes_per_token * 8.0
    return avg_tokens_per_word, bits_per_token


def compute_unigram_counts(dataset, max_tokens: int | None) -> Tuple[Counter, int]:
    counts = Counter()
    total = 0
    for ex in dataset:
        for tid in ex["input_ids"]:
            counts[tid] += 1
            total += 1
            if max_tokens is not None and total >= max_tokens:
                return counts, total
    return counts, total


def compute_unigram_perplexity(
    counts: Counter,
    total: int,
    eval_dataset,
    vocab_size: int,
    max_eval_tokens: int | None,
) -> float:
    denom = total + vocab_size  # add-one smoothing
    n = 0
    neg_log_likelihood = 0.0
    for ex in eval_dataset:
        for tid in ex["input_ids"]:
            prob = (counts.get(tid, 0) + 1) / denom
            neg_log_likelihood += -math.log(prob)
            n += 1
            if max_eval_tokens is not None and n >= max_eval_tokens:
                break
        if max_eval_tokens is not None and n >= max_eval_tokens:
            break
    return math.exp(neg_log_likelihood / max(n, 1))


def train_and_eval_mlm(
    name: str,
    tokenizer: PreTrainedTokenizerFast,
    train_dataset,
    eval_dataset,
    args,
    output_dir: Path,
) -> Tuple[float, float]:
    # Debug info before model init
    print(f"\n[{name}] Debug: Vocab size (len): {len(tokenizer)}")
    print(f"[{name}] Debug: config.vocab_size (internal): {tokenizer.vocab_size}")
    print(f"[{name}] Debug: Pad token ID: {tokenizer.pad_token_id}")

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.block_size,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = BertForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "evaluation_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": 0,
        "no_cuda": (args.device == "cpu"),
    }
    # Dynamically filter valid kwargs to prevent TypeError on older/custom versions
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    training_args = TrainingArguments(**valid_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }
    # Dynamically filter valid kwargs for Trainer
    trainer_sig = inspect.signature(Trainer.__init__)
    valid_trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters}
    trainer = Trainer(**valid_trainer_kwargs)

    start = time.time()
    trainer.train()
    eval_metrics = trainer.evaluate()
    elapsed = time.time() - start

    eval_loss = float(eval_metrics.get("eval_loss", 0.0))
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    print(f"\n[{name}] Eval loss: {eval_loss:.4f} | PPL: {perplexity:.2f} | Time: {elapsed/60:.1f} min")
    return eval_loss, perplexity


def improvement(amb: float, base: float, higher_is_better: bool) -> float:
    if base == 0:
        return 0.0
    if higher_is_better:
        return (amb - base) / base * 100.0
    return (base - amb) / base * 100.0


def print_table(results: Dict[str, Dict[str, float]]):
    amb = results["AMB"]
    base = results["Baseline"]

    rows = [
        ("Unigram Perplexity", amb["unigram_ppl"], base["unigram_ppl"], False, ""),
        ("Neural Perplexity (LM)", amb["neural_ppl"], base["neural_ppl"], False, "PASS"),
        ("Avg Tokens/Word", amb["tokens_per_word"], base["tokens_per_word"], False, ""),
        ("Bits/Token (Density)", amb["bits_per_token"], base["bits_per_token"], True, ""),
        ("Final Eval Loss", amb["eval_loss"], base["eval_loss"], False, ""),
    ]

    print("\nMetric                         AMB             Baseline        Improvement")
    print("---------------------------------------------------------------------------")
    for metric, amb_val, base_val, higher_better, pass_flag in rows:
        imp = improvement(amb_val, base_val, higher_better)
        suffix = f"  {pass_flag}" if pass_flag else ""
        print(f"{metric:<30} {amb_val:>10.2f} {base_val:>16.2f} {imp:>14.1f}%{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Compare AMB vs baseline tokenizer perplexity")
    parser.add_argument("--config", default="tokenizer/config.yaml", help="Path to config.yaml")
    parser.add_argument("--amb-tokenizer", default=None, help="Path to AMB tokenizer.json")
    parser.add_argument("--baseline-tokenizer", default=None, help="Path to baseline tokenizer.json")
    parser.add_argument("--train-file", default=None, help="Training text file")
    parser.add_argument("--eval-file", default=None, help="Eval text file")
    parser.add_argument("--output", default=None, help="Output JSON report")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--max-eval-lines", type=int, default=None)
    parser.add_argument("--unigram-max-tokens", type=int, default=1_000_000)
    parser.add_argument("--unigram-eval-max-tokens", type=int, default=300_000)
    parser.add_argument("--stats-max-lines", type=int, default=100_000)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_config(config_path)
    base_dir = config_path.parent

    amb_tok_path = args.amb_tokenizer
    if amb_tok_path is None:
        amb_dir = cfg["tokenizer"]["output_dir"]
        # Try root models dir first, then nested tokenizer/models
        path1 = Path(amb_dir) / "tokenizer.json"
        path2 = resolve_path(base_dir, amb_dir) / "tokenizer.json"
        amb_tok_path = str(path1) if path1.exists() else str(path2)

    base_tok_path = args.baseline_tokenizer
    if base_tok_path is None:
        base_dir_cfg = cfg["huggingface"]["output_dir"]
        path1 = Path(base_dir_cfg) / "tokenizer.json"
        path2 = resolve_path(base_dir, base_dir_cfg) / "tokenizer.json"
        base_tok_path = str(path1) if path1.exists() else str(path2)

    train_path = args.train_file or str(resolve_path(base_dir, cfg["corpus"]["output_file"]))
    eval_path = args.eval_file or str(resolve_path(base_dir, cfg["corpus"]["eval_dir"]) / "eval_corpus.txt")

    amb_tok_str = str(amb_tok_path)
    base_tok_str = str(base_tok_path)

    train_path = Path(train_path)
    eval_path = Path(eval_path)

    # Check data files
    for p in [train_path, eval_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required data file: {p}")

    # Check tokenizers (only if they look like local paths)
    def is_local_path(s):
        return "/" in s or "\\" in s or s.endswith(".json") or Path(s).exists()

    if is_local_path(amb_tok_str):
        if not Path(amb_tok_str).exists():
             raise FileNotFoundError(f"Missing required AMB tokenizer file: {amb_tok_path}")

    if is_local_path(base_tok_str):
        if not Path(base_tok_str).exists():
             # If it's a simple name like 'gpt2', it's NOT a local path. 
             # Only error if it specifically looks like a path but is missing.
             if "/" in base_tok_str or "\\" in base_tok_str or base_tok_str.endswith(".json"):
                 raise FileNotFoundError(f"Missing required Baseline tokenizer file: {base_tok_path}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    args.device = device

    print(f"Device: {args.device}")
    print(f"Train file: {train_path}")
    print(f"Eval file:  {eval_path}")

    print("\nLoading tokenizers...")
    # Use AutoTokenizer to support both local files and HF HUB IDs
    def load_tok(p):
        path_str = str(p)
        if path_str.endswith(".json") and Path(path_str).exists():
            # If it's a direct path to a tokenizer.json, we can use PreTrainedTokenizerFast
            return PreTrainedTokenizerFast(tokenizer_file=path_str)
        return AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)

    amb_tokenizer = ensure_special_tokens(load_tok(amb_tok_path))
    base_tokenizer = ensure_special_tokens(load_tok(base_tok_path))

    print("\nBuilding datasets (AMB)...")
    amb_train, amb_eval = build_tokenized_datasets(
        amb_tokenizer, train_path, eval_path, args.block_size, args.max_train_lines, args.max_eval_lines
    )

    print("Building datasets (Baseline)...")
    base_train, base_eval = build_tokenized_datasets(
        base_tokenizer, train_path, eval_path, args.block_size, args.max_train_lines, args.max_eval_lines
    )

    print("\nComputing unigram stats (AMB)...")
    amb_counts, amb_total = compute_unigram_counts(amb_train, args.unigram_max_tokens)
    amb_unigram_ppl = compute_unigram_perplexity(
        amb_counts, amb_total, amb_eval, amb_tokenizer.vocab_size, args.unigram_eval_max_tokens
    )

    print("Computing unigram stats (Baseline)...")
    base_counts, base_total = compute_unigram_counts(base_train, args.unigram_max_tokens)
    base_unigram_ppl = compute_unigram_perplexity(
        base_counts, base_total, base_eval, base_tokenizer.vocab_size, args.unigram_eval_max_tokens
    )

    print("\nTokenization efficiency stats (AMB)...")
    amb_tokens_per_word, amb_bits_per_token = compute_token_stats(
        amb_tokenizer, eval_path, args.stats_max_lines
    )

    print("Tokenization efficiency stats (Baseline)...")
    base_tokens_per_word, base_bits_per_token = compute_token_stats(
        base_tokenizer, eval_path, args.stats_max_lines
    )

    reports_dir = base_dir / "reports" / "perplexity_runs"
    amb_out = reports_dir / "amb"
    base_out = reports_dir / "baseline"

    print("\nTraining AMB model...")
    amb_eval_loss, amb_neural_ppl = train_and_eval_mlm(
        "AMB",
        amb_tokenizer,
        amb_train,
        amb_eval,
        args,
        amb_out,
    )

    print("\nTraining Baseline model...")
    base_eval_loss, base_neural_ppl = train_and_eval_mlm(
        "Baseline",
        base_tokenizer,
        base_train,
        base_eval,
        args,
        base_out,
    )

    results = {
        "AMB": {
            "unigram_ppl": amb_unigram_ppl,
            "neural_ppl": amb_neural_ppl,
            "tokens_per_word": amb_tokens_per_word,
            "bits_per_token": amb_bits_per_token,
            "eval_loss": amb_eval_loss,
        },
        "Baseline": {
            "unigram_ppl": base_unigram_ppl,
            "neural_ppl": base_neural_ppl,
            "tokens_per_word": base_tokens_per_word,
            "bits_per_token": base_bits_per_token,
            "eval_loss": base_eval_loss,
        },
    }

    print_table(results)

    output_path = args.output
    if output_path is None:
        output_path = str(base_dir / "reports" / "perplexity_comparison.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
