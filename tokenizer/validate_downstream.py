"""
validate_downstream.py - Downstream LM Validation (AMB vs Baseline)
==================================================================
Trains a small 2-layer Transformer LM on the pilot tokenizer and a
standard BPE baseline, then compares perplexity on the eval split.

Usage:
  python validate_downstream.py --config config_pilot.yaml
  python validate_downstream.py --train-tokens 1000000 --eval-tokens 200000
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import yaml


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_docs(path: Path) -> Iterator[str]:
    """Yield documents separated by blank lines."""
    buffer: List[str] = []
    with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if buffer:
                    yield " ".join(buffer).strip()
                    buffer = []
            else:
                buffer.append(line.strip())
        if buffer:
            yield " ".join(buffer).strip()


def build_blocks(
    tokenizer: PreTrainedTokenizerFast,
    docs: Iterator[str],
    target_tokens: int,
    block_size: int,
) -> Tuple[List[List[int]], int]:
    eos_id = tokenizer.eos_token_id
    buffer: List[int] = []
    blocks: List[List[int]] = []
    total_tokens = 0

    for doc in docs:
        if not doc:
            continue
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if eos_id is not None:
            ids.append(eos_id)
        if not ids:
            continue

        buffer.extend(ids)
        total_tokens += len(ids)

        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]

        if target_tokens > 0 and total_tokens >= target_tokens:
            break

    return blocks, total_tokens


class BlockDataset(Dataset):
    def __init__(self, blocks: List[List[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.blocks[idx], dtype=torch.long)


def load_tokenizer(path: Path, cfg: dict) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))
    specials = cfg.get("tokenizer", {}).get("special_tokens", [])
    eos = specials[0] if specials else None
    pad = specials[1] if len(specials) > 1 else None

    if eos and tokenizer.eos_token is None and eos in tokenizer.get_vocab():
        tokenizer.eos_token = eos
    if pad and tokenizer.pad_token is None and pad in tokenizer.get_vocab():
        tokenizer.pad_token = pad
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_model(vocab_size: int, block_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=256,
        n_positions=block_size,
        n_ctx=block_size,
    )
    return GPT2LMHeadModel(config)


def train_one(
    name: str,
    tokenizer: PreTrainedTokenizerFast,
    train_blocks: List[List[int]],
    eval_blocks: List[List[int]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    model = build_model(tokenizer.vocab_size, len(train_blocks[0]))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(BlockDataset(train_blocks), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(BlockDataset(eval_blocks), batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"[{name}] Epoch {epoch}/{epochs} - train loss: {avg_loss:.4f}")

    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item()
            steps += 1

    eval_loss = total_loss / max(steps, 1)
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    return {
        "eval_loss": round(eval_loss, 6),
        "perplexity": round(perplexity, 4),
        "train_steps": steps,
        "train_blocks": len(train_blocks),
        "eval_blocks": len(eval_blocks),
    }


def main():
    parser = argparse.ArgumentParser(description="Downstream LM validation: AMB vs Baseline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--amb-tokenizer", default=None, help="Path to AMB tokenizer.json")
    parser.add_argument("--baseline-tokenizer", default=None, help="Path to baseline tokenizer.json")
    parser.add_argument("--train-corpus", default=None, help="Path to training corpus")
    parser.add_argument("--eval-corpus", default=None, help="Path to eval corpus")
    parser.add_argument("--train-tokens", type=int, default=1_000_000, help="Max tokens for training")
    parser.add_argument("--eval-tokens", type=int, default=200_000, help="Max tokens for eval")
    parser.add_argument("--block-size", type=int, default=256, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-path", default="reports/downstream_report.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)

    corpus_path = Path(args.train_corpus or cfg["corpus"]["output_file"])
    eval_path = Path(args.eval_corpus or (Path(cfg["corpus"]["eval_dir"]) / "eval_corpus.txt"))

    amb_path = Path(args.amb_tokenizer or (Path(cfg["tokenizer"]["output_dir"]) / "tokenizer.json"))
    base_path = Path(args.baseline_tokenizer or (Path(cfg["huggingface"]["output_dir"]) / "tokenizer.json"))

    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval corpus not found: {eval_path}")
    if not amb_path.exists():
        raise FileNotFoundError(f"AMB tokenizer not found: {amb_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"Baseline tokenizer not found: {base_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amb_tok = load_tokenizer(amb_path, cfg)
    base_tok = load_tokenizer(base_path, cfg)

    print(f"AMB vocab size: {amb_tok.vocab_size}")
    print(f"Baseline vocab size: {base_tok.vocab_size}")

    print("Building train/eval blocks...")
    amb_train_blocks, amb_train_tokens = build_blocks(
        amb_tok, iter_docs(corpus_path), args.train_tokens, args.block_size
    )
    amb_eval_blocks, amb_eval_tokens = build_blocks(
        amb_tok, iter_docs(eval_path), args.eval_tokens, args.block_size
    )
    base_train_blocks, base_train_tokens = build_blocks(
        base_tok, iter_docs(corpus_path), args.train_tokens, args.block_size
    )
    base_eval_blocks, base_eval_tokens = build_blocks(
        base_tok, iter_docs(eval_path), args.eval_tokens, args.block_size
    )

    if not amb_train_blocks or not base_train_blocks:
        raise RuntimeError("Insufficient training blocks. Increase --train-tokens or check corpus.")
    if not amb_eval_blocks or not base_eval_blocks:
        raise RuntimeError("Insufficient eval blocks. Increase --eval-tokens or check eval corpus.")

    print(f"AMB: train tokens={amb_train_tokens:,}, eval tokens={amb_eval_tokens:,}")
    print(f"Baseline: train tokens={base_train_tokens:,}, eval tokens={base_eval_tokens:,}")

    amb_result = train_one(
        "AMB", amb_tok, amb_train_blocks, amb_eval_blocks,
        device, args.epochs, args.batch_size, args.lr
    )
    base_result = train_one(
        "BASE", base_tok, base_train_blocks, base_eval_blocks,
        device, args.epochs, args.batch_size, args.lr
    )

    report = {
        "amb": amb_result,
        "baseline": base_result,
        "params": {
            "train_tokens": args.train_tokens,
            "eval_tokens": args.eval_tokens,
            "block_size": args.block_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDownstream report saved to {report_path}")
    print(f"AMB perplexity: {amb_result['perplexity']}")
    print(f"Baseline perplexity: {base_result['perplexity']}")


if __name__ == "__main__":
    main()
