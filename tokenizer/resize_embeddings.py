"""
resize_embeddings.py - Resize LLM Embeddings for Merged Tokenizer
===================================================================
After merging new Tamil tokens into the base tokenizer, the LLM's
embedding matrices (embed_tokens + lm_head) must be resized.

Initialization strategies:
  1. "wechsel" — WECHSEL (Minixhofer et al., 2022): Warm-start from
     byte-fallback token embeddings. Strongly recommended.
  2. "mean"   — Mean of all existing embeddings + Gaussian noise
  3. "random" — N(0, 0.02) — NOT recommended for production

WECHSEL gives each new Tamil token a semantic "warm start" by
averaging the embeddings of its byte-level decomposition in the
original model.

Usage:
    python resize_embeddings.py [--config config.yaml]
    python resize_embeddings.py --base-model meta-llama/Llama-3-8B
    python resize_embeddings.py --strategy wechsel
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List

import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_wechsel_embeddings(
    model,
    tokenizer_old,
    tokenizer_new,
    new_token_ids: List[int],
    noise_std: float = 0.02,
):
    """
    WECHSEL-style embedding initialization.

    For each new token, tokenize it with the OLD tokenizer (which will use
    byte-fallback), get those embeddings, and average them. This gives each
    new Tamil token a semantic warm-start from its byte-level representation.
    """
    import torch

    embed_weight = model.get_input_embeddings().weight.data
    old_vocab_size = tokenizer_old.vocab_size if hasattr(tokenizer_old, 'vocab_size') else len(tokenizer_old)

    new_embeddings = []
    fallback_count = 0

    for new_id in new_token_ids:
        token_str = tokenizer_new.decode([new_id]).strip()

        if not token_str:
            new_emb = embed_weight[:old_vocab_size].mean(dim=0)
            new_embeddings.append(new_emb)
            fallback_count += 1
            continue

        # Tokenize with OLD tokenizer (will byte-decompose Tamil text)
        old_ids = tokenizer_old.encode(token_str, add_special_tokens=False)

        if old_ids and all(oid < embed_weight.shape[0] for oid in old_ids):
            old_embeds = embed_weight[old_ids]
            avg_emb = old_embeds.mean(dim=0)
        else:
            avg_emb = embed_weight[:old_vocab_size].mean(dim=0)
            fallback_count += 1

        # Add noise for symmetry breaking
        noise = torch.randn_like(avg_emb) * noise_std
        new_emb = avg_emb + noise
        new_embeddings.append(new_emb)

    if fallback_count > 0:
        log.info(f"  WECHSEL: {fallback_count} tokens used mean fallback")

    return torch.stack(new_embeddings)


def resize_and_initialize(
    base_model_id: str,
    merged_tokenizer_dir: str,
    output_dir: str,
    init_strategy: str = "wechsel",
    noise_std: float = 0.02,
    torch_dtype: str = "float16",
) -> dict:
    """Load base model, resize embeddings, initialize, and save."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        log.error("Install torch and transformers")
        sys.exit(1)

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    # Load tokenizers
    log.info(f"Loading base model: {base_model_id}")
    log.info(f"  This may take several minutes for large models...")

    tokenizer_old = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer_new = AutoTokenizer.from_pretrained(merged_tokenizer_dir, trust_remote_code=True)

    old_vocab_size = len(tokenizer_old)
    new_vocab_size = len(tokenizer_new)
    num_new_tokens = new_vocab_size - old_vocab_size

    log.info(f"  Old vocab: {old_vocab_size:,}")
    log.info(f"  New vocab: {new_vocab_size:,}")
    log.info(f"  New tokens: {num_new_tokens:,}")

    if num_new_tokens <= 0:
        log.warning("No new tokens to add. Skipping.")
        return {"status": "skipped", "reason": "no new tokens"}

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings
    log.info(f"Resizing model embeddings: {old_vocab_size:,} -> {new_vocab_size:,}")
    model.resize_token_embeddings(new_vocab_size)

    # Initialize new embeddings
    new_token_ids = list(range(old_vocab_size, new_vocab_size))

    if init_strategy == "wechsel":
        log.info(f"Initializing with WECHSEL strategy (noise_std={noise_std})...")
        new_embeds = compute_wechsel_embeddings(
            model, tokenizer_old, tokenizer_new, new_token_ids, noise_std
        )
    elif init_strategy == "mean":
        log.info(f"Initializing with MEAN strategy (noise_std={noise_std})...")
        import torch
        embed_weight = model.get_input_embeddings().weight.data
        mean_emb = embed_weight[:old_vocab_size].mean(dim=0)
        noise = torch.randn(num_new_tokens, mean_emb.shape[0]) * noise_std
        new_embeds = mean_emb.unsqueeze(0) + noise
    elif init_strategy == "random":
        log.warning("Using RANDOM initialization. NOT recommended for production.")
        import torch
        embed_dim = model.get_input_embeddings().weight.shape[1]
        new_embeds = torch.randn(num_new_tokens, embed_dim) * noise_std
    else:
        log.error(f"Unknown init strategy: {init_strategy}")
        sys.exit(1)

    # Write embeddings
    import torch
    with torch.no_grad():
        model.get_input_embeddings().weight.data[old_vocab_size:] = new_embeds.to(
            model.get_input_embeddings().weight.dtype
        )

    # Also initialize lm_head if separate
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings is not model.get_input_embeddings():
        log.info("Initializing output embeddings (lm_head)...")
        with torch.no_grad():
            output_embeddings.weight.data[old_vocab_size:] = new_embeds.to(
                output_embeddings.weight.dtype
            )

    # Verify norms
    log.info("Verifying embedding norms...")
    old_norms = model.get_input_embeddings().weight.data[:old_vocab_size].float().norm(dim=1)
    new_norms = model.get_input_embeddings().weight.data[old_vocab_size:].float().norm(dim=1)

    log.info(f"  Old embedding norm: mean={old_norms.mean():.4f}, std={old_norms.std():.4f}")
    log.info(f"  New embedding norm: mean={new_norms.mean():.4f}, std={new_norms.std():.4f}")

    norm_ratio = new_norms.mean() / old_norms.mean()
    if norm_ratio < 0.1 or norm_ratio > 10:
        log.warning(f"  Norm ratio is {norm_ratio:.2f}. May cause training instability.")
    else:
        log.info(f"  Norm ratio: {norm_ratio:.2f} (healthy)")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving resized model to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer_new.save_pretrained(str(output_path))

    model_size_gb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 3)

    stats = {
        "base_model": base_model_id,
        "init_strategy": init_strategy,
        "noise_std": noise_std,
        "torch_dtype": torch_dtype,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "num_new_tokens": num_new_tokens,
        "old_embedding_norm_mean": float(old_norms.mean()),
        "new_embedding_norm_mean": float(new_norms.mean()),
        "norm_ratio": float(norm_ratio),
        "model_size_gb": round(model_size_gb, 2),
        "output_dir": str(output_path),
    }

    log.info(f"\n{'='*60}")
    log.info(f"EMBEDDING RESIZE COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Model: {output_path}")
    log.info(f"  Size:  {model_size_gb:.2f} GB")
    log.info(f"  Strategy: {init_strategy}")
    log.info(f"  New tokens: {num_new_tokens:,}")
    log.info(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Resize LLM embeddings for merged tokenizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--base-model", default=None, help="Base HF model ID")
    parser.add_argument("--tokenizer-dir", default=None, help="Merged tokenizer directory")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--strategy", default=None, help="Init strategy: wechsel/mean/random")
    parser.add_argument("--dtype", default=None, help="Torch dtype: float16/bfloat16/float32")
    args = parser.parse_args()

    cfg = load_config(args.config)
    emb_cfg = cfg["embeddings"]
    merge_cfg = cfg["merge"]

    base_model_id = args.base_model or merge_cfg["base_model"]
    tokenizer_dir = args.tokenizer_dir or merge_cfg["output_dir"]
    output_dir = args.output or emb_cfg["output_dir"]
    strategy = args.strategy or emb_cfg["init_strategy"]
    noise_std = emb_cfg["noise_std"]
    torch_dtype = args.dtype or emb_cfg.get("torch_dtype", "float16")

    if not os.path.exists(tokenizer_dir):
        log.error(f"Merged tokenizer not found: {tokenizer_dir}")
        log.error("Run 'python merge_vocabularies.py' first.")
        sys.exit(1)

    stats = resize_and_initialize(
        base_model_id=base_model_id,
        merged_tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        init_strategy=strategy,
        noise_std=noise_std,
        torch_dtype=torch_dtype,
    )

    # Save stats
    report_path = Path("reports") / "embedding_stats.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved to {report_path}")

    log.info(f"\n=== PIPELINE COMPLETE ===")
    log.info(f"Your Tamil-adapted model is at: {output_dir}")
    log.info(f"Next: Continued Pretraining (CPT) on Tamil text.")


if __name__ == "__main__":
    main()
