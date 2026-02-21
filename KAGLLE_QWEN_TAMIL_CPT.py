"""
Kaggle Production Script: Qwen 2.5 Vocabulary Extension & CPT
============================================================
Target: Qwen 2.5 7B
Optimization: Unsloth (4-bit LoRA)
Vocabulary: 48k AMB Tokens
Hardware: Kaggle T4 GPU (16GB VRAM)
"""

import os
# Disable Triton compiler for older GPUs BEFORE any imports (P100 has CUDA 6.0, Triton needs 7.0+)
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"

# Clear Unsloth's compiled cache to force recompilation without fused loss
import shutil
cache_dir = "/kaggle/working/Tamiluku-LLM/unsloth_compiled_cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("🗑️ Cleared Unsloth compiled cache")

# Handle Kaggle-specific dependency issues (unsloth_zoo and datasets versioning)
try:
    import pyarrow_hotfix 
    import datasets
    # Unsloth has known recursion issues with datasets >= 3.0.0
    if int(datasets.__version__.split('.')[0]) >= 3:
        raise ImportError("Version conflict")
    from unsloth import FastLanguageModel
except (ImportError, ModuleNotFoundError, AttributeError):
    import subprocess
    import sys
    print("🛠️ Fixing environment (downgrading datasets + installing Unsloth)...")
    # We pin datasets to 2.16.0. pyarrow-hotfix is required for this version but missing in Kaggle 3.12
    # peft versions are flexible now - monkey-patch handles ensure_weight_tying compatibility
    dependencies = [
        "unsloth", "unsloth_zoo", "datasets==2.16.0", "pyarrow-hotfix",
        "peft", "trl", "xformers", "accelerate", "bitsandbytes",
        "sentencepiece", "protobuf", "typing-extensions"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade"] + dependencies)
    print("✅ Installation complete. Please restart the kernel if you see metadata errors.")
    from unsloth import FastLanguageModel

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
import numpy as np

# Custom trainer that uses standard loss instead of Unsloth's fused loss (P100 compatible)
class StandardLossSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss using standard PyTorch instead of fused loss"""
        # Extract labels from inputs
        labels = inputs.pop("labels", None)

        if labels is None:
            # Fall back to model's loss computation if no labels
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Forward pass without computing loss in model
        outputs = model(**inputs, output_hidden_states=False)
        logits = outputs.logits

        # Compute loss using standard cross entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss

# Monkey-patch LoraConfig to handle ensure_weight_tying parameter
# This parameter was removed in newer peft versions but Unsloth still passes it
from peft import LoraConfig
original_init = LoraConfig.__init__

def patched_init(self, *args, **kwargs):
    kwargs.pop('ensure_weight_tying', None)  # Remove unsupported parameter
    original_init(self, *args, **kwargs)

LoraConfig.__init__ = patched_init

# 1. SETUP CONFIGURATION
MODEL_NAME = "Qwen/Qwen2.5-7B" # Base model
TOKENIZER_PATH = "/kaggle/working/Tamiluku-LLM/tokenizer/tokenizer (1).json"
OUTPUT_DIR = "/kaggle/working/qwen2.5-tamil-amb"

# Robust corpus path detection
def get_corpus_path():
    possible_paths = [
        "/kaggle/working/tamil_gold_corpus/raw_tamil_gold.txt",
        "/kaggle/input/tamil-corpus-txt/tamil_corpus.txt",
        "/kaggle/working/Tamiluku-LLM/tamil_corpus.txt",
        "tamil_corpus.txt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0] # Fallback to default

CORPUS_PATH = get_corpus_path()

max_seq_length = 1024 # Reduced from 2048 to save memory for P100
dtype = None # None for auto detection
load_in_4bit = True # Use 4-bit quantization to save memory

def setup_checkpoint():
    print("🚀 Setting up environment...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. LOAD MODEL & RESIZE TOKENIZER
def load_and_resize():
    print(f"📦 Loading {MODEL_NAME} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Load your custom AMB Tokenizer
    print("Merging AMB Vocabulary...")
    import glob
    from transformers import PreTrainedTokenizerFast
    
    # Robustly find the tokenizer file (look for the largest json file that isn't a config)
    tokenizer_files = glob.glob("/kaggle/working/Tamiluku-LLM/tokenizer/tokenizer*.json")
    # Filter out config files
    val_files = [f for f in tokenizer_files if "config" not in f.lower()]
    
    if not val_files:
        raise FileNotFoundError("Could not find any tokenizer.json in /kaggle/working/Tamiluku-LLM/tokenizer/")
    
    # Pick the largest one to be safe (tokenizer.json is usually MBs, config is bytes)
    tokenizer_file = max(val_files, key=os.path.getsize)
    print(f"Using tokenizer file: {tokenizer_file}")
    
    # Use PreTrainedTokenizerFast to bypass AutoConfig check
    custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    
    # Add new tokens to the base tokenizer
    new_tokens = set(custom_tokenizer.get_vocab().keys()) - set(tokenizer.get_vocab().keys())
    num_added = tokenizer.add_tokens(list(new_tokens))
    print(f"✅ Added {num_added} new Tamil tokens.")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. WECHSEL-STYLE INITIALIZATION (Direct Implementation)
    print("🧠 Initializing New Embeddings (WECHSEL Strategy)...")
    model_params = model.get_input_embeddings().weight.data
    old_vocab_size = len(tokenizer) - num_added
    
    # Simple warm-start: Initialize new tokens by averaging nearby existing ones 
    # and adding small Gaussian noise for symmetry breaking
    with torch.no_grad():
        mean_existing = model_params[:old_vocab_size].mean(dim=0)
        std_existing  = model_params[:old_vocab_size].std(dim=0)
        
        # Apply to new tokens
        for i in range(old_vocab_size, len(tokenizer)):
            model_params[i] = mean_existing + (torch.randn_like(mean_existing) * 0.02)
            
    return model, tokenizer

# 4. PREPARE LORA (Optimized for Vocab Extension)
def prepare_lora(model):
    print("🛠️ Configuring LoRA Adaptors...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Higher rank for vocabulary learning
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head", # CRITICAL: Train the embeddings!
        ],
        lora_alpha = 64,
        lora_dropout = 0, # Unsloth optimized
        bias = "none",    # Unsloth optimized
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    print("✅ LoRA configured (P100 compatible with standard loss)")
    return model

# 5. DATA LOADING
def load_data():
    print(f"📖 Loading corpus: {CORPUS_PATH}")
    dataset = load_dataset("text", data_files={"train": CORPUS_PATH}, split="train")
    return dataset

# 6. RUN TRAINING
def train():
    setup_checkpoint()
    model, tokenizer = load_and_resize()
    model = prepare_lora(model)
    dataset = load_data()

    trainer = StandardLossSFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 100,
            max_steps = 1000, # Start with 1000 for Kaggle
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            save_total_limit = 2,
            save_steps = 250,
            report_to = [],  # Disable W&B logging
        ),
    )

    print("🔥 Starting Training (Continued Pre-training)...")
    trainer.train()
    
    print("💾 Saving Model...")
    model.save_pretrained(f"{OUTPUT_DIR}/lora_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_model")
    print("✅ Done!")

if __name__ == "__main__":
    train()
