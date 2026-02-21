"""
Kaggle Production Script: Qwen 2.5 Vocabulary Extension & CPT
============================================================
Target: Qwen 2.5 7B
Optimization: Unsloth (4-bit LoRA)
Vocabulary: 48k AMB Tokens
Hardware: Kaggle T4 GPU (16GB VRAM)
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from unsloth import FastLanguageModel
import numpy as np

# 1. SETUP CONFIGURATION
MODEL_NAME = "Qwen/Qwen2.5-7B" # Base model
TOKENIZER_PATH = "/kaggle/working/Tamiluku-LLM/tokenizer/tokenizer (1).json"
CORPUS_PATH = "/kaggle/working/tamil_gold_corpus/raw_tamil_gold.txt"
OUTPUT_DIR = "/kaggle/working/qwen2.5-tamil-amb"

max_seq_length = 2048 # Adjust based on need
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
    custom_tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/Tamiluku-LLM/tokenizer")
    
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

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
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
