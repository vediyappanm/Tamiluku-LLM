import math
import os
import sys
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer
try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False
    from transformers import AutoTokenizer

def calculate_unigram_entropy(tokenizer_type, tokenizer_obj, texts):
    token_counts = Counter()
    total_token_len = 0
    total_words = 0
    
    for text in texts:
        if tokenizer_type == "amb":
            ids = tokenizer_obj.encode(text).ids
            tokens = [str(i) for i in ids]
        elif tokenizer_type == "tiktoken":
            tokens = tokenizer_obj.encode(text)
        elif tokenizer_type == "hf":
            tokens = tokenizer_obj.encode(text)
        
        token_counts.update(tokens)
        total_token_len += len(tokens)
        total_words += len(text.split())

    if total_token_len == 0:
        return 0, 0, 0

    # Entropy in bits per token
    entropy = -sum((count/total_token_len) * math.log2(count/total_token_len) 
                   for count in token_counts.values())
    
    # Fertility: tokens per word
    fertility = total_token_len / max(total_words, 1)
    
    return entropy, fertility, total_token_len

def run_experiments():
    print("🔬 --- AMB Quick Experiments --- 🔬\n")
    
    # Load AMB Tokenizer
    amb_path = "models/amb_tokenizer/tokenizer.json"
    if not os.path.exists(amb_path):
        print(f"❌ Error: {amb_path} not found. Train with a small sample first.")
        return
    
    amb_tokenizer = Tokenizer.from_file(amb_path)
    
    if USE_TIKTOKEN:
        gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")
        comp_engine = gpt4_tokenizer
        comp_type = "tiktoken"
    else:
        print("⚠️ tiktoken not found, using GPT-2 HF as baseline.")
        comp_engine = AutoTokenizer.from_pretrained("gpt2")
        comp_type = "hf"
    
    # Try to load from corpus if available for better stats
    corpus_path = "data/cleaned/tamil_corpus.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = [l.strip() for l in f.readlines() if l.strip()]
            if len(corpus_text) > 1000:
                import random
                test_sentences = random.sample(corpus_text, 1000)
                print(f"✅ Sampled 1000 sentences from {corpus_path}")
            else:
                test_sentences = corpus_text
    else:
        # Fallback to defaults
        test_sentences = [
            "இந்திய அரசியலமைப்புச் சட்டம் அனைவருக்கும் சமத்துவத்தை உறுதிசெய்கிறது.",
            "தமிழ்நாடு அரசு கல்வித் துறையில் பல்வேறு சீர்திருத்தங்களை மேற்கொண்டு வருகிறது.",
            "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது.",
            "யாதும் ஊரே யாவரும் கேளிர் தீதும் நன்றும் பிறர்தர வாரா.",
            "மின்னணுவியல் துறை மாற்றங்களை சந்தித்து வருகிறது."
        ]
    
    print(f"📊 [Experiment 1] Entropy & Fertility Showdown (vs {comp_type})")
    amb_ent, amb_fert, amb_toks = calculate_unigram_entropy("amb", amb_tokenizer, test_sentences)
    gpt_ent, gpt_fert, gpt_toks = calculate_unigram_entropy(comp_type, comp_engine, test_sentences)
    
    print(f"{'Metric':<15} | {'AMB':<10} | {'GPT-4':<10} | {'Improvement'}")
    print("-" * 55)
    print(f"{'Fertility':<15} | {amb_fert:<10.2f} | {gpt_fert:<10.2f} | {((gpt_fert/amb_fert)-1)*100:>7.1f}% better")
    print(f"{'Entropy (bits)':<15} | {amb_ent:<10.2f} | {gpt_ent:<10.2f} | {((gpt_ent/amb_ent)-1)*100:>7.1f}% dense")
    print(f"{'Total Tokens':<15} | {amb_toks:<10} | {gpt_toks:<10} | {gpt_toks-amb_toks} saved")
    print("\n")

    # 2. Code-Mixing Robustness
    print("💻 [Experiment 2] Code-Mixing Robustness")
    code_mixed_samples = [
        "Python-ல coding செய்கிறேன்",
        "AI-யின் future மிகவும் bright-ஆக இருக்கிறது",
        "Netflix-இல் Tamil movies பார்க்கிறேன்"
    ]
    
    for sample in code_mixed_samples:
        encoded = amb_tokenizer.encode(sample)
        # Check for cross-script tokens manually
        tokens = [amb_tokenizer.decode([i]) for i in encoded.ids]
        leaky = any(any(0x0B80 <= ord(c) <= 0x0BFF for c in t) and any(c.isascii() and c.isalpha() for c in t) for t in tokens)
        print(f"Sample: {sample}")
        print(f"Tokens: {' | '.join(tokens)}")
        print(f"Leaky:  {'❌ YES' if leaky else '✅ NO'}")
    print("\n")

    # 3. Morpheme Boundary Stress Test (Oblique stems)
    print("🏛️ [Experiment 3] Morpheme Stress Test (Linguistic Nuance)")
    critical_word = "சென்னையிலிருந்துதான்"
    # The user noted "சென்னையி" + "லிருந்து" + "தான்" is slightly off.
    # It should ideally be "சென்னை" + "யில்" + "இருந்து" + "தான்"
    
    encoded = amb_tokenizer.encode(critical_word)
    token_texts = [amb_tokenizer.decode([i]) for i in encoded.ids]
    print(f"Word:   {critical_word}")
    print(f"Splits: {' | '.join(token_texts)}")
    
if __name__ == "__main__":
    run_experiments()
