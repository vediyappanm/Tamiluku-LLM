# Fix #1: Syllable Coverage (62.75% → 95%+)

## Problem
Your tokenizer only covers 155 of 247 Tamil syllables as single tokens. Syllables like "ச்", "ஞ்", "கீ" are being split into consonant + vowel sign.

## Root Cause
The BPE trainer learns merges from scratch based on frequency. Rare syllables don't appear often enough in the training corpus to be learned as atomic tokens.

## Solution: Pre-populate Vocabulary with All Tamil Syllables

### Step 1: Modify `train_amb_tokenizer.py`

Add this function at the top (after imports):

```python
def generate_tamil_syllables() -> List[str]:
    """Generate the complete Tamil syllable inventory (247 syllables)."""
    vowels = list("அஆஇஈஉஊஎஏஐஒஓஔ")
    consonant_bases = list("கஙசஞடணதநபமயரலவழளறன")
    vowel_signs = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
    
    syllables = []
    # Pure vowels
    syllables.extend(vowels)
    # Consonant + pulli (்)
    for c in consonant_bases:
        syllables.append(c + "்")
    # Consonant + vowel sign combinations
    for c in consonant_bases:
        for vs in vowel_signs:
            syllables.append(c + vs)
    # Aaytham
    syllables.append("ஃ")
    
    return syllables
```

### Step 2: Modify the Trainer Initialization

Find this section in `train_amb_tokenizer()`:

```python
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
```

Replace with:

```python
# Generate all Tamil syllables
tamil_syllables = generate_tamil_syllables()
log.info(f"Pre-populating vocabulary with {len(tamil_syllables)} Tamil syllables...")

# Combine special tokens + Tamil syllables
# These will be locked into the vocabulary regardless of frequency
protected_tokens = special_tokens + tamil_syllables

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=protected_tokens,  # Lock syllables in vocab
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
```

### Step 3: Adjust Vocabulary Size

Since you're adding 247 syllables to the vocabulary, you need to account for this:

In `config.yaml`:

```yaml
tokenizer:
  vocab_size: 48247  # 48000 + 247 syllables
  # OR increase to 64000 for more morpheme coverage
```

### Step 4: Verify After Training

Add this validation function to `train_amb_tokenizer.py`:

```python
def verify_syllable_coverage(tokenizer_path: str):
    """Verify that all Tamil syllables are in the vocabulary."""
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    
    tamil_syllables = generate_tamil_syllables()
    covered = 0
    uncovered = []
    
    for syllable in tamil_syllables:
        # Encode and check if it's a single token
        encoded = tokenizer.encode(syllable)
        if len(encoded.tokens) == 1:
            covered += 1
        else:
            uncovered.append((syllable, encoded.tokens))
    
    coverage = covered / len(tamil_syllables)
    log.info(f"Syllable Coverage: {coverage:.2%} ({covered}/{len(tamil_syllables)})")
    
    if uncovered:
        log.warning(f"Uncovered syllables ({len(uncovered)}):")
        for syl, tokens in uncovered[:10]:
            log.warning(f"  {syl} → {tokens}")
    
    return coverage
```

Call it after training:

```python
model_path = train_amb_tokenizer(cfg, corpus_path)
coverage = verify_syllable_coverage(model_path)

if coverage < 0.95:
    log.error(f"Syllable coverage {coverage:.2%} is below target 95%!")
else:
    log.info(f"✅ Syllable coverage target met: {coverage:.2%}")
```

## Expected Results

After this fix:
- Syllable Coverage: 95-100% (up from 62.75%)
- Tokens/Word: Should improve by 10-15%
- Morpheme Boundary Accuracy: Should improve (fewer broken syllables)

## Testing

Run this command to verify:

```bash
python train_tokenizer.py --engine amb --vocab-size 64000
python evaluate_tokenizer.py
```

Check the `tamil_syllable_coverage` metric in the evaluation report.
