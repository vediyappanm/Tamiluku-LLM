# Fix #2: Cross-Script Leakage (126 → 0)

## Problem
Your tokenizer has 126 tokens that mix Tamil and Latin characters, like:
- `"à®®à¯įĠ(B"` (Tamil bytes + Latin)
- `"à®¿Ġ(L"` (Tamil bytes + Latin)

This defeats the purpose of script isolation.

## Root Cause
1. Script isolation regex is not strong enough
2. ByteLevel pre-tokenizer operates on bytes, which can merge across boundaries
3. The training corpus may contain mixed-script text that wasn't cleaned

## Solution: Multi-Layer Script Isolation

### Step 1: Fix Normalization (Apply BEFORE Training)

Modify `normalize.py` to add script isolation:

```python
def isolate_scripts(text: str) -> str:
    """
    Physically separate Tamil, Latin, and digits with whitespace.
    This prevents BPE from creating cross-script tokens.
    """
    import regex
    
    # Pattern: Capture Tamil blocks, Latin blocks, digit blocks separately
    # Insert space between different script types
    result = []
    prev_script = None
    
    for char in text:
        if '\u0B80' <= char <= '\u0BFF':  # Tamil
            curr_script = 'tamil'
        elif char.isascii() and char.isalpha():  # Latin
            curr_script = 'latin'
        elif char.isdigit():  # Digits
            curr_script = 'digit'
        else:  # Punctuation, whitespace, etc.
            curr_script = 'other'
        
        # Insert space when script changes (except for 'other')
        if prev_script and curr_script != prev_script and \
           prev_script != 'other' and curr_script != 'other':
            result.append(' ')
        
        result.append(char)
        prev_script = curr_script
    
    return ''.join(result)
```

Add this to the `process_document()` function:

```python
def process_document(text: str, norm_cfg: dict) -> Optional[str]:
    """Process a single document through the normalization pipeline."""
    
    # ... existing normalization code ...
    
    # NEW: Apply script isolation BEFORE returning
    text = isolate_scripts(text)
    
    return text
```

### Step 2: Strengthen Pre-tokenizer in `train_amb_tokenizer.py`

Replace the pre-tokenizer configuration:

```python
# OLD (weak isolation):
SCRIPT_ISOLATOR = r"[\u0B80-\u0BFF]+|[a-zA-Z]+|[0-9]+|[^\s\u0B80-\u0BFFa-zA-Z0-9]+"

tokenizer.pre_tokenizer = Sequence([
    Split(pattern=" @@ ", behavior="isolated"),
    Split(pattern=SCRIPT_ISOLATOR, behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
])
```

```python
# NEW (strong isolation):
# Three-stage isolation: morpheme boundaries → script boundaries → byte-level
tokenizer.pre_tokenizer = Sequence([
    # Stage 1: Respect morpheme boundaries (@@)
    Split(pattern=r" @@ ", behavior="removed"),
    
    # Stage 2: Hard script isolation (CRITICAL)
    # This MUST come before ByteLevel to prevent cross-script merges
    Split(
        pattern=r"([\u0B80-\u0BFF]+)|([a-zA-Z]+)|([0-9]+)",
        behavior="isolated",
        invert=False
    ),
    
    # Stage 3: Byte-level encoding (for unknown characters)
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
```

### Step 3: Post-Training Validation

Add this function to detect cross-script tokens:

```python
def detect_cross_script_leakage(tokenizer_path: str) -> int:
    """
    Scan vocabulary for tokens that mix Tamil and Latin/Digit characters.
    Returns count of leaky tokens.
    """
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    
    leaky_tokens = []
    
    for token_str, token_id in vocab.items():
        # Decode the token to get actual text
        try:
            decoded = tokenizer.decode([token_id])
        except:
            decoded = token_str
        
        # Skip special tokens
        if decoded.startswith("<") and decoded.endswith(">"):
            continue
        
        # Check for mixed scripts
        has_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in decoded)
        has_latin = any(ch.isascii() and ch.isalpha() for ch in decoded)
        has_digit = any(ch.isdigit() for ch in decoded)
        
        # Flag if mixing Tamil with Latin or Digits
        if has_tamil and (has_latin or has_digit):
            leaky_tokens.append({
                "id": token_id,
                "token": token_str,
                "decoded": decoded
            })
    
    if leaky_tokens:
        log.error(f"❌ Found {len(leaky_tokens)} cross-script tokens!")
        log.error("Sample leaky tokens:")
        for item in leaky_tokens[:10]:
            log.error(f"  ID {item['id']}: {item['decoded']}")
    else:
        log.info(f"✅ No cross-script leakage detected!")
    
    return len(leaky_tokens)
```

Call after training:

```python
model_path = train_amb_tokenizer(cfg, corpus_path)
leakage_count = detect_cross_script_leakage(model_path)

if leakage_count > 0:
    log.error(f"Training failed: {leakage_count} cross-script tokens found")
    log.error("Check normalization and pre-tokenizer configuration")
else:
    log.info("✅ Script isolation successful!")
```

### Step 4: Re-normalize Corpus

If you've already collected your corpus, you need to re-normalize it:

```bash
# Backup old corpus
cp tokenizer/data/cleaned/tamil_corpus.txt tokenizer/data/cleaned/tamil_corpus.backup.txt

# Re-run normalization with script isolation
python tokenizer/normalize.py

# Verify the output
head -n 100 tokenizer/data/cleaned/tamil_corpus.txt
```

Look for proper spacing between Tamil and Latin text.

## Expected Results

After this fix:
- Cross-Script Leakage: 0 tokens (down from 126)
- Vocabulary will be cleaner
- Tokens will be more interpretable

## Testing

```bash
# Re-normalize corpus
python tokenizer/normalize.py

# Re-train tokenizer
python tokenizer/train_tokenizer.py --engine amb

# Verify
python tokenizer/evaluate_tokenizer.py
```

Check the `cross_script_leakage_count` metric - it should be 0.

## Why This Matters

Cross-script tokens are:
1. **Uninterpretable**: Humans can't read them
2. **Inefficient**: They waste vocabulary space
3. **Harmful**: They confuse the model during training
4. **Unprofessional**: Reviewers will reject papers with this issue

Zero cross-script leakage is a MUST for publication.
