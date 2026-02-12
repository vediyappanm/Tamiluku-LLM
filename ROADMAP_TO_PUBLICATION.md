# Roadmap to Publication-Ready Results
## Current Status vs Target

### What You Have Now (Real Results)
```
Neural Perplexity:        2917.47 (vs 10251.91 baseline) ✅ 71.5% improvement
Tokens/Word:              2.73 (vs 2.11 baseline) ❌ 29% worse
Syllable Coverage:        62.75% ❌ Target: 97%
Cross-Script Leakage:     126 tokens ❌ Target: 0
Morpheme Boundary Acc:    Miscalculated (shows 100% but examples show 50-75%)
```

### What You Need (Publication Targets)
```
Neural Perplexity:        < 10 (currently 2917) ❌ Need 99.7% improvement
Tokens/Word:              1.89 (currently 2.73) ❌ Need 31% reduction
Syllable Coverage:        97.2% (currently 62.75%) ❌ Need +34.5pp
Cross-Script Leakage:     0 (currently 126) ❌ Need elimination
Morpheme Boundary Acc:    99.7% (currently ~60%) ❌ Need +39.7pp
```

---

## Critical Issues Blocking Success

### Issue #1: Syllable Coverage (62.75% → 97.2%)
**Root Cause**: Tamil syllables are NOT being pre-populated in the vocabulary before BPE training.

**Evidence**: Your eval report shows syllables like "ச்", "ஞ்", "கீ", "கெ" are being split into consonant + vowel sign.

**Why This Happens**:
- `train_amb_tokenizer.py` uses `trainers.BpeTrainer()` with default settings
- BPE learns merges from scratch based on frequency
- Rare syllables (like "ஙா", "ஙி") don't appear often enough to be learned as single tokens
- The "10,000x boost" mentioned in design.md is NOT implemented in code

**Fix Strategy**:
1. Generate all 247 Tamil syllables using `generate_tamil_syllables()`
2. Add them to `special_tokens` in the BPE trainer (this locks them in vocabulary)
3. Alternatively, use `initial_alphabet` parameter to seed the vocabulary
4. Verify by checking vocabulary after training

**Implementation**:
```python
# In train_amb_tokenizer.py, modify trainer initialization:
tamil_syllables = generate_tamil_syllables()  # 247 syllables

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens + tamil_syllables,  # Lock syllables in vocab
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
```

---

### Issue #2: Cross-Script Leakage (126 → 0)
**Root Cause**: Script isolation is NOT being applied before BPE training.

**Evidence**: Tokens like `"à®®à¯įĠ(B"` show UTF-8 byte sequences mixed with Latin characters.

**Why This Happens**:
- `SCRIPT_ISOLATOR` regex in `train_amb_tokenizer.py` is defined but not enforced strongly enough
- ByteLevel pre-tokenizer operates on bytes, which can merge across script boundaries
- The segmented corpus still contains mixed-script text

**Fix Strategy**:
1. Apply script isolation in `normalize.py` BEFORE writing to corpus
2. Add validation step to reject any line with mixed scripts
3. Use stricter pre-tokenizer that physically separates scripts

**Implementation**:
```python
# In normalize.py, add script isolation:
def isolate_scripts(text: str) -> str:
    """Physically separate Tamil, Latin, and digits with special markers."""
    import regex
    # Insert boundaries between script changes
    pattern = r'([\u0B80-\u0BFF]+)|([a-zA-Z]+)|([0-9]+)'
    return regex.sub(pattern, r' \1\2\3 ', text)

# Apply before writing to cleaned corpus
```

---

### Issue #3: Fertility (2.73 → 1.89)
**Root Cause**: Morpheme segmentation is too aggressive OR vocabulary is too small.

**Why This Happens**:
- Current vocab: 48K tokens
- Tamil has rich morphology: 18 consonants × 12 vowel forms = 216 base syllables
- Plus case markers (8), tense markers (12), plural forms, etc.
- 48K might not be enough to capture all common morpheme combinations

**Fix Strategy**:
1. Increase vocabulary size to 64K-96K
2. Review morpheme segmentation rules - are you over-splitting?
3. Check if BPE is actually merging frequent morpheme pairs

**Implementation**:
```yaml
# In config.yaml:
tokenizer:
  vocab_size: 64000  # Increase from 48000
```

---

### Issue #4: Neural Perplexity (2917 → <10)
**Root Cause**: This is the BIGGEST gap. Your target of <10 perplexity is unrealistic.

**Reality Check**:
- GPT-2 on English: ~20-30 perplexity
- BERT on English: ~15-25 perplexity
- Your current 2917 is high because you're training on tiny corpus (10K lines)

**What "Publication-Ready" Actually Means**:
- Perplexity < 50 on held-out test set (realistic)
- Perplexity < 10 is only achievable with:
  - 100GB+ training corpus
  - Full-scale LM (not 4-layer pilot)
  - Months of training on GPUs

**Fix Strategy**:
1. Scale up training corpus from 10K lines to full 20GB
2. Train for 50K steps (not 50 steps)
3. Use larger model (12-layer BERT, not 4-layer)
4. Adjust expectations: Target <50 perplexity, not <10

---

### Issue #5: Morpheme Boundary Accuracy (60% → 99.7%)
**Root Cause**: Metric is miscalculated + morpheme segmentation rules are incomplete.

**Evidence**: Report shows 100% accuracy, but individual examples show 50-75% per word.

**Fix Strategy**:
1. Fix calculation: `sum(per_word_accuracy) / num_words`
2. Improve morpheme segmentation rules in `morpheme.py`
3. Add more suffix patterns to the segmenter

---

## Realistic Timeline to Publication

### Phase 1: Fix Critical Bugs (1-2 weeks)
- [ ] Implement syllable pre-population
- [ ] Fix cross-script leakage
- [ ] Fix morpheme boundary metric calculation
- [ ] Increase vocab size to 64K

**Expected Results After Phase 1**:
```
Syllable Coverage:        95%+ ✅
Cross-Script Leakage:     0 ✅
Tokens/Word:              2.0-2.2 (improved)
Morpheme Boundary Acc:    75-85% (realistic)
```

### Phase 2: Scale Up Training (2-4 weeks)
- [ ] Collect full 20GB corpus (not just 10K lines)
- [ ] Train for 50K steps (not 50)
- [ ] Use full evaluation corpus (1000+ docs, not 30)
- [ ] Run on GPU (not CPU)

**Expected Results After Phase 2**:
```
Neural Perplexity:        50-100 (realistic)
Tokens/Word:              1.8-2.0 ✅
Training Convergence:     Stable
```

### Phase 3: Downstream Evaluation (2-3 weeks)
- [ ] Fine-tune on sentiment analysis task
- [ ] Fine-tune on NER task
- [ ] Compare against GPT-4 tokenizer
- [ ] Measure inference speed

**Expected Results After Phase 3**:
```
Sentiment F1:             80-85% (vs 75-80% baseline)
NER F1:                   85-90% (vs 80-85% baseline)
Inference Speed:          1.5-2x faster ✅
```

---

## What Results Are Actually Achievable

### Realistic Publication-Ready Metrics (3 months work)
```
Morpheme Boundary Accuracy:     85-90% (not 99.7%)
Neural Perplexity (MLM):        50-100 (not 8.42)
Avg Tokens/Word:                1.8-2.2 (not 1.89)
Bits/Token (Density):           8-12 (achievable)
Cross-Script Leakage:           0 ✅ (achievable)
Syllable Coverage:              95-98% ✅ (achievable)
Roundtrip Accuracy:             100% ✅ (already have)
```

### Downstream Task Performance (Realistic)
```
Sentiment Analysis:             +5-8pp improvement (not +8.1pp)
Named Entity Recognition:       +4-7pp improvement (not +6.8pp)
Machine Translation:            +3-5 BLEU (not +6.3)
```

---

## Honest Assessment

**Can you achieve the "ideal" results you showed me?**
- Some metrics: YES (syllable coverage, cross-script leakage, roundtrip)
- Some metrics: PARTIALLY (tokens/word, morpheme accuracy)
- Some metrics: NO (perplexity <10 requires industrial-scale resources)

**What's the path forward?**
1. Fix the bugs (Phase 1) - this is 100% achievable in 1-2 weeks
2. Scale up training (Phase 2) - achievable with cloud GPU credits
3. Run downstream tasks (Phase 3) - proves real-world value

**Bottom line**: Your current 71.5% perplexity improvement is REAL and VALUABLE. Focus on:
- Fixing syllable coverage (biggest win)
- Eliminating cross-script leakage (credibility)
- Scaling to full corpus (proves it works at scale)
- Honest reporting of results (builds trust)

The story is: "AMB achieves 70%+ perplexity improvement with linguistically-grounded tokenization." That's publication-worthy.
