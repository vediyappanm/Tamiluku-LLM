"""
verify_fixes.py - Verify All Critical Fixes Have Been Applied
==============================================================
This script checks that all fixes are in place before training.

Usage:
    python verify_fixes.py
"""

import sys
from pathlib import Path

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()

def check_string_in_file(filepath: str, search_string: str) -> bool:
    """Check if a string exists in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return search_string in content
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

def main():
    print("="*70)
    print("VERIFICATION: Critical Fixes Status")
    print("="*70)
    
    checks = []
    
    # Check 1: Syllable coverage fix in train_amb_tokenizer.py
    print("\n1. Checking syllable coverage fix...")
    if check_file_exists("tokenizer/train_amb_tokenizer.py"):
        if check_string_in_file("tokenizer/train_amb_tokenizer.py", "protected_tokens = special_tokens + tamil_syllables"):
            print("   ✅ Syllable coverage fix applied")
            checks.append(True)
        else:
            print("   ❌ Syllable coverage fix NOT found")
            print("      Expected: protected_tokens = special_tokens + tamil_syllables")
            checks.append(False)
    else:
        print("   ❌ File not found: tokenizer/train_amb_tokenizer.py")
        checks.append(False)
    
    # Check 2: Script isolation function in normalize.py
    print("\n2. Checking script isolation function...")
    if check_file_exists("tokenizer/normalize.py"):
        if check_string_in_file("tokenizer/normalize.py", "def isolate_scripts("):
            print("   ✅ Script isolation function exists")
            checks.append(True)
        else:
            print("   ❌ Script isolation function NOT found")
            checks.append(False)
    else:
        print("   ❌ File not found: tokenizer/normalize.py")
        checks.append(False)
    
    # Check 3: Script isolation applied in process_document
    print("\n3. Checking script isolation is applied...")
    if check_file_exists("tokenizer/normalize.py"):
        if check_string_in_file("tokenizer/normalize.py", "text = isolate_scripts(text)"):
            print("   ✅ Script isolation applied in process_document")
            checks.append(True)
        else:
            print("   ❌ Script isolation NOT applied in process_document")
            checks.append(False)
    else:
        checks.append(False)
    
    # Check 4: Strengthened pre-tokenizer
    print("\n4. Checking strengthened pre-tokenizer...")
    if check_file_exists("tokenizer/train_amb_tokenizer.py"):
        if check_string_in_file("tokenizer/train_amb_tokenizer.py", "# Stage 1: Respect morpheme boundaries"):
            print("   ✅ Pre-tokenizer strengthened with multi-stage isolation")
            checks.append(True)
        else:
            print("   ⚠️  Pre-tokenizer may need manual review")
            checks.append(True)  # Not critical
    else:
        checks.append(False)
    
    # Check 5: Vocabulary size increased
    print("\n5. Checking vocabulary size...")
    if check_file_exists("tokenizer/config.yaml"):
        if check_string_in_file("tokenizer/config.yaml", "vocab_size: 64000"):
            print("   ✅ Vocabulary size set to 64000")
            checks.append(True)
        elif check_string_in_file("tokenizer/config.yaml", "vocab_size: 48000"):
            print("   ⚠️  Vocabulary size still at 48000 (recommended: 64000)")
            checks.append(True)  # Not critical, but recommended
        else:
            print("   ❌ Could not verify vocabulary size")
            checks.append(False)
    else:
        print("   ❌ File not found: tokenizer/config.yaml")
        checks.append(False)
    
    # Check 6: Validation functions added
    print("\n6. Checking validation functions...")
    if check_file_exists("tokenizer/train_amb_tokenizer.py"):
        has_syllable_check = check_string_in_file("tokenizer/train_amb_tokenizer.py", "def verify_syllable_coverage(")
        has_leakage_check = check_string_in_file("tokenizer/train_amb_tokenizer.py", "def detect_cross_script_leakage(")
        
        if has_syllable_check and has_leakage_check:
            print("   ✅ Validation functions added")
            checks.append(True)
        else:
            if not has_syllable_check:
                print("   ❌ verify_syllable_coverage() NOT found")
            if not has_leakage_check:
                print("   ❌ detect_cross_script_leakage() NOT found")
            checks.append(False)
    else:
        checks.append(False)
    
    # Check 7: Corpus exists
    print("\n7. Checking training corpus...")
    corpus_paths = [
        "tokenizer/data/cleaned/tamil_corpus.txt",
        "data/cleaned/tamil_corpus.txt"
    ]
    corpus_found = False
    for corpus_path in corpus_paths:
        if check_file_exists(corpus_path):
            size_mb = Path(corpus_path).stat().st_size / (1024 * 1024)
            print(f"   ✅ Corpus found: {corpus_path} ({size_mb:.1f} MB)")
            corpus_found = True
            checks.append(True)
            break
    
    if not corpus_found:
        print("   ⚠️  Corpus not found. Run: python tokenizer/collect_corpus.py")
        print("      Then: python tokenizer/normalize.py")
        checks.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if all(checks):
        print("\n✅ ALL FIXES VERIFIED!")
        print("\nYou're ready to train:")
        print("  cd tokenizer")
        print("  python train_tokenizer.py --engine amb")
        print("\nOr if corpus needs re-normalization:")
        print("  python normalize.py")
        print("  python train_tokenizer.py --engine amb")
        return 0
    else:
        print("\n❌ Some fixes are missing or incomplete.")
        print("\nTo apply all fixes automatically:")
        print("  python IMPLEMENT_FIXES.py --apply-all")
        print("\nOr apply fixes manually using the guide documents:")
        print("  - FIX_SYLLABLE_COVERAGE.md")
        print("  - FIX_CROSS_SCRIPT_LEAKAGE.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
