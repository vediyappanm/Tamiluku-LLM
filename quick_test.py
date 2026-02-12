"""
quick_test.py - Quick Test of Fixes with Minimal Corpus
========================================================
This script creates a minimal test corpus and trains the tokenizer
to verify all fixes are working correctly.

Usage:
    python quick_test.py
"""

import os
import sys
from pathlib import Path

def create_test_corpus():
    """Create a minimal Tamil corpus for testing."""
    print("Creating test corpus...")
    
    # Sample Tamil text covering various syllables and morphemes
    test_text = """
தமிழ் ஒரு அழகான மொழி. இது உலகின் பழமையான மொழிகளில் ஒன்று.
வீடுகளிலிருந்து மக்கள் வெளியே வருகிறார்கள்.
போகவேண்டியிருந்தது என்னால் முடியவில்லை.
அரசியலமைப்புச் சட்டம் அனைவருக்கும் சமத்துவத்தை உறுதிசெய்கிறது.
படிக்கிறான் எழுதுகிறான் பேசுகிறான் கேட்கிறான்.
கற்றுக்கொள்ள வேண்டிய பாடங்கள் நிறைய இருக்கின்றன.
செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது.
மாணவர்களுக்கு நல்ல கல்வி அவசியம்.
பள்ளியில் ஆசிரியர்கள் பாடம் நடத்துகிறார்கள்.
நாட்டின் வளர்ச்சிக்கு கல்வி முக்கியம்.
""" * 100  # Repeat 100 times for minimal training data
    
    # Create directory structure
    data_dir = Path("tokenizer/data/cleaned")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_path = data_dir / "tamil_corpus.txt"
    
    # Write test corpus
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    size_kb = corpus_path.stat().st_size / 1024
    print(f"✅ Test corpus created: {corpus_path} ({size_kb:.1f} KB)")
    
    return str(corpus_path)

def run_training():
    """Run training with the test corpus."""
    print("\n" + "="*70)
    print("TRAINING AMB TOKENIZER (TEST RUN)")
    print("="*70)
    
    os.chdir("tokenizer")
    
    # Run training
    result = os.system("python train_tokenizer.py --engine amb --vocab-size 8000")
    
    os.chdir("..")
    
    return result == 0

def run_evaluation():
    """Run evaluation."""
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    os.chdir("tokenizer")
    
    result = os.system("python evaluate_tokenizer.py")
    
    os.chdir("..")
    
    return result == 0

def main():
    print("="*70)
    print("QUICK TEST: Verify All Fixes Are Working")
    print("="*70)
    print("\nThis will:")
    print("  1. Create a minimal test corpus")
    print("  2. Train the tokenizer with all fixes")
    print("  3. Run evaluation")
    print("  4. Verify syllable coverage and cross-script leakage")
    print("\nThis should take 2-5 minutes.")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        # Step 1: Create test corpus
        corpus_path = create_test_corpus()
        
        # Step 2: Train
        if not run_training():
            print("\n❌ Training failed. Check errors above.")
            return 1
        
        # Step 3: Evaluate
        if not run_evaluation():
            print("\n⚠️  Evaluation had issues. Check errors above.")
        
        # Step 4: Summary
        print("\n" + "="*70)
        print("QUICK TEST COMPLETE")
        print("="*70)
        print("\nCheck the training output above for:")
        print("  ✅ Syllable Coverage: Should be > 95%")
        print("  ✅ Cross-Script Leakage: Should be 0")
        print("\nIf both checks passed, all fixes are working!")
        print("\nNext steps:")
        print("  1. Collect full corpus: python tokenizer/collect_corpus.py")
        print("  2. Normalize: python tokenizer/normalize.py")
        print("  3. Train on full corpus: python tokenizer/train_tokenizer.py --engine amb")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
