"""
morpheme.py — AMB Layer 3: Morpheme Boundary Detection
========================================================
Segments Tamil words into their constituent morphemes.

Tamil is a highly agglutinative language. Words like 
"போகவேண்டியிருந்தது" are not atomic units; they are sentences:
  "போக" (go) + "வேண்டி" (must/need) + "இருந்தது" (was)

Standard BPE fails here because it statistically shreds these
meaningful units. AMB Layer 3 restores linguistic sanity.

Architecture:
  1. **Rule-Based Splitter**: A high-precision regex engine that
     detects 50+ common Tamil suffixes (plurals, cases, clitics).
     Fast, deterministic, and handles 80% of noun morphology.
  
  2. **Statistical Segmenter (Morfessor)**: A data-driven model
     trained on the corpus to find deeper splits (verb tenses,
     sandhi rules). Used when rules aren't enough.

Usage:
    from morpheme import MorphemeSegmenter
    
    seg = MorphemeSegmenter()
    # "வீடுகளிலிருந்து" -> "வீடு @@கள் @@இலிருந்து"
    splits = seg.segment_word("வீடுகளிலிருந்து") 
"""

import re
import os
import logging
import pickle
import unicodedata
from typing import List, Tuple, Optional, Set
from pathlib import Path

# Try to import Morfessor, but don't crash if missing
try:
    import morfessor
    HAS_MORFESSOR = True
except ImportError:
    HAS_MORFESSOR = False

log = logging.getLogger(__name__)


# ===========================================================================
# Common Tamil Suffixes (Rule-Based Knowledge)
# ===========================================================================

# These are high-frequency suffixes that should ALMOST ALWAYS be split.
# We use "@@" as the boundary marker for BPE constraints.

# Plurals (including oblique forms used before case markers)
PLURALS = [
    "கள்", "க்கள்", "ங்கள்", "ற்கள்", "ட்கள்",
    "களி", "க்களி", "ங்களி", "ற்களி", "ட்களி", # Oblique forms (e.g., வீடுகளி-ல்)
]

# Case Markers (Vettrumai Urubugal) - Including Sandhi Variants
CASE_MARKERS = [
    # Accusative (ai)
    "ஐ", "யை", "வை",      # -ai variants
    
    # Instrumental (aal)
    "ஆல்", "வால்", "லால்",  # -aal variants
    
    # Dative (ku)
    "கு", "க்கு", "உக்கு",   # -ku variants
    
    # Genitive (in/athu)
    "இன்", "வின்", "னின்",   # -in variants
    "அது", "வது",          # -athu variants
    
    # Locative / Oblique insertions
    "கண்", "இடம்",
    "இல்", "வில்", "லில்", "யில்",  # -il (in) variants
    "இ", "யை", "வை", "ஐ",           # -i/-ai variants (oblique bases)
    
    # Ablative (ilirunthu)
    "இருந்து", "லிருந்து", "விலிருந்து", "திலிருந்து",
    
    # Sociative
    "உடன்", "வுடன்", "னுடன்",
    "ஓடு", "வோடு", "ளோடு",
]

# Clitics (Emphatic & Interrogative particles)
CLITICS = [
    "ஏ",       # Emphatic (e - தான்)
    "ஓ",       # Doubt/Question (o)
    "ஆ",       # Question (aa)
    "உம்",     # Conjunctive (um - and)
    "தான்",    # Emphatic (thaan - only)
    "ஆவது",    # Ordinal (aavathu - th)
]

# Common Verb Suffixes (Tense + PNG + Auxiliaries)
VERB_SUFFIXES = [
    # Present
    "கின்றன", "கிறான்", "கிறாள்", "கிறோம்", "கிறாய்", "கார்", "கிற",
    
    # Past
    "ந்தன", "ந்தான்", "ந்தாள்", "ந்தோம்", "ந்தாய்", "ந்த",
    "த்தன", "த்தான்", "த்தாள்", "த்தோம்", "த்தாய்", "த்த",
    "டன", "ட்டான்", "ட்டாள்", "ட்டோம்", "ட்டாய்", "ட்ட",
    
    # Future/Infinitive
    "ப்பார்", "ப்பார்", "ப்பான்", "ப்பாள்", "ப்போமா", "ப்பும்", "ப்ப",
    "வும்", "க்க", "க",
    
    # Auxiliaries (Negatives/Modals)
    "மாட்டார்கள்", "மாட்டான்", "மாட்டாள்", "மாட்டோம்", "மாட்டார்",
    "வில்லை", "க்கூடாது", "முடியாது", "முடியும்", "வேண்டும்", "வேண்டாம்",
    
    # Passives/Honorifics
    "ப்படுகிறது", "பட்டது", "படுகிறது", "மார்", "ீர்கள்", "ுங்கள்", "ங்கள்",
]

# Sandhi Consonants that often bridge morphemes
SANDHI = ["க்", "ச்", "த்", "ப்"]


# ===========================================================================
# Regex Compilation
# ===========================================================================

def get_all_suffixes() -> List[str]:
    """
    Return all known suffixes sorted by length (longest first).
    """
    all_suffixes = (
        PLURALS + CASE_MARKERS + CLITICS + VERB_SUFFIXES
    )
    return sorted(all_suffixes, key=len, reverse=True)


def build_suffix_regex() -> re.Pattern:
    """
    Compile a massive regex to match common suffixes at the END of words.
    Sorted by length (longest first) to ensure greedy matching.
    """
    all_suffixes = get_all_suffixes()
    
    # Create group: (கள்|க்கள்|...)
    pattern = "|".join(re.escape(s) for s in all_suffixes)
    
    # Anchor to end of string ($)
    full_pattern = f"({pattern})$"
    return re.compile(full_pattern)


def is_combining_mark(ch: str) -> bool:
    """
    True if `ch` is a Unicode combining mark.
    This blocks splits that would detach a vowel sign from its consonant.
    """
    return unicodedata.category(ch) in ("Mn", "Mc")


def is_invalid_split_boundary(word: str, boundary: int) -> bool:
    """
    A split is invalid if the suffix starts with a combining mark.
    That would separate a vowel sign (or virama) from its base consonant.
    """
    if boundary <= 0 or boundary >= len(word):
        return False
    return is_combining_mark(word[boundary])


ALL_SUFFIXES = get_all_suffixes()
SUCCESSIVE_SUFFIX_REGEX = build_suffix_regex()


# ===========================================================================
# MorphemeSegmenter Class
# ===========================================================================

class MorphemeSegmenter:
    """
    Hybrid morpheme segmenter combining rules and statistical models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.use_morfessor = False
        self.min_stem_len = 4
        
        # Load Morfessor model if provided
        if HAS_MORFESSOR and model_path and os.path.exists(model_path):
            try:
                io = morfessor.MorfessorIO()
                self.model = io.read_binary_model_file(model_path)
                self.use_morfessor = True
                log.info(f"Loaded Morfessor model from {model_path}")
            except Exception as e:
                log.warning(f"Failed to load Morfessor model: {e}")
        
        # Regex fallback
        self.suffix_re = SUCCESSIVE_SUFFIX_REGEX
        self.suffixes = ALL_SUFFIXES

    def segment_word(self, word: str) -> str:
        """
        Segment a single word into morphemes separated by spaces.
        Example: "வீடுகளிலிருந்து" -> "வீடு கள் இருந்து"
        
        Returns space-separated string.
        """
        # If we have a statistical model, prioritize it (it learns rules too)
        if self.use_morfessor:
            segments = self.model.viterbi_segment(word)[0]
            return " ".join(segments)
            
        # Fallback: Iterative suffix stripping (Rule-Based)
        # 1. Strip suffixes from right-to-left
        # 2. Handle Sandhi (grammatical junctions)
        # 3. Validation to prevent over-stemming
        
        current = word
        segments = []
        
        # Peel off suffixes from the right
        # We loop until no more known suffixes are found at the end
        # Limit to 5 suffixes max to avoid infinite loops or over-stripping
        for _ in range(5):
            suffix = None
            boundary = None
            for cand in self.suffixes:
                if not current.endswith(cand):
                    continue
                boundary = len(current) - len(cand)
                if is_invalid_split_boundary(current, boundary):
                    # Don't split between a consonant and its vowel sign.
                    continue
                suffix = cand
                break

            if suffix:
                
<<<<<<< HEAD
                # Safety check: don't strip if stem becomes too short (<4 chars)
                stem_len = len(current) - len(suffix)
                if stem_len < self.min_stem_len:
=======
                # Safety check: don't strip if stem becomes too short (<2 chars)
                # Exception: specific short roots if known, but generally 2 is safe
                stem_candidate = current[:match.start()]
                
                if len(stem_candidate) < 2:
>>>>>>> c25ce88512efbe7542bd955cdc129498e67b2035
                    break
                    
                # We found a suffix!
                # 1. Add it to our list (at the front)
                segments.insert(0, suffix)
                
                # 2. Update 'current' to be the stem
<<<<<<< HEAD
                current = current[:boundary]
=======
                current = stem_candidate
>>>>>>> c25ce88512efbe7542bd955cdc129498e67b2035
                
                # 3. Handle Sandhi (Consonant Doubling)
                # In Tamil, suffixes starting with vowels often double the previous consonant
                # or inject a glide (y/v).
                # ex: மர + த்து (atthu) -> மரத்து
                # ex: பூ + கள் -> பூக்கள் (k doubled)
                
                # Check for Sandhi consonants: க், ச், த், ப் at the end of the NEW stem
                for s in SANDHI:
                    if current.endswith(s):
                        # Heuristic: Only strip sandhi if the PREVIOUS char is a vowel
                        # or it looks like a valid junction. 
                        # This prevents stripping root-final consonants.
                        # For now, we strip clearly identified sandhi.
                        segments.insert(0, s)
                        current = current[:-len(s)]
                        break
            else:
                break
                
        # Add the remaining stem
        segments.insert(0, current)
        
        return " ".join(segments)

    def train(self, corpus_file: str, save_path: str, vocab_size: int = 50000):
        """
        Train a Morfessor model on a text corpus.
        This is a wrapper around the `morfessor` library training loop.
        """
        if not HAS_MORFESSOR:
            log.error("Morfessor library not installed. Cannot train.")
            return

        log.info(f"Training Morfessor model on {corpus_file}...")
        
        # Initialize model
        model = morfessor.BaselineModel()
        
        # Stream data
        io = morfessor.MorfessorIO()
        data = io.read_corpus_file(corpus_file)
        
        # Train
        model.load_data(data)
        model.train_batch()
        
        # Save
        io.write_binary_model_file(save_path, model)
        log.info(f"Morfessor model saved to {save_path}")
        self.model = model
        self.use_morfessor = True


# ===========================================================================
# Boundary Injection (Connecting Layers 3 & 4)
# ===========================================================================

def apply_boundary_marker(segmented_text: str, marker: str = "@@") -> str:
    """
    Convert space-separated morphemes into BPE-ready format.
    
    Input:  "வீடு கள் இல்"
    Output: "வீடு@@ கள்@@ இல்"
    
    The '@@' marker tells the BPE trainer: "You can merge characters
    within this block, but this is a hard linguistic boundary."
    """
    # Simply replace spaces with the marker + space
    # NOTE: We need to be careful not to double-mark existing spaces
    
    morphemes = segmented_text.split()
    if not morphemes:
        return ""
        
    # Join with the boundary marker
    # "word" -> "word"
    # "word suffix" -> "word@@ suffix"
    return f"{marker} ".join(morphemes)
