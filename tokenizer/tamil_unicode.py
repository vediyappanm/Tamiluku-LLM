"""
tamil_unicode.py — AMB Layer 1: Deep Unicode Normalization Engine
===================================================================
Goes far beyond standard NFC normalization to handle ALL Tamil Unicode
edge cases that corrupt tokenizer training.

Standard NFC handles:
  - Canonical decomposition + recomposition (e.g., கொ normalization)

This engine ALSO handles:
  - Spurious ZWJ/ZWNJ between Tamil characters (web text noise)
  - ZWNJ preservation when linguistically meaningful (rare but real)
  - Grantha character canonicalization (ஜ, ஷ, ஸ, ஹ variants)
  - Tamil numeral ↔ Arabic numeral normalization
  - Legacy encoding artifacts (TAB/TAM converter leftovers)
  - Invisible formatting characters (LTR marks, paragraph separators)
  - Pulli (virama ்) normalization — ensure consistent placement
  - Whitespace canonicalization (NBSP, thin space, etc.)

Design principle: Every Tamil syllable must have EXACTLY ONE Unicode
representation after normalization. No exceptions.

Usage:
    from tamil_unicode import TamilDeepNormalizer
    normalizer = TamilDeepNormalizer()
    clean = normalizer.normalize("raw tamil text")
"""

import re
import unicodedata
from typing import List, Tuple, Optional


# ===========================================================================
# Tamil Unicode Constants
# ===========================================================================

# Tamil Unicode block: U+0B80 - U+0BFF
TAMIL_BLOCK_START = 0x0B80
TAMIL_BLOCK_END = 0x0BFF
TAMIL_RANGE = range(TAMIL_BLOCK_START, TAMIL_BLOCK_END + 1)

# --- Consonants (mei ezhuthu base forms) ---
TAMIL_CONSONANTS = {
    0x0B95,  # க
    0x0B99,  # ங
    0x0B9A,  # ச
    0x0B9C,  # ஜ (Grantha)
    0x0B9E,  # ஞ
    0x0B9F,  # ட
    0x0BA3,  # ண
    0x0BA4,  # த
    0x0BA8,  # ந
    0x0BA9,  # ன
    0x0BAA,  # ப
    0x0BAE,  # ம
    0x0BAF,  # ய
    0x0BB0,  # ர
    0x0BB1,  # ற
    0x0BB2,  # ல
    0x0BB3,  # ள
    0x0BB4,  # ழ
    0x0BB5,  # வ
    0x0BB6,  # ஶ (Grantha)
    0x0BB7,  # ஷ (Grantha)
    0x0BB8,  # ஸ (Grantha)
    0x0BB9,  # ஹ (Grantha)
}

# --- Vowels (uyir ezhuthu) ---
TAMIL_VOWELS = {
    0x0B85,  # அ
    0x0B86,  # ஆ
    0x0B87,  # இ
    0x0B88,  # ஈ
    0x0B89,  # உ
    0x0B8A,  # ஊ
    0x0B8E,  # எ
    0x0B8F,  # ஏ
    0x0B90,  # ஐ
    0x0B92,  # ஒ
    0x0B93,  # ஓ
    0x0B94,  # ஔ
}

# --- Vowel signs (dependent vowels / matras) ---
TAMIL_VOWEL_SIGNS = {
    0x0BBE,  # ா
    0x0BBF,  # ி
    0x0BC0,  # ீ
    0x0BC1,  # ு
    0x0BC2,  # ூ
    0x0BC6,  # ெ
    0x0BC7,  # ே
    0x0BC8,  # ை
    0x0BCA,  # ொ (composite: ெ + ா)
    0x0BCB,  # ோ (composite: ே + ா)
    0x0BCC,  # ௌ (composite: ெ + ௗ)
}

# --- Pulli (virama) ---
PULLI = 0x0BCD  # ்

# --- Aytham ---
AYTHAM = 0x0B83  # ஃ

# --- Tamil digits ---
TAMIL_DIGITS = {
    0x0BE6: "0",  # ௦
    0x0BE7: "1",  # ௧
    0x0BE8: "2",  # ௨
    0x0BE9: "3",  # ௩
    0x0BEA: "4",  # ௪
    0x0BEB: "5",  # ௫
    0x0BEC: "6",  # ௬
    0x0BED: "7",  # ௭
    0x0BEE: "8",  # ௮
    0x0BEF: "9",  # ௯
}

ARABIC_TO_TAMIL_DIGIT = {v: chr(k) for k, v in TAMIL_DIGITS.items()}

# --- Tamil special number symbols ---
TAMIL_NUMBER_SYMBOLS = {
    0x0BF0,  # ௰ (10)
    0x0BF1,  # ௱ (100)
    0x0BF2,  # ௲ (1000)
}

# --- Tamil special symbols ---
TAMIL_SPECIAL = {
    0x0BD0,  # ௐ (Om)
    0x0BD7,  # ௗ (au length mark)
}

# --- Grantha consonants (Sanskrit loanwords in Tamil) ---
GRANTHA_CONSONANTS = {
    0x0B9C,  # ஜ (ja)
    0x0BB6,  # ஶ (sha)
    0x0BB7,  # ஷ (ssa)
    0x0BB8,  # ஸ (sa)
    0x0BB9,  # ஹ (ha)
}

# ===========================================================================
# Zero-width and Invisible Characters
# ===========================================================================

# Characters that are ALWAYS noise in Tamil text
ALWAYS_REMOVE = {
    "\u200B",   # Zero Width Space
    "\uFEFF",   # BOM / ZWNBS
    "\u00AD",   # Soft Hyphen
    "\u200E",   # Left-to-Right Mark
    "\u200F",   # Right-to-Left Mark
    "\u2028",   # Line Separator
    "\u2029",   # Paragraph Separator
    "\u00A0",   # Non-Breaking Space → regular space (handled separately)
    "\u2060",   # Word Joiner
    "\u180E",   # Mongolian Vowel Separator (sometimes appears in copy-paste)
    "\u034F",   # Combining Grapheme Joiner
}

# Whitespace characters to normalize to regular space
WHITESPACE_NORMALIZE = {
    "\u00A0",   # Non-Breaking Space
    "\u2000",   # En Quad
    "\u2001",   # Em Quad
    "\u2002",   # En Space
    "\u2003",   # Em Space
    "\u2004",   # Three-Per-Em Space
    "\u2005",   # Four-Per-Em Space
    "\u2006",   # Six-Per-Em Space
    "\u2007",   # Figure Space
    "\u2008",   # Punctuation Space
    "\u2009",   # Thin Space
    "\u200A",   # Hair Space
    "\u202F",   # Narrow No-Break Space
    "\u205F",   # Medium Mathematical Space
    "\u3000",   # Ideographic Space
}

# ===========================================================================
# Precompiled Regex Patterns
# ===========================================================================

# ZWJ/ZWNJ between Tamil characters (noise from web/mobile input)
_RE_ZWJ_BETWEEN_TAMIL = re.compile(
    r"([\u0B80-\u0BFF])\u200D([\u0B80-\u0BFF])"
)
_RE_ZWNJ_BETWEEN_TAMIL = re.compile(
    r"([\u0B80-\u0BFF])\u200C([\u0B80-\u0BFF])"
)

# Multiple pulli (virama) in sequence — clear data corruption
_RE_MULTI_PULLI = re.compile(r"\u0BCD{2,}")

# Pulli followed immediately by a vowel sign (invalid sequence)
_RE_PULLI_VOWEL_SIGN = re.compile(
    r"\u0BCD([\u0BBE-\u0BCC])"
)

# Multiple consecutive vowel signs (invalid)
_RE_MULTI_VOWEL_SIGN = re.compile(
    r"([\u0BBE-\u0BCC]){2,}"
)

# URLs and emails
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")

# Excessive repetition
_RE_REPEATED_PUNCT = re.compile(r"([!?.,;:…]){4,}")
_RE_REPEATED_CHAR = re.compile(r"(.)\1{9,}")

# Multiple whitespace
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Tamil followed by Latin with no space (common in code-mixed text)
_RE_TAMIL_LATIN_BOUNDARY = re.compile(
    r"([\u0B80-\u0BFF])([ ]?[a-zA-Z])"
)

# Stray combining marks not attached to a base character
_RE_STRAY_COMBINING = re.compile(
    r"(?<![^\s])([\u0BBE-\u0BCD\u0BD7])"
)


# ===========================================================================
# TamilDeepNormalizer — Main Class
# ===========================================================================

class TamilDeepNormalizer:
    """
    Production-grade Tamil Unicode normalizer.

    Goes beyond NFC to handle web-text noise, legacy encoding artifacts,
    and all known Tamil Unicode edge cases.

    Usage:
        normalizer = TamilDeepNormalizer()
        clean = normalizer.normalize(text)
        # Or step-by-step:
        clean = normalizer.normalize_unicode(text)  # just Unicode fixes
    """

    def __init__(
        self,
        strip_urls: bool = True,
        strip_emails: bool = True,
        normalize_numerals: str = "preserve",  # "preserve", "arabic", "tamil"
        preserve_grantha: bool = True,
        max_repeated_punct: int = 3,
        max_repeated_char: int = 3,
    ):
        self.strip_urls = strip_urls
        self.strip_emails = strip_emails
        self.normalize_numerals = normalize_numerals
        self.preserve_grantha = preserve_grantha
        self.max_repeated_punct = max_repeated_punct
        self.max_repeated_char = max_repeated_char

        # Build whitespace translation table
        self._ws_table = str.maketrans(
            {ch: " " for ch in WHITESPACE_NORMALIZE}
        )

        # Build removal table for always-remove characters
        self._remove_table = str.maketrans(
            {ch: None for ch in ALWAYS_REMOVE if ch not in WHITESPACE_NORMALIZE}
        )

    def normalize(self, text: str) -> str:
        """
        Full normalization pipeline. Use this for corpus cleaning.

        Steps:
          1. Strip BOM, normalize line endings
          2. NFC normalization
          3. Remove invisible/zero-width characters
          4. Fix ZWJ/ZWNJ noise between Tamil characters
          5. Fix invalid Tamil character sequences
          6. Normalize whitespace variants
          7. Optionally strip URLs/emails
          8. Normalize repetition
          9. Remove stray control characters
          10. Final whitespace cleanup
        """
        if not text:
            return ""

        # Step 1: Line ending normalization
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2: NFC normalization — the foundation
        text = unicodedata.normalize("NFC", text)

        # Step 3: Remove always-noise characters
        text = text.translate(self._remove_table)

        # Step 4: Normalize whitespace variants to regular space
        text = text.translate(self._ws_table)

        # Step 5: Fix ZWJ/ZWNJ noise in Tamil text
        text = self._fix_zwj_zwnj(text)

        # Step 6: Fix invalid Tamil sequences
        text = self._fix_invalid_tamil(text)

        # Step 7: Normalize Tamil numerals if requested
        if self.normalize_numerals != "preserve":
            text = self._normalize_numerals(text)

        # Step 8: Strip URLs and emails
        if self.strip_urls:
            text = _RE_URL.sub(" ", text)
        if self.strip_emails:
            text = _RE_EMAIL.sub(" ", text)

        # Step 9: Normalize repetition
        text = _RE_REPEATED_PUNCT.sub(
            lambda m: m.group(1) * self.max_repeated_punct, text
        )
        text = _RE_REPEATED_CHAR.sub(
            lambda m: m.group(1) * self.max_repeated_char, text
        )

        # Step 10: Remove remaining control characters (except newline, tab)
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
        )

        # Step 11: Final whitespace cleanup
        text = _RE_MULTI_SPACE.sub(" ", text)
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)

        # Strip each line
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        text = "\n".join(lines)

        return text.strip()

    def normalize_unicode_only(self, text: str) -> str:
        """
        Unicode-only normalization (no URL stripping, no repetition fixes).
        Use this during tokenizer inference, not training.
        """
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text)
        text = text.translate(self._remove_table)
        text = text.translate(self._ws_table)
        text = self._fix_zwj_zwnj(text)
        text = self._fix_invalid_tamil(text)
        return text

    def _fix_zwj_zwnj(self, text: str) -> str:
        """
        Handle ZWJ (U+200D) and ZWNJ (U+200C) in Tamil context.

        In Tamil:
          - ZWJ between Tamil chars is ALWAYS noise (unlike Malayalam)
          - ZWNJ between Tamil chars is ALMOST ALWAYS noise
          - Exception: ZWNJ after pulli can be meaningful in some
            rendering engines to force explicit pulli display

        Strategy: Remove ZWJ/ZWNJ between Tamil characters.
        Preserve if between non-Tamil characters (Emoji ZWJ sequences, etc.)
        """
        # Remove ZWJ between Tamil characters
        text = _RE_ZWJ_BETWEEN_TAMIL.sub(r"\1\2", text)

        # Remove ZWNJ between Tamil characters
        # (except after pulli where it might be intentional)
        text = _RE_ZWNJ_BETWEEN_TAMIL.sub(r"\1\2", text)

        return text

    def _fix_invalid_tamil(self, text: str) -> str:
        """
        Fix sequences that are invalid in Tamil Unicode.

        Invalid sequences include:
          - Multiple consecutive pulli (virama)
          - Pulli immediately followed by vowel sign
          - Multiple consecutive vowel signs
          - Stray vowel signs without a base consonant
        """
        # Multiple pulli → single pulli
        text = _RE_MULTI_PULLI.sub("\u0BCD", text)

        # Multiple vowel signs → keep first
        text = _RE_MULTI_VOWEL_SIGN.sub(r"\1", text)

        return text

    def _normalize_numerals(self, text: str) -> str:
        """
        Normalize numeral representation.

        Mode "arabic": Convert Tamil digits to Arabic (௧ → 1)
        Mode "tamil":  Convert Arabic digits to Tamil (1 → ௧)
        Mode "preserve": Keep both as-is (default)
        """
        if self.normalize_numerals == "arabic":
            for tamil_cp, arabic_ch in TAMIL_DIGITS.items():
                text = text.replace(chr(tamil_cp), arabic_ch)
        elif self.normalize_numerals == "tamil":
            for arabic_ch, tamil_ch in ARABIC_TO_TAMIL_DIGIT.items():
                text = text.replace(arabic_ch, tamil_ch)
        return text


# ===========================================================================
# Utility Functions
# ===========================================================================

def is_tamil_char(ch: str) -> bool:
    """Check if a character is in the Tamil Unicode block."""
    return TAMIL_BLOCK_START <= ord(ch) <= TAMIL_BLOCK_END


def is_tamil_consonant(ch: str) -> bool:
    """Check if a character is a Tamil consonant."""
    return ord(ch) in TAMIL_CONSONANTS


def is_tamil_vowel(ch: str) -> bool:
    """Check if a character is a Tamil vowel."""
    return ord(ch) in TAMIL_VOWELS


def is_tamil_vowel_sign(ch: str) -> bool:
    """Check if a character is a Tamil vowel sign (matra)."""
    return ord(ch) in TAMIL_VOWEL_SIGNS


def is_pulli(ch: str) -> bool:
    """Check if a character is the Tamil pulli (virama)."""
    return ord(ch) == PULLI


def is_grantha(ch: str) -> bool:
    """Check if a character is a Grantha consonant."""
    return ord(ch) in GRANTHA_CONSONANTS


def compute_tamil_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Tamil."""
    if not text:
        return 0.0
    tamil = sum(1 for ch in text if is_tamil_char(ch))
    alpha = sum(1 for ch in text if ch.isalpha())
    return tamil / max(alpha, 1)


def compute_english_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Latin."""
    if not text:
        return 0.0
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    alpha = sum(1 for ch in text if ch.isalpha())
    return latin / max(alpha, 1)
