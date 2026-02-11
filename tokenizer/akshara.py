"""
akshara.py — AMB Layer 2: Akshara (Grapheme Cluster) Segmentation
===================================================================
Segments Tamil text into Aksharas — the atomic perceptual units of
Tamil script.

What is an Akshara?
  An akshara (அக்ஷரம்) is a grapheme cluster — a consonant with its
  vowel sign, or a standalone vowel, or a consonant with pulli (dot).
  It is what a Tamil reader perceives as a single "letter."

  Examples:
    க   → consonant + implicit 'a' vowel (one akshara)
    கா  → க + ா  (one akshara: "kaa")
    கி  → க + ி  (one akshara: "ki")
    க்  → க + ்  (one akshara: "k" with pulli, no vowel)
    கொ  → க + ெ + ா  (one akshara: "ko", composite vowel sign)
    க்ஷ → க + ் + ஷ  (conjunct consonant, one akshara)
    ஸ்ரீ → ஸ + ் + ர + ீ  (conjunct + vowel sign, one akshara)

Why this matters for tokenization:
  Standard BPE treats these as individual codepoints. This means "கா"
  is TWO tokens (க, ா) that BPE may or may not merge. If it doesn't,
  the model sees a meaningless floating vowel sign.

  AMB guarantees that aksharas are NEVER split. They are the atomic
  units, the TRUE Tamil alphabet of ~300 unique forms.

Algorithm:
  Tamil aksharas follow a regular grammar:
    AKSHARA → VOWEL
            | CONSONANT (PULLI CONSONANT)* [VOWEL_SIGN | PULLI]
            | AYTHAM
            | NON_TAMIL_CHAR

  We implement this as a state machine for maximum performance.

Usage:
    from akshara import AksharaSegmenter
    segmenter = AksharaSegmenter()
    aksharas = segmenter.segment("தமிழ்")
    # ['த', 'மி', 'ழ்']
"""

import re
from typing import List, Tuple, Optional, Iterator
from tamil_unicode import (
    TAMIL_CONSONANTS, TAMIL_VOWELS, TAMIL_VOWEL_SIGNS,
    PULLI, AYTHAM, TAMIL_DIGITS, TAMIL_NUMBER_SYMBOLS,
    TAMIL_SPECIAL, GRANTHA_CONSONANTS,
    TAMIL_BLOCK_START, TAMIL_BLOCK_END,
    is_tamil_char, is_tamil_consonant, is_tamil_vowel,
    is_tamil_vowel_sign, is_pulli,
)


# ===========================================================================
# Akshara Categories (for downstream morpheme analysis)
# ===========================================================================

class AksharaType:
    """Classification of an akshara for morphological analysis."""
    VOWEL = "vowel"                # அ, ஆ, இ, ஈ, etc.
    CONSONANT_VOWEL = "cv"         # க, கா, கி, கு, etc. (consonant + inherent/explicit vowel)
    CONSONANT_PULLI = "cp"         # க், ங், ச், etc. (consonant with pulli = pure consonant)
    CONJUNCT = "conjunct"          # க்ஷ, ஸ்ரீ, etc. (consonant clusters)
    AYTHAM = "aytham"              # ஃ
    DIGIT = "digit"                # ௦-௯
    SYMBOL = "symbol"              # ௐ, ௗ, etc.
    SPACE = "space"                # whitespace
    PUNCTUATION = "punctuation"    # . , ! ? etc.
    LATIN = "latin"                # ASCII letters
    OTHER = "other"                # everything else


class Akshara:
    """
    Represents a single akshara (grapheme cluster).

    Attributes:
        text:     The Unicode string of this akshara
        type:     AksharaType classification
        position: Character offset in the original string
        codepoints: List of Unicode codepoints
    """
    __slots__ = ("text", "type", "position", "codepoints")

    def __init__(self, text: str, akshara_type: str, position: int):
        self.text = text
        self.type = akshara_type
        self.position = position
        self.codepoints = [ord(ch) for ch in text]

    def __repr__(self):
        return f"Akshara({self.text!r}, {self.type}, pos={self.position})"

    def __str__(self):
        return self.text

    def __eq__(self, other):
        if isinstance(other, Akshara):
            return self.text == other.text
        return self.text == other

    def __hash__(self):
        return hash(self.text)

    @property
    def is_tamil(self) -> bool:
        """Is this akshara a Tamil script unit?"""
        return self.type in (
            AksharaType.VOWEL,
            AksharaType.CONSONANT_VOWEL,
            AksharaType.CONSONANT_PULLI,
            AksharaType.CONJUNCT,
            AksharaType.AYTHAM,
        )

    @property
    def base_consonant(self) -> Optional[str]:
        """Return the base consonant, if any."""
        for ch in self.text:
            if ord(ch) in TAMIL_CONSONANTS:
                return ch
        return None

    @property
    def vowel_component(self) -> Optional[str]:
        """Return the vowel sign, if any."""
        for ch in self.text:
            if ord(ch) in TAMIL_VOWEL_SIGNS:
                return ch
        return None

    @property
    def has_pulli(self) -> bool:
        """Does this akshara end with pulli (no vowel)?"""
        return len(self.text) > 0 and ord(self.text[-1]) == PULLI


# ===========================================================================
# AksharaSegmenter — Core State Machine
# ===========================================================================

class AksharaSegmenter:
    """
    Segments Tamil text into aksharas using a finite state machine.

    The grammar for Tamil aksharas:
      AKSHARA → VOWEL
              | CONSONANT (PULLI CONSONANT)* [VOWEL_SIGN | PULLI]
              | AYTHAM
              | NON_TAMIL_CHAR

    This correctly handles:
      - Simple consonant-vowel syllables: கா → [கா]
      - Pulli consonants: க் → [க்]
      - Conjunct consonants: க்ஷ → [க்ஷ] (via Pulli+Consonant chain)
      - Complex conjuncts: ஸ்ரீ → [ஸ்ரீ]
      - Composite vowel signs: கொ → [கொ] (ெ + ா are kept together)
      - Mixed text: "hello தமிழ்" → ['h','e','l','l','o',' ','த','மி','ழ்']
    """

    # Maximum consonants in a conjunct chain (safety limit)
    MAX_CONJUNCT_DEPTH = 4

    def __init__(self):
        # Precompute lookup sets for O(1) checks
        self._consonants = frozenset(TAMIL_CONSONANTS)
        self._vowels = frozenset(TAMIL_VOWELS)
        self._vowel_signs = frozenset(TAMIL_VOWEL_SIGNS)
        self._digits = frozenset(TAMIL_DIGITS.keys())
        self._symbols = frozenset(TAMIL_NUMBER_SYMBOLS | TAMIL_SPECIAL)

        # Build the complete akshara inventory for stats
        self._inventory = None

    def segment(self, text: str) -> List[Akshara]:
        """
        Segment text into a list of Akshara objects.

        This is the primary API. Returns a list of Akshara objects,
        each with its text, type, and position in the original string.
        """
        aksharas = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            cp = ord(ch)

            # --- Tamil Vowel (standalone) ---
            if cp in self._vowels:
                aksharas.append(Akshara(ch, AksharaType.VOWEL, i))
                i += 1

            # --- Tamil Consonant (start of syllable) ---
            elif cp in self._consonants:
                akshara_text, akshara_type, consumed = self._parse_consonant_cluster(text, i, n)
                aksharas.append(Akshara(akshara_text, akshara_type, i))
                i += consumed

            # --- Aytham ---
            elif cp == AYTHAM:
                aksharas.append(Akshara(ch, AksharaType.AYTHAM, i))
                i += 1

            # --- Tamil digit ---
            elif cp in self._digits:
                aksharas.append(Akshara(ch, AksharaType.DIGIT, i))
                i += 1

            # --- Tamil symbol ---
            elif cp in self._symbols:
                aksharas.append(Akshara(ch, AksharaType.SYMBOL, i))
                i += 1

            # --- Stray vowel sign (not attached to consonant — malformed) ---
            elif cp in self._vowel_signs or cp == PULLI:
                # Treat as its own unit (the normalizer should have caught this)
                aksharas.append(Akshara(ch, AksharaType.OTHER, i))
                i += 1

            # --- Whitespace ---
            elif ch.isspace():
                aksharas.append(Akshara(ch, AksharaType.SPACE, i))
                i += 1

            # --- Latin letters ---
            elif ch.isascii() and ch.isalpha():
                aksharas.append(Akshara(ch, AksharaType.LATIN, i))
                i += 1

            # --- Punctuation / Digits / Other ---
            elif ch in ".,;:!?\"'()-–—…/\\@#$%^&*_+=[]{}|<>~`":
                aksharas.append(Akshara(ch, AksharaType.PUNCTUATION, i))
                i += 1

            # --- Everything else ---
            else:
                aksharas.append(Akshara(ch, AksharaType.OTHER, i))
                i += 1

        return aksharas

    def segment_text(self, text: str) -> List[str]:
        """Segment text and return only the text strings (not Akshara objects)."""
        return [a.text for a in self.segment(text)]

    def segment_to_string(self, text: str, separator: str = "|") -> str:
        """Segment text and join with a separator for visualization."""
        return separator.join(self.segment_text(text))

    def _parse_consonant_cluster(
        self, text: str, start: int, length: int
    ) -> Tuple[str, str, int]:
        """
        Parse a Tamil consonant cluster starting at position `start`.

        Grammar:
          CLUSTER → CONSONANT (PULLI CONSONANT)* [VOWEL_SIGN | PULLI]

        Returns: (akshara_text, akshara_type, characters_consumed)
        """
        i = start
        conjunct_depth = 0

        # Eat the initial consonant
        i += 1

        # Try to eat PULLI + CONSONANT chains (conjuncts like க்ஷ)
        while (
            i < length
            and conjunct_depth < self.MAX_CONJUNCT_DEPTH
            and i + 1 < length
            and ord(text[i]) == PULLI
            and ord(text[i + 1]) in self._consonants
        ):
            i += 2  # eat pulli + next consonant
            conjunct_depth += 1

        # Now check what follows the consonant (chain)
        if i < length:
            next_cp = ord(text[i])

            # Vowel sign → consonant-vowel syllable
            if next_cp in self._vowel_signs:
                i += 1
                # Handle composite vowel signs (ொ = ெ + ா, ோ = ே + ா)
                # NFC should have composed these, but be safe
                if i < length and ord(text[i]) in self._vowel_signs:
                    i += 1

                akshara_text = text[start:i]
                if conjunct_depth > 0:
                    return akshara_text, AksharaType.CONJUNCT, i - start
                return akshara_text, AksharaType.CONSONANT_VOWEL, i - start

            # Pulli → pure consonant (no vowel)
            elif next_cp == PULLI:
                i += 1
                akshara_text = text[start:i]
                if conjunct_depth > 0:
                    return akshara_text, AksharaType.CONJUNCT, i - start
                return akshara_text, AksharaType.CONSONANT_PULLI, i - start

        # No vowel sign or pulli → consonant with inherent 'a' vowel
        akshara_text = text[start:i]
        if conjunct_depth > 0:
            return akshara_text, AksharaType.CONJUNCT, i - start
        return akshara_text, AksharaType.CONSONANT_VOWEL, i - start

    def get_tamil_aksharas_only(self, text: str) -> List[Akshara]:
        """Return only the Tamil aksharas, filtering out spaces, punctuation, etc."""
        return [a for a in self.segment(text) if a.is_tamil]

    def count_aksharas(self, text: str) -> int:
        """Count the number of Tamil aksharas in the text."""
        return len(self.get_tamil_aksharas_only(text))


# ===========================================================================
# Akshara Inventory Builder
# ===========================================================================

def build_akshara_inventory() -> dict:
    """
    Build the complete inventory of valid Tamil aksharas.

    Returns a dict mapping akshara_text → akshara_type.

    The Tamil writing system has:
      - 12 vowels (uyir)
      - 18 consonants × 12 vowel forms = 216 consonant-vowel syllables (uyirmei)
      - 18 consonants with pulli = 18 pure consonants (mei)
      - 5 Grantha consonants × 12 vowel forms = 60 Grantha syllables
      - 5 Grantha consonants with pulli = 5 Grantha pure consonants
      - 1 Aytham
      - Common conjuncts (க்ஷ, ஸ்ரீ, etc.)
    Total: ~312 base aksharas + conjuncts

    This is the TRUE Tamil alphabet — not the 256 bytes that BPE starts with.
    """
    inventory = {}

    # Vowels
    for cp in sorted(TAMIL_VOWELS):
        ch = chr(cp)
        inventory[ch] = AksharaType.VOWEL

    # Consonant + vowel sign combinations
    for cons_cp in sorted(TAMIL_CONSONANTS):
        cons = chr(cons_cp)

        # Consonant alone (inherent 'a' vowel)
        inventory[cons] = AksharaType.CONSONANT_VOWEL

        # Consonant + pulli (pure consonant)
        inventory[cons + chr(PULLI)] = AksharaType.CONSONANT_PULLI

        # Consonant + each vowel sign
        for vs_cp in sorted(TAMIL_VOWEL_SIGNS):
            inventory[cons + chr(vs_cp)] = AksharaType.CONSONANT_VOWEL

    # Aytham
    inventory[chr(AYTHAM)] = AksharaType.AYTHAM

    # Common conjuncts
    common_conjuncts = [
        "க்ஷ", "க்ஷி", "க்ஷை",  # ksha
        "ஸ்ர", "ஸ்ரீ",            # sri
        "த்ர", "த்ரி",            # tra, tri
    ]
    for conj in common_conjuncts:
        inventory[conj] = AksharaType.CONJUNCT

    return inventory


def get_akshara_stats(text: str) -> dict:
    """
    Compute akshara-level statistics for a text.

    Returns:
        dict with keys:
          - total_aksharas: total count
          - tamil_aksharas: count of Tamil script aksharas
          - unique_aksharas: set of unique akshara strings
          - type_distribution: count of each AksharaType
    """
    seg = AksharaSegmenter()
    aksharas = seg.segment(text)

    type_dist = {}
    unique = set()

    for a in aksharas:
        type_dist[a.type] = type_dist.get(a.type, 0) + 1
        unique.add(a.text)

    return {
        "total_aksharas": len(aksharas),
        "tamil_aksharas": sum(1 for a in aksharas if a.is_tamil),
        "unique_aksharas": len(unique),
        "type_distribution": type_dist,
    }
