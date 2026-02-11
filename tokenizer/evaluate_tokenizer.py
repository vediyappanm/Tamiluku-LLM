"""
evaluate_tokenizer.py - Production Tamil Tokenizer Benchmark Suite
===================================================================
Comprehensive evaluation framework with:
  - 8 core metrics (fertility, compression, roundtrip, coverage, etc.)
  - Multi-domain evaluation (wiki, news, social, classical, technical)
  - Cross-tokenizer comparison (GPT-4, LLaMA-3, Gemma, Qwen)
  - Extended morpheme test suite (100+ agglutinated Tamil words)
  - Support for both HuggingFace and SentencePiece tokenizers

Usage:
    python evaluate_tokenizer.py [--config config.yaml]
    python evaluate_tokenizer.py --interactive
    python evaluate_tokenizer.py --compare tiktoken,meta-llama/Llama-3-8B
    python evaluate_tokenizer.py --engine huggingface
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        script_dir = Path(__file__).parent
        script_relative_path = script_dir / path
        if script_relative_path.exists():
            path = str(script_relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Ensure the AMB modules can be imported
sys.path.append(str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Tamil Syllable Inventory (247 syllables)
# ---------------------------------------------------------------------------

def generate_tamil_syllables() -> List[str]:
    """Generate the complete Tamil syllable inventory (247 syllables)."""
    vowels = list("அஆஇஈஉஊஎஏஐஒஓஔ")
    consonant_bases = list("கஙசஞடணதநபமயரலவழளறன")
    vowel_signs = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]

    syllables = []
    syllables.extend(vowels)
    for c in consonant_bases:
        syllables.append(c + "்")  # Pure consonant form
    for c in consonant_bases:
        for vs in vowel_signs:
            syllables.append(c + vs)
    syllables.append("ஃ")  # Aaytham
    return syllables


# ---------------------------------------------------------------------------
# Extended Morpheme Test Suite (100+ words)
# ---------------------------------------------------------------------------

MORPHEME_TEST_CASES = [
    # Verb conjugations
    ("போகவேண்டியிருந்தது", ["போக", "வேண்டிய", "இருந்த", "து"]),
    ("படிக்கிறான்", ["படி", "க்கிறான்"]),
    ("வருகிறார்கள்", ["வரு", "கிறார்கள்"]),
    ("எழுதுவதற்கு", ["எழுது", "வதற்கு"]),
    ("சொல்லவில்லை", ["சொல்ல", "வில்லை"]),
    ("நடக்கும்", ["நட", "க்கும்"]),
    ("பேசிக்கொண்டிருந்தான்", ["பேசி", "க்கொண்டிருந்தான்"]),
    ("பார்க்கலாம்", ["பார்", "க்கலாம்"]),
    ("செய்யப்படுகிறது", ["செய்ய", "ப்படுகிறது"]),
    ("கற்றுக்கொள்ள", ["கற்று", "க்கொள்ள"]),

    # Tense forms
    ("ஓடினான்", ["ஓடி", "னான்"]),
    ("பாடுவாள்", ["பாடு", "வாள்"]),
    ("சாப்பிட்டேன்", ["சாப்பிட்", "டேன்"]),
    ("தூங்கினார்கள்", ["தூங்கி", "னார்கள்"]),
    ("விளையாடுகிறோம்", ["விளையாடு", "கிறோம்"]),
    ("படிப்பார்கள்", ["படி", "ப்பார்கள்"]),
    ("எழுதினேன்", ["எழுதி", "னேன்"]),
    ("கேட்டான்", ["கேட்", "டான்"]),

    # Case markers
    ("வீட்டிற்கு", ["வீட்", "டிற்கு"]),
    ("பள்ளியில்", ["பள்ளி", "யில்"]),
    ("மாணவர்களுக்கு", ["மாணவர்கள்", "உக்கு"]),
    ("நாட்டின்", ["நாட்", "டின்"]),
    ("ஊரிலிருந்து", ["ஊரில்", "இருந்து"]),
    ("கடையிடம்", ["கடை", "யிடம்"]),

    # Noun derivations
    ("அழகான", ["அழக", "ான"]),
    ("நல்லவன்", ["நல்ல", "வன்"]),
    ("பெரியது", ["பெரிய", "து"]),
    ("சிறியவள்", ["சிறிய", "வள்"]),
    ("புதியவை", ["புதிய", "வை"]),

    # Complex agglutination
    ("படிக்கவேண்டியதில்லை", ["படிக்க", "வேண்டிய", "தில்லை"]),
    ("சொல்லிக்கொண்டிருக்கிறான்", ["சொல்லி", "க்கொண்டிருக்கிறான்"]),
    ("வந்துகொண்டிருந்தார்கள்", ["வந்து", "கொண்டிருந்தார்கள்"]),
    ("செய்யமுடியாதவர்கள்", ["செய்ய", "முடியாத", "வர்கள்"]),
    ("போகவேண்டாம்", ["போக", "வேண்டாம்"]),

    # Participial nouns
    ("படித்தவன்", ["படித்த", "வன்"]),
    ("வந்தவர்கள்", ["வந்த", "வர்கள்"]),
    ("செய்தது", ["செய்த", "து"]),
    ("கொடுத்தவள்", ["கொடுத்த", "வள்"]),

    # Verbal nouns
    ("படிப்பது", ["படிப்", "பது"]),
    ("செய்வது", ["செய்", "வது"]),
    ("போவது", ["போ", "வது"]),
    ("வருவது", ["வரு", "வது"]),

    # Postpositions
    ("மேசையின்மேல்", ["மேசை", "யின்மேல்"]),
    ("வீட்டுக்குள்", ["வீட்", "டுக்குள்"]),
    ("மரத்தடியில்", ["மரத்தடி", "யில்"]),

    # Compound words
    ("தலைநகரம்", ["தலை", "நகரம்"]),
    ("கல்வித்துறை", ["கல்வி", "த்துறை"]),
    ("விமானநிலையம்", ["விமான", "நிலையம்"]),
    ("புத்தகக்கடை", ["புத்தக", "க்கடை"]),
    ("மருத்துவமனை", ["மருத்துவ", "மனை"]),
    ("பல்கலைக்கழகம்", ["பல்கலை", "க்கழகம்"]),

    # Negative forms
    ("வரமாட்டான்", ["வர", "மாட்டான்"]),
    ("செய்யவில்லை", ["செய்ய", "வில்லை"]),
    ("படிக்காதவன்", ["படிக்", "காதவன்"]),
    ("போகாமல்", ["போகா", "மல்"]),

    # Causative / passive
    ("எழுதச்செய்தான்", ["எழுத", "ச்செய்தான்"]),
    ("படிக்கவைத்தாள்", ["படிக்க", "வைத்தாள்"]),

    # Honorific forms
    ("வாருங்கள்", ["வாரு", "ங்கள்"]),
    ("சொல்லுங்கள்", ["சொல்லு", "ங்கள்"]),
    ("பாருங்கள்", ["பாரு", "ங்கள்"]),

    # Conditional/concessive
    ("வந்தால்", ["வந்", "தால்"]),
    ("படித்தாலும்", ["படித்", "தாலும்"]),
    ("போனாலும்", ["போனா", "லும்"]),
    ("செய்தாலும்", ["செய்", "தாலும்"]),

    # Infinitive + auxiliary
    ("போய்விட்டான்", ["போய்", "விட்டான்"]),
    ("வந்துவிட்டாள்", ["வந்து", "விட்டாள்"]),
    ("செய்துமுடித்தான்", ["செய்து", "முடித்தான்"]),

    # Question forms
    ("போகிறாயா", ["போகிறாய்", "ஆ"]),
    ("வருவாளா", ["வருவாள்", "ஆ"]),

    # Modern/technical
    ("கணினிமயமாக்கல்", ["கணினி", "மயமாக்கல்"]),
    ("தொழில்நுட்பவியலாளர்", ["தொழில்நுட்ப", "வியலாளர்"]),
    ("மின்னணுவியல்", ["மின்", "னணுவியல்"]),
    ("செயற்கைநுண்ணறிவு", ["செயற்கை", "நுண்ணறிவு"]),
    ("தரவுத்தளம்", ["தரவு", "த்தளம்"]),

    # Relative participle constructions
    ("படிக்கின்ற", ["படிக்", "கின்ற"]),
    ("எழுதுகின்ற", ["எழுது", "கின்ற"]),
    ("வருகின்ற", ["வரு", "கின்ற"]),

    # Emphatic forms
    ("அவன்தான்", ["அவன்", "தான்"]),
    ("இதுவே", ["இது", "வே"]),
    ("அவர்களும்", ["அவர்கள்", "உம்"]),

    # More complex forms
    ("நடத்திக்கொண்டிருக்கிறார்கள்", ["நடத்தி", "க்கொண்டிருக்கிறார்கள்"]),
    ("பேசிக்கொள்ளலாம்", ["பேசி", "க்கொள்ளலாம்"]),
    ("படிக்கவேண்டியிருக்கிறது", ["படிக்க", "வேண்டியிருக்கிறது"]),
    ("செய்யப்பட்டிருக்கிறது", ["செய்ய", "ப்பட்டிருக்கிறது"]),
    ("கொண்டுவரப்பட்டது", ["கொண்டுவர", "ப்பட்டது"]),

    # Adverbial forms
    ("வேகமாக", ["வேகம்", "ஆக"]),
    ("அழகாக", ["அழக", "ாக"]),
    ("நன்றாக", ["நன்ற", "ாக"]),
    ("மெதுவாக", ["மெதுவ", "ாக"]),

    # Plural forms
    ("மாணவர்கள்", ["மாணவர்", "கள்"]),
    ("குழந்தைகள்", ["குழந்தை", "கள்"]),
    ("பறவைகள்", ["பறவை", "கள்"]),
    ("நாடுகள்", ["நாடு", "கள்"]),
    ("மொழிகள்", ["மொழி", "கள்"]),
]

# ---------------------------------------------------------------------------
# Multi-Domain Tamil Eval Sentences
# ---------------------------------------------------------------------------

DOMAIN_EVAL_SENTENCES = {
    "formal_prose": [
        "இந்திய அரசியலமைப்புச் சட்டம் அனைவருக்கும் சமத்துவத்தை உறுதிசெய்கிறது.",
        "ஜனநாயக அரசாங்கத்தின் அடிப்படைக் கொள்கைகளில் கருத்து சுதந்திரமும் ஒன்றாகும்.",
        "தமிழ்நாடு அரசு கல்வித் துறையில் பல்வேறு சீர்திருத்தங்களை மேற்கொண்டு வருகிறது.",
        "சுற்றுச்சூழல் பாதுகாப்பு இன்றைய காலகட்டத்தின் மிக முக்கியமான தேவையாகும்.",
        "பொருளாதார வளர்ச்சிக்கு தொழில்நுட்ப முன்னேற்றம் இன்றியமையாததாகும்.",
    ],
    "colloquial": [
        "என்ன மாப்பிள்ளை எப்படி இருக்க நல்லா இருக்கியா?",
        "போன வாரம் சினிமா பார்த்தேன் அட்டகாசமா இருந்துச்சு!",
        "இந்த கடை சாப்பாடு செம்ம டேஸ்ட்டா இருக்கு பாஸ்.",
        "நாளைக்கு ஆபீஸ் லீவ் போடலாமா என்ன சொல்ற?",
        "ரொம்ப நாளா உன்னை பார்க்கணும்னு நினைச்சேன்.",
    ],
    "classical": [
        "அறத்துப்பால் இல்லறவியல் அன்பு உடைமை என்பது அறிவியலின் அடிப்படையாகும்.",
        "யாதும் ஊரே யாவரும் கேளிர் தீதும் நன்றும் பிறர்தர வாரா.",
        "கல்தோன்றி மண்தோன்றா காலத்தே வாளொடு முன்தோன்றிய மூத்தகுடி.",
        "தமிழுக்கு அமுதென்று பேர் அந்தத் தமிழ் இன்பத் தமிழ் எங்கள் உயிருக்கு நேர்.",
        "ஒன்றே குலம் ஒருவனே தேவன் என்று சொன்ன திருமூலரின் வாக்கு.",
    ],
    "technical": [
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது.",
        "கணினி வலையமைப்பில் TCP/IP நெறிமுறை அடிப்படையானது.",
        "இயந்திர கற்றல் வழிமுறைகள் தரவு பகுப்பாய்வில் பயன்படுகின்றன.",
        "மரபணு மாற்றத் தொழில்நுட்பம் வேளாண்மையில் புரட்சியை ஏற்படுத்தியுள்ளது.",
        "குவாண்டம் கணினிகள் எதிர்கால கணிப்பொறி அறிவியலின் திசையை மாற்றும்.",
    ],
    "code_mixed": [
        "Machine learning model-ஐ train பண்ணனும்.",
        "React component-ல state management செய்ய Redux use பண்ணலாம்.",
        "Docker container-ல deploy பண்ணி Kubernetes-ல orchestrate செய்யலாம்.",
        "API endpoint-ஐ test பண்ண Postman use பண்ணு.",
        "Git branch create பண்ணி pull request raise பண்ணு.",
    ],
    "numeric": [
        "2024-ல் தமிழ்நாட்டின் மக்கள்தொகை 8.5 கோடி.",
        "இந்த திட்டத்திற்கு ₹50,000 கோடி ஒதுக்கீடு செய்யப்பட்டுள்ளது.",
        "சென்னையின் சராசரி வெப்பநிலை 28.4°C ஆகும்.",
        "GDP வளர்ச்சி விகிதம் 7.2% என எதிர்பார்க்கப்படுகிறது.",
        "தமிழ் மொழிக்கு 2,500+ ஆண்டு வரலாறு உள்ளது.",
    ],
}


# ---------------------------------------------------------------------------
# Tokenizer Wrappers (support both HF and SentencePiece)
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """Abstract wrapper for different tokenizer backends."""

    def encode_tokens(self, text: str) -> List[str]:
        raise NotImplementedError

    def encode_ids(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError

    def decode_pieces(self, ids: List[int]) -> List[str]:
        """Decode each id as an individual piece."""
        return [self.decode([i]) for i in ids]




class HuggingFaceTokenizerWrapper(TokenizerWrapper):
    """Wrapper for HuggingFace tokenizers."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        self.name = Path(path).parent.name
        log.info(f"Loaded HF tokenizer: {path} (vocab={self.tokenizer.get_vocab_size():,})")

    def encode_tokens(self, text: str) -> List[str]:
        # Encode to IDs first, then decode each ID individually to get true UTF-8 tokens
        # HF ByteLevel BPE token strings (e.g. 'à®±') are not valid Tamil strings.
        ids = self.tokenizer.encode(text).ids
        return [self.tokenizer.decode([i]) for i in ids]

    def encode_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()


class AMBTokenizerWrapper(HuggingFaceTokenizerWrapper):
    """Wrapper for AMB tokenizers (HF + Deep Normalization)."""

    def __init__(self, path: str):
        super().__init__(path)
        from tamil_unicode import TamilDeepNormalizer
        self.normalizer = TamilDeepNormalizer()
        self.name = f"amb-{Path(path).parent.name}"

    def encode_tokens(self, text: str) -> List[str]:
        # Apply Layer 1 before tokenizing
        clean_text = self.normalizer.normalize(text)
        return super().encode_tokens(clean_text)

    def encode_ids(self, text: str) -> List[int]:
        clean_text = self.normalizer.normalize(text)
        return super().encode_ids(clean_text)


class SentencePieceTokenizerWrapper(TokenizerWrapper):
    """Wrapper for SentencePiece tokenizers."""

    def __init__(self, path: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)
        self.name = Path(path).stem
        log.info(f"Loaded SP tokenizer: {path} (vocab={self.sp.get_piece_size():,})")

    def encode_tokens(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def encode_ids(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}


class TransformersTokenizerWrapper(TokenizerWrapper):
    """Wrapper for HuggingFace transformers tokenizers (for comparison)."""

    def __init__(self, model_id: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.name = model_id.split("/")[-1]
        log.info(f"Loaded transformers tokenizer: {model_id} (vocab={len(self.tokenizer):,})")

    def encode_tokens(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def encode_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()


class TiktokenWrapper(TokenizerWrapper):
    """Wrapper for OpenAI's tiktoken (GPT-4 tokenizer)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken-{encoding_name}"
        log.info(f"Loaded tiktoken: {encoding_name} (vocab={self.enc.n_vocab:,})")

    def encode_tokens(self, text: str) -> List[str]:
        ids = self.enc.encode(text)
        return [self.enc.decode([i]) for i in ids]

    def encode_ids(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_vocab(self) -> Dict[str, int]:
        return {}  # tiktoken doesn't expose full vocab easily


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

class TokenizerEvaluator:
    """Comprehensive evaluation suite for Tamil tokenizers."""

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tok = tokenizer

    @staticmethod
    def _clean_piece(piece: str) -> str:
        # Remove common token markers and surrounding whitespace
        return piece.replace("▁", "").replace("Ġ", "").replace("Ċ", "").strip()

    def _decoded_pieces(self, text: str) -> List[str]:
        ids = self.tok.encode_ids(text)
        pieces = [self._clean_piece(p) for p in self.tok.decode_pieces(ids)]
        return [p for p in pieces if p]

    def measure_fertility(self, texts: List[str]) -> Dict:
        """Fertility = average tokens per word. Target: <= 1.5."""
        total_words = 0
        total_tokens = 0
        per_doc_fertility = []

        for text in texts:
            words = text.split()
            token_ids = self.tok.encode_ids(text)
            n_words = len(words)
            n_tokens = len(token_ids)
            if n_words > 0:
                total_words += n_words
                total_tokens += n_tokens
                per_doc_fertility.append(n_tokens / n_words)

        avg_fertility = total_tokens / max(total_words, 1)

        return {
            "avg_fertility": round(avg_fertility, 4),
            "median_fertility": round(
                sorted(per_doc_fertility)[len(per_doc_fertility) // 2], 4
            ) if per_doc_fertility else 0,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "num_documents": len(texts),
        }

    def measure_compression(self, texts: List[str]) -> Dict:
        """Compression = bytes per token. Target: >= 4.0."""
        total_bytes = 0
        total_tokens = 0

        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(self.tok.encode_ids(text))

        ratio = total_bytes / max(total_tokens, 1)
        return {
            "bytes_per_token": round(ratio, 4),
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
        }

    def measure_roundtrip(self, texts: List[str]) -> Dict:
        """Roundtrip = encode->decode fidelity. Target: 100%."""
        total = len(texts)
        matches = 0
        failures = []

        for text in texts:
            ids = self.tok.encode_ids(text)
            decoded = self.tok.decode(ids)

            if decoded.strip() == text.strip():
                matches += 1
            else:
                failures.append({
                    "input": text[:200],
                    "decoded": decoded[:200],
                })

        accuracy = matches / max(total, 1)
        return {
            "roundtrip_accuracy": round(accuracy, 6),
            "total_tested": total,
            "passed": matches,
            "failed": total - matches,
            "failure_samples": failures[:5],
        }

    def measure_tamil_coverage(self) -> Dict:
        """Tamil syllable coverage. Target: >= 95%."""
        syllables = generate_tamil_syllables()
        total = len(syllables)
        covered = 0
        uncovered = []

        for syl in syllables:
            pieces = self._decoded_pieces(syl)
            if len(pieces) == 1 and pieces[0] == syl:
                covered += 1
            elif syl in pieces:
                covered += 1
            else:
                uncovered.append({"syllable": syl, "tokenized_as": pieces})

        coverage = covered / max(total, 1)
        return {
            "tamil_syllable_coverage": round(coverage, 4),
            "total_syllables": total,
            "covered": covered,
            "uncovered_count": len(uncovered),
            "uncovered_samples": uncovered[:20],
        }

    def measure_morpheme_boundary_accuracy(self) -> Dict:
        """Morpheme boundary accuracy on extended test suite. Target: >= 70%."""
        total = len(MORPHEME_TEST_CASES)
        aligned = 0
        results = []

        for word, expected_morphemes in MORPHEME_TEST_CASES:
            clean_tokens = self._decoded_pieces(word)

            # Check boundary alignment
            token_text = ""
            token_boundaries = set()
            for t in clean_tokens:
                token_boundaries.add(len(token_text))
                token_text += t

            morpheme_text = ""
            morpheme_boundaries = set()
            for m in expected_morphemes:
                morpheme_boundaries.add(len(morpheme_text))
                morpheme_text += m

            if morpheme_boundaries:
                overlap = len(morpheme_boundaries & token_boundaries)
                boundary_acc = overlap / len(morpheme_boundaries)
            else:
                boundary_acc = 0

            if boundary_acc >= 0.5:
                aligned += 1

            results.append({
                "word": word,
                "expected_morphemes": expected_morphemes,
                "actual_tokens": clean_tokens,
                "boundary_accuracy": round(boundary_acc, 2),
            })

        accuracy = aligned / max(total, 1)
        return {
            "morpheme_boundary_accuracy": round(accuracy, 4),
            "total_tested": total,
            "aligned": aligned,
            "results": results[:30],  # First 30 for report
        }

    def measure_cross_script_leakage(self) -> Dict:
        """Detect tokens containing both Tamil and Latin. Target: 0."""
        vocab = self.tok.get_vocab()
        leaky_tokens = []

        for token, token_id in vocab.items():
            if token.startswith("<") and token.endswith(">"):
                continue

            piece_clean = self._clean_piece(self.tok.decode([token_id]))
            if not piece_clean:
                continue

            has_tamil = any(0x0B80 <= ord(ch) <= 0x0BFF for ch in piece_clean)
            has_latin = any(ch.isascii() and ch.isalpha() for ch in piece_clean)

            if has_tamil and has_latin:
                leaky_tokens.append({"id": token_id, "token": token})

        return {
            "cross_script_leakage_count": len(leaky_tokens),
            "leaky_tokens": leaky_tokens[:20],
        }

    def measure_domain_fertility(self) -> Dict:
        """Fertility breakdown by domain."""
        domain_results = {}
        for domain, sentences in DOMAIN_EVAL_SENTENCES.items():
            fertility = self.measure_fertility(sentences)
            domain_results[domain] = {
                "avg_fertility": fertility["avg_fertility"],
                "total_words": fertility["total_words"],
                "total_tokens": fertility["total_tokens"],
            }
        return domain_results

    def run_full_evaluation(self, texts: List[str]) -> Dict:
        """Run all evaluation metrics."""
        log.info("Running full evaluation suite...")

        report = {
            "tokenizer_name": self.tok.name,
            "vocab_size": self.tok.vocab_size(),
        }

        log.info("  [1/8] Measuring fertility...")
        report["fertility"] = self.measure_fertility(texts)

        log.info("  [2/8] Measuring compression ratio...")
        report["compression"] = self.measure_compression(texts)

        log.info("  [3/8] Measuring roundtrip accuracy...")
        report["roundtrip"] = self.measure_roundtrip(texts)

        log.info("  [4/8] Measuring Tamil syllable coverage...")
        report["tamil_coverage"] = self.measure_tamil_coverage()

        log.info("  [5/8] Measuring morpheme boundary accuracy...")
        report["morpheme_boundaries"] = self.measure_morpheme_boundary_accuracy()

        log.info("  [6/8] Detecting cross-script leakage...")
        report["cross_script"] = self.measure_cross_script_leakage()

        log.info("  [7/8] Measuring domain-specific fertility...")
        report["domain_fertility"] = self.measure_domain_fertility()

        log.info("  [8/8] Computing token length distribution...")
        report["token_length_distribution"] = self._token_length_dist(texts)

        return report

    def _token_length_dist(self, texts: List[str]) -> Dict:
        """Distribution of token lengths (in characters)."""
        lengths = Counter()
        for text in texts:
            for token in self._decoded_pieces(text):
                clean = token
                lengths[len(clean)] += 1

        total = sum(lengths.values())
        dist = {str(k): round(v / max(total, 1), 4) for k, v in sorted(lengths.items())[:20]}
        avg_len = sum(k * v for k, v in lengths.items()) / max(total, 1)

        return {
            "avg_token_length_chars": round(avg_len, 2),
            "distribution": dist,
        }


# ---------------------------------------------------------------------------
# Target Checking
# ---------------------------------------------------------------------------

def check_targets(report: Dict, targets: Dict) -> Dict:
    checks = [
        ("fertility", "avg_fertility", "max_fertility", "<="),
        ("compression", "bytes_per_token", "min_compression_ratio", ">="),
        ("roundtrip", "roundtrip_accuracy", "roundtrip_accuracy", ">="),
        ("tamil_coverage", "tamil_syllable_coverage", "min_tamil_coverage", ">="),
        ("morpheme_boundaries", "morpheme_boundary_accuracy", "min_morpheme_boundary_acc", ">="),
        ("cross_script", "cross_script_leakage_count", "max_cross_script_leakage", "<="),
    ]

    results = {}
    all_pass = True

    for section, metric, target_key, op in checks:
        actual = report.get(section, {}).get(metric, None)
        target = targets.get(target_key, None)

        if actual is None or target is None:
            results[metric] = {"status": "SKIP", "actual": actual, "target": target}
            continue

        if op == "<=" and actual <= target:
            passed = True
        elif op == ">=" and actual >= target:
            passed = True
        else:
            passed = False
            all_pass = False

        results[metric] = {
            "status": "PASS" if passed else "FAIL",
            "actual": actual,
            "target": target,
            "operator": op,
        }

    results["overall"] = "PASS" if all_pass else "FAIL"
    return results


# ---------------------------------------------------------------------------
# Cross-Tokenizer Comparison
# ---------------------------------------------------------------------------

def run_comparison(our_tokenizer: TokenizerWrapper, compare_ids: List[str]):
    """Compare our tokenizer against other tokenizers on Tamil text."""
    # Gather all eval sentences
    all_sentences = []
    for sentences in DOMAIN_EVAL_SENTENCES.values():
        all_sentences.extend(sentences)

    log.info(f"\n{'='*70}")
    log.info(f"CROSS-TOKENIZER COMPARISON ({len(all_sentences)} Tamil sentences)")
    log.info(f"{'='*70}")

    results = {}

    # Our tokenizer
    our_eval = TokenizerEvaluator(our_tokenizer)
    our_fertility = our_eval.measure_fertility(all_sentences)
    our_compression = our_eval.measure_compression(all_sentences)
    results[our_tokenizer.name] = {
        "vocab_size": our_tokenizer.vocab_size(),
        "fertility": our_fertility["avg_fertility"],
        "compression": our_compression["bytes_per_token"],
        "is_ours": True,
    }

    # Comparison tokenizers
    for comp_id in compare_ids:
        try:
            if comp_id == "tiktoken" or comp_id == "gpt4":
                wrapper = TiktokenWrapper("cl100k_base")
            elif comp_id.startswith("tiktoken:"):
                wrapper = TiktokenWrapper(comp_id.split(":")[1])
            else:
                wrapper = TransformersTokenizerWrapper(comp_id)

            comp_eval = TokenizerEvaluator(wrapper)
            comp_fertility = comp_eval.measure_fertility(all_sentences)
            comp_compression = comp_eval.measure_compression(all_sentences)

            results[wrapper.name] = {
                "vocab_size": wrapper.vocab_size(),
                "fertility": comp_fertility["avg_fertility"],
                "compression": comp_compression["bytes_per_token"],
                "is_ours": False,
            }
        except Exception as e:
            log.warning(f"  Could not load {comp_id}: {e}")

    # Print comparison table
    print(f"\n{'Tokenizer':<30} {'Vocab':>10} {'Fertility':>10} {'Bytes/Tok':>10}")
    print("-" * 62)
    for name, data in sorted(results.items(), key=lambda x: x[1]["fertility"]):
        marker = " <-- OURS" if data["is_ours"] else ""
        print(f"{name:<30} {data['vocab_size']:>10,} {data['fertility']:>10.3f} {data['compression']:>10.2f}{marker}")
    print("-" * 62)

    return results


# ---------------------------------------------------------------------------
# Eval Text Loading
# ---------------------------------------------------------------------------

def load_eval_texts(eval_dir: str) -> List[str]:
    texts = []
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        log.info(f"Eval directory not found: {eval_dir}. Using internal domain evaluation sentences only.")
    else:
        for txt_file in sorted(eval_path.glob("*.txt")):
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
                docs = [d.strip() for d in content.split("\n\n") if d.strip()]
                texts.extend(docs)

    # Add built-in domain eval sentences
    for sentences in DOMAIN_EVAL_SENTENCES.values():
        texts.extend(sentences)

    log.info(f"Loaded {len(texts):,} eval documents (file + built-in domains)")
    return texts


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------

def interactive_mode(tokenizer: TokenizerWrapper):
    print(f"\n{'='*60}")
    print(f"Tamil Tokenizer Interactive Mode")
    print(f"Tokenizer: {tokenizer.name} | Vocab: {tokenizer.vocab_size():,}")
    print(f"Type Tamil text to see tokenization. Type 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        ids = tokenizer.encode_ids(text)
        tokens = tokenizer.decode_pieces(ids)
        decoded = tokenizer.decode(ids)
        words = text.split()
        fertility = len(tokens) / max(len(words), 1)

        print(f"  Tokens:    {tokens}")
        print(f"  IDs:       {ids}")
        print(f"  Decoded:   {decoded}")
        print(f"  Fertility: {fertility:.2f} ({len(tokens)} tokens / {len(words)} words)")
        print(f"  Roundtrip: {'OK' if decoded.strip() == text.strip() else 'MISMATCH!'}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tamil tokenizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", default=None, help="Path to tokenizer file")
    parser.add_argument(
        "--engine", default=None, choices=["amb", "huggingface", "sentencepiece"],
        help="Tokenizer engine"
    )
    parser.add_argument("--test-dir", default=None, help="Test corpus directory")
    parser.add_argument("--report", default=None, help="Output report path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--compare", default=None,
        help="Comma-separated tokenizer IDs to compare (e.g., tiktoken,meta-llama/Llama-3-8B)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = args.engine or cfg.get("tokenizer", {}).get("engine", "huggingface")

    # Load tokenizer
    if args.model:
        model_path = args.model
    elif engine == "amb":
        output_dir = cfg.get("tokenizer", {}).get("output_dir", "models/amb_tokenizer")
        model_path = str(Path(output_dir) / "tokenizer.json")
    elif engine == "huggingface":
        output_dir = cfg.get("tokenizer", {}).get("output_dir", "models/tamil_tokenizer")
        model_path = str(Path(output_dir) / "tokenizer.json")
    else:
        model_path = f"{cfg['sentencepiece']['model_prefix']}.model"

    if engine == "amb":
        tokenizer = AMBTokenizerWrapper(model_path)
    elif engine == "huggingface":
        tokenizer = HuggingFaceTokenizerWrapper(model_path)
    else:
        tokenizer = SentencePieceTokenizerWrapper(model_path)

    # Interactive mode
    if args.interactive:
        interactive_mode(tokenizer)
        return

    # Load eval texts
    eval_dir = args.test_dir or cfg["evaluation"]["test_dir"]
    report_path = args.report or cfg["evaluation"]["report_path"]
    targets = cfg["evaluation"]["targets"]

    texts = load_eval_texts(eval_dir)
    if not texts:
        log.error("No evaluation texts found (neither files nor internal domains).")
        sys.exit(1)

    # Run evaluation
    evaluator = TokenizerEvaluator(tokenizer)
    report = evaluator.run_full_evaluation(texts)

    # Check targets
    target_results = check_targets(report, targets)
    report["target_check"] = target_results

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {tokenizer.name}")
    print(f"{'='*60}")
    print(f"  Vocab size:          {report['vocab_size']:,}")
    print(f"  Fertility:           {report['fertility']['avg_fertility']:.3f} (target: <={targets['max_fertility']})")
    print(f"  Compression:         {report['compression']['bytes_per_token']:.2f} bytes/tok (target: >={targets['min_compression_ratio']})")
    print(f"  Roundtrip:           {report['roundtrip']['roundtrip_accuracy']:.4%} (target: 100%)")
    print(f"  Tamil coverage:      {report['tamil_coverage']['tamil_syllable_coverage']:.2%} (target: >={targets['min_tamil_coverage']:.0%})")
    print(f"  Morpheme boundary:   {report['morpheme_boundaries']['morpheme_boundary_accuracy']:.2%} (target: >={targets['min_morpheme_boundary_acc']:.0%})")
    print(f"  Cross-script leak:   {report['cross_script']['cross_script_leakage_count']} (target: {targets['max_cross_script_leakage']})")
    print(f"  Avg token length:    {report['token_length_distribution']['avg_token_length_chars']:.1f} chars")
    print(f"")
    print(f"  Domain fertility:")
    for domain, data in report.get("domain_fertility", {}).items():
        print(f"    {domain:<20} {data['avg_fertility']:.3f}")
    
    print(f"\n  Tokenization Samples:")
    # We use some common words to see how they are split
    debug_words = ["தமிழ்", "வீடுகளிலிருந்து", "கணினி", "அம்மா", "123456"]
    for word in debug_words:
        # Use our wrapper to get human-readable tokens
        tokens = tokenizer.encode_tokens(word)
        print(f"    {word:<20} -> {tokens}")

    if report['tamil_coverage']['uncovered_count'] > 0:
        print(f"\n  Top Uncovered Syllables:")
        for item in report['tamil_coverage']['uncovered_samples'][:10]:
            print(f"    {item['syllable']}: {item['tokenized_as']}")
    print(f"{'='*60}")
    print(f"  OVERALL: {target_results['overall']}")
    print(f"{'='*60}")

    # Save report
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_file), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Full report saved to {report_file}")

    # Cross-tokenizer comparison
    if args.compare:
        compare_ids = [c.strip() for c in args.compare.split(",")]
        comparison = run_comparison(tokenizer, compare_ids)
        comp_path = Path("reports") / "comparison_report.json"
        with open(str(comp_path), "w") as f:
            json.dump(comparison, f, indent=2)
        log.info(f"Comparison report saved to {comp_path}")

    # Guidance
    if target_results["overall"] == "PASS":
        log.info("All targets met! Next step: python merge_vocabularies.py")
    else:
        log.warning("Some targets not met. Review recommendations:")
        for metric, result in target_results.items():
            if isinstance(result, dict) and result.get("status") == "FAIL":
                log.warning(f"  FAIL: {metric} = {result['actual']} (target {result['operator']} {result['target']})")


if __name__ == "__main__":
    main()
