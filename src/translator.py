"""Translator module using CTranslate2 with NLLB-200 for translation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ctranslate2
import yaml
from transformers import AutoTokenizer


# Default model repository (pre-converted CT2 int8 model)
# Options:
# - "JustFrederik/nllb-200-distilled-600M-ct2-int8" (600M - faster, lower quality)
# - "michaelfeil/ct2fast-nllb-200-1.3B" (1.3B - better quality, still fast on M4 Max)
# - "michaelfeil/ct2fast-nllb-200-3.3B" (3.3B - highest quality, slower)
DEFAULT_MODEL_REPO = "michaelfeil/ct2fast-nllb-200-1.3B"  # Upgraded for better quality

# Cache directory for models
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nllb-ct2"

# FLORES-200 language code mapping
# Maps simple ISO 639-1 codes to NLLB FLORES-200 codes
LANGUAGE_CODE_MAP = {
    # Major European languages
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "fi": "fin_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "is": "isl_Latn",
    # Asian languages
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
    # Middle Eastern languages
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "tr": "tur_Latn",
    # African languages
    "sw": "swh_Latn",
    "am": "amh_Ethi",
    "ha": "hau_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "af": "afr_Latn",
}

# Reverse mapping for display
NLLB_TO_SIMPLE = {v: k for k, v in LANGUAGE_CODE_MAP.items()}

# Common language names
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "fa": "Persian",
    "bn": "Bengali",
    "ta": "Tamil",
    "ur": "Urdu",
    "sw": "Swahili",
}


@dataclass
class TranslatorConfig:
    """Configuration for the NLLB translator."""

    enabled: bool = True
    source_lang: Optional[str] = None  # None = auto from transcriber
    target_lang: str = "es"
    model_repo: str = DEFAULT_MODEL_REPO
    cache_dir: Optional[Path] = None
    device: str = "auto"  # cpu, cuda, auto
    compute_type: str = "auto"  # auto, int8, float16
    glossary_path: Optional[str] = None  # Path to custom glossary YAML


@dataclass
class TranslationResult:
    """Result of a translation."""

    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str


class Translator:
    """Text translator using CTranslate2 and NLLB-200."""

    def __init__(self, config: TranslatorConfig):
        """Initialize the translator.

        Args:
            config: Translator configuration.
        """
        self.config = config
        self._model: Optional[ctranslate2.Translator] = None
        self._tokenizer = None
        self._cache_dir = config.cache_dir or DEFAULT_CACHE_DIR
        self._glossary = self._load_glossary()

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self.config.device != "auto":
            return self.config.device

        # Try to use CUDA if available
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _get_compute_type(self, device: str) -> str:
        """Determine the compute type based on device."""
        if self.config.compute_type != "auto":
            return self.config.compute_type

        # Use int8 on CPU, float16 on CUDA
        return "float16" if device == "cuda" else "int8"

    def _download_model(self) -> Path:
        """Download the model from HuggingFace Hub.

        Returns:
            Path to the downloaded model directory.
        """
        from huggingface_hub import snapshot_download

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model
        model_path = snapshot_download(
            repo_id=self.config.model_repo,
            cache_dir=str(self._cache_dir),
            local_dir=str(self._cache_dir / "model"),
        )

        return Path(model_path)

    def _ensure_model(self) -> tuple[ctranslate2.Translator, any]:
        """Ensure the model is loaded (lazy loading)."""
        if self._model is None:
            # Download model if needed
            model_path = self._download_model()

            device = self._get_device()
            compute_type = self._get_compute_type(device)

            # Load CTranslate2 model
            self._model = ctranslate2.Translator(
                str(model_path),
                device=device,
                compute_type=compute_type,
            )

            # Load tokenizer from the original NLLB model
            self._tokenizer = AutoTokenizer.from_pretrained(
                "facebook/nllb-200-distilled-600M",
                cache_dir=str(self._cache_dir / "tokenizer"),
            )

        return self._model, self._tokenizer

    def _to_nllb_code(self, lang_code: str) -> str:
        """Convert a language code to NLLB FLORES-200 format.

        Args:
            lang_code: Simple language code (e.g., 'en', 'es').

        Returns:
            NLLB FLORES-200 code (e.g., 'eng_Latn', 'spa_Latn').
        """
        # Already in NLLB format
        if "_" in lang_code:
            return lang_code

        # Map from simple code
        code = lang_code.lower()
        if code in LANGUAGE_CODE_MAP:
            return LANGUAGE_CODE_MAP[code]

        # Try with region stripped
        if "-" in code:
            base_code = code.split("-")[0]
            if base_code in LANGUAGE_CODE_MAP:
                return LANGUAGE_CODE_MAP[base_code]

        raise ValueError(f"Unknown language code: {lang_code}")

    def _load_glossary(self) -> dict:
        """Load custom glossary from YAML file.

        Returns:
            Dictionary with glossary mappings.
        """
        if not self.config.glossary_path:
            return {"en_to_ja": {}, "ja_to_en": {}}

        try:
            glossary_path = Path(self.config.glossary_path).expanduser()
            if not glossary_path.exists():
                return {"en_to_ja": {}, "ja_to_en": {}}

            with open(glossary_path) as f:
                data = yaml.safe_load(f)

            if not data or "glossary" not in data:
                return {"en_to_ja": {}, "ja_to_en": {}}

            glossary = data["glossary"]
            return {
                "en_to_ja": glossary.get("en_to_ja", {}),
                "ja_to_en": glossary.get("ja_to_en", {}),
            }
        except Exception:
            # If glossary loading fails, continue without it
            return {"en_to_ja": {}, "ja_to_en": {}}

    def _apply_glossary(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Apply glossary replacements to translated text.

        Args:
            text: Text to apply glossary to.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Text with glossary replacements applied.
        """
        if not text or not self._glossary:
            return text

        # Determine which glossary to use
        glossary_map = None
        if source_lang == "en" and target_lang == "ja":
            glossary_map = self._glossary.get("en_to_ja", {})
        elif source_lang == "ja" and target_lang == "en":
            glossary_map = self._glossary.get("ja_to_en", {})

        if not glossary_map:
            return text

        # Sort by length (longest first) to avoid partial replacements
        sorted_terms = sorted(glossary_map.items(), key=lambda x: len(x[0]), reverse=True)

        result = text
        for source_term, target_term in sorted_terms:
            # For English, use case-insensitive matching
            if source_lang == "en":
                # Use word boundaries for English terms
                pattern = re.compile(
                    r'\b' + re.escape(source_term) + r'\b',
                    re.IGNORECASE
                )
                result = pattern.sub(target_term, result)
            else:
                # For Japanese, exact match
                result = result.replace(source_term, target_term)

        return result

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> TranslationResult:
        """Translate text.

        Args:
            text: Text to translate.
            source_lang: Source language (overrides config).
            target_lang: Target language (overrides config).

        Returns:
            TranslationResult with original and translated text.
        """
        if not self.config.enabled:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang or "unknown",
                target_lang=target_lang or "unknown",
            )

        # Handle empty text
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang or "unknown",
                target_lang=target_lang or self.config.target_lang,
            )

        model, tokenizer = self._ensure_model()

        # Resolve language codes
        src_lang = source_lang or self.config.source_lang
        tgt_lang = target_lang or self.config.target_lang

        # Convert to NLLB codes
        src_nllb = self._to_nllb_code(src_lang) if src_lang else "eng_Latn"
        tgt_nllb = self._to_nllb_code(tgt_lang)

        # Skip translation if source and target are the same
        if src_nllb == tgt_nllb:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang or "unknown",
                target_lang=tgt_lang,
            )

        # Tokenize with source language prefix
        tokenizer.src_lang = src_nllb
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Translate
        target_prefix = [[tgt_nllb]]
        results = model.translate_batch(
            [input_tokens],
            target_prefix=target_prefix,
            beam_size=4,
            max_decoding_length=512,
        )

        # Decode output
        output_tokens = results[0].hypotheses[0]
        translated_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )

        # Apply custom glossary
        translated_text = self._apply_glossary(
            translated_text,
            source_lang=src_lang or "unknown",
            target_lang=tgt_lang,
        )

        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_lang=src_lang or "unknown",
            target_lang=tgt_lang,
        )

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def get_supported_languages() -> dict[str, str]:
    """Get dictionary of supported languages.

    Returns:
        Dictionary mapping language codes to names.
    """
    return LANGUAGE_NAMES.copy()


def get_language_code_map() -> dict[str, str]:
    """Get mapping of simple codes to NLLB codes.

    Returns:
        Dictionary mapping ISO 639-1 codes to NLLB FLORES-200 codes.
    """
    return LANGUAGE_CODE_MAP.copy()


def is_language_supported(lang_code: str) -> bool:
    """Check if a language is supported.

    Args:
        lang_code: Language code to check.

    Returns:
        True if language is supported.
    """
    code = lang_code.lower()
    if code in LANGUAGE_CODE_MAP:
        return True
    if "-" in code:
        base_code = code.split("-")[0]
        return base_code in LANGUAGE_CODE_MAP
    # Check if it's already an NLLB code
    return "_" in code
