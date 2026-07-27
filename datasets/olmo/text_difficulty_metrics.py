"""Document-level difficulty metrics for curriculum labeling.

Definitions (stored verbatim so offline training can sort either direction):

- compression_ratio: utf8_bytes / zlib_bytes (level 6). Higher => more
  compressible / redundant (Yin et al. 2024 style information density).
- flesch_reading_ease: classic Kincaid (1975) score. Higher => easier English.
- mtld: Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010),
  bidirectional average with TTR factor threshold 0.72. Higher => more diverse.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import asdict, dataclass

# Letters with optional ASCII digits / internal apostrophes. Digits matter for
# code and math documents in OLMo-mix; pure punctuation is ignored.
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[A-Za-z]*\d+[A-Za-z0-9]*")
_SENTENCE_RE = re.compile(r"[.!?]+")
_MTLD_THRESHOLD = 0.72


@dataclass(frozen=True)
class DifficultyMetrics:
    compression_ratio: float
    flesch_reading_ease: float
    mtld: float
    raw_bytes: int
    zlib_bytes: int
    n_chars: int
    n_words: int
    n_sentences: int
    n_syllables: int

    def as_dict(self) -> dict:
        return asdict(self)


def compression_ratio(text: str) -> tuple[float, int, int]:
    raw = text.encode("utf-8", errors="surrogatepass")
    raw_bytes = len(raw)
    if raw_bytes == 0:
        return float("nan"), 0, 0
    compressed = zlib.compress(raw, level=6)
    zlib_bytes = len(compressed)
    if zlib_bytes == 0:
        return float("nan"), raw_bytes, 0
    return raw_bytes / zlib_bytes, raw_bytes, zlib_bytes


def _count_syllables(word: str) -> int:
    w = word.lower()
    if len(w) <= 3:
        return 1
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> tuple[float, int, int, int]:
    words = _WORD_RE.findall(text)
    n_words = len(words)
    if n_words == 0:
        return float("nan"), 0, 0, 0
    # At least one sentence even if terminal punctuation is missing.
    n_sentences = max(1, len(_SENTENCE_RE.findall(text)))
    n_syllables = sum(_count_syllables(w) for w in words)
    score = (
        206.835
        - 1.015 * (n_words / n_sentences)
        - 84.6 * (n_syllables / n_words)
    )
    return score, n_words, n_sentences, n_syllables


def _mtld_forward(tokens: list[str], threshold: float = _MTLD_THRESHOLD) -> float:
    if not tokens:
        return float("nan")
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for token in tokens:
        token_count += 1
        types.add(token)
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            types = set()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        # Partial factor: how far TTR has moved from 1.0 toward threshold.
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)
        else:
            factors += 0.0
    if factors == 0.0:
        return float(len(tokens))
    return len(tokens) / factors


def mtld(text: str, threshold: float = _MTLD_THRESHOLD) -> float:
    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    if len(tokens) < 10:
        # Too short for a stable MTLD estimate; still return a finite value.
        return float(len(set(tokens))) if tokens else float("nan")
    forward = _mtld_forward(tokens, threshold)
    backward = _mtld_forward(list(reversed(tokens)), threshold)
    if math.isnan(forward) or math.isnan(backward):
        return float("nan")
    return 0.5 * (forward + backward)


def compute_difficulty_metrics(text: str) -> DifficultyMetrics:
    ratio, raw_bytes, zlib_bytes = compression_ratio(text)
    flesch, n_words, n_sentences, n_syllables = flesch_reading_ease(text)
    return DifficultyMetrics(
        compression_ratio=ratio,
        flesch_reading_ease=flesch,
        mtld=mtld(text),
        raw_bytes=raw_bytes,
        zlib_bytes=zlib_bytes,
        n_chars=len(text),
        n_words=n_words,
        n_sentences=n_sentences,
        n_syllables=n_syllables,
    )
