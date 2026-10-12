"""Eigen byte-level BPE-tokenizer.

Waarom byte-level: elk denkbaar teken (Nederlands, Frans, Duits, emoji, wiskundige
symbolen) is op te bouwen uit bytes, dus er bestaat nooit een "onbekend token".

De woordenschat wordt getraind op jóuw data (`ai/data`), niet gedownload. Het
resultaat is een klein JSON-bestand dat naast het model bewaard wordt.

Speciale tokens (altijd dezelfde id's, zodat ze nooit uit elkaar gesplitst worden):

    0  <|pad|>        opvulling
    1 <|fim_suffix|>       einde van het gesprek
    2 <|fim_prefix|>        opening van een rol
    3 <|fim_middle|>        einde van een beurt
    4 <|fim_suffix|>        hulpmiddel-antwoord
    5  <|tool_call|>   het model wil een hulpmiddel gebruiken
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

PAD = "<|pad|>"
BOS = "<|fim_suffix|>"
TURN = "<|fim_prefix|>"
EOT = "<|fim_middle|>"
TOOL = "<|fim_suffix|>"
TOOL_CALL = "<|tool_call|>"

SPECIAL_TOKENS: Tuple[str, ...] = (PAD, BOS, TURN, EOT, TOOL, TOOL_CALL)
SPECIAL_IDS: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
N_SPECIAL = len(SPECIAL_TOKENS)

# GPT-2-achtig splitspatroon: woorden, getallen, losse leestekens, en
# "woorden die bij elkaar horen" (zoals in "don't") blijven heel.
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D"""
    r"""|[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]+"""
    r"""|[0-9](?:[0-9.,]*[0-9])?"""
    r"""|\s+"""
    r"""|[^\sA-Za-zÀ-ÖØ-öø-ÿĀ-ž0-9]""",
    re.UNICODE,
)

# De speciale tokens worden vóór het splitspatroon gevonden, met voorrang.
SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


def pre_tokenize(text: str) -> List[str]:
    """Knipt tekst in stukjes waarbinnen BPE mag mergen."""
    return PAT.findall(text)


def _pairs(seq: Sequence[int]) -> Counter:
    c: Counter = Counter()
    for i in range(len(seq) - 1):
        c[(seq[i], seq[i + 1])] += 1
    return c


def _merge(seq: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class ByteLevelBPE:
    """BPE over bytes. `vocab_size` is het totaal inclusief speciale tokens."""

    def __init__(self, merges: List[Tuple[int, int]], vocab_size: int) -> None:
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: Dict[Tuple[int, int], int] = {
            (a, b): i for i, (a, b) in enumerate(self.merges)
        }
        self.vocab_size = int(vocab_size)
        self.pad_id = SPECIAL_IDS[PAD]
        self.bos_id = SPECIAL_IDS[BOS]
        self.eot_id = SPECIAL_IDS[EOT]
        # id -> bytes; de eerste N_SPECIAL zijn de speciale tokens
        self.id_to_bytes: Dict[int, bytes] = {i: t.encode("utf-8") for i, t in enumerate(SPECIAL_TOKENS)}
        for i in range(256):
            self.id_to_bytes[N_SPECIAL + i] = bytes([i])
        nxt = N_SPECIAL + 256
        for a, b in self.merges:
            self.id_to_bytes[nxt] = self.id_to_bytes[a] + self.id_to_bytes[b]
            nxt += 1

    # ------------------------------------------------------------------
    # Trainen
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, min_freq: int = 2, verbose: bool = False) -> "ByteLevelBPE":
        if vocab_size <= N_SPECIAL + 256:
            raise ValueError(
                f"vocab_size moet groter zijn dan {N_SPECIAL + 256} "
                f"(speciale tokens + 256 bytes); kreeg {vocab_size}"
            )
        # Elk uniek stukje maar één keer tellen, met zijn frequentie als gewicht.
        counts: Counter = Counter()
        for piece in pre_tokenize(text):
            counts[piece] += 1
        # woorden: tuple van byte-ids, plus hoe vaak het voorkomt
        words: Dict[Tuple[int, ...], int] = {}
        for piece, freq in counts.items():
            key = tuple(b for b in piece.encode("utf-8"))
            words[key] = words.get(key, 0) + freq

        pair_counts: Counter = Counter()
        for word, freq in words.items():
            for pair, n in _pairs(word).items():
                pair_counts[pair] += n * freq

        merges: List[Tuple[int, int]] = []
        target_merges = vocab_size - N_SPECIAL - 256
        while len(merges) < target_merges:
            if not pair_counts:
                break
            # hoogste frequentie wint; bij gelijkspel alfabetisch (deterministisch)
            best = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if pair_counts[best] < min_freq:
                break
            new_id = N_SPECIAL + 256 + len(merges)
            merges.append(best)
            next_counts: Counter = Counter()
            new_words: Dict[Tuple[int, ...], int] = {}
            for word, freq in words.items():
                if best[0] in word and best[1] in word:
                    word = tuple(_merge(word, best, new_id))
                new_words[word] = freq
            words = new_words
            for word, freq in words.items():
                for pair, n in _pairs(word).items():
                    next_counts[pair] += n * freq
            pair_counts = next_counts
            if verbose and (len(merges) % 200 == 0 or len(merges) == target_merges):
                print(f"  [bpe] {len(merges)}/{target_merges} merges, "
                      f"{len(pair_counts)} paren over")
        vocab = N_SPECIAL + 256 + len(merges)
        if verbose:
            print(f"  [bpe] klaar: woordenschat van {vocab} tokens")
        return cls(merges, vocab)

    # ------------------------------------------------------------------
    # Coderen / decoderen
    # ------------------------------------------------------------------

    def encode_ordinary(self, text: str) -> List[int]:
        """Codeert tekst; speciale tokens worden als gewone tekst behandeld."""
        out: List[int] = []
        for piece in pre_tokenize(text):
            ids = [N_SPECIAL + b for b in piece.encode("utf-8")]
            while len(ids) > 1:
                candidates = [(self.ranks[(ids[i], ids[i + 1])], i)
                              for i in range(len(ids) - 1) if (ids[i], ids[i + 1]) in self.ranks]
                if not candidates:
                    break
                rank, i = min(candidates)
                ids = _merge(ids, (ids[i], ids[i + 1]), N_SPECIAL + 256 + rank)
            out.extend(ids)
        return out

    def encode(self, text: str, allowed_special: Iterable[str] = SPECIAL_TOKENS) -> List[int]:
        """Codeert tekst en laat de opgegeven speciale tokens heel."""
        allowed = set(allowed_special)
        out: List[int] = []
        for chunk in SPECIAL_RE.split(text):
            if not chunk:
                continue
            if chunk in allowed:
                out.append(SPECIAL_IDS[chunk])
            else:
                out.extend(self.encode_ordinary(chunk))
        return out

    def decode(self, ids: Sequence[int], drop_special: bool = True) -> str:
        skip = set(SPECIAL_IDS.values()) if drop_special else set()
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in skip:
                continue
            b = self.id_to_bytes.get(i)
            if b is None:
                continue
            buf.extend(b)
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Opslaan / laden
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "version": 1,
            "kind": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "merges": [list(m) for m in self.merges],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPE":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls([tuple(m) for m in data["merges"]], int(data["vocab_size"]))

    def stats(self, text: str) -> Dict[str, float]:
        """Handig om te zien hoe efficiënt de tokenizer is (tekens per token)."""
        ids = self.encode_ordinary(text)
        n = max(1, len(ids))
        return {
            "tokens": len(ids),
            "chars": len(text),
            "chars_per_token": round(len(text) / n, 3),
            # hoeveel bits het model nodig heeft per teken (lager = beter)
            "bits_per_char": round(n * math.log2(max(2, self.vocab_size)) / max(1, len(text)), 3),
        }
