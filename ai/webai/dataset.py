"""Dataset: van ruwe bestanden in `ai/data` naar trainingsreeksen.

Ondersteunde invoer:
  * `.txt` / `.md`      – platte tekst (taalmodel-training: elk token telt mee)
  * `.jsonl` / `.json`  – gesprekken, in één van deze vormen per regel:
        {"messages": [{"role": "user", "content": "..."}, ...]}
        {"prompt": "...", "completion": "..."}
        {"instruction": "...", "input": "...", "output": "..."}   (Alpaca-stijl)
        {"text": "..."}
        ["...", "..."]                                            (lijst gesprekken)

Bij gesprekken telt alleen het antwoord van de assistant mee voor het verlies:
het model leert zo *antwoorden*, niet de vragen uit zijn hoofd.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .chat import EOT, TURN, norm_role, render_prompt, render_turn
from .config import LEARNED_DIR, SEED_DIR
from .tokenizer import ByteLevelBPE

RAW_EXT = (".txt", ".md", ".markdown", ".rst")
JSON_EXT = (".jsonl", ".json", ".ndjson")


@dataclass
class Sample:
    ids: np.ndarray       # [n] token-id's
    mask: np.ndarray      # [n] 1.0 = telt mee voor het verlies
    source: str = ""
    kind: str = "chat"    # "chat" | "text"


# ---------------------------------------------------------------------------
# Bestanden lezen
# ---------------------------------------------------------------------------


def iter_data_files(extra_dirs: Sequence[str] = ()) -> List[str]:
    """Alle databestanden, in een vaste volgorde (seed eerst, dan eigen data)."""
    dirs = [d for d in (SEED_DIR, LEARNED_DIR, *extra_dirs) if d]
    files: List[str] = []
    seen = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in RAW_EXT + JSON_EXT:
                continue
            real = os.path.abspath(path)
            if real in seen:
                continue
            seen.add(real)
            files.append(path)
    return files


def _load_json_any(path: str) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if isinstance(data, dict):
            return [data]
        return data if isinstance(data, list) else []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # Een kapotte regel in .jsonl stil als tekst trainen zou de fout
            # verbergen; sla hem over zodat hij in de statistiek zichtbaar blijft.
            out.append({"_invalid": True})
    return out


def record_to_messages(rec: Any) -> Optional[List[Dict[str, Any]]]:
    """Zet één JSON-record om naar een lijst messages (of None als het tekst is)."""
    if isinstance(rec, str):
        return [{"role": "user", "content": rec}, {"role": "assistant", "content": rec}]
    if not isinstance(rec, dict):
        if isinstance(rec, list) and rec and isinstance(rec[0], dict) and "role" in rec[0]:
            return rec
        return None
    if "messages" in rec and isinstance(rec["messages"], list):
        return [m for m in rec["messages"] if isinstance(m, dict) and m.get("role")]
    if "conversations" in rec and isinstance(rec["conversations"], list):
        msgs = []
        for m in rec["conversations"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from") or m.get("sender") or "user"
            content = m.get("content") or m.get("value") or m.get("text") or ""
            msgs.append({"role": str(role), "content": content})
        return msgs or None
    if "prompt" in rec and "completion" in rec:
        return [
            {"role": "user", "content": str(rec["prompt"])},
            {"role": "assistant", "content": str(rec["completion"])},
        ]
    if "instruction" in rec or "output" in rec:
        instruction = str(rec.get("instruction", "")).strip()
        extra = str(rec.get("input", "") or "").strip()
        user = instruction if not extra else f"{instruction}\n\n{extra}"
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(rec.get("output", ""))},
        ]
    if "question" in rec or "answer" in rec:
        return [
            {"role": "user", "content": str(rec.get("question", ""))},
            {"role": "assistant", "content": str(rec.get("answer", ""))},
        ]
    return None


# ---------------------------------------------------------------------------
# Rijken bouwen
# ---------------------------------------------------------------------------


def build_chat_sample(
    tok: ByteLevelBPE,
    messages: List[Dict[str, Any]],
    block_size: int,
    source: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Sample]:
    """Een gesprek omzetten naar tokens; verlies alleen op assistant-beurten."""
    msgs = [{"role": norm_role(m.get("role")), "content": m.get("content")} for m in messages]
    msgs = [m for m in msgs if m["content"] is not None]
    if not any(m["role"] == "assistant" for m in msgs):
        return None

    system_text = "\n\n".join(
        str(m["content"]) for m in msgs if m["role"] == "system" if str(m["content"]).strip()
    )
    body = [m for m in msgs if m["role"] != "system"]

    ids: List[int] = [tok.bos_id]
    mask: List[float] = [0.0]

    def add(text: str, learn: bool) -> None:
        part = tok.encode(text)
        ids.extend(part)
        mask.extend([1.0 if learn else 0.0] * len(part))

    if system_text:
        add(render_turn("system", system_text), False)

    for m in body:
        if m["role"] == "assistant":
            add(f"{TURN}assistant\n", False)
            # het antwoord zélf leren we, inclusief het beurt-einde
            add(str(m["content"]) + EOT + "\n", True)
        elif m["role"] == "tool":
            add(f"{TURN}tool\n{m['content']}{EOT}\n", False)
        else:
            add(f"{TURN}{m['role']}\n{m['content']}{EOT}\n", False)

    if len(ids) > block_size:
        # te lang: bewaar het begin (systeem + eerste vraag) en het einde (laatste antwoord)
        head = max(1, block_size // 4)
        tail = block_size - head
        ids = ids[:head] + ids[-tail:]
        mask = mask[:head] + mask[-tail:]
    if not any(mask):
        return None
    return Sample(
        ids=np.asarray(ids, dtype=np.int64),
        mask=np.asarray(mask, dtype=np.float32),
        source=source,
        kind="chat",
    )


def build_text_samples(
    tok: ByteLevelBPE, text: str, block_size: int, source: str = ""
) -> List[Sample]:
    """Platte tekst in blokken van `block_size` hakken (klassieke taalmodel-training)."""
    text = text.strip()
    if not text:
        return []
    ids = tok.encode(text)
    if len(ids) < 2:
        return []
    out: List[Sample] = []
    step = block_size
    for start in range(0, len(ids), step):
        chunk = ids[start : start + block_size + 1]
        if len(chunk) < 8:
            break
        arr = np.asarray(chunk, dtype=np.int64)
        out.append(Sample(
            ids=arr[:-1],
            mask=np.ones(len(arr) - 1, dtype=np.float32),
            source=source,
            kind="text",
        ))
    return out


def build_dataset(
    tok: ByteLevelBPE,
    block_size: int,
    files: Optional[Sequence[str]] = None,
    repeat_small: int = 1,
    verbose: bool = False,
) -> Tuple[List[Sample], Dict[str, Any]]:
    """Leest alle databestanden en maakt de lijst trainingssamples."""
    files = list(files) if files else iter_data_files()
    samples: List[Sample] = []
    stats = {"files": 0, "chat": 0, "text": 0, "skipped": 0, "chars": 0}
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        rel = os.path.relpath(path)
        stats["files"] += 1
        if ext in RAW_EXT:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            stats["chars"] += len(text)
            made = build_text_samples(tok, text, block_size, rel)
            samples.extend(made)
            stats["text"] += len(made)
        else:
            n_before = len(samples)
            for rec in _load_json_any(path):
                msgs = record_to_messages(rec)
                if msgs is None:
                    if isinstance(rec, dict) and rec.get("text"):
                        made = build_text_samples(tok, str(rec["text"]), block_size, rel)
                        samples.extend(made)
                        stats["text"] += len(made)
                        stats["chars"] += len(str(rec["text"]))
                    else:
                        stats["skipped"] += 1
                    continue
                sample = build_chat_sample(tok, msgs, block_size, rel)
                if sample is None:
                    stats["skipped"] += 1
                    continue
                samples.append(sample)
            stats["chat"] += len(samples) - n_before
    if not samples:
        return [], stats
    # Kleine datasets vaker herhalen, zodat een training van N stappen zinvol blijft.
    if repeat_small > 1 and len(samples) < 2000:
        samples = samples * repeat_small
    if verbose:
        total = sum(len(s.ids) for s in samples)
        print(f"  [data] {stats['files']} bestanden -> {len(samples)} samples "
              f"({stats['chat']} gesprek, {stats['text']} tekst), {total} tokens")
    return samples, stats


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


class Batcher:
    """Maakt batches van samples: kort bij elkaar houden scheelt rekenwerk (padding)."""

    def __init__(
        self,
        samples: List[Sample],
        batch_size: int,
        block_size: int,
        seed: int = 0,
        sort_by_length: bool = True,
    ) -> None:
        self.samples = samples
        self.batch_size = max(1, batch_size)
        self.block_size = block_size
        self.rng = random.Random(seed)
        self.sort_by_length = sort_by_length
        self.epoch = 0
        self._order: List[int] = []
        self._pos = 0
        self.reshuffle()

    def __len__(self) -> int:
        return max(1, (len(self.samples) + self.batch_size - 1) // self.batch_size)

    def reshuffle(self) -> None:
        idx = list(range(len(self.samples)))
        self.rng.shuffle(idx)
        if self.sort_by_length:
            # groepen van ~50 batches: binnen een groep op lengte sorteren
            chunk = self.batch_size * 50
            grouped: List[int] = []
            for i in range(0, len(idx), chunk):
                part = idx[i : i + chunk]
                part.sort(key=lambda j: len(self.samples[j].ids))
                grouped.extend(part)
            idx = grouped
        self._order = idx
        self._pos = 0
        self.epoch += 1

    def batches(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        while self._pos < len(self._order):
            take = self._order[self._pos : self._pos + self.batch_size]
            self._pos += self.batch_size
            if not take:
                break
            yield self.collate(take)
        self.reshuffle()

    def collate(self, indices: Sequence[int]):
        items = [self.samples[i] for i in indices]
        # Eén positie korter dan de langste sample: het laatste token is alleen
        # maar doel en heeft zelf geen invoer. Anders blijft er een kolom padding.
        T = min(self.block_size, max(len(s.ids) for s in items) - 1)
        T = max(1, T)
        B = len(items)
        ids = np.zeros((B, T), dtype=np.int64)
        targets = np.full((B, T), -100, dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.float32)
        for b, s in enumerate(items):
            n = min(T, len(s.ids) - 1)
            if n <= 0:
                continue
            ids[b, :n] = s.ids[:n]
            targets[b, :n] = s.ids[1 : n + 1]
            mask[b, :n] = s.mask[1 : n + 1]
        return ids, targets, mask

    def eval_batches(self, count: int = 8):
        """Vaste evaluatie-batches (zelfde volgorde elke keer, dus vergelijkbaar)."""
        rng = random.Random(9999)
        idx = list(range(len(self.samples)))
        rng.shuffle(idx)
        out = []
        for i in range(min(count, max(1, len(idx) // self.batch_size))):
            take = idx[i * self.batch_size : (i + 1) * self.batch_size]
            if not take:
                break
            out.append(self.collate(take))
        return out


def split_samples(
    samples: List[Sample], eval_fraction: float = 0.05, seed: int = 0
) -> Tuple[List[Sample], List[Sample]]:
    """Splits in train/eval.

    Bij een kleine dataset is 5% te weinig om iets zinnigs te meten (2 voorbeelden
    geven een eval-verlies dat alleen maar ruis is). Daarom: hoe minder data, hoe
    groter het evaluatiedeel, met een minimum van 4 en een maximum van een kwart.
    """
    n = len(samples)
    if n < 6 or eval_fraction <= 0:
        return samples, []
    frac = eval_fraction
    if n < 400:
        frac = max(eval_fraction, 0.10)
    n_eval = int(round(n * frac))
    n_eval = max(1, min(n_eval, max(4, n // 4)))
    if n - n_eval < 1:
        n_eval = max(0, n - 1)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    eval_idx = set(idx[:n_eval])
    train = [s for i, s in enumerate(samples) if i not in eval_idx]
    evl = [s for i, s in enumerate(samples) if i in eval_idx]
    return train, evl


def dataset_tokens(samples: Sequence[Sample]) -> int:
    return int(sum(len(s.ids) for s in samples))


def dataset_chars(files: Optional[Sequence[str]] = None) -> int:
    total = 0
    for path in files or iter_data_files():
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total
