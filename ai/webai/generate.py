"""Tekst maken: sampling met temperatuur, top-k, top-p en herhalingsstraf.

`stream()` geeft token voor token terug, zodat de server (en dus het zijpaneel
in de browser) het antwoord al kan laten zien terwijl het nog gemaakt wordt.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .config import GenConfig
from .model import TransformerLM

F32 = np.float32


def apply_repetition_penalty(
    logits: np.ndarray, past_ids: Sequence[int], penalty: float, window: int = 256
) -> np.ndarray:
    """Maakt tokens die net gebruikt zijn minder aantrekkelijk (zonder ze te schrappen)."""
    if penalty == 1.0 or not past_ids:
        return logits
    recent = list(past_ids)[-window:]
    uniq = np.unique(np.asarray(recent, dtype=np.int64))
    out = logits.copy()
    vals = out[uniq]
    out[uniq] = np.where(vals > 0, vals / penalty, vals * penalty)
    return out


def top_k_top_p_filter(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    out = logits
    V = out.shape[-1]
    if top_k and 0 < top_k < V:
        kth = np.partition(out, -top_k)[-top_k]
        out = np.where(out < kth, -np.inf, out)
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(out)[::-1]
        sorted_logits = out[order]
        with np.errstate(divide="ignore"):
            probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs = probs / max(probs.sum(), 1e-12)
        cum = np.cumsum(probs)
        cutoff = int(np.searchsorted(cum, top_p) + 1)
        cutoff = max(1, min(cutoff, len(order)))
        remove = order[cutoff:]
        if remove.size:
            out = out.copy()
            out[remove] = -np.inf
    if np.all(np.isneginf(out)):
        return logits
    return out


def sample_next(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    x = logits.astype(np.float64, copy=False)
    if temperature <= 0:
        return int(np.argmax(x))
    x = x / max(1e-6, temperature)
    x = top_k_top_p_filter(x, top_k, top_p)
    x = x - np.max(x)
    p = np.exp(x)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    p = p / total
    return int(rng.choice(len(p), p=p))


class Generator:
    """Houdt de KV-cache bij voor één gesprek en streamt tokens."""

    def __init__(
        self,
        model: TransformerLM,
        prompt_ids: Sequence[int],
        cfg: Optional[GenConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or GenConfig()
        self.block_size = model.cfg.block_size
        ids = [int(i) for i in prompt_ids]
        if not ids:
            raise ValueError("Lege prompt: er is minstens één token nodig om mee te beginnen.")
        if len(ids) > self.block_size:
            # prompt te lang: bewaar het einde (dat is wat er toe doet voor het antwoord)
            ids = ids[-self.block_size :]
        self.state = model.new_state(1)
        self.rng = np.random.default_rng(seed)
        self.generated: List[int] = []
        self._fed: List[int] = []
        # Voer de prompt in brokjes in (snel + spaarzaam geheugen). De logits van
        # het laatste brokje voorspellen het eerste nieuwe token; die bewaren we.
        pending = None
        i = 0
        chunk = 128
        while i < len(ids):
            piece = ids[i : i + chunk]
            logits = model.step(np.asarray([piece], dtype=np.int64), self.state)
            pending = logits[0, -1]
            self._fed.extend(piece)
            i += chunk
        self._pending = pending
        self.prompt_len = len(ids)

    @property
    def context_used(self) -> int:
        return self.state["len"]

    @property
    def context_left(self) -> int:
        return max(0, self.block_size - self.state["len"])

    def next_token(self, extra_stop_ids: Sequence[int] = ()) -> Optional[int]:
        """Kiest het volgende token. `None` = klaar (stop-token of context vol)."""
        if self._pending is not None:
            row = self._pending
            self._pending = None
        elif self.context_left <= 0:
            return None
        else:
            last = np.asarray([[self._fed[-1]]], dtype=np.int64)
            row = self.model.step(last, self.state)[0, -1]
        row = apply_repetition_penalty(row, self._fed, self.cfg.repetition_penalty)
        token = sample_next(row, self.rng, self.cfg.temperature, self.cfg.top_k, self.cfg.top_p)
        self._fed.append(token)
        if token in extra_stop_ids:
            return None
        self.generated.append(token)
        return token

    def stream(
        self,
        extra_stop_ids: Sequence[int] = (),
        on_token: Optional[Callable[[int], None]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[int]:
        limit = max_new_tokens or self.cfg.max_new_tokens
        for _ in range(limit):
            tok = self.next_token(extra_stop_ids)
            if tok is None:
                break
            if on_token:
                on_token(tok)
            yield tok
            if self.context_left <= 0:
                break


def generate_ids(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    cfg: Optional[GenConfig] = None,
    extra_stop_ids: Sequence[int] = (),
    seed: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> List[int]:
    gen = Generator(model, prompt_ids, cfg=cfg, seed=seed)
    return list(gen.stream(extra_stop_ids=extra_stop_ids, max_new_tokens=max_new_tokens))


def generate_text(
    model: TransformerLM,
    tokenizer,
    prompt_text: str,
    cfg: Optional[GenConfig] = None,
    seed: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Gemakkelijke functie: tekst erin, tekst eruit (prompt wordt niet herhaald)."""
    ids = tokenizer.encode(prompt_text)
    out = generate_ids(model, ids, cfg=cfg, seed=seed, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out)


def sampling_params(overrides: Dict[str, Any]) -> GenConfig:
    """Maakt een GenConfig uit een OpenAI-verzoek (temperature, top_p, max_tokens...)."""
    base = GenConfig()
    cfg = GenConfig(
        max_new_tokens=int(overrides.get("max_tokens") or overrides.get("max_completion_tokens") or base.max_new_tokens),
        temperature=float(overrides.get("temperature", base.temperature)),
        top_k=int(overrides.get("top_k") or base.top_k),
        top_p=float(overrides.get("top_p", base.top_p)),
        repetition_penalty=float(overrides.get("repetition_penalty", base.repetition_penalty)),
        stop=tuple(overrides.get("stop") or base.stop),
    )
    cfg.max_new_tokens = max(1, min(cfg.max_new_tokens, 4096))
    cfg.temperature = max(0.0, min(cfg.temperature, 2.0))
    return cfg
