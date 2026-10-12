"""Kleine numpy-bouwdoos: de wiskundige laagjes van het netwerk, mét terugrekening.

Elke functie heeft de vorm:

    out, cache = <laag>_forward(x, params...)
    dx, dparams = <laag>_backward(dout, cache)

Er is geen automatisch differentiatie-framework nodig: de afgeleiden staan er
expliciet bij. Dat houdt de afhankelijkheden op `numpy` en maakt het leesbaar.

Alles rekent in float32 (snel + weinig geheugen); de optimalisator werkt in
float32 maar bewaart zijn momenten apart.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

F32 = np.float32
EPS = 1e-5


# ---------------------------------------------------------------------------
# Linear  (y = x @ W + b)
# ---------------------------------------------------------------------------


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None):
    y = x @ W
    if b is not None:
        y = y + b
    return y, (x, W)


def linear_backward(dy: np.ndarray, cache: Tuple[np.ndarray, np.ndarray]):
    x, W = cache
    # x: [..., in], W: [in, out], dy: [..., out]
    dx = dy @ W.T
    flat_dy = dy.reshape(-1, dy.shape[-1])
    flat_x = x.reshape(-1, x.shape[-1])
    dW = flat_x.T @ flat_dy
    db = flat_dy.sum(axis=0)
    return dx, dW, db


# ---------------------------------------------------------------------------
# LayerNorm (over de laatste as)
# ---------------------------------------------------------------------------


def layernorm_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + EPS)
    rstd = 1.0 / std
    xhat = xc * rstd
    y = xhat * g + b
    return y, (xhat, g, rstd)


def layernorm_backward(dy: np.ndarray, cache):
    xhat, g, rstd = cache
    D = dy.shape[-1]
    dg = (dy * xhat).reshape(-1, D).sum(axis=0)
    db = dy.reshape(-1, D).sum(axis=0)
    dxhat = dy * g
    # d/dx van layernorm, in de standaard compacte vorm
    term = dxhat.mean(axis=-1, keepdims=True) + xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    dx = (dxhat - term) * rstd
    dt = np.result_type(dy.dtype, np.float32)
    return dx, dg.astype(dt, copy=False), db.astype(dt, copy=False)


# ---------------------------------------------------------------------------
# GELU (tanh-benadering, zoals GPT-2) + softmax + cross-entropy
# ---------------------------------------------------------------------------


def gelu_forward(x: np.ndarray):
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    return y, (x, t)


def gelu_backward(dy: np.ndarray, cache):
    x, t = cache
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x**3)
    sech2 = 1.0 - t * t
    dinner = c * (1.0 + 3 * 0.044715 * x**2)
    dx = 0.5 * (1.0 + t) * dy + 0.5 * x * sech2 * dinner * dy
    return dx


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None):
    """Gemiddelde kruisentropie over geldige posities.

    logits  : [..., V]
    targets : [...]  (token-id's; waarden < 0 worden genegeerd)
    mask    : [...]  (1.0 = meetellen, 0.0 = padding)
    """
    shape = logits.shape[:-1]
    V = logits.shape[-1]
    # dtype van de invoer aanhouden: float32 tijdens trainen, float64 bij een
    # gradient-check (dan is de numerieke benadering precies genoeg).
    dt = np.result_type(logits.dtype, np.float32)
    flat = logits.reshape(-1, V).astype(dt, copy=False)
    tgt = targets.reshape(-1).astype(np.int64, copy=False)
    if mask is None:
        keep = tgt >= 0
    else:
        keep = (tgt >= 0) & (mask.reshape(-1) > 0)
    n = max(int(keep.sum()), 1)

    m = flat.max(axis=1, keepdims=True)
    z = flat - m
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True)) + m
    logprob = flat - logsumexp  # log softmax

    idx = np.clip(tgt, 0, V - 1)
    nll = -logprob[np.arange(flat.shape[0]), idx]
    loss = float((nll * keep).sum() / n)

    probs = np.exp(logprob)
    dlogits = probs
    rows = np.arange(flat.shape[0])
    dlogits[rows, idx] -= 1.0
    dlogits *= keep[:, None]
    dlogits /= n
    return loss, dlogits.reshape(shape + (V,)).astype(dt, copy=False), n


# ---------------------------------------------------------------------------
# Initialisatie
# ---------------------------------------------------------------------------


def normal(rng: np.random.Generator, shape: Tuple[int, ...], std: float) -> np.ndarray:
    return (rng.standard_normal(shape) * std).astype(F32, copy=False)


def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=F32)


def ones(shape: Tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=F32)


# ---------------------------------------------------------------------------
# AdamW + leerplanschema + gradiënt-afkapping
# ---------------------------------------------------------------------------


class AdamW:
    """Adam met ontkoppelde weight decay (AdamW), per parameter-groep.

    `no_decay` bevat namen die géén weight decay krijgen (LayerNorm + biases +
    embeddings). Dat is de standaardaanpak bij transformers.
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.05,
    ) -> None:
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self._no_decay = {
            k for k in params if k.endswith("_b") or k.endswith("_g") or k in ("wte", "wpe")
        }

    def step(self, grads: Dict[str, np.ndarray], lr: float | None = None) -> float:
        lr = self.lr if lr is None else lr
        self.t += 1
        b1, b2 = self.b1, self.b2
        bc1 = 1.0 - b1**self.t
        bc2 = 1.0 - b2**self.t
        total_norm = 0.0
        for name, p in self.params.items():
            g = grads.get(name)
            if g is None:
                continue
            g = g.astype(F32, copy=False)
            total_norm += float((g.astype(np.float64) ** 2).sum())
            m = self.m[name]
            v = self.v[name]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            mhat = m / bc1
            vhat = v / bc2
            update = mhat / (np.sqrt(vhat) + self.eps)
            if name not in self._no_decay and self.weight_decay:
                p -= lr * self.weight_decay * p
            p -= lr * update
        return math.sqrt(total_norm)


def clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float) -> float:
    """Schaalt alle gradiënten zodat de totale norm <= max_norm. Geeft de norm vóór afkappen."""
    if max_norm <= 0:
        return 0.0
    total = 0.0
    for g in grads.values():
        total += float((g.astype(np.float64) ** 2).sum())
    norm = math.sqrt(total)
    if norm > max_norm and norm > 0:
        scale = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] = (grads[k] * scale).astype(F32, copy=False)
    return norm


def lr_at(step: int, cfg) -> float:
    """Leerplanning: warmup (lineair) daarna afbouw naar min_lr."""
    base = cfg.learning_rate
    if step < cfg.warmup_steps:
        return base * (step + 1) / max(1, cfg.warmup_steps)
    if cfg.decay_style == "none":
        return base
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if cfg.decay_style == "linear":
        return base + (cfg.min_lr - base) * progress
    # cosine
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (base - cfg.min_lr) * coeff
