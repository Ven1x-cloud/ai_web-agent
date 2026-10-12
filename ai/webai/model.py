"""Het model: een GPT-achtige decoder-transformer, volledig in numpy.

Wat erin zit:
  * `TransformerLM.forward()`  – hele reeks in één keer, mét verlies én alle
                                 gradiënten (voor trainen)
  * `TransformerLM.step()`     – één nieuw token, met KV-cache (voor生成/snel
                                 antwoord geven zonder alles opnieuw te rekenen)
  * opslaan/laden van checkpoints (`.npz` + `config.json`)

De terugrekening is met de hand geschreven (zie `nn.py`); daardoor is er geen
zwaar framework nodig en draait trainen op een gewone laptop.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import nn
from .config import ModelConfig, load_json, save_json

F32 = np.float32


class TransformerLM:
    def __init__(self, cfg: ModelConfig, params: Optional[Dict[str, np.ndarray]] = None) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.params: Dict[str, np.ndarray] = params if params is not None else self._init_params(rng)
        # Rekenprecisie volgt de parameters: float32 normaal, float64 voor een
        # gradient-check.
        self.dtype = self.params["wte"].dtype
        self._causal_masks: Dict[Any, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _init_params(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        c = self.cfg
        D, V, T = c.n_embd, c.vocab_size, c.block_size
        std = c.init_std
        p: Dict[str, np.ndarray] = {}
        p["wte"] = nn.normal(rng, (V, D), std)
        p["wpe"] = nn.normal(rng, (T, D), std * 0.5)
        for i in range(c.n_layer):
            p[f"h{i}.ln1_g"] = nn.ones((D,))
            p[f"h{i}.ln1_b"] = nn.zeros((D,))
            p[f"h{i}.attn.qkv_w"] = nn.normal(rng, (D, 3 * D), std)
            p[f"h{i}.attn.qkv_b"] = nn.zeros((3 * D,))
            p[f"h{i}.attn.proj_w"] = nn.normal(rng, (D, D), std)
            p[f"h{i}.attn.proj_b"] = nn.zeros((D,))
            p[f"h{i}.ln2_g"] = nn.ones((D,))
            p[f"h{i}.ln2_b"] = nn.zeros((D,))
            p[f"h{i}.mlp.fc_w"] = nn.normal(rng, (D, c.mlp_dim), std)
            p[f"h{i}.mlp.fc_b"] = nn.zeros((c.mlp_dim,))
            p[f"h{i}.mlp.proj_w"] = nn.normal(rng, (c.mlp_dim, D), std)
            p[f"h{i}.mlp.proj_b"] = nn.zeros((D,))
            # GPT-2 truc: de restakkel-uitgangen kleiner initialiseren, zodat een
            # dieper netwerk stabiel begint.
            scale = 1.0 / math.sqrt(2 * c.n_layer)
            p[f"h{i}.attn.proj_w"] *= F32(scale)
            p[f"h{i}.mlp.proj_w"] *= F32(scale)
        p["lnf_g"] = nn.ones((D,))
        p["lnf_b"] = nn.zeros((D,))
        if not c.tied_embeddings:
            p["lm_w"] = nn.normal(rng, (D, V), std)
        p["lm_b"] = nn.zeros((V,))
        return p

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(int(v.size) for v in self.params.values())
        if non_embedding:
            n -= int(self.params["wte"].size) + int(self.params["wpe"].size)
        return n

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {k: tuple(v.shape) for k, v in self.params.items()}

    # ------------------------------------------------------------------
    # Hulpmiddelen
    # ------------------------------------------------------------------

    def out_weight(self) -> np.ndarray:
        """Gewicht van de uitvoerlaag: [D, V]. Bij 'tied' is dat wte.T."""
        if self.cfg.tied_embeddings:
            return self.params["wte"].T
        return self.params["lm_w"]

    def _causal_mask(self, T: int) -> np.ndarray:
        m = self._causal_masks.get(T)
        if m is None:
            m = np.triu(np.full((T, T), -1e9, dtype=F32), k=1)
            self._causal_masks[T] = m
        return m

    def _chunk_mask(self, n: int, end: int) -> np.ndarray:
        """Causaal masker voor een brokje van `n` nieuwe tokens op een cache van `end`.

        Rij j (globale positie end-n+j) mag alleen kolommen <= die positie zien.
        """
        key = ("chunk", n, end)
        m = self._causal_masks.get(key)
        if m is None:
            start = end - n
            qpos = np.arange(start, end, dtype=np.int64)[:, None]
            kpos = np.arange(end, dtype=np.int64)[None, :]
            m = np.where(kpos <= qpos, 0.0, -1e9).astype(F32, copy=False)
            self._causal_masks[key] = m
        return m

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """[B, T, D] -> [B, H, T, dh]"""
        B, T, _ = x.shape
        H, dh = self.cfg.n_head, self.cfg.head_dim
        return x.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """[B, H, T, dh] -> [B, T, D]"""
        B, H, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)

    # ------------------------------------------------------------------
    # Trainings-doorloop (hele reeks, met gradiënten)
    # ------------------------------------------------------------------

    def forward(
        self,
        ids: np.ndarray,
        targets: Optional[np.ndarray] = None,
        loss_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
        """Rekent het verlies en (optioneel) de gradiënten uit.

        ids      : [B, T] token-id's
        targets  : [B, T] id's die voorspeld moeten worden (-100 = negeren)
        loss_mask: [B, T] 1.0 = telt mee voor het verlies

        Geeft: (loss, grads, logits, cache). `grads` is leeg als targets None is.
        """
        c = self.cfg
        p = self.params
        ids = np.asarray(ids, dtype=np.int64)
        B, T = ids.shape
        if T > c.block_size:
            raise ValueError(f"Reeks van {T} tokens is langer dan block_size={c.block_size}")

        cache: Dict[str, Any] = {"ids": ids, "B": B, "T": T}

        pos = np.arange(T, dtype=np.int64)
        emb = p["wte"][ids]  # [B,T,D]
        x = (emb + p["wpe"][pos][None, :, :]).astype(self.dtype, copy=False)
        cache["emb"] = emb

        scale = 1.0 / math.sqrt(c.head_dim)
        mask = self._causal_mask(T)

        for i in range(c.n_layer):
            pre = f"h{i}."
            x0 = x

            h, ln1 = nn.layernorm_forward(x, p[pre + "ln1_g"], p[pre + "ln1_b"])
            qkv, qkv_c = nn.linear_forward(h, p[pre + "attn.qkv_w"], p[pre + "attn.qkv_b"])
            D = c.n_embd
            q = self._split_heads(qkv[..., 0:D])
            k = self._split_heads(qkv[..., D : 2 * D])
            v = self._split_heads(qkv[..., 2 * D :])

            att = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B,H,T,T]
            att = att + mask[None, None]
            probs = nn.softmax(att, axis=-1)
            ctx = probs @ v  # [B,H,T,dh]
            ctx = self._merge_heads(ctx)

            a, proj_c = nn.linear_forward(ctx, p[pre + "attn.proj_w"], p[pre + "attn.proj_b"])
            x1 = x0 + a  # eerste restakkel

            h2, ln2 = nn.layernorm_forward(x1, p[pre + "ln2_g"], p[pre + "ln2_b"])
            f, fc_c = nn.linear_forward(h2, p[pre + "mlp.fc_w"], p[pre + "mlp.fc_b"])
            g, gelu_c = nn.gelu_forward(f)
            m, mp_c = nn.linear_forward(g, p[pre + "mlp.proj_w"], p[pre + "mlp.proj_b"])
            x = x1 + m  # tweede restakkel

            cache[i] = dict(
                ln1=ln1, qkv_c=qkv_c, q=q, k=k, v=v, probs=probs, proj_c=proj_c,
                ln2=ln2, fc_c=fc_c, gelu_c=gelu_c, mp_c=mp_c, scale=scale,
            )

        xf, lnf = nn.layernorm_forward(x, p["lnf_g"], p["lnf_b"])
        W_out = self.out_weight()
        logits, out_c = nn.linear_forward(xf, W_out, p["lm_b"])
        cache["lnf"] = lnf
        cache["out_c"] = out_c
        cache["xf"] = xf

        if targets is None:
            return None, {}, logits, cache

        loss, dlogits, ntok = nn.cross_entropy(logits, targets, loss_mask)
        cache["ntok"] = ntok

        grads = self._backward(dlogits, cache)
        return loss, grads, logits, cache

    # ------------------------------------------------------------------
    # Terugrekening
    # ------------------------------------------------------------------

    def _backward(self, dlogits: np.ndarray, cache: Dict[str, Any]) -> Dict[str, np.ndarray]:
        c = self.cfg
        p = self.params
        dt = self.dtype
        ids = cache["ids"]
        B, T = cache["B"], cache["T"]
        g: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}

        # uitvoerlaag
        dxf, dW_out, dlm_b = nn.linear_backward(dlogits, cache["out_c"])
        g["lm_b"] = dlm_b.astype(dt, copy=False)
        if c.tied_embeddings:
            g["wte"] += dW_out.T.astype(dt, copy=False)
        else:
            g["lm_w"] = dW_out.astype(dt, copy=False)

        dx, dlnf_g, dlnf_b = nn.layernorm_backward(dxf, cache["lnf"])
        g["lnf_g"] = dlnf_g
        g["lnf_b"] = dlnf_b

        for i in reversed(range(c.n_layer)):
            pre = f"h{i}."
            blk = cache[i]

            # Het blok rekent:  x1 = x0 + a   en   x = x1 + m
            # Dus: dL/dm = dx, en dL/dx1 = dx + (bijdrage via ln2 -> MLP).

            # --- MLP-tak ---
            dg, dmp_w, dmp_b = nn.linear_backward(dx, blk["mp_c"])
            g[pre + "mlp.proj_w"] = dmp_w.astype(dt, copy=False)
            g[pre + "mlp.proj_b"] = dmp_b.astype(dt, copy=False)
            df = nn.gelu_backward(dg, blk["gelu_c"])
            dh2, dfc_w, dfc_b = nn.linear_backward(df, blk["fc_c"])
            g[pre + "mlp.fc_w"] = dfc_w.astype(dt, copy=False)
            g[pre + "mlp.fc_b"] = dfc_b.astype(dt, copy=False)
            dx_ln2, dln2_g, dln2_b = nn.layernorm_backward(dh2, blk["ln2"])
            g[pre + "ln2_g"] = dln2_g
            g[pre + "ln2_b"] = dln2_b

            # x1 voedt zowel de restakkel (x = x1 + m) als ln2 -> dit is de
            # gradiënt die de aandachtstak én de invoer van het blok in gaat.
            dx1 = dx + dx_ln2

            # --- Aandachtstak ---
            dctx, dproj_w, dproj_b = nn.linear_backward(dx1, blk["proj_c"])
            g[pre + "attn.proj_w"] = dproj_w.astype(dt, copy=False)
            g[pre + "attn.proj_b"] = dproj_b.astype(dt, copy=False)

            dctx_h = self._split_heads(dctx)  # [B,H,T,dh]
            probs = blk["probs"]
            dv = probs.transpose(0, 1, 3, 2) @ dctx_h  # [B,H,T,dh]
            dprobs = dctx_h @ blk["v"].transpose(0, 1, 3, 2)  # [B,H,T,T]
            datt = probs * (dprobs - (dprobs * probs).sum(axis=-1, keepdims=True))
            datt = datt * blk["scale"]
            dq = datt @ blk["k"]                        # [B,H,T,dh]
            dk = datt.transpose(0, 1, 3, 2) @ blk["q"]  # [B,H,T,dh]

            D = c.n_embd
            dqkv = np.zeros((B, T, 3 * D), dtype=dt)
            dqkv[..., 0:D] = self._merge_heads(dq)
            dqkv[..., D : 2 * D] = self._merge_heads(dk)
            dqkv[..., 2 * D :] = self._merge_heads(dv)

            dh, dqkv_w, dqkv_b = nn.linear_backward(dqkv, blk["qkv_c"])
            g[pre + "attn.qkv_w"] = dqkv_w.astype(dt, copy=False)
            g[pre + "attn.qkv_b"] = dqkv_b.astype(dt, copy=False)
            dx_ln1, dln1_g, dln1_b = nn.layernorm_backward(dh, blk["ln1"])
            g[pre + "ln1_g"] = dln1_g
            g[pre + "ln1_b"] = dln1_b

            # x1 = x0 + a  →  beide krijgen dezelfde gradiënt
            dx = dx1 + dx_ln1

        # --- posities en tokens ---
        gwpe = dx.sum(axis=0)  # [T, D]
        g["wpe"][:T] += gwpe.astype(dt, copy=False)

        demb = dx.reshape(-1, c.n_embd)
        flat_ids = ids.reshape(-1)
        np.add.at(g["wte"], flat_ids, demb)

        return {k: np.ascontiguousarray(v, dtype=dt) for k, v in g.items()}

    # ------------------------------------------------------------------
    # Stapsgewijs genereren (KV-cache)
    # ------------------------------------------------------------------

    def new_state(self, batch_size: int = 1) -> Dict[str, Any]:
        c = self.cfg
        H, dh = c.n_head, c.head_dim
        S = c.block_size
        return {
            "k": [np.zeros((batch_size, H, S, dh), dtype=F32) for _ in range(c.n_layer)],
            "v": [np.zeros((batch_size, H, S, dh), dtype=F32) for _ in range(c.n_layer)],
            "len": 0,
        }

    def step(self, ids: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """Voert `ids` ([B, n] nieuwe tokens) in en geeft logits [B, n, V] terug."""
        c = self.cfg
        p = self.params
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        B, n = ids.shape
        start = state["len"]
        end = start + n
        if end > c.block_size:
            raise ValueError(
                f"Context vol: {end} > block_size={c.block_size}. Start een nieuw antwoord."
            )
        scale = 1.0 / math.sqrt(c.head_dim)

        pos = np.arange(start, end, dtype=np.int64)
        x = (p["wte"][ids] + p["wpe"][pos][None, :, :]).astype(self.dtype, copy=False)

        for i in range(c.n_layer):
            pre = f"h{i}."
            D = c.n_embd
            x0 = x
            h, _ = nn.layernorm_forward(x, p[pre + "ln1_g"], p[pre + "ln1_b"])
            qkv, _ = nn.linear_forward(h, p[pre + "attn.qkv_w"], p[pre + "attn.qkv_b"])
            q = self._split_heads(qkv[..., 0:D])
            k = self._split_heads(qkv[..., D : 2 * D])
            v = self._split_heads(qkv[..., 2 * D :])
            state["k"][i][:, :, start:end, :] = k
            state["v"][i][:, :, start:end, :] = v
            K = state["k"][i][:, :, :end, :]
            V = state["v"][i][:, :, :end, :]
            att = (q @ K.transpose(0, 1, 3, 2)) * scale
            if n > 1:
                # Binnen dit brokje mag een token niet naar latere tokens in
                # hetzelfde brokje kijken (anders lekt de toekomst door).
                att = att + self._chunk_mask(n, end)[None, None]
            probs = nn.softmax(att, axis=-1)
            ctx = self._merge_heads(probs @ V)
            a, _ = nn.linear_forward(ctx, p[pre + "attn.proj_w"], p[pre + "attn.proj_b"])
            x = x0 + a
            h2, _ = nn.layernorm_forward(x, p[pre + "ln2_g"], p[pre + "ln2_b"])
            f, _ = nn.linear_forward(h2, p[pre + "mlp.fc_w"], p[pre + "mlp.fc_b"])
            gg, _ = nn.gelu_forward(f)
            m, _ = nn.linear_forward(gg, p[pre + "mlp.proj_w"], p[pre + "mlp.proj_b"])
            x = x + m

        xf, _ = nn.layernorm_forward(x, p["lnf_g"], p["lnf_b"])
        logits, _ = nn.linear_forward(xf, self.out_weight(), p["lm_b"])
        state["len"] = end
        return logits.astype(self.dtype, copy=False)

    # ------------------------------------------------------------------
    # Opslaan / laden
    # ------------------------------------------------------------------

    def save(self, run_dir: str, name: str = "model", extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(run_dir, exist_ok=True)
        np.savez(os.path.join(run_dir, f"{name}.npz"), **{k: v for k, v in self.params.items()})
        meta = {"model": self.cfg.to_dict(), "num_params": self.num_params()}
        if extra:
            meta.update(extra)
        save_json(os.path.join(run_dir, f"{name}.json"), meta)
        return run_dir

    @classmethod
    def load(cls, run_dir: str, name: str = "model") -> Tuple["TransformerLM", Dict[str, Any]]:
        npz_path = os.path.join(run_dir, f"{name}.npz")
        json_path = os.path.join(run_dir, f"{name}.json")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Geen checkpoint gevonden: {npz_path}")
        meta = load_json(json_path, {}) or {}
        cfg = ModelConfig.from_dict(meta.get("model", {}))
        with np.load(npz_path) as npz:
            params = {k: npz[k].astype(F32, copy=False) for k in npz.files}
        model = cls(cfg, params=params)
        return model, meta


def random_model(cfg: ModelConfig) -> TransformerLM:
    return TransformerLM(cfg)
