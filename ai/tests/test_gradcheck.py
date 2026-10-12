"""Gradient-check: vergelijkt de handgeschreven terugrekening met eindige verschillen.

Draaien:  python ai/tests/test_gradcheck.py
Of:        python -m unittest discover -s ai/tests -t ai -v

De controle gebeurt in float64, zodat de numerieke benadering zelf geen ruis
veroorzaakt. Daarnaast wordt de KV-cache-stap (float32, zoals in productie)
tegen de volledige doorloop gehouden.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai import nn  # noqa: E402
from webai.config import ModelConfig  # noqa: E402
from webai.model import TransformerLM  # noqa: E402


def tiny_model(seed: int = 0, dtype=np.float64) -> TransformerLM:
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=16, block_size=12, vocab_size=23, seed=seed)
    model = TransformerLM(cfg)
    model.params = {k: v.astype(dtype) for k, v in model.params.items()}
    model.dtype = np.dtype(dtype)
    return model


def numeric_grad(model, ids, targets, mask, name, index, eps=1e-4):
    p = model.params[name]
    orig = float(p[index])
    p[index] = p.dtype.type(orig + eps)
    loss_hi, _, _, _ = model.forward(ids, targets, mask)
    p[index] = p.dtype.type(orig - eps)
    loss_lo, _, _, _ = model.forward(ids, targets, mask)
    p[index] = p.dtype.type(orig)
    return (loss_hi - loss_lo) / (2 * eps)


class TestGradients(unittest.TestCase):
    def test_full_model_gradcheck(self):
        rng = np.random.default_rng(1234)
        model = tiny_model(seed=7)
        B, T = 3, 9
        V = model.cfg.vocab_size
        ids = rng.integers(0, V, size=(B, T))
        targets = rng.integers(0, V, size=(B, T))
        mask = np.ones((B, T), dtype=np.float32)
        mask[2, 6:] = 0.0  # een beetje padding

        loss, grads, _, _ = model.forward(ids, targets, mask)
        self.assertTrue(np.isfinite(loss))

        checked = 0
        worst = ("", 0.0)
        for name in model.params:
            p = model.params[name]
            g = grads[name]
            self.assertEqual(p.shape, g.shape, f"vorm van gradiënt {name} klopt niet")
            if p.size == 0:
                continue
            for fi in rng.choice(p.size, size=min(3, p.size), replace=False):
                index = np.unravel_index(int(fi), p.shape)
                num = numeric_grad(model, ids, targets, mask, name, index)
                ana = float(g[index])
                abs_err = abs(num - ana)
                rel_err = abs_err / max(1e-9, abs(num), abs(ana))
                checked += 1
                if rel_err > worst[1]:
                    worst = (f"{name}{index}", rel_err)
                self.assertTrue(
                    rel_err < 1e-4 or abs_err < 1e-10,
                    f"Gradiënt klopt niet voor {name}{index}: numeriek={num:.8g} "
                    f"analytisch={ana:.8g} (rel={rel_err:.3g})",
                )
        print(f"\n[gradcheck] {checked} waarden gecontroleerd; slechtste rel-afwijking "
              f"{worst[1]:.2e} bij {worst[0]}")

    def test_loss_mask_ignored_positions(self):
        """Maskerde posities mogen verlies noch gradiënten beïnvloeden."""
        rng = np.random.default_rng(5)
        model = tiny_model(seed=3)
        B, T = 2, 8
        V = model.cfg.vocab_size
        ids = rng.integers(0, V, size=(B, T))
        base_targets = rng.integers(0, V, size=(B, T))
        other_targets = base_targets.copy()
        other_targets[1, 5:] = rng.integers(0, V, size=(3,))

        mask = np.ones((B, T), dtype=np.float32)
        mask[1, 5:] = 0.0  # dezelfde geldige posities in beide gevallen

        loss_a, grads_a, _, _ = model.forward(ids, base_targets, mask)
        loss_b, grads_b, _, _ = model.forward(ids, other_targets, mask)
        self.assertAlmostEqual(loss_a, loss_b, places=9)
        for k in grads_a:
            self.assertTrue(
                np.allclose(grads_a[k], grads_b[k], atol=1e-12, rtol=1e-9),
                f"mask lekt door in de gradiënt van {k}",
            )

    def test_step_matches_forward_float32(self):
        """De KV-cache-stap moet dezelfde logits geven als de hele reeks in één keer."""
        rng = np.random.default_rng(99)
        model = tiny_model(seed=11, dtype=np.float32)  # productieprecisie
        T = 10
        ids = rng.integers(0, model.cfg.vocab_size, size=(1, T))
        _, _, logits_full, _ = model.forward(ids)

        # in brokjes van verschillende grootte (zo gebeurt het bij genereren ook)
        state = model.new_state(1)
        pieces = []
        i = 0
        for chunk in (3, 1, 4, 2):
            pieces.append(model.step(ids[:, i : i + chunk], state))
            i += chunk
        logits_step = np.concatenate(pieces, axis=1)
        diff = float(np.abs(logits_full - logits_step).max())
        self.assertLess(diff, 1e-5, f"KV-cache wijkt af: max verschil {diff:.2e}")

        # token voor token moet hetzelfde geven
        state2 = model.new_state(1)
        one = np.concatenate([model.step(ids[:, i : i + 1], state2) for i in range(T)], axis=1)
        self.assertLess(float(np.abs(one - logits_step).max()), 1e-5)

    def test_context_limit_respected(self):
        model = tiny_model(seed=2, dtype=np.float32)
        state = model.new_state(1)
        ids = np.zeros((1, model.cfg.block_size), dtype=np.int64)
        model.step(ids, state)
        with self.assertRaises(ValueError):
            model.step(np.zeros((1, 1), dtype=np.int64), state)

    def test_layer_gradchecks(self):
        """Ook de losse laagjes apart controleren."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 5, 8))

        # LayerNorm
        g = rng.standard_normal(8)
        b = rng.standard_normal(8)
        y, cache = nn.layernorm_forward(x, g, b)
        dy = rng.standard_normal(y.shape)
        dx, dg, db = nn.layernorm_backward(dy, cache)
        self._check(lambda t: nn.layernorm_forward(t, g, b)[0], x, dx, dy)
        self._check_param(lambda gg: nn.layernorm_forward(x, gg, b)[0], g, dg, dy)
        self._check_param(lambda bb: nn.layernorm_forward(x, g, bb)[0], b, db, dy)

        # Linear
        W = rng.standard_normal((8, 6))
        bb = rng.standard_normal(6)
        y, cache = nn.linear_forward(x, W, bb)
        dy = rng.standard_normal(y.shape)
        dx, dW, db2 = nn.linear_backward(dy, cache)
        self._check(lambda t: nn.linear_forward(t, W, bb)[0], x, dx, dy)
        self._check_param(lambda w: nn.linear_forward(x, w, bb)[0], W, dW, dy)
        self._check_param(lambda b3: nn.linear_forward(x, W, b3)[0], bb, db2, dy)

        # GELU
        y, cache = nn.gelu_forward(x)
        dy = rng.standard_normal(y.shape)
        dx = nn.gelu_backward(dy, cache)
        self._check(lambda t: nn.gelu_forward(t)[0], x, dx, dy)

    def test_cross_entropy_gradcheck(self):
        rng = np.random.default_rng(4)
        logits = rng.standard_normal((2, 3, 7))
        targets = rng.integers(0, 7, size=(2, 3))
        mask = np.ones((2, 3))
        mask[0, 2] = 0.0
        loss, dlogits, n = nn.cross_entropy(logits, targets, mask)
        self.assertEqual(n, 5)
        eps = 1e-6
        for _ in range(20):
            i = tuple(int(v) for v in rng.integers(0, np.array(logits.shape)))
            orig = float(logits[i])
            logits[i] = orig + eps
            hi, _, _ = nn.cross_entropy(logits, targets, mask)
            logits[i] = orig - eps
            lo, _, _ = nn.cross_entropy(logits, targets, mask)
            logits[i] = orig
            num = (hi - lo) / (2 * eps)
            ana = float(dlogits[i])
            self.assertLess(abs(num - ana), 1e-6 + 1e-4 * max(abs(num), abs(ana)))

    def _check(self, fn, x, dx, dy, eps=1e-6, n=15):
        rng = np.random.default_rng(0)
        x = x.copy()
        for _ in range(n):
            i = tuple(int(v) for v in rng.integers(0, np.array(x.shape)))
            orig = float(x[i])
            x[i] = orig + eps
            hi = float((fn(x) * dy).sum())
            x[i] = orig - eps
            lo = float((fn(x) * dy).sum())
            x[i] = orig
            num = (hi - lo) / (2 * eps)
            ana = float(dx[i])
            self.assertLess(
                abs(num - ana), 1e-7 + 1e-4 * max(abs(num), abs(ana)), f"{i}: num={num} ana={ana}"
            )

    def _check_param(self, fn, prm, dprm, dy, eps=1e-6, n=10):
        rng = np.random.default_rng(1)
        prm = prm.copy()
        for _ in range(n):
            i = tuple(int(v) for v in rng.integers(0, np.array(prm.shape)))
            orig = float(prm[i])
            prm[i] = orig + eps
            hi = float((fn(prm) * dy).sum())
            prm[i] = orig - eps
            lo = float((fn(prm) * dy).sum())
            prm[i] = orig
            num = (hi - lo) / (2 * eps)
            ana = float(dprm[i])
            self.assertLess(
                abs(num - ana), 1e-7 + 1e-4 * max(abs(num), abs(ana)), f"{i}: num={num} ana={ana}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
