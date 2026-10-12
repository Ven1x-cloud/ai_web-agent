"""Test voor de trainingslus: daalt het verlies, en kan hij hervatten?

Draaien: python ai/tests/test_training.py

Gebruikt expres een piepklein model en heel weinig stappen, zodat de test in
seconden klaar is. Het gaat om de *machinerie* — loggen, checkpoints, hervatten,
vroeg stoppen — en niet om de kwaliteit van het model.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai.config import PRESETS, ModelConfig, TrainConfig, configs_from_preset  # noqa: E402
from webai.dataset import build_dataset  # noqa: E402
from webai.model import TransformerLM  # noqa: E402
from webai.tokenizer import ByteLevelBPE  # noqa: E402
from webai import train as trainer  # noqa: E402

def read_jsonl(path: str):
    """Leest een .jsonl-bestand en sluit het weer netjes."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


CORPUS = ("Wat is twee plus twee? Vier.\n" * 40 +
          "De hoofdstad van Nederland is Amsterdam.\n" * 40)


def tiny_config(tok: ByteLevelBPE) -> ModelConfig:
    return ModelConfig(n_layer=1, n_head=2, n_embd=16, block_size=48,
                       vocab_size=tok.vocab_size, seed=11)


def tiny_train_cfg(run: str, steps: int, **kw) -> TrainConfig:
    base = dict(batch_size=4, max_steps=steps, learning_rate=3e-3, warmup_steps=3,
                eval_every=5, eval_batches=2, log_every=5, save_every=10,
                run_name=run, seed=7)
    base.update(kw)
    return TrainConfig(**base)


class TrainFixture:
    """Een tokenizer + dataset in een tijdelijke map, gedeeld door alle tests."""

    def __init__(self) -> None:
        self.tok = ByteLevelBPE.train(CORPUS, vocab_size=500)
        self.datadir = tempfile.TemporaryDirectory()
        path = os.path.join(self.datadir.name, "corpus.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(CORPUS)
        self.samples, self.stats = build_dataset(
            self.tok, block_size=48, files=[path], verbose=False
        )

    def train(self, root: str, run: str, steps: int, **kw) -> trainer.TrainResult:
        tcfg = tiny_train_cfg(run, steps, **kw)
        return trainer.train(
            tiny_config(self.tok), tcfg, tokenizer=self.tok, samples=self.samples,
            root=root, verbose=False,
        )

    def close(self) -> None:
        self.datadir.cleanup()


class TestTrainLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = TrainFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_dataset_was_built(self):
        self.assertGreater(len(self.fx.samples), 1, "zonder samples valt er niets te testen")

    def test_loss_goes_down(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.fx.train(root, "r1", steps=40)
            log = read_jsonl(trainer.log_path("r1", root))
            self.assertGreater(len(log), 3)
            first = float(np.mean([e["loss"] for e in log[:3]]))
            last = float(np.mean([e["loss"] for e in log[-3:]]))
            self.assertLess(last, first, "het verlies moet dalen tijdens trainen")
            self.assertTrue(all("lr" in e and "tokens_seen" in e for e in log))
            self.assertEqual(result.steps, 40)

    def test_checkpoint_and_state_are_written(self):
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r2", steps=20)
            directory = trainer.run_dir("r2", root)
            self.assertTrue(os.path.exists(os.path.join(directory, "model.npz")))
            self.assertTrue(os.path.exists(os.path.join(directory, "model.json")))
            state = read_json(trainer.state_path("r2", root))
            self.assertEqual(state["step"], 20)
            self.assertIn("loss", state)
            self.assertIn("model", state)
            self.assertIn("train", state)

    def test_resume_continues_instead_of_restarting(self):
        """Hervatten moet doortellen, en hetzelfde model gebruiken."""
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r3", steps=15)
            state = read_json(trainer.state_path("r3", root))
            self.assertEqual(state["step"], 15)

            tcfg = tiny_train_cfg("r3", 25)
            result = trainer.train(
                tiny_config(self.fx.tok), tcfg, tokenizer=self.fx.tok,
                samples=self.fx.samples, root=root, verbose=False, resume=True,
            )
            self.assertEqual(result.steps, 25, "hervatten telt door naar het nieuwe totaal")

            state2 = read_json(trainer.state_path("r3", root))
            self.assertEqual(state2["step"], 25)

            # het log is aangevuld, niet overschreven
            log = read_jsonl(trainer.log_path("r3", root))
            steps = [e["step"] for e in log]
            self.assertEqual(max(steps), 25)
            self.assertIn(15, steps, "de eerdere stappen horen nog in het log te staan")

    def test_best_checkpoint_is_tagged(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.fx.train(root, "r4", steps=30, eval_every=5)
            directory = trainer.run_dir("r4", root)
            log = read_jsonl(trainer.log_path("r4", root))
            evals = [e["eval_loss"] for e in log if e.get("eval_loss") is not None]
            self.assertGreater(len(evals), 1, "er moet meer dan één evaluatie zijn")
            self.assertTrue(all(np.isfinite(evals)), "eval-verlies mag geen NaN zijn")
            if result.best_eval is not None:
                self.assertTrue(os.path.exists(os.path.join(directory, "best.npz")),
                                "bij een beste eval hoort een best.npz")
                self.assertAlmostEqual(result.best_eval, min(evals), places=3)

    def test_losses_are_finite(self):
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r5", steps=20)
            log = read_jsonl(trainer.log_path("r5", root))
            for entry in log:
                self.assertTrue(np.isfinite(entry["loss"]),
                                f"NaN/inf verlies bij stap {entry['step']}")
                self.assertTrue(np.isfinite(entry["grad_norm"]),
                                f"NaN/inf gradiëntnorm bij stap {entry['step']}")

    def test_learning_rate_warms_up_then_decays(self):
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r6", steps=30, warmup_steps=5)
            log = read_jsonl(trainer.log_path("r6", root))
            lrs = [e["lr"] for e in log]
            # met log_every=5 en warmup_steps=5 valt de eerste logregel op de piek
            self.assertAlmostEqual(lrs[0], max(lrs), places=6,
                                   msg="de eerste gelogde leerkracht is de piek")
            self.assertGreater(lrs[0], lrs[-1], "daarna moet hij afnemen (cosine)")
            self.assertEqual(lrs, sorted(lrs, reverse=True),
                             "na de warmup daalt de leerkracht stap voor stap")
            self.assertTrue(all(lr > 0 for lr in lrs))
            self.assertLessEqual(max(lrs), 3e-3 + 1e-12, "hij mag niet boven de ingestelde lr komen")

    def test_tokens_seen_increases(self):
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r7", steps=15)
            log = read_jsonl(trainer.log_path("r7", root))
            seen = [e["tokens_seen"] for e in log]
            self.assertEqual(seen, sorted(seen), "tokens_seen loopt alleen maar op")
            self.assertGreater(seen[-1], seen[0])

    def test_model_is_reusable_after_training(self):
        """Het opgeslagen model moet weer geladen kunnen worden en vooruit kunnen."""
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r8", steps=15)
            model, meta = TransformerLM.load(trainer.run_dir("r8", root), "model")
            self.assertEqual(meta.get("step"), 15)
            ids = np.array([[1, 5, 9, 13, 2]], dtype=np.int64)
            targets = np.array([[5, 9, 13, 2, 7]], dtype=np.int64)
            loss, _grads, logits, _cache = model.forward(ids, targets)
            self.assertTrue(np.isfinite(loss), "het verlies moet een eindig getal zijn")
            self.assertEqual(logits.shape[-1], model.cfg.vocab_size)
            # zonder targets is er niets te meten
            none_loss, _g2, _l2, _c2 = model.forward(ids)
            self.assertIsNone(none_loss)

    def test_model_info_reports_status(self):
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r9", steps=15)
            info = trainer.model_info("r9", root)
            self.assertTrue(info["exists"])
            self.assertGreater(info["params"], 0)
            self.assertEqual(info["step"], 15)
            self.assertEqual(info["context"], 48)
            self.assertTrue(len(info["log_tail"]) > 0)

    def test_model_info_for_unknown_run(self):
        with tempfile.TemporaryDirectory() as root:
            info = trainer.model_info("bestaat-niet", root)
            self.assertFalse(info["exists"])
            self.assertEqual(info["run_name"], "bestaat-niet")

    def test_early_stopping_halts_when_eval_stagnates(self):
        """Met weinig data en veel stappen moet hij stoppen in plaats van dooroverfitten."""
        with tempfile.TemporaryDirectory() as root:
            result = self.fx.train(root, "r10", steps=500, eval_every=5,
                                   early_stop_patience=2)
            self.assertLess(result.steps, 500,
                            "vroeg stoppen moet vóór de 500 stappen ingrijpen")
            directory = trainer.run_dir("r10", root)
            meta = read_json(os.path.join(directory, "model.json"))
            self.assertTrue(meta.get("early_stopped"))
            self.assertTrue(meta.get("best_step") is not None)

    def test_gradient_accumulation_runs(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.fx.train(root, "r11", steps=10, grad_accum=2, batch_size=2)
            self.assertEqual(result.steps, 10)
            self.assertGreater(result.tokens_seen, 0)


    def test_best_step_survives_resume(self):
        """Na hervatten moet nog bekend zijn wáár het optimum lag.

        Zonder dit viel de melding terug op de startstap: "beste op stap 856"
        terwijl het optimum op stap 400 lag.
        """
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r12", steps=20, eval_every=5)
            state = read_json(trainer.state_path("r12", root))
            self.assertIn("best_step", state, "de status moet de beste stap bewaren")
            self.assertLessEqual(state["best_step"], 20)

            tcfg = tiny_train_cfg("r12", 30)
            result = trainer.train(
                tiny_config(self.fx.tok), tcfg, tokenizer=self.fx.tok,
                samples=self.fx.samples, root=root, verbose=False, resume=True,
            )
            self.assertEqual(result.steps, 30)
            state2 = read_json(trainer.state_path("r12", root))
            self.assertLessEqual(state2["best_step"], 30)
            self.assertIn("best_step", state2)

            # het checkpoint met het beste eval-verlies moet diezelfde stap noemen
            best = read_json(os.path.join(trainer.run_dir("r12", root), "best.json"))
            self.assertEqual(best["step"], state2["best_step"],
                             "best.json en train_state.json horen dezelfde stap te noemen")


    def test_resume_with_nothing_to_do_changes_nothing(self):
        """Hervatten met een stap-aantal dat al gepasseerd is, mag niets overschrijven.

        Eerder trainde dat nul stappen en schreef tóch een NaN-verlies weg over
        een goede status heen — en NaN is geen geldige JSON.
        """
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r13", steps=20, eval_every=5)
            before = read_json(trainer.state_path("r13", root))
            self.assertIsNotNone(before.get("loss"))
            self.assertTrue(np.isfinite(before["loss"]))

            tcfg = tiny_train_cfg("r13", 10)          # lager dan de huidige 20
            result = trainer.train(
                tiny_config(self.fx.tok), tcfg, tokenizer=self.fx.tok,
                samples=self.fx.samples, root=root, verbose=False, resume=True,
            )
            self.assertEqual(result.steps, 20, "hij meldt de bestaande stand, niet 10")

            after = read_json(trainer.state_path("r13", root))
            self.assertEqual(after["loss"], before["loss"],
                             "de status mag niet verslechteren")
            self.assertTrue(np.isfinite(after["loss"]), "geen NaN op schijf")

    def test_tokens_seen_is_cumulative_across_resumes(self):
        """Een hervatting mag de eerder getrainde tokens niet weggooien.

        Anders lijkt een model dat in drie sessies getraind is net zo ver als
        de laatste sessie alleen.
        """
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r15", steps=20, eval_every=5)
            first = read_json(trainer.state_path("r15", root))["tokens_seen"]
            self.assertGreater(first, 0)

            tcfg = tiny_train_cfg("r15", 40)
            trainer.train(tiny_config(self.fx.tok), tcfg, tokenizer=self.fx.tok,
                          samples=self.fx.samples, root=root, verbose=False,
                          resume=True)
            second = read_json(trainer.state_path("r15", root))["tokens_seen"]
            self.assertGreater(second, first,
                               "hervatten moet doortellen, niet opnieuw beginnen")

    def test_saved_state_is_always_valid_json(self):
        """NaN mag nooit in een statusbestand belanden: JSON.parse weigert dat."""
        with tempfile.TemporaryDirectory() as root:
            self.fx.train(root, "r14", steps=10, eval_every=5)
            with open(trainer.state_path("r14", root), encoding="utf-8") as fh:
                raw = fh.read()
            self.assertNotIn("NaN", raw, "NaN is geen geldige JSON")
            self.assertNotIn("Infinity", raw)
            json.loads(raw)  # strikt parsen; gooit bij ongeldige JSON


class TestAutoEarlyStop(unittest.TestCase):
    """Wanneer moet vroeg stoppen vanzelf aan gaan?

    De aanleiding is een echte meting: bij 484 voorbeelden, batch 16 en 2500
    stappen lag het optimum op stap 400 (eval 2,78) en liep het daarna op naar
    3,04. De oude regel keek alleen naar "minder dan 300 samples" en greep dus
    niet in, terwijl er wel degelijk overfit werd.
    """

    def test_negative_patience_means_deliberately_off(self):
        """--no-early-stop geeft -1 door; dat is iets anders dan "niet opgegeven"."""
        self.assertEqual(trainer.auto_early_stop_patience(50, 16, 1000, patience=-1), 0)
        self.assertEqual(trainer.auto_early_stop_patience(484, 16, 2500, patience=-1), 0)
        # 0 blijft "laat train zelf beslissen"
        self.assertEqual(trainer.auto_early_stop_patience(484, 16, 2500, patience=0), 4)

    def test_explicit_patience_always_wins(self):
        self.assertEqual(trainer.auto_early_stop_patience(50, 16, 1000, patience=7), 7)
        self.assertEqual(trainer.auto_early_stop_patience(50000, 16, 100, patience=2), 2)

    def test_few_samples_turns_it_on(self):
        self.assertEqual(trainer.auto_early_stop_patience(100, 16, 500), 4)

    def test_many_epochs_turn_it_on(self):
        """484 samples boven de oude grens van 300, maar 83 ronden is te veel."""
        got = trainer.auto_early_stop_patience(484, 16, 2500)
        self.assertEqual(got, 4, "de echte gemeten casus moet nu wél ingrijpen")

    def test_plenty_of_data_with_few_epochs_stays_off(self):
        # 100.000 voorbeelden, batch 16, 2000 stappen = 0,3 ronden
        self.assertEqual(trainer.auto_early_stop_patience(100000, 16, 2000), 0)

    def test_borderline_cases(self):
        # precies op de grenzen: net onder de sample-grens → aan
        self.assertEqual(trainer.auto_early_stop_patience(299, 16, 10), 4)
        # net erboven en weinig ronden → uit
        self.assertEqual(trainer.auto_early_stop_patience(300, 16, 10), 0)

    def test_planned_epochs_math(self):
        self.assertAlmostEqual(trainer.planned_epochs(484, 16, 2500), 2500 / 30, places=6)
        self.assertAlmostEqual(trainer.planned_epochs(16, 16, 31), 31.0)
        self.assertEqual(trainer.planned_epochs(0, 16, 100), 100.0,
                         "delen door nul mag niet: minimaal één voorbeeld per ronde")

    def test_thresholds_are_tunable(self):
        self.assertEqual(trainer.auto_early_stop_patience(484, 16, 2500, epoch_ceiling=1000), 0)
        self.assertEqual(trainer.auto_early_stop_patience(484, 16, 2500, sample_floor=500), 4)


class TestPresets(unittest.TestCase):
    def test_presets_exist_and_grow(self):
        self.assertEqual(list(PRESETS), ["tiny", "small", "medium", "large"])
        params = []
        for name in PRESETS:
            mcfg, _tcfg = configs_from_preset(name)
            params.append(TransformerLM(mcfg).num_params())
        self.assertEqual(params, sorted(params),
                         "de presets moeten oplopend groter worden")

    def test_preset_context_matches_train_config(self):
        for name in PRESETS:
            mcfg, tcfg = configs_from_preset(name)
            self.assertEqual(mcfg.vocab_size, tcfg.__dict__.get("vocab_size", mcfg.vocab_size))
            self.assertGreater(mcfg.block_size, 0)
            self.assertGreater(tcfg.max_steps, 0)
            self.assertLess(tcfg.min_lr, tcfg.learning_rate)

    def test_overrides_reach_the_right_config(self):
        mcfg, tcfg = configs_from_preset("tiny", n_embd=64, batch_size=2)
        self.assertEqual(mcfg.n_embd, 64, "een modeloverride hoort in ModelConfig")
        self.assertEqual(tcfg.batch_size, 2, "een trainoverride hoort in TrainConfig")

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            configs_from_preset("gigantisch")

    def test_none_overrides_are_ignored(self):
        mcfg, _tcfg = configs_from_preset("tiny", n_embd=None)
        self.assertEqual(mcfg.n_embd, PRESETS["tiny"]["model"]["n_embd"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
