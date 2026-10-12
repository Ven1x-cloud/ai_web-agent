"""Test voor datasetbouw: welke tokens telt het model mee voor het verlies?

Draaien: python ai/tests/test_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai.dataset import (  # noqa: E402
    Batcher,
    Sample,
    build_chat_sample,
    build_dataset,
    build_text_samples,
    record_to_messages,
    split_samples,
)
from webai.tokenizer import EOT, TURN, ByteLevelBPE  # noqa: E402

CORPUS = ("Hallo wereld, dit is een testtekst. " * 30) + ("What is the answer? Yes. " * 30)


def make_tok(vocab: int = 600) -> ByteLevelBPE:
    return ByteLevelBPE.train(CORPUS, vocab_size=vocab)


class TestChatSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = make_tok()

    def test_loss_only_on_assistant(self):
        """Het model leert alléén zijn eigen antwoorden, niet de vragen."""
        sample = build_chat_sample(self.tok, [
            {"role": "system", "content": "Je bent aardig."},
            {"role": "user", "content": "Hoi"},
            {"role": "assistant", "content": "Hallo!"},
        ], block_size=512)
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample.ids), len(sample.mask))
        # drop_special=False: anders wordt het beurt-einde juist weggelaten
        decoded_learned = self.tok.decode(
            [int(t) for t, m in zip(sample.ids, sample.mask) if m > 0], drop_special=False
        )
        decoded_masked = self.tok.decode(
            [int(t) for t, m in zip(sample.ids, sample.mask) if m == 0], drop_special=False
        )
        self.assertIn("Hallo!", decoded_learned, "het antwoord moet meegeteld worden")
        self.assertIn(EOT, decoded_learned, "het beurt-einde hoort bij het antwoord")
        self.assertNotIn("Hallo!", decoded_masked)
        self.assertIn("Hoi", decoded_masked, "de vraag mag het verlies niet beïnvloeden")
        self.assertIn("Je bent aardig.", decoded_masked)

    def test_multi_turn_learns_every_assistant_turn(self):
        sample = build_chat_sample(self.tok, [
            {"role": "user", "content": "vraag één"},
            {"role": "assistant", "content": "antwoord één"},
            {"role": "user", "content": "vraag twee"},
            {"role": "assistant", "content": "antwoord twee"},
        ], block_size=1024)
        learned = self.tok.decode([int(t) for t, m in zip(sample.ids, sample.mask) if m > 0])
        self.assertIn("antwoord één", learned)
        self.assertIn("antwoord twee", learned)
        self.assertNotIn("vraag één", learned)
        self.assertNotIn("vraag twee", learned)

    def test_tool_turns_are_not_learned(self):
        sample = build_chat_sample(self.tok, [
            {"role": "user", "content": "lees de pagina"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "read_page -> de tekst van de pagina"},
            {"role": "assistant", "content": "De pagina gaat over dijken."},
        ], block_size=1024)
        learned = self.tok.decode([int(t) for t, m in zip(sample.ids, sample.mask) if m > 0])
        self.assertIn("De pagina gaat over dijken.", learned)
        self.assertNotIn("de tekst van de pagina", learned,
                         "toolresultaten zijn invoer, geen voorbeeld om na te zeggen")

    def test_without_assistant_turn_is_skipped(self):
        self.assertIsNone(build_chat_sample(self.tok, [
            {"role": "user", "content": "alleen een vraag"},
        ], block_size=512))

    def test_long_conversation_is_truncated_to_block_size(self):
        long_answer = "Dit is een heel lang antwoord. " * 400
        sample = build_chat_sample(self.tok, [
            {"role": "system", "content": "Systeem."},
            {"role": "user", "content": "Vertel alles."},
            {"role": "assistant", "content": long_answer},
        ], block_size=128)
        self.assertLessEqual(len(sample.ids), 128)
        self.assertTrue(any(m > 0 for m in sample.mask),
                        "na afkappen moet er nog iets te leren over zijn")

    def test_starts_with_bos(self):
        sample = build_chat_sample(self.tok, [
            {"role": "user", "content": "Hoi"},
            {"role": "assistant", "content": "Hallo"},
        ], block_size=512)
        self.assertEqual(int(sample.ids[0]), self.tok.bos_id)
        self.assertEqual(float(sample.mask[0]), 0.0, "het begin-token leren we niet")

    def test_turn_markers_are_present(self):
        sample = build_chat_sample(self.tok, [
            {"role": "user", "content": "Hoi"},
            {"role": "assistant", "content": "Hallo"},
        ], block_size=512)
        text = self.tok.decode(sample.ids, drop_special=False)
        self.assertIn(f"{TURN}user", text)
        self.assertIn(f"{TURN}assistant", text)
        self.assertTrue(text.rstrip().endswith(EOT), f"moet eindigen op {EOT}, kreeg {text[-40:]!r}")


class TestTextSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = make_tok()

    def test_chunks_cover_the_text(self):
        text = "Een zin die herhaald wordt. " * 300
        samples = build_text_samples(self.tok, text, block_size=64)
        self.assertGreater(len(samples), 1)
        for s in samples:
            self.assertLessEqual(len(s.ids), 64)
            self.assertTrue(all(m == 1.0 for m in s.mask),
                            "bij platte tekst telt elk token mee")
            self.assertEqual(s.kind, "text")

    def test_empty_text_gives_nothing(self):
        self.assertEqual(build_text_samples(self.tok, "   ", 64), [])


class TestRecords(unittest.TestCase):
    def test_openai_messages(self):
        msgs = record_to_messages({"messages": [{"role": "user", "content": "a"}]})
        self.assertEqual(msgs[0]["content"], "a")

    def test_prompt_completion(self):
        msgs = record_to_messages({"prompt": "vraag", "completion": "antwoord"})
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])

    def test_alpaca_style(self):
        msgs = record_to_messages({"instruction": "Los op", "input": "2+2", "output": "4"})
        self.assertIn("2+2", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "4")

    def test_question_answer(self):
        msgs = record_to_messages({"question": "q", "answer": "a"})
        self.assertEqual(msgs[0]["content"], "q")

    def test_sharegpt_conversations(self):
        msgs = record_to_messages({"conversations": [
            {"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]})
        self.assertEqual(len(msgs), 2)

    def test_plain_text_field(self):
        self.assertIsNone(record_to_messages({"text": "gewone tekst"}))

    def test_bare_list_of_messages(self):
        msgs = record_to_messages([{"role": "user", "content": "x"},
                                   {"role": "assistant", "content": "y"}])
        self.assertEqual(len(msgs), 2)

    def test_garbage_is_none(self):
        self.assertIsNone(record_to_messages(42))
        self.assertIsNone(record_to_messages({"foo": "bar"}))


class TestBatcher(unittest.TestCase):
    def _samples(self, n=12, lengths=None):
        lengths = lengths or [20 + i * 5 for i in range(n)]
        out = []
        for ln in lengths:
            ids = np.arange(ln, dtype=np.int64) % 500
            mask = np.ones(ln, dtype=np.float32)
            out.append(Sample(ids=ids, mask=mask, source="t", kind="chat"))
        return out

    def test_collate_shapes_and_shift(self):
        samples = self._samples(4, [10, 12, 11, 13])
        b = Batcher(samples, batch_size=4, block_size=64)
        ids, targets, mask = b.collate([0, 1, 2, 3])
        self.assertEqual(ids.shape[0], 4)
        self.assertEqual(ids.shape, targets.shape)
        self.assertEqual(ids.shape, mask.shape)
        # targets zijn de id's één opgeschoven
        row = 1
        n = min(ids.shape[1], len(samples[row].ids) - 1)
        self.assertTrue(np.array_equal(ids[row, :n], samples[row].ids[:n]))
        self.assertTrue(np.array_equal(targets[row, :n], samples[row].ids[1 : n + 1]))

    def test_padding_is_masked_out(self):
        samples = self._samples(2, [8, 30])
        b = Batcher(samples, batch_size=2, block_size=64)
        ids, targets, mask = b.collate([0, 1])
        self.assertEqual(ids.shape[1], 29)  # langste sample bepaalt de breedte (min 1 voor shift)
        self.assertEqual(mask[0, 7:].sum(), 0.0, "padding telt niet mee")
        self.assertTrue((targets[0, 7:] == -100).all(), "padding krijgt -100")

    def test_respects_block_size(self):
        samples = self._samples(2, [200, 210])
        b = Batcher(samples, batch_size=2, block_size=64)
        ids, _, _ = b.collate([0, 1])
        self.assertLessEqual(ids.shape[1], 64)

    def test_all_samples_are_used_each_epoch(self):
        samples = self._samples(25, [15] * 25)
        b = Batcher(samples, batch_size=8, block_size=64)
        seen = 0
        for _ids, _t, _m in b.batches():
            seen += _ids.shape[0]
            if seen >= 25:
                break
        self.assertEqual(seen, 25, "een hele ronde moet alle samples langs laten komen")

    def test_eval_batches_are_stable(self):
        samples = self._samples(20, [15] * 20)
        b = Batcher(samples, batch_size=4, block_size=64)
        first = [x[0].copy() for x in b.eval_batches(3)]
        b.reshuffle()
        second = [x[0].copy() for x in b.eval_batches(3)]
        self.assertTrue(all(np.array_equal(a, c) for a, c in zip(first, second)),
                        "evaluatieset moet vast staan, anders is het verlies niet vergelijkbaar")


class TestSplit(unittest.TestCase):
    def _samples(self, n):
        return [Sample(ids=np.arange(20, dtype=np.int64), mask=np.ones(20, np.float32))
                for _ in range(n)]

    def test_small_dataset_gets_a_useful_eval_set(self):
        train, evl = split_samples(self._samples(48), 0.05)
        self.assertGreaterEqual(len(evl), 4, "met weinig data moet eval groter zijn dan 5%")
        self.assertEqual(len(train) + len(evl), 48)

    def test_large_dataset_uses_the_fraction(self):
        train, evl = split_samples(self._samples(2000), 0.05)
        self.assertEqual(len(evl), 100)
        self.assertEqual(len(train), 1900)

    def test_tiny_dataset_keeps_everything_for_training(self):
        train, evl = split_samples(self._samples(4), 0.05)
        self.assertEqual(len(train), 4)
        self.assertEqual(evl, [])

    def test_no_overlap(self):
        samples = [Sample(ids=np.full(5, i, dtype=np.int64), mask=np.ones(5, np.float32))
                   for i in range(40)]
        train, evl = split_samples(samples, 0.1, seed=3)
        keys_train = {int(s.ids[0]) for s in train}
        keys_eval = {int(s.ids[0]) for s in evl}
        self.assertFalse(keys_train & keys_eval, "train en eval mogen niet overlappen")


class TestBuildDataset(unittest.TestCase):
    def test_reads_mixed_files(self):
        tok = make_tok()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "chat.jsonl"), "w", encoding="utf-8") as fh:
                for i in range(6):
                    fh.write(json.dumps({"messages": [
                        {"role": "user", "content": f"vraag {i}"},
                        {"role": "assistant", "content": f"antwoord {i} met wat tekst"},
                    ]}, ensure_ascii=False) + "\n")
            with open(os.path.join(d, "text.txt"), "w", encoding="utf-8") as fh:
                fh.write("Platte tekst om te leren. " * 60)
            with open(os.path.join(d, "lijst.json"), "w", encoding="utf-8") as fh:
                json.dump([{"prompt": "p1", "completion": "c1"}], fh)
            files = [os.path.join(d, n) for n in ("chat.jsonl", "text.txt", "lijst.json")]
            samples, stats = build_dataset(tok, block_size=128, files=files)
            self.assertEqual(stats["files"], 3)
            self.assertGreaterEqual(stats["chat"], 7)
            self.assertGreaterEqual(stats["text"], 1)
            self.assertTrue(all(len(s.ids) <= 128 for s in samples))

    def test_bad_lines_are_not_fatal(self):
        tok = make_tok()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kapot.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"messages": [{"role": "user", "content": "ok"},'
                         ' {"role": "assistant", "content": "prima"}]}\n')
                fh.write("{ dit is geen json\n")
                fh.write("\n")
            samples, stats = build_dataset(tok, block_size=128, files=[path])
            self.assertEqual(len(samples), 1)
            self.assertEqual(stats["skipped"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
