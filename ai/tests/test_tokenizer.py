"""Test voor de eigen BPE-tokenizer.

Draaien: python ai/tests/test_tokenizer.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai.tokenizer import (  # noqa: E402
    EOT,
    TOOL,
    TOOL_CALL,
    TURN,
    ByteLevelBPE,
    pre_tokenize,
)

CORPUS = (
    "Hallo wereld! Dit is een kleine testtekst om de tokenizer te trainen. "
    "De kat zit op de mat. Ik ga naar school en reken wiskunde: 3,5 + 2 = 5,5. "
    "What is the answer? The answer is yes, of course. "
) * 25


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteLevelBPE.train(CORPUS, vocab_size=700)

    def test_roundtrip_dutch(self):
        text = "Nederlandse tekst met diakrieten: café, naïef, geïnteresseerd — 3,5 euro."
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)

    def test_roundtrip_multilingual(self):
        for text in (
            "Bonjour, comment ça va ? Je m'appelle Léa.",
            "Guten Tag! Wie heißen Sie? Ich heiße Müller.",
            "Wiskunde: $$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$",
            "Emoji blijft heel: 🎉🧠✅ en 中文 ook.",
            "",
            "   \n\t  ",
        ):
            with self.subTest(text=text[:30]):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_special_tokens_stay_intact(self):
        text = f"{TURN}system\nJij bent aardig.{EOT}\n{TURN}user\nHoi{EOT}"
        ids = self.tok.encode(text)
        # speciale tokens zijn één id en staan op de verwachte plekken
        self.assertEqual(ids[0], self.tok.encode(TURN)[0])
        decoded = self.tok.decode(ids, drop_special=False)
        self.assertEqual(decoded, text)
        self.assertIn(TOOL_CALL, self.tok.decode(self.tok.encode(TOOL_CALL), drop_special=False))

    def test_specials_never_split(self):
        for special in (TURN, EOT, TOOL, TOOL_CALL):
            ids = self.tok.encode(special)
            self.assertEqual(len(ids), 1, f"{special} werd in {len(ids)} stukken gesplitst")

    def test_encode_ordinary_escapes_specials(self):
        """Zonder toestemming wordt een speciale token als gewone tekst behandeld."""
        ids = self.tok.encode(EOT, allowed_special=())
        self.assertGreater(len(ids), 1)
        self.assertEqual(self.tok.decode(ids, drop_special=False), EOT)

    def test_compression_is_better_than_bytes(self):
        plain = len(CORPUS.encode("utf-8"))
        toks = len(self.tok.encode_ordinary(CORPUS))
        self.assertLess(toks, plain, "BPE moet minder tokens dan bytes geven")
        stats = self.tok.stats(CORPUS)
        self.assertGreater(stats["chars_per_token"], 1.0)

    def test_deterministic_training(self):
        a = ByteLevelBPE.train(CORPUS, vocab_size=600)
        b = ByteLevelBPE.train(CORPUS, vocab_size=600)
        self.assertEqual(a.merges, b.merges)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tokenizer.json")
            self.tok.save(path)
            loaded = ByteLevelBPE.load(path)
            self.assertEqual(loaded.vocab_size, self.tok.vocab_size)
            self.assertEqual(loaded.merges, self.tok.merges)
            text = "Roundtrip na opslaan: hé, 42 %!"
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_pretokenize_covers_everything(self):
        text = "abc 123\n\n\txyz! @# € ½"
        self.assertEqual("".join(pre_tokenize(text)), text)

    def test_min_vocab_guard(self):
        with self.assertRaises(ValueError):
            ByteLevelBPE.train("te kort", vocab_size=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
