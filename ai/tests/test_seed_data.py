"""Test voor de meegeleverde trainingsdata in ai/data/seed.

Draaien: python ai/tests/test_seed_data.py

Deze bestanden zijn met de hand geschreven en worden meegeleverd als het
startpunt van iemands eigen model. Een typefout of een kapotte regel zou
stilletjes in elke training terechtkomen, dus hier wordt de data zelf
gecontroleerd — niet alleen de code die hem leest.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai.config import PRESETS, configs_from_preset  # noqa: E402
from webai.dataset import build_dataset, dataset_tokens, record_to_messages  # noqa: E402
from webai.tokenizer import ByteLevelBPE  # noqa: E402

SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "seed")
VALID_ROLES = {"system", "user", "assistant", "tool"}


def seed_files():
    return sorted(glob.glob(os.path.join(SEED_DIR, "*.jsonl")))


def all_records():
    """Alle records uit alle seedbestanden, met vindplaats voor foutmeldingen."""
    out = []
    for path in seed_files():
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                out.append((os.path.basename(path), lineno, line))
    return out


class TestSeedFilesExist(unittest.TestCase):
    def test_directory_and_files_present(self):
        self.assertTrue(os.path.isdir(SEED_DIR), f"{SEED_DIR} ontbreekt")
        files = seed_files()
        self.assertGreaterEqual(len(files), 3, "er hoort meer dan een handvol databestanden te zijn")

    def test_files_are_numbered_for_reading_order(self):
        """De namen beginnen met een nummer, zodat de volgorde logisch is."""
        names = [os.path.basename(f) for f in seed_files()]
        for name in names:
            self.assertRegex(name, r"^\d\d_.+_nl\.jsonl$",
                             f"{name} volgt niet het schema NN_onderwerp_nl.jsonl")
        self.assertEqual(names, sorted(names), "bestanden horen op nummer gesorteerd te staan")


class TestRecordShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = all_records()

    def test_there_is_a_meaningful_amount_of_data(self):
        self.assertGreaterEqual(len(self.records), 300,
                                "de meegeleverde dataset hoort substantieel te zijn")

    def test_every_line_is_valid_json(self):
        for name, lineno, line in self.records:
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                self.fail(f"{name}:{lineno} is geen geldige JSON — {exc}")

    def test_every_record_has_messages(self):
        for name, lineno, line in self.records:
            rec = json.loads(line)
            msgs = rec.get("messages")
            self.assertIsInstance(msgs, list, f"{name}:{lineno} heeft geen berichtenlijst")
            self.assertGreaterEqual(len(msgs), 2, f"{name}:{lineno} heeft te weinig beurten")

    def test_roles_are_known(self):
        for name, lineno, line in self.records:
            for m in json.loads(line)["messages"]:
                self.assertIn(m.get("role"), VALID_ROLES,
                              f"{name}:{lineno} bevat een onbekende rol {m.get('role')!r}")

    def test_conversation_starts_with_user_or_system(self):
        for name, lineno, line in self.records:
            first = json.loads(line)["messages"][0]["role"]
            self.assertIn(first, ("user", "system"),
                          f"{name}:{lineno} begint met {first!r}")

    def test_every_conversation_has_an_assistant_answer(self):
        """Zonder antwoord valt er niets te leren."""
        for name, lineno, line in self.records:
            roles = [m["role"] for m in json.loads(line)["messages"]]
            self.assertIn("assistant", roles, f"{name}:{lineno} heeft geen antwoord")

    def test_conversation_ends_with_assistant(self):
        for name, lineno, line in self.records:
            last = json.loads(line)["messages"][-1]["role"]
            self.assertEqual(last, "assistant",
                             f"{name}:{lineno} eindigt met {last!r} in plaats van een antwoord")

    def test_content_is_non_empty_text(self):
        for name, lineno, line in self.records:
            for m in json.loads(line)["messages"]:
                content = m.get("content")
                self.assertIsInstance(content, str,
                                      f"{name}:{lineno} heeft geen tekst als inhoud")
                if m["role"] != "assistant" or content.strip():
                    self.assertTrue(content.strip(),
                                    f"{name}:{lineno} heeft een lege {m['role']}-beurt")

    def test_no_leading_or_trailing_whitespace_in_answers(self):
        for name, lineno, line in self.records:
            for m in json.loads(line)["messages"]:
                self.assertEqual(m["content"], m["content"].strip(),
                                 f"{name}:{lineno} heeft witruimte aan het begin of eind")


class TestRecordContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsed = []
        for name, lineno, line in all_records():
            cls.parsed.append((name, lineno, json.loads(line)))

    def test_questions_are_unique(self):
        """Dubbele vragen zouden het model onnodig in één richting duwen."""
        seen = {}
        for name, lineno, rec in self.parsed:
            first_user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            key = first_user.strip().lower().rstrip("?.!")
            if key in seen:
                self.fail(f"Dubbele vraag {first_user!r} in {name}:{lineno} "
                          f"en eerder in {seen[key]}")
            seen[key] = f"{name}:{lineno}"

    def test_answers_are_not_identical_everywhere(self):
        """Als elk antwoord hetzelfde is, leert het model alleen dat ene antwoord."""
        answers = [rec["messages"][-1]["content"] for _n, _l, rec in self.parsed]
        unique = set(answers)
        self.assertGreater(len(unique), len(answers) * 0.9,
                           "te veel identieke antwoorden")

    def test_answers_are_substantive(self):
        for name, lineno, rec in self.parsed:
            answer = rec["messages"][-1]["content"]
            self.assertGreater(len(answer), 1,
                               f"{name}:{lineno} heeft een leeg antwoord")
            self.assertLess(len(answer), 4000,
                            f"{name}:{lineno} heeft een extreem lang antwoord "
                            f"({len(answer)} tekens) dat niet in de context past")

    def test_records_convert_to_messages(self):
        """De datasetlaag moet elk record kunnen omzetten."""
        for name, lineno, rec in self.parsed:
            msgs = record_to_messages(rec)
            self.assertTrue(msgs, f"{name}:{lineno} wordt niet herkend als gesprek")
            self.assertEqual(msgs[-1]["role"], "assistant")

    def test_dutch_content_uses_dutch_punctuation(self):
        """Steekproef: de antwoorden horen Nederlands te zijn."""
        dutch_markers = ("de", "het", "een", "is", "van", "en", "je", "wat")
        hits = 0
        for _name, _lineno, rec in self.parsed:
            text = rec["messages"][-1]["content"].lower()
            if any(f" {w} " in f" {text} " for w in dutch_markers):
                hits += 1
        self.assertGreater(hits, len(self.parsed) * 0.6,
                           "minder dan 60% van de antwoorden lijkt Nederlands")

    def test_no_placeholder_text(self):
        for name, lineno, line in self.parsed:
            text = line["messages"][-1]["content"].lower()
            for marker in ("lorem ipsum", "todo", "xxx", "placeholder", "<insert"):
                self.assertNotIn(marker, text, f"{name}:{lineno} bevat {marker!r}")


class TestDatasetBuilds(unittest.TestCase):
    """De hele dataset moet zonder overslagen regels verwerkt kunnen worden."""

    @classmethod
    def setUpClass(cls):
        files = seed_files()
        with open(files[0], encoding="utf-8") as fh:
            text = fh.read()[:40000]
        cls.tok = ByteLevelBPE.train(text, vocab_size=1500, verbose=False)
        cls.files = files

    def test_builds_without_skipped_records(self):
        _samples, stats = build_dataset(self.tok, block_size=512,
                                        files=self.files, verbose=False)
        self.assertEqual(stats["skipped"], 0,
                         "geen enkel meegeleverd record mag overgeslagen worden")
        self.assertEqual(stats["files"], len(self.files))
        self.assertEqual(stats["chat"], len(all_records()))

    def test_yields_a_substantial_token_count(self):
        samples, _stats = build_dataset(self.tok, block_size=512,
                                        files=self.files, verbose=False)
        tokens = dataset_tokens(samples)
        self.assertGreater(tokens, 50000,
                           f"te weinig tokens om iets te leren: {tokens}")

    def test_every_sample_fits_the_largest_context(self):
        """Een sample dat langer is dan de context wordt afgekapt en verliest inhoud."""
        biggest = PRESETS["large"]["model"]["block_size"]
        samples, _stats = build_dataset(self.tok, block_size=biggest,
                                        files=self.files, verbose=False)
        for s in samples:
            self.assertLessEqual(len(s.ids), biggest)

    def test_small_context_still_keeps_learnable_content(self):
        """Ook bij de kleinste context moet er iets te leren overblijven."""
        smallest = PRESETS["tiny"]["model"]["block_size"]
        samples, _stats = build_dataset(self.tok, block_size=smallest,
                                        files=self.files, verbose=False)
        with_loss = [s for s in samples if any(m > 0 for m in s.mask)]
        self.assertEqual(len(with_loss), len(samples),
                         "elk sample moet minstens één token hebben dat meetelt")

    def test_works_with_every_preset_context(self):
        for name in PRESETS:
            mcfg, _tcfg = configs_from_preset(name)
            samples, _stats = build_dataset(self.tok, block_size=mcfg.block_size,
                                            files=self.files, verbose=False)
            self.assertGreater(len(samples), 0, f"{name} levert geen samples op")


class TestCoverage(unittest.TestCase):
    def test_several_subjects_are_covered(self):
        """Het startpakket hoort breed te zijn, niet één onderwerp."""
        names = " ".join(os.path.basename(f) for f in seed_files()).lower()
        for topic in ("basis", "school", "tools", "feiten", "rekenen", "taal"):
            self.assertIn(topic, names, f"het onderwerp {topic!r} ontbreekt in de seeddata")

    def test_files_are_balanced(self):
        """Eén bestand met 90% van de data zou alles scheeftrekken."""
        counts = []
        for path in seed_files():
            with open(path, encoding="utf-8") as fh:
                counts.append(sum(1 for line in fh if line.strip()))
        total = sum(counts)
        for path, n in zip(seed_files(), counts):
            share = n / total
            self.assertLess(share, 0.35,
                            f"{os.path.basename(path)} bevat {share:.0%} van alle records")
            self.assertGreaterEqual(n, 10,
                                    f"{os.path.basename(path)} bevat maar {n} records")


if __name__ == "__main__":
    unittest.main(verbosity=2)
