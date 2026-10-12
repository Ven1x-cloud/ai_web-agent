"""Test voor de leerlus: feedback bewaren en omzetten naar trainingsdata.

Draaien: python ai/tests/test_feedback.py

Dit is de kern van "een eigen AI die je kunt trainen": wat de gebruiker goed-
of afkeurt in de extensie moet als trainingsmateriaal terugkomen. Daarom wordt
hier streng gecontroleerd welke feedback wél bruikbaar is.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai import feedback as fb  # noqa: E402
from webai.feedback import FeedbackStore, VALID_KINDS  # noqa: E402

USER = {"role": "user", "content": "Wat is de hoofdstad van Nederland?"}
ASSISTANT = {"role": "assistant", "content": "Rotterdam"}


def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestStoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "feedback.jsonl")
        self.store = FeedbackStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_thumbs_up_is_stored(self):
        entry = self.store.add({
            "kind": "thumbs_up", "messages": [USER],
            "response": "Amsterdam", "model": "webai",
        })
        self.assertEqual(entry["kind"], "thumbs_up")
        self.assertEqual(entry["response"], "Amsterdam")
        self.assertFalse(entry["exported"])
        self.assertTrue(entry["id"])
        self.assertEqual(len(read_jsonl(self.path)), 1)

    def test_kind_is_normalised(self):
        entry = self.store.add({
            "kind": "  THUMBS_UP ", "messages": [USER], "response": "Amsterdam",
        })
        self.assertEqual(entry["kind"], "thumbs_up")

    def test_all_four_kinds_are_accepted(self):
        for kind in VALID_KINDS:
            record = {"kind": kind, "messages": [USER], "response": "Amsterdam"}
            if kind == "correction":
                record["correction"] = "Amsterdam"
            entry = self.store.add(record)
            self.assertEqual(entry["kind"], kind)
        self.assertEqual(len(read_jsonl(self.path)), len(VALID_KINDS))

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.store.add({"kind": "super", "messages": [USER], "response": "x"})
        self.assertIn("kind", str(ctx.exception))
        self.assertEqual(read_jsonl(self.path), [], "een geweigerde melding mag niet op schijf")

    def test_correction_without_correction_field_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.store.add({"kind": "correction", "messages": [USER], "response": "x"})
        self.assertIn("correction", str(ctx.exception))

    def test_thumbs_needs_a_response(self):
        for kind in ("thumbs_up", "thumbs_down"):
            with self.assertRaises(ValueError):
                self.store.add({"kind": kind, "messages": [USER], "response": "   "})

    def test_id_is_stable_for_the_same_conversation(self):
        a = self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        b = self.store.add({"kind": "thumbs_down", "messages": [USER], "response": "Amsterdam"})
        self.assertEqual(a["id"], b["id"],
                         "zelfde gesprek →zelfde id, zodat markeren als geëxporteerd werkt")
        c = self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Utrecht"})
        self.assertNotEqual(a["id"], c["id"], "ander antwoord → ander id")

    def test_answer_field_is_an_alias_for_response(self):
        entry = self.store.add({"kind": "thumbs_up", "messages": [USER], "answer": "Amsterdam"})
        self.assertEqual(entry["response"], "Amsterdam")

    def test_missing_messages_does_not_crash(self):
        entry = self.store.add({"kind": "note", "note": "gewoon een notitie"})
        self.assertEqual(entry["messages"], [])
        self.assertEqual(entry["note"], "gewoon een notitie")

    def test_all_reads_back_everything(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.add({"kind": "note", "note": "leuk"})
        entries = self.store.all()
        self.assertEqual(len(entries), 2)
        self.assertEqual([e["kind"] for e in entries], ["thumbs_up", "note"])

    def test_all_survives_a_broken_line(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("{ kapot\n")
        self.store.add({"kind": "note", "note": "tweede"})
        entries = self.store.all()
        self.assertEqual(len(entries), 2, "een kapotte regel mag de rest niet verpesten")


class TestStatsAndPending(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(os.path.join(self.tmp.name, "feedback.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats_counts_by_kind(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Utrecht"})
        self.store.add({"kind": "thumbs_down", "messages": [USER], "response": "Rotterdam"})
        self.store.add({"kind": "note", "note": "hmm"})
        stats = self.store.stats()
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["by_kind"]["thumbs_up"], 2)
        self.assertEqual(stats["by_kind"]["thumbs_down"], 1)
        self.assertEqual(stats["by_kind"]["correction"], 0)

    def test_usable_counts_only_learning_material(self):
        """Alleen 👍 en verbeteringen leren het model iets; 👎 zonder tekst niet."""
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.add({"kind": "correction", "messages": [USER], "response": "Rotterdam",
                        "correction": "Amsterdam"})
        self.store.add({"kind": "thumbs_down", "messages": [USER], "response": "Rotterdam"})
        self.store.add({"kind": "note", "note": "niks"})
        self.assertEqual(self.store.stats()["usable"], 2)

    def test_pending_excludes_exported(self):
        first = self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "A"})
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "B"})
        self.assertEqual(len(self.store.pending()), 2)
        self.store.mark_exported([first["id"]])
        pending = self.store.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["response"], "B")

    def test_mark_exported_persists_to_disk(self):
        entry = self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "A"})
        self.store.mark_exported([entry["id"]])
        # een nieuwe store op hetzelfde bestand moet het ook zien
        again = FeedbackStore(self.store.path)
        self.assertEqual(again.pending(), [])
        self.assertTrue(again.all()[0]["exported"])

    def test_mark_exported_ignores_unknown_ids(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "A"})
        self.store.mark_exported(["bestaat-niet"])
        self.assertEqual(len(self.store.pending()), 1)


class TestToTrainingRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(os.path.join(self.tmp.name, "feedback.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_thumbs_up_becomes_a_training_sample(self):
        entry = self.store.add({"kind": "thumbs_up", "messages": [USER],
                                "response": "Amsterdam"})
        records = self.store.to_training_records([entry])
        self.assertEqual(len(records), 1)
        msgs = records[0]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[-1]["content"], "Amsterdam")
        self.assertEqual(records[0]["_source"], "feedback")
        self.assertEqual(records[0]["_kind"], "thumbs_up")

    def test_correction_replaces_the_wrong_answer(self):
        """Bij een verbetering leert het model de verbetering, niet het foute antwoord."""
        entry = self.store.add({"kind": "correction", "messages": [USER],
                                "response": "Rotterdam", "correction": "Amsterdam"})
        records = self.store.to_training_records([entry])
        self.assertEqual(records[0]["messages"][-1]["content"], "Amsterdam")

    def test_thumbs_down_is_not_learning_material(self):
        entry = self.store.add({"kind": "thumbs_down", "messages": [USER],
                                "response": "Rotterdam"})
        self.assertEqual(self.store.to_training_records([entry]), [],
                         "een 👎 zonder verbetering zegt alleen wat het NIET moet zijn")

    def test_note_is_not_learning_material(self):
        entry = self.store.add({"kind": "note", "note": "interessant"})
        self.assertEqual(self.store.to_training_records([entry]), [])

    def test_trailing_assistant_turn_is_dropped(self):
        """De geschiedenis eindigt vaak op het antwoord zelf; dat moet eraf."""
        entry = self.store.add({
            "kind": "correction", "response": "Rotterdam", "correction": "Amsterdam",
            "messages": [USER, ASSISTANT],
        })
        records = self.store.to_training_records([entry])
        msgs = records[0]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[-1]["content"], "Amsterdam")
        self.assertNotIn("Rotterdam", [m["content"] for m in msgs])

    def test_trailing_tool_turns_are_dropped(self):
        entry = self.store.add({
            "kind": "thumbs_up", "response": "De pagina gaat over dijken.",
            "messages": [USER,
                         {"role": "assistant", "content": ""},
                         {"role": "tool", "content": "read_page -> dijken"}],
        })
        records = self.store.to_training_records([entry])
        self.assertEqual([m["role"] for m in records[0]["messages"]], ["user", "assistant"])

    def test_needs_a_user_turn(self):
        """Zonder vraag is er niets om aan te leren."""
        entry = self.store.add({"kind": "thumbs_up", "messages": [], "response": "Amsterdam"})
        self.assertEqual(self.store.to_training_records([entry]), [])

    def test_multi_turn_history_is_kept(self):
        entry = self.store.add({
            "kind": "thumbs_up", "response": "Vier",
            "messages": [{"role": "system", "content": "Je bent kort."},
                         {"role": "user", "content": "Reken eens"},
                         {"role": "assistant", "content": "Zeker"},
                         {"role": "user", "content": "Wat is 2+2?"}],
        })
        records = self.store.to_training_records([entry])
        roles = [m["role"] for m in records[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant"])

    def test_content_parts_are_flattened_to_text(self):
        """De extensie stuurt content soms als lijst met tekst en afbeeldingen."""
        entry = self.store.add({
            "kind": "thumbs_up", "response": "Een kat.",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Wat zie je?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ]}],
        })
        records = self.store.to_training_records([entry])
        self.assertEqual(records[0]["messages"][0]["content"], "Wat zie je?")

    def test_unknown_roles_are_dropped(self):
        entry = self.store.add({
            "kind": "thumbs_up", "response": "Oké",
            "messages": [{"role": "wizard", "content": "geheim"}, USER],
        })
        records = self.store.to_training_records([entry])
        roles = [m["role"] for m in records[0]["messages"]]
        self.assertEqual(roles, ["user", "assistant"])

    def test_empty_messages_are_dropped(self):
        entry = self.store.add({
            "kind": "thumbs_up", "response": "Oké",
            "messages": [{"role": "user", "content": "   "}, USER],
        })
        records = self.store.to_training_records([entry])
        self.assertEqual(len(records[0]["messages"]), 2)

    def test_garbage_records_are_skipped(self):
        skipped = self.store.to_training_records([
            {"kind": "thumbs_up", "messages": "geen lijst", "response": "x"},
            {"kind": "onbekend", "messages": [USER], "response": "x"},
            {"kind": "correction", "messages": [USER], "response": "x", "correction": "  "},
            {"geen": "record"},
        ])
        self.assertEqual(skipped, [], "rommel mag geen trainingsdata worden")

    def test_records_are_directly_usable_als_dataset(self):
        """De uitvoer moet door build_dataset geaccepteerd worden."""
        from webai.dataset import record_to_messages
        from webai.tokenizer import ByteLevelBPE

        entry = self.store.add({"kind": "correction", "messages": [USER],
                                "response": "Rotterdam", "correction": "Amsterdam"})
        record = self.store.to_training_records([entry])[0]
        msgs = record_to_messages(record)
        self.assertEqual(msgs[-1]["content"], "Amsterdam")
        tok = ByteLevelBPE.train("Wat is de hoofdstad van Nederland? Amsterdam", vocab_size=300)
        from webai.dataset import build_chat_sample
        sample = build_chat_sample(tok, msgs, block_size=128)
        self.assertIsNotNone(sample)
        self.assertTrue(any(m > 0 for m in sample.mask))


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(os.path.join(self.tmp.name, "feedback.jsonl"))
        self.learned = os.path.join(self.tmp.name, "learned")
        # de export schrijft naar LEARNED_DIR; leid dat om naar de tijdelijke map
        self._old = fb.LEARNED_DIR
        fb.LEARNED_DIR = self.learned

    def tearDown(self):
        fb.LEARNED_DIR = self._old
        self.tmp.cleanup()

    def test_export_writes_jsonl(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.add({"kind": "correction", "messages": [USER], "response": "Rotterdam",
                        "correction": "Amsterdam"})
        count = self.store.export_training_data("webai")
        self.assertEqual(count, 2)
        path = os.path.join(self.learned, "feedback_webai.jsonl")
        records = read_jsonl(path)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["messages"][-1]["role"] == "assistant" for r in records))

    def test_export_marks_entries_as_exported(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.export_training_data("webai")
        self.assertEqual(self.store.pending(), [], "geëxporteerd is niet meer in afwachting")

    def test_export_twice_does_not_duplicate(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.assertEqual(self.store.export_training_data("webai"), 1)
        self.assertEqual(self.store.export_training_data("webai"), 0,
                         "tweede keer: niets nieuws te exporteren")
        path = os.path.join(self.learned, "feedback_webai.jsonl")
        self.assertEqual(len(read_jsonl(path)), 1, "en het bestand groeit niet dubbel")

    def test_export_appends_new_feedback_to_the_same_file(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.export_training_data("webai")
        self.store.add({"kind": "correction", "messages": [USER], "response": "X",
                        "correction": "Utrecht"})
        self.assertEqual(self.store.export_training_data("webai"), 1)
        path = os.path.join(self.learned, "feedback_webai.jsonl")
        records = read_jsonl(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["messages"][-1]["content"], "Utrecht")

    def test_export_without_usable_feedback_writes_nothing(self):
        self.store.add({"kind": "thumbs_down", "messages": [USER], "response": "Rotterdam"})
        self.store.add({"kind": "note", "note": "hmm"})
        count = self.store.export_training_data("webai")
        self.assertEqual(count, 0)
        self.assertFalse(os.path.exists(os.path.join(self.learned, "feedback_webai.jsonl")),
                         "leeg bestand zou bij het trainen als databron meegeteld worden")

    def test_export_all_ignores_the_exported_flag(self):
        self.store.add({"kind": "thumbs_up", "messages": [USER], "response": "Amsterdam"})
        self.store.export_training_data("webai")
        count = self.store.export_training_data("webai", only_pending=False)
        self.assertEqual(count, 1, "only_pending=False exporteert alles opnieuw")

    def test_exported_file_is_valid_training_input(self):
        from webai.dataset import build_dataset
        from webai.tokenizer import ByteLevelBPE

        self.store.add({"kind": "correction", "messages": [USER],
                        "response": "Rotterdam", "correction": "Amsterdam"})
        self.store.export_training_data("webai")
        path = os.path.join(self.learned, "feedback_webai.jsonl")
        tok = ByteLevelBPE.train(
            "Wat is de hoofdstad van Nederland? Amsterdam Rotterdam Utrecht", vocab_size=400)
        samples, stats = build_dataset(tok, block_size=64, files=[path], verbose=False)
        self.assertGreater(len(samples), 0, "de export moet trainbare samples opleveren")
        self.assertEqual(stats["chat"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
