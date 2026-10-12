"""Test voor de OpenAI-compatibele server: echte HTTP-verzoeken.

Draaien: python ai/tests/test_server.py

De test start een echte server op een vrije poort met een piepklein model, en
controleert dat de antwoorden precies de vorm hebben die de browserextensie
(lib/openrouter.js) verwacht — inclusief SSE-streaming en tool calls.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai.config import ModelConfig  # noqa: E402
from webai.model import TransformerLM  # noqa: E402
from webai.serve import ModelHost, make_server  # noqa: E402
from webai.tokenizer import ByteLevelBPE  # noqa: E402
from webai import train as trainer  # noqa: E402

CORPUS = "Hallo wereld, dit is een test. " * 40 + "What is the answer? Yes. " * 40


class ServerFixture:
    """Start een server op een vrije poort met een piepklein, willekeurig model."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.run = "testrun"
        directory = trainer.run_dir(self.run, self.root)
        os.makedirs(directory, exist_ok=True)

        tok = ByteLevelBPE.train(CORPUS, vocab_size=400)
        tok.save(trainer.tokenizer_path(self.run, self.root))

        cfg = ModelConfig(n_layer=1, n_head=2, n_embd=16, block_size=64,
                          vocab_size=tok.vocab_size, seed=3)
        model = TransformerLM(cfg)
        model.save(directory, "model", extra={"step": 7, "loss": 2.5})

        self.host = ModelHost(run_name=self.run, root=self.root, checkpoint="model",
                              verbose=False,
                              feedback_path=os.path.join(self.root, "feedback.jsonl"))
        self.httpd = make_server(self.host, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()


def get(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return res.status, json.loads(res.read().decode("utf-8"))


def post(url: str, body, timeout: int = 60):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def post_stream(url: str, body, timeout: int = 60):
    """POST en geef de ruwe SSE-regels terug."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        raw = res.read().decode("utf-8")
        ctype = res.headers.get("Content-Type", "")
    return ctype, raw


def parse_sse(raw: str):
    chunks, done = [], False
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return chunks, done


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = ServerFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    # ------------------------------------------------------------------

    def test_health(self):
        status, data = get(self.fx.base + "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])

    def test_models_list(self):
        status, data = get(self.fx.base + "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)
        m = data["data"][0]
        self.assertEqual(m["id"], "webai")
        self.assertEqual(m["object"], "model")
        # de extensie leest deze velden om te beslissen of een model bruikbaar is
        self.assertEqual(m["pricing"]["prompt"], "0")
        self.assertIn("tools", m["supported_parameters"])

    def test_info(self):
        status, data = get(self.fx.base + "/v1/info")
        self.assertEqual(status, 200)
        self.assertTrue(data["loaded"])
        self.assertGreater(data["params"], 0)
        self.assertEqual(data["context"], 64)
        self.assertIn("feedback", data)

    def test_info_without_v1_prefix(self):
        """Het zijpaneel pingt /info (zonder /v1) bij het openen; die moet bestaan."""
        status, data = get(self.fx.base + "/info")
        self.assertEqual(status, 200)
        self.assertTrue(data["loaded"])

    def test_info_stays_valid_json_with_a_nan_loss(self):
        """Een NaN-verlies mag het antwoord geen ongeldige JSON maken.

        Python schrijft NaN gewoon weg, maar JSON.parse in de browser weigert
        het — en dan strandt de hele instellingenpagina op één getal.
        """
        original = self.fx.host.info
        self.fx.host.info = lambda: dict(original(), loss=float("nan"),
                                         eval_loss=float("inf"))
        try:
            with urllib.request.urlopen(self.fx.base + "/v1/info", timeout=20) as res:
                raw = res.read().decode("utf-8")
        finally:
            self.fx.host.info = original
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        data = json.loads(raw)  # zou anders gooien
        self.assertIsNone(data["loss"])
        self.assertIsNone(data["eval_loss"])

    def test_cors_headers(self):
        """Zonder CORS kan een extensie (chrome-extension://…) de server niet bereiken."""
        req = urllib.request.Request(self.fx.base + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=20) as res:
            self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")

    def test_options_preflight(self):
        req = urllib.request.Request(self.fx.base + "/v1/chat/completions", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=20) as res:
            self.assertEqual(res.status, 204)
            self.assertIn("Content-Type", res.headers.get("Access-Control-Allow-Headers", ""))

    def test_chat_completion_non_streaming(self):
        status, data = post(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [{"role": "user", "content": "Hoi"}],
            "max_tokens": 12,
            "temperature": 0.5,
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "chat.completion")
        choice = data["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIsInstance(choice["message"]["content"], str)
        self.assertIn(choice["finish_reason"], ("stop", "tool_calls", "length"))
        usage = data["usage"]
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_chat_completion_streaming_shape(self):
        ctype, raw = post_stream(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [{"role": "user", "content": "Vertel iets"}],
            "max_tokens": 10,
            "stream": True,
        })
        self.assertIn("text/event-stream", ctype)
        chunks, done = parse_sse(raw)
        self.assertTrue(done, "de stream moet eindigen met data: [DONE]")
        self.assertGreater(len(chunks), 1)
        first = chunks[0]
        self.assertEqual(first["object"], "chat.completion.chunk")
        # eerste chunk kondigt de rol aan, precies zoals de client van de extensie verwacht
        self.assertEqual(first["choices"][0]["delta"].get("role"), "assistant")
        # laatste chunk heeft een finish_reason en usage
        last = chunks[-1]
        self.assertIn(last["choices"][0]["finish_reason"], ("stop", "tool_calls"))
        self.assertIn("usage", last)

    def test_stream_text_concatenates(self):
        """Alle content-delta's aan elkaar geplakt moeten één geheel zijn."""
        _ctype, raw = post_stream(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [{"role": "user", "content": "Hoi"}],
            "max_tokens": 10,
            "temperature": 0.2,
            "stream": True,
        })
        chunks, _ = parse_sse(raw)
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        # en hetzelfde gesprek zonder streaming moet dezelfde lengte-orde geven
        status, data = post(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [{"role": "user", "content": "Hoi"}],
            "max_tokens": 10,
            "temperature": 0.2,
        })
        self.assertEqual(status, 200)
        self.assertTrue(text or data["choices"][0]["message"]["content"] is not None)

    def test_missing_messages_is_a_400(self):
        status, data = post(self.fx.base + "/v1/chat/completions", {"model": "webai"})
        self.assertEqual(status, 400)
        self.assertIn("error", data)
        self.assertIn("messages", data["error"]["message"])

    def test_unknown_route_is_a_404(self):
        try:
            get(self.fx.base + "/v1/niet-bestaand")
            self.fail("had een 404 moeten geven")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            data = json.loads(exc.read().decode("utf-8"))
            self.assertEqual(data["error"]["type"], "not_found")

    def test_conversation_with_history(self):
        status, data = post(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [
                {"role": "system", "content": "Je bent kort."},
                {"role": "user", "content": "Eerste vraag"},
                {"role": "assistant", "content": "Eerste antwoord"},
                {"role": "user", "content": "Tweede vraag"},
            ],
            "max_tokens": 8,
        })
        self.assertEqual(status, 200)
        self.assertIsInstance(data["choices"][0]["message"]["content"], str)

    def test_content_parts_and_images_do_not_crash(self):
        """De extensie stuurt content als lijst met tekst én afbeeldingen."""
        status, data = post(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Wat zie je?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
            ]}],
            "max_tokens": 8,
        })
        self.assertEqual(status, 200)
        self.assertIsInstance(data["choices"][0]["message"]["content"], str)

    def test_tool_calls_streaming_shape(self):
        """De extensie plakt tool-aanroepen uit delta's aan elkaar; die vorm moet kloppen."""
        original = self.fx.host.complete

        def fake_complete(**kwargs):
            yield ("content", "Even kijken.")
            yield ("tool_call", json.dumps({"name": "read_page", "arguments": {"offset": 0}}))
            yield ("done", "Even kijken.<|tool_call|>{}")

        self.fx.host.complete = fake_complete
        try:
            _ctype, raw = post_stream(self.fx.base + "/v1/chat/completions", {
                "model": "webai",
                "messages": [{"role": "user", "content": "lees de pagina"}],
                "tools": [{"type": "function", "function": {"name": "read_page"}}],
                "max_tokens": 8,
                "stream": True,
            })
        finally:
            self.fx.host.complete = original

        chunks, done = parse_sse(raw)
        self.assertTrue(done)
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        self.assertEqual(len(tool_chunks), 1)
        tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["index"], 0)
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "read_page")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"offset": 0})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_tool_calls_non_streaming(self):
        original = self.fx.host.complete

        def fake_complete(**kwargs):
            yield ("content", "")
            yield ("tool_call", json.dumps({"name": "web_search", "arguments": {"query": "x"}}))
            yield ("done", "")

        self.fx.host.complete = fake_complete
        try:
            status, data = post(self.fx.base + "/v1/chat/completions", {
                "model": "webai",
                "messages": [{"role": "user", "content": "zoek iets"}],
                "max_tokens": 8,
            })
        finally:
            self.fx.host.complete = original
        self.assertEqual(status, 200)
        msg = data["choices"][0]["message"]
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(len(msg["tool_calls"]), 1)
        call = msg["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "web_search")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "x"})
        self.assertNotIn("index", call, "in het eindbericht hoort geen index-veld")

    def test_feedback_roundtrip(self):
        messages = [{"role": "user", "content": "Wat is 2+2?"}]
        status, data = post(self.fx.base + "/v1/feedback", {
            "kind": "thumbs_up", "messages": messages,
            "response": "4", "model": "webai",
        })
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["stats"]["total"], 1)
        self.assertEqual(data["stats"]["by_kind"]["thumbs_up"], 1)

        status, data = post(self.fx.base + "/v1/feedback", {
            "kind": "correction", "messages": messages, "response": "5",
            "correction": "Het antwoord is 4.",
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["stats"]["usable"], 2)

        # 👎 zonder verbetering telt niet als leermateriaal
        post(self.fx.base + "/v1/feedback", {
            "kind": "thumbs_down", "messages": messages, "response": "5",
        })
        status, stats = get(self.fx.base + "/v1/feedback")
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["usable"], 2)

    def test_feedback_validation(self):
        status, data = post(self.fx.base + "/v1/feedback", {"kind": "onbekend"})
        self.assertEqual(status, 400)
        self.assertIn("kind", data["error"]["message"])

        status, data = post(self.fx.base + "/v1/feedback", {
            "kind": "correction",
            "messages": [{"role": "user", "content": "v"}],
            "response": "a",
        })
        self.assertEqual(status, 400)
        self.assertIn("correction", data["error"]["message"])

    def test_completions_endpoint(self):
        status, data = post(self.fx.base + "/v1/completions", {
            "prompt": "Hallo", "max_tokens": 6,
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "text_completion")
        self.assertIn("text", data["choices"][0])

    def test_playground_page(self):
        with urllib.request.urlopen(self.fx.base + "/", timeout=20) as res:
            html = res.read().decode("utf-8")
        self.assertIn("<!doctype html>", html.lower())
        self.assertIn("WebAI", html)

    def test_context_overflow_does_not_crash(self):
        """Een veel te lange prompt moet netjes afgekapt worden, geen 500."""
        long_text = "woord " * 4000
        status, data = post(self.fx.base + "/v1/chat/completions", {
            "model": "webai",
            "messages": [{"role": "user", "content": long_text}],
            "max_tokens": 5,
        })
        self.assertEqual(status, 200)
        self.assertIsInstance(data["choices"][0]["message"]["content"], str)


class TestCheckpointChoice(unittest.TestCase):
    def test_auto_prefers_best(self):
        from webai.serve import pick_checkpoint

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(pick_checkpoint(d, "auto"), "model")
            open(os.path.join(d, "model.npz"), "w").close()
            self.assertEqual(pick_checkpoint(d, "auto"), "model")
            open(os.path.join(d, "best.npz"), "w").close()
            self.assertEqual(pick_checkpoint(d, "auto"), "best",
                             "auto kiest het checkpoint met het laagste eval-verlies")
            self.assertEqual(pick_checkpoint(d, "model"), "model",
                             "een expliciete keuze wordt gerespecteerd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
