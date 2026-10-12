"""Test voor het chatsjabloon en het teruglezen van tool-aanroepen.

Draaien: python ai/tests/test_chat.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from webai.chat import (  # noqa: E402
    content_to_text,
    describe_tools,
    parse_assistant_output,
    render_prompt,
    split_tool_calls,
)
from webai.tokenizer import EOT, TOOL_CALL, TURN  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Lees de tekst van de huidige pagina.",
            "parameters": {
                "type": "object",
                "properties": {"offset": {"type": "integer"}},
                "required": ["offset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Zoek op internet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


class TestRendering(unittest.TestCase):
    def test_simple_conversation(self):
        text = render_prompt([
            {"role": "system", "content": "Je bent aardig."},
            {"role": "user", "content": "Hoi"},
            {"role": "assistant", "content": "Hallo!"},
            {"role": "user", "content": "Hoe gaat het?"},
        ])
        self.assertIn(f"{TURN}system\nJe bent aardig.{EOT}", text)
        self.assertIn(f"{TURN}user\nHoi{EOT}", text)
        self.assertIn(f"{TURN}assistant\nHallo!{EOT}", text)
        self.assertTrue(text.endswith(f"{TURN}assistant\n"),
                        "de prompt moet eindigen met een open assistant-beurt")

    def test_no_generation_prompt(self):
        text = render_prompt([{"role": "user", "content": "Hoi"}], add_generation_prompt=False)
        self.assertFalse(text.endswith(f"{TURN}assistant\n"))

    def test_role_aliases(self):
        text = render_prompt([{"role": "human", "content": "vraag"}])
        self.assertIn(f"{TURN}user\nvraag{EOT}", text)
        text = render_prompt([{"role": "function", "content": "resultaat"}])
        self.assertIn(f"{TURN}tool\n", text)

    def test_tools_are_described(self):
        text = render_prompt([{"role": "user", "content": "lees de pagina"}], tools=TOOLS)
        self.assertIn(TOOL_CALL, text, "de uitleg moet het aanroepformaat laten zien")
        self.assertIn("read_page", text)
        self.assertIn("web_search", text)
        self.assertIn("offset", text)

    def test_describe_tools_marks_optional(self):
        desc = describe_tools(TOOLS)
        self.assertIn("offset:integer", desc)
        tools_optional = [{
            "type": "function",
            "function": {"name": "x", "description": "", "parameters": {
                "type": "object", "properties": {"y": {"type": "string"}}, "required": []}},
        }]
        self.assertIn("y?:string", describe_tools(tools_optional),
                      "niet-verplichte argumenten horen een ? te krijgen")

    def test_images_become_text_note(self):
        text, images = content_to_text([
            {"type": "text", "text": "Wat zie je?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ])
        self.assertEqual(text, "Wat zie je?")
        self.assertEqual(images, ["data:image/png;base64,AAAA"])
        prompt = render_prompt([{"role": "user", "content": [
            {"type": "text", "text": "Wat zie je?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}])
        self.assertIn("afbeelding #1", prompt)
        self.assertIn("kan nog geen plaatjes zien", prompt,
                      "eerlijk zijn: dit eigen model heeft geen ogen")

    def test_string_and_none_content(self):
        self.assertEqual(content_to_text("gewoon tekst"), ("gewoon tekst", []))
        self.assertEqual(content_to_text(None), ("", []))
        self.assertEqual(content_to_text(42), ("42", []))


class TestToolCallParsing(unittest.TestCase):
    def test_plain_text(self):
        content, calls = split_tool_calls("Gewoon een antwoord.")
        self.assertEqual(content, "Gewoon een antwoord.")
        self.assertEqual(calls, [])

    def test_single_call(self):
        raw = f'{TOOL_CALL}{{"name": "read_page", "arguments": {{"offset": 0}}}}'
        content, calls = split_tool_calls(raw)
        self.assertEqual(calls, [{"name": "read_page", "arguments": {"offset": 0}, "index": 0}])
        self.assertEqual(content, "")

    def test_text_then_call(self):
        raw = f'Even kijken.{TOOL_CALL}{{"name": "web_search", "arguments": {{"query": "pythagoras"}}}}'
        content, calls = split_tool_calls(raw)
        self.assertEqual(content, "Even kijken.")
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["arguments"], {"query": "pythagoras"})

    def test_two_calls_get_indices(self):
        raw = (f'{TOOL_CALL}{{"name": "a", "arguments": {{}}}}\n'
               f'{TOOL_CALL}{{"name": "b", "arguments": {{"x": 1}}}}')
        _, calls = split_tool_calls(raw)
        self.assertEqual([c["name"] for c in calls], ["a", "b"])
        self.assertEqual([c["index"] for c in calls], [0, 1])

    def test_nested_braces_in_arguments(self):
        raw = f'{TOOL_CALL}{{"name": "type_text", "arguments": {{"text": "{{haakjes}}", "id": 3}}}}'
        _, calls = split_tool_calls(raw)
        self.assertEqual(calls[0]["arguments"]["text"], "{haakjes}")
        self.assertEqual(calls[0]["arguments"]["id"], 3)

    def test_arguments_as_json_string(self):
        raw = f'{TOOL_CALL}{{"name": "read_page", "arguments": "{{\\"offset\\": 5}}"}}'
        _, calls = split_tool_calls(raw)
        self.assertEqual(calls[0]["arguments"], {"offset": 5})

    def test_broken_json_is_salvaged(self):
        # afgebroken tijdens genereren: de naam is er wel, de argumenten niet helemaal
        raw = f'{TOOL_CALL}{{"name": "web_search", "arguments": {{"query": "deltawer'
        _, calls = split_tool_calls(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "web_search")

    def test_nonsense_call_is_left_alone(self):
        raw = f"Ik noem {TOOL_CALL} even in een zin zonder JSON."
        content, calls = split_tool_calls(raw)
        self.assertEqual(calls, [])
        self.assertEqual(content, raw.strip())

    def test_parse_stops_at_eot(self):
        parsed = parse_assistant_output(f"Hallo!{EOT}\n{TURN}user\ndit mag niet mee")
        self.assertEqual(parsed["content"], "Hallo!")
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(parsed["finish_reason"], "stop")

    def test_parse_finish_reason_tool_calls(self):
        parsed = parse_assistant_output(
            f'{TOOL_CALL}{{"name": "read_page", "arguments": {{}}}}{EOT}'
        )
        self.assertEqual(parsed["finish_reason"], "tool_calls")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read_page")

    def test_tool_calls_to_text_roundtrip(self):
        from webai.chat import tool_calls_to_text

        calls = [{
            "id": "c1", "type": "function",
            "function": {"name": "read_page", "arguments": json.dumps({"offset": 3})},
        }]
        text = tool_calls_to_text(calls)
        self.assertTrue(text.startswith(TOOL_CALL))
        _, parsed = split_tool_calls(text)
        self.assertEqual(parsed[0]["arguments"], {"offset": 3})




class TestStreamingCompleteness(unittest.TestCase):
    """Tijdens streamen mag een halve aanroep niet al doorgespeeld worden."""

    def test_incomplete_call_is_withheld(self):
        partial = f'{TOOL_CALL}{{"name": "read_page", "arguments": {{"off'
        _content, calls = split_tool_calls(partial, require_complete=True)
        self.assertEqual(calls, [], "een halve JSON-aanroep mag nog niet mee")

    def test_same_call_is_accepted_once_closed(self):
        full = f'{TOOL_CALL}{{"name": "read_page", "arguments": {{"offset": 0}}}}'
        _content, calls = split_tool_calls(full, require_complete=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {"offset": 0})

    def test_growing_buffer_stream(self):
        """Simuleer token voor token: pas bij de sluithaak komt de aanroep vrij."""
        full = f'{TOOL_CALL}{{"name": "web_search", "arguments": {{"query": "dijken"}}}}'
        released_at = None
        for i in range(len(full) + 1):   # +1: ook de hele string meenemen
            _c, calls = split_tool_calls(full[:i], require_complete=True)
            if calls and released_at is None:
                released_at = i
        self.assertIsNotNone(released_at)
        self.assertEqual(released_at, len(full),
                         "de aanroep mag pas helemaal aan het eind vrijkomen")

    def test_without_require_complete_salvage_still_works(self):
        """Zonder de vlag blijft een kapotte aanroep gered worden (einde van de stream)."""
        partial = f'{TOOL_CALL}{{"name": "read_page", "arguments": {{"offset": 0'
        _content, calls = split_tool_calls(partial, require_complete=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_page")


if __name__ == "__main__":
    unittest.main(verbosity=2)
