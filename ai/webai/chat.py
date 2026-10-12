"""Chatsjabloon: van OpenAI-achtige `messages` naar tokens en weer terug.

Het model leert één vast formaat (ChatML-achtig):

    <|fim_suffix|>system
    <systeemprompt><|fim_middle|>
    <|fim_suffix|>user
    <vraag><|fim_middle|>
    <|fim_suffix|>assistant
    <antwoord><|fim_middle|>

Een hulpmiddel-aanroep ziet er zo uit:

    <|fim_middle|>assistant
    <|tool_call|>{"name": "read_page", "arguments": {"offset": 0}}<|fim_middle|>

en het antwoord van dat hulpmiddel komt terug als:

    <|fim_suffix|>tool
    read_page -> <resultaat><|fim_middle|>

Daardoor heeft het eigen model precies dezelfde gereedschappen als een
cloudmodel, en hoeft de extensie niets te weten van het verschil.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .tokenizer import EOT, TOOL, TOOL_CALL, TURN, ByteLevelBPE

ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}

_TURN_RE = re.compile(re.escape(TURN) + r"\s*(\w+)\s*\n?", re.IGNORECASE)


def norm_role(role: str) -> str:
    return ROLE_ALIASES.get(str(role or "").strip().lower(), "user")


def content_to_text(content: Any) -> Tuple[str, List[str]]:
    """OpenAI-content mag een string zijn of een lijst onderdelen (tekst/afbeelding).

    Geeft (tekst, afbeeldingen) terug. Afbeeldingen worden als tekstbeschrijving
    meegenomen: dit eigen model heeft (nog) geen ogen.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        texts: List[str] = []
        images: List[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text" or "text" in part:
                texts.append(str(part.get("text", "")))
            elif kind in ("image_url", "image"):
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                images.append(str(url or ""))
            elif kind == "input_audio":
                texts.append("[audio]")
        return "\n".join(t for t in texts if t), images
    return str(content), []


def tool_calls_to_text(tool_calls: Any) -> str:
    """Zet `tool_calls` uit een assistant-bericht om naar ons eigen formaat."""
    if not tool_calls:
        return ""
    out: List[str] = []
    for call in tool_calls:
        fn = (call or {}).get("function", call) or {}
        name = fn.get("name") or fn.get("tool") or ""
        args = fn.get("arguments", fn.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if not name:
            continue
        out.append(f"{TOOL_CALL}{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}")
    return "\n".join(out)


def render_turn(role: str, body: str) -> str:
    return f"{TURN}{role}\n{body}{EOT}\n"


def render_prompt(
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    system_suffix: str = "",
) -> str:
    """Bouwt de volledige prompt-tekst voor een gesprek.

    `tools` (OpenAI-vorm) wordt als leesbare uitleg aan de systeembeurt
    toegevoegd, zodat een klein model snapt welke gereedschappen er zijn.
    """
    parts: List[str] = []
    system_chunks: List[str] = []
    pending: List[Dict[str, Any]] = []

    for msg in messages or []:
        role = norm_role((msg or {}).get("role"))
        text, images = content_to_text((msg or {}).get("content"))
        if role == "system":
            if text:
                system_chunks.append(text)
            continue
        if role == "assistant":
            tc = tool_calls_to_text((msg or {}).get("tool_calls"))
            body = (text + ("\n" if text and tc else "") + tc).strip()
            pending.append({"role": "assistant", "body": body or "(geen tekst)"})
        elif role == "tool":
            name = (msg or {}).get("name") or "tool"
            body = f"{name} -> {text}"
            pending.append({"role": "tool", "body": body})
        else:
            body = text
            if images:
                marks = "\n".join(f"[afbeelding #{i + 1} meegegeven: {describe_image(u)}]"
                                  for i, u in enumerate(images))
                body = (body + "\n\n" + marks).strip()
            pending.append({"role": "user", "body": body or "(leeg bericht)"})

    if system_chunks or tools or system_suffix:
        parts.append(render_turn("system", build_system_body(system_chunks, tools, system_suffix)))
    for turn in pending:
        parts.append(render_turn(turn["role"], turn["body"]))
    if add_generation_prompt:
        parts.append(f"{TURN}assistant\n")
    return "".join(parts)


def build_system_body(
    system_chunks: List[str],
    tools: Optional[List[Dict[str, Any]]],
    system_suffix: str,
) -> str:
    body = "\n\n".join(c for c in system_chunks if c)
    if tools:
        body = (body + "\n\n" if body else "") + describe_tools(tools)
    if system_suffix:
        body = (body + "\n\n" if body else "") + system_suffix
    return body.strip()


def describe_tools(tools: List[Dict[str, Any]]) -> str:
    """Maakt van de OpenAI-tooldefinities een korte, leesbare uitleg."""
    lines = ["Je hebt deze gereedschappen. Roep er één aan door op een nieuwe regel te schrijven:"]
    lines.append(f"{TOOL_CALL}" + '{"name": "<naam>", "arguments": { ... }}')
    lines.append("Schrijf daarna niets meer; je krijgt het resultaat te zien en mag dan antwoorden.")
    lines.append("")
    lines.append("Beschikbare gereedschappen:")
    for tool in tools:
        fn = (tool or {}).get("function", tool) or {}
        name = fn.get("name", "")
        if not name:
            continue
        desc = str(fn.get("description", "")).strip().replace("\n", " ")
        params = ((fn.get("parameters") or {}).get("properties")) or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        bits = []
        for pname, spec in params.items():
            ptype = (spec or {}).get("type", "")
            mark = "" if pname in required else "?"
            bits.append(f"{pname}{mark}:{ptype}")
        argtxt = f" ({', '.join(bits)})" if bits else " (geen argumenten)"
        lines.append(f"- {name}{argtxt}: {desc[:300]}")
    lines.append("")
    lines.append(
        f"Antwoord je zonder gereedschap, schrijf dan gewoon tekst en eindig met {EOT}."
    )
    return "\n".join(lines)


def describe_image(url: str) -> str:
    if url.startswith("data:"):
        head = url.split(",", 1)[0]
        kind = head[5:].split(";")[0] or "afbeelding"
        size = len(url)
        return f"{kind}, {size // 1024} kB base64 (dit eigen model kan nog geen plaatjes zien)"
    return url[:200]


# ---------------------------------------------------------------------------
# Uitvoer van het model teruglezen
# ---------------------------------------------------------------------------


def split_tool_calls(text: str, require_complete: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """Haalt `<|tool_call|>{json}` uit de gegenereerde tekst.

    Geeft (overgebleven tekst, [tool calls]) terug. Een tool call heeft de vorm
    {"name": ..., "arguments": {...}, "index": n} zodat de server er
    OpenAI-`tool_calls` van kan maken.

    Met `require_complete=True` wordt een nog niet afgesloten JSON-object
    genegeerd. Dat is nodig tijdens streamen: anders zou een halve aanroep
    al uitgevoerd worden met onvolledige argumenten.
    """
    calls: List[Dict[str, Any]] = []
    rest = text
    while TOOL_CALL in rest:
        head, tail = rest.split(TOOL_CALL, 1)
        payload, remainder, complete = _read_json_object(tail)
        if require_complete and not complete:
            # nog niet compleet: laat alles staan en wacht op meer tokens
            return text, calls
        parsed = None
        if payload:
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = _salvage_tool_call(payload)
        if isinstance(parsed, dict) and parsed.get("name"):
            args = parsed.get("arguments", parsed.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append({
                "name": str(parsed["name"]),
                "arguments": args if isinstance(args, dict) else {},
                "index": len(calls),
            })
            rest = (head + remainder).strip()
        else:
            # geen geldige aanroep: laat de tekst met rust en stop met zoeken
            rest = text
            break
    if not calls:
        return text.strip(), []
    return rest.strip(), calls


def _read_json_object(text: str) -> Tuple[str, str, bool]:
    """Leest het eerste {...} object uit `text`.

    Geeft (object, rest, compleet) terug. `compleet` is False als de
    sluithaak nog niet binnen is — dat gebeurt tijdens streamen, wanneer de
    aanroep token voor token binnenkomt.
    """
    start = text.find("{")
    if start < 0:
        return "", text, False
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], text[i + 1 :], True
    return text[start:], "", False  # onafgemaakt object (afgebroken tijdens genereren)


def _salvage_tool_call(payload: str) -> Optional[Dict[str, Any]]:
    """Redt wat er te redden valt van een halve/kapotte JSON-aanroep."""
    name_m = re.search(r'"name"\s*:\s*"([A-Za-z0-9_\-]+)"', payload)
    if not name_m:
        return None
    args_m = re.search(r'"arguments"\s*:\s*(\{.*)$', payload, re.DOTALL)
    args: Dict[str, Any] = {}
    if args_m:
        raw = args_m.group(1)
        obj, _, _ = _read_json_object(raw)
        try:
            loaded = json.loads(obj)
            if isinstance(loaded, dict):
                args = loaded
        except Exception:
            # Haal losse sleutel/waarde-paren uit het kapotte object. De regex
            # heeft drie groepen: sleutel, hele waarde, en de waarde zonder
            # aanhalingstekens.
            for key, value, unquoted in re.findall(
                r'"(\w+)"\s*:\s*("([^"]*)"|[-\d.]+|true|false|null)', raw
            ):
                try:
                    args[key] = json.loads(value)
                except Exception:
                    args[key] = unquoted
    return {"name": name_m.group(1), "arguments": args}


def parse_assistant_output(text: str, require_complete: bool = False) -> Dict[str, Any]:
    """Volledige parse: tekst + tool calls, en stop bij het eerste beurt-einde."""
    body = text
    if EOT in body:
        body = body.split(EOT, 1)[0]
    content, calls = split_tool_calls(body, require_complete=require_complete)
    return {
        "content": content,
        "tool_calls": calls,
        "finish_reason": "tool_calls" if calls else "stop",
    }


def encode_prompt(tok: ByteLevelBPE, prompt_text: str) -> List[int]:
    """Prompt -> id's, voorafgegaan door het gesprek-begin-token."""
    ids = [tok.bos_id] if tok.bos_id >= 0 else []
    ids.extend(tok.encode(prompt_text))
    return ids


def stop_tokens(tok: ByteLevelBPE) -> List[int]:
    return [tok.encode(t)[0] for t in (EOT, TOOL, TURN) if tok.encode(t)]
