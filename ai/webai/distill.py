"""Distilleren: laat een groot cloudmodel trainingsdata maken voor jóuw eigen model.

Een klein eigen model wordt nooit zo slim als GPT of Gemini — maar het wél veel
beter als het getraind is op antwoorden die precies bij jouw gebruik passen
(jouw vakken, jouw niveau, jouw toon). Dit script doet dat:

  1. het stelt een lijst vragen aan een cloudmodel (bijv. een gratis OpenRouter-model)
  2. het bewaart vraag + antwoord als `ai/data/learned/distill.jsonl`
  3. daarna train je je eigen model opnieuw: `python -m webai train`

Je hebt hier een API-sleutel voor nodig (of een lokale Ollama). Zonder sleutel
kun je ook gewoon eigen bestanden in `ai/data/learned/` zetten.

    python -m webai distill --limit 40
    python -m webai distill --questions mijn_vragen.txt --model google/gemini-3.5-flash
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from .config import LEARNED_DIR, ensure_dirs

DEFAULT_SYSTEM = (
    "Je maakt trainingsmateriaal voor een kleine, Nederlandstalige leerling-assistent "
    "voor de middelbare school. Antwoord duidelijk, correct en compact: korte uitleg, "
    "waar nodig een tussenstap of een tabel, in Markdown. Geen inleidende zin zoals "
    "'Natuurlijk!'. Begin meteen met het antwoord."
)

# Startvragen zodat `distill` meteen iets nuttigs doet zonder eigen bestand.
DEFAULT_QUESTIONS: List[str] = [
    "Leg uit wat een samenvatting is en hoe je er een maakt.",
    "Wat is de stelling van Pythagoras? Geef een rekenvoorbeeld.",
    "Hoe los je een kwadratische vergelijking op met de abc-formule?",
    "Wat is een afgeleide en waar gebruik je hem voor?",
    "Leg uit hoe je procenten uitrekent, met drie voorbeelden.",
    "Wat is het verschil tussen gemiddelde, mediaan en modus?",
    "Hoe schrijf je een betoog? Geef de opbouw.",
    "Wat zijn de vier Duitse naamvallen? Geef voorbeelden.",
    "Leg de passé composé uit in het Frans, inclusief wanneer je être gebruikt.",
    "Wat is fotosynthese? Geef de reactievergelijking.",
    "Leg de wet van Ohm uit met een rekenvoorbeeld.",
    "Hoe balanceer je een scheikundige vergelijking? Laat het zien.",
    "Wat waren de oorzaken van de Eerste Wereldoorlog?",
    "Wat waren de oorzaken van de Tweede Wereldoorlog?",
    "Leg vraag en aanbod uit en wat de evenwichtsprijs is.",
    "Wat is een cel en welke organellen heeft hij?",
    "Hoe werkt het broeikaseffect?",
    "Wat is een while-lus in Python? Geef code.",
    "Leg binair rekenen uit en reken 45 om naar binair.",
    "Wat is het verschil tussen weten en kennen in het Nederlands?",
    "Hoe citeer je een website in een bronnenlijst?",
    "Wat is plagiëren en hoe voorkom je het?",
    "Hoe leer je het beste voor een toets?",
    "Wat is een mindmap en hoe maak je die?",
    "Leg uit wat een werkwoordelijk en naamwoordelijk gezegde is.",
    "Wat zijn de regels voor d en t in het Nederlands?",
    "Leg uit wat een bijzin is en waar het werkwoord dan staat.",
    "Wat is de verschil tussen their, there en they're in het Engels?",
    "Leg de present perfect en de past simple uit in het Engels.",
    "Wat is een lineair verband en hoe stel je de formule op?",
    "Hoe reken je met machten en wortels? Geef de regels.",
    "Wat is de eenheidscirkel en hoe gebruik je die bij sinus en cosinus?",
    "Leg uit wat kansrekening is en hoe je een kans berekent.",
    "Wat is zwaartekracht en hoe reken je met de valversnelling?",
    "Leg uit wat snelheid, versnelling en een eenparige beweging zijn.",
    "Wat is het verschil tussen een element en een verbinding in de scheikunde?",
    "Leg uit wat een mol is en hoe je ermee rekent.",
    "Wat zijn de drie aggregatietoestanden en hoe heten de overgangen?",
    "Leg uit wat evolutie is en wat natuurlijke selectie betekent.",
    "Wat is het verschil tussen DNA en RNA?",
    "Leg uit hoe een ecosysteem werkt: producenten, consumenten, reducenten.",
    "Wat zijn de belangrijkste kenmerken van de Nederlandse grondwet?",
    "Leg uit wat de Europese Unie is en wat zij doet.",
    "Wat is globalisering? Noem voor- en nadelen.",
    "Leg uit wat inflatie is en hoe zij ontstaat.",
    "Wat is het verschil tussen bruto en netto inkomen?",
    "Hoe werkt een verkiezingsstelsel met evenredige vertegenwoordiging?",
    "Leg uit wat de Waterloopkundige en Deltawerken betekenen voor Nederland.",
    "Wat zijn de belangrijkste klimaatgebieden op aarde?",
    "Leg uit hoe het weer ontstaat: luchtdruk, wind en fronten.",
    "Geef vijf tips om een presentatie goed te houden.",
    "Hoe maak je een goede planning voor een werkweek?",
    "Leg uit wat bronnenkritiek is en hoe je een betrouwbare bron herkent.",
    "Wat is het verschil tussen een feit en een mening?",
    "Hoe schrijf je een formele e-mail? Geef een voorbeeld.",
]


def _load_questions(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_QUESTIONS)
    if not os.path.exists(path):
        raise SystemExit(f"Vragenbestand niet gevonden: {path}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if path.lower().endswith((".jsonl", ".ndjson")):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                out.append(line)
                continue
            if isinstance(rec, str):
                out.append(rec)
            elif isinstance(rec, dict):
                q = rec.get("question") or rec.get("prompt") or rec.get("instruction") or rec.get("text")
                if q:
                    out.append(str(q))
    elif path.lower().endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                q = item.get("question") or item.get("prompt") or item.get("text")
                if q:
                    out.append(str(q))
    else:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 900,
    timeout: int = 120,
) -> str:
    """Roep een OpenAI-compatibel /chat/completions aan en geef de tekst terug."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "X-Title": "WebAI distill"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        data = json.loads(res.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Geen choices in antwoord: {str(data)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"Leeg antwoord van {model}: {str(message)[:200]}")
    return str(content)


def distill(
    api_base: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
    model: str = "qwen/qwen3.8-27b:free",
    questions_file: Optional[str] = None,
    out: Optional[str] = None,
    limit: int = 40,
    system: Optional[str] = None,
    verbose: bool = True,
    delay: float = 1.5,
) -> int:
    """Haalt antwoorden op bij een cloudmodel en bewaart ze als trainingsdata."""
    ensure_dirs()
    os.makedirs(LEARNED_DIR, exist_ok=True)
    out_path = out or os.path.join(LEARNED_DIR, "distill.jsonl")

    if not api_key and "localhost" not in api_base and "127.0.0.1" not in api_base:
        print(
            "Geen API-sleutel opgegeven. Dat kan op twee manieren:\n"
            "  1. omgevingsvariabele:  export OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  2. optie:              --api-key sk-or-v1-...\n"
            "Zonder sleutel werkt distilleren alleen tegen een lokale server "
            "(bijv. Ollama op http://localhost:11434/v1).\n",
            file=sys.stderr,
        )
        return 2

    questions = _load_questions(questions_file)[: max(1, limit)]
    if verbose:
        print(f"[distill] {len(questions)} vragen via {model} op {api_base}")
        print(f"[distill] uitvoer: {out_path}")

    written = 0
    failed = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, question in enumerate(questions, start=1):
            try:
                answer = call_chat_api(
                    api_base, api_key, model,
                    [{"role": "system", "content": system or DEFAULT_SYSTEM},
                     {"role": "user", "content": question}],
                )
            except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, TimeoutError,
                    json.JSONDecodeError, OSError) as exc:
                failed += 1
                if verbose:
                    print(f"  {i:>3}/{len(questions)} MISLUKT: {exc}")
                if failed >= 5 and written == 0:
                    print("[distill] te veel fouten — controleer sleutel, modelnaam en "
                          "internettoegang.", file=sys.stderr)
                    return 1
                continue
            answer = answer.strip()
            if len(answer) < 20:
                failed += 1
                continue
            fh.write(json.dumps({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "_source": f"distill:{model}",
            }, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if verbose:
                print(f"  {i:>3}/{len(questions)} ok ({len(answer)} tekens)")
            time.sleep(max(0.0, delay))

    if verbose:
        print(f"\n[distill] {written} voorbeelden toegevoegd, {failed} mislukt.")
        if written:
            print("[distill] Bouw nu een nieuwe tokenizer en train opnieuw:")
            print("  python -m webai build")
            print("  python -m webai train --init-from webai")
    return 0 if written else 1
