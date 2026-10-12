"""Leren van wat de gebruiker vindt: 👍, 👎 en verbeteringen.

De extensie stuurt feedback naar `POST /v1/feedback`. Die wordt bewaard in
`ai/data/feedback.jsonl`. Bij de volgende training (`POST /v1/train` of
`python -m webai train`) wordt daar automatisch leermateriaal van gemaakt in
`ai/data/learned/`.

Wat er gebeurt met elk soort feedback:

  thumbs_up     het gesprek wordt toegevoegd als voorbeeld → het model doet het
                vaker zo
  thumbs_down   alleen het slechte antwoord wordt *niet* geleerd; zonder
                verbetering is er niets om van te leren, dus die wordt bewaard
                voor later (en geteld in de statistieken)
  correction    jóuw verbeterde antwoord wordt het voorbeeld → het sterkste
                signaal dat er is
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from .config import FEEDBACK_PATH, LEARNED_DIR, ensure_dirs

VALID_KINDS = ("thumbs_up", "thumbs_down", "correction", "note")


def _conversation_id(messages: Optional[List[Dict[str, Any]]], response: str) -> str:
    blob = json.dumps({"m": messages or [], "r": response}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class FeedbackStore:
    """Bewaart feedback en zet hem om in trainingsdata."""

    def __init__(self, path: str = FEEDBACK_PATH) -> None:
        self.path = path
        ensure_dirs()

    # ------------------------------------------------------------------
    # Bewaren
    # ------------------------------------------------------------------

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        kind = str(record.get("kind") or "").strip().lower()
        if kind not in VALID_KINDS:
            raise ValueError(
                f"kind moet één van {', '.join(VALID_KINDS)} zijn, kreeg '{kind}'"
            )
        messages = record.get("messages")
        if not isinstance(messages, list):
            messages = []
        response = str(record.get("response") or record.get("answer") or "")
        correction = str(record.get("correction") or "")

        if kind == "correction" and not correction.strip():
            raise ValueError("Bij een verbetering hoort het veld 'correction'.")
        if kind in ("thumbs_up", "thumbs_down") and not response.strip():
            raise ValueError("Zonder antwoord (veld 'response') is er niets om feedback op te geven.")

        entry = {
            "id": _conversation_id(messages, response),
            "kind": kind,
            "time": time.time(),
            "messages": messages,
            "response": response,
            "correction": correction,
            "note": str(record.get("note") or ""),
            "model": str(record.get("model") or ""),
            "exported": False,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def pending(self) -> List[Dict[str, Any]]:
        return [e for e in self.all() if not e.get("exported")]

    def mark_exported(self, ids: List[str]) -> None:
        """Zet de zojuist verwerkte feedback op 'exported' (herschrijft het bestand)."""
        if not ids:
            return
        wanted = set(ids)
        entries = self.all()
        for entry in entries:
            if entry.get("id") in wanted:
                entry["exported"] = True
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def stats(self) -> Dict[str, Any]:
        entries = self.all()
        by_kind: Dict[str, int] = {k: 0 for k in VALID_KINDS}
        for e in entries:
            k = e.get("kind")
            if k in by_kind:
                by_kind[k] += 1
        usable = sum(1 for e in entries if e.get("kind") in ("thumbs_up", "correction"))
        return {
            "total": len(entries),
            "by_kind": by_kind,
            "usable": usable,
            "pending": sum(1 for e in entries if not e.get("exported")
                           and e.get("kind") in ("thumbs_up", "correction")),
            "path": self.path,
        }

    # ------------------------------------------------------------------
    # Omzetten naar trainingsdata
    # ------------------------------------------------------------------

    def to_training_records(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Maakt van feedback gesprekken in het trainingsformaat."""
        out: List[Dict[str, Any]] = []
        for entry in entries:
            kind = entry.get("kind")
            if kind not in ("thumbs_up", "correction"):
                continue
            messages: List[Dict[str, Any]] = []
            for m in entry.get("messages") or []:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").lower()
                content = m.get("content")
                if role not in ("system", "user", "assistant", "tool"):
                    continue
                if isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    text = " ".join(
                        str(p.get("text", "")) for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
                    if text:
                        messages.append({"role": role, "content": text})
            # Antwoord dat we willen aanleren. Bij een verbetering is dat alléén
            # de verbetering: terugvallen op het oorspronkelijke antwoord zou het
            # model precies datgene aanleren wat de gebruiker net corrigeerde.
            correction_text = (entry.get("correction") or "").strip()
            response_text = (entry.get("response") or "").strip()
            if kind == "correction":
                good = correction_text
            else:
                good = response_text or correction_text
            if not good:
                continue
            # Een eerder assistant-antwoord in de geschiedenis mag blijven staan,
            # maar het gesprek moet eindigen op een user-beurt gevolgd door het
            # goedgekeurde antwoord.
            while messages and messages[-1]["role"] in ("assistant", "tool"):
                messages.pop()
            if not messages or messages[-1]["role"] != "user":
                continue
            messages.append({"role": "assistant", "content": good})
            out.append({
                "messages": messages,
                "_source": "feedback",
                "_kind": kind,
                "_id": entry.get("id"),
            })
        return out

    def export_training_data(self, run_name: str = "webai",
                             only_pending: bool = True) -> int:
        """Schrijf bruikbare feedback weg als trainingsdata. Geeft het aantal records."""
        ensure_dirs()
        os.makedirs(LEARNED_DIR, exist_ok=True)
        entries = self.pending() if only_pending else self.all()
        records = self.to_training_records(entries)
        if not records:
            return 0
        path = os.path.join(LEARNED_DIR, f"feedback_{run_name}.jsonl")
        existing: List[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                existing = [l for l in fh.read().splitlines() if l.strip()]
        with open(path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.mark_exported([r["_id"] for r in records if r.get("_id")])
        return len(records)
