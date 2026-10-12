"""OpenAI-compatibele server voor het eigen model.

Daardoor praat de browserextensie met dit lokale model precies zoals met
OpenRouter, Ollama of LM Studio: dezelfde `/v1/chat/completions`, met streaming
en tool calls. Geen aanpassingen in de extensie nodig.

    python -m webai serve            # http://localhost:8787
    python -m webai serve --port 9000

Extra's bovenop de OpenAI-routes (voor de extensie en het trainen):

    GET  /v1/info       statistieken van het model (parameters, verlies, stappen)
    POST /v1/feedback   👍/👎 of een verbetering uit de extensie -> leermateriaal
    POST /v1/train      een training starten/voortzetten (draait op de achtergrond)
    GET  /v1/status     hoe ver de training is
    GET  /              een klein test-paneel in de browser
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import train as trainer
from .chat import EOT, TOOL, TOOL_CALL, TURN, parse_assistant_output, render_prompt
from .config import GenConfig, RUNS_DIR, load_json
from .feedback import FeedbackStore
from .generate import Generator, sampling_params
from .model import TransformerLM
from .tokenizer import ByteLevelBPE

DEFAULT_PORT = 8787
DEFAULT_HOST = "127.0.0.1"
MODEL_NAME = "webai"
# "auto": gebruik het checkpoint met het laagste eval-verlies als dat bestaat.
# Dat is bij een kleine dataset bijna altijd beter dan het laatste checkpoint,
# want dat heeft de voorbeelden uit zijn hoofd geleerd.
DEFAULT_CHECKPOINT = "auto"


def pick_checkpoint(directory: str, wanted: str) -> str:
    if wanted and wanted != "auto":
        return wanted
    best = os.path.join(directory, "best.npz")
    return "best" if os.path.exists(best) else "model"


# ---------------------------------------------------------------------------
# Model-gastheer: laadt, bewaakt en ververst het model
# ---------------------------------------------------------------------------


class ModelHost:
    """Houdt het model + de tokenizer in het geheugen, met een slot erop."""

    def __init__(self, run_name: str = MODEL_NAME, root: str = RUNS_DIR,
                 checkpoint: str = "model", verbose: bool = True,
                 feedback_path: Optional[str] = None) -> None:
        self.run_name = run_name
        self.root = root
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.lock = threading.Lock()
        self.model: Optional[TransformerLM] = None
        self.tokenizer: Optional[ByteLevelBPE] = None
        self.meta: Dict[str, Any] = {}
        self.loaded_at: Optional[float] = None
        self.feedback = FeedbackStore(feedback_path) if feedback_path else FeedbackStore()
        self.train_thread: Optional[threading.Thread] = None
        self.train_status: Dict[str, Any] = {"state": "idle"}

    @property
    def directory(self) -> str:
        return trainer.run_dir(self.run_name, self.root)

    def load(self, force: bool = False) -> None:
        with self.lock:
            self._load_locked(force)

    def _load_locked(self, force: bool = False) -> None:
        directory = self.directory
        # "auto" blijft een beleid en wordt elke keer opnieuw gekozen, zodat een
        # later weggeschreven 'best' alsnog gebruikt wordt.
        name = pick_checkpoint(directory, self.checkpoint)
        npz = os.path.join(directory, f"{name}.npz")
        if not os.path.exists(npz):
            raise FileNotFoundError(
                f"Geen model gevonden in {directory}. Train er eerst één:  "
                f"python -m webai build && python -m webai train --preset tiny"
            )
        if not force and self.model is not None and getattr(self, "_loaded_name", None) == name:
            mtime = os.path.getmtime(npz)
            if self.loaded_at and mtime <= self.loaded_at:
                return
        t0 = time.time()
        model, meta = TransformerLM.load(directory, name)
        tok = trainer.load_tokenizer(self.run_name, self.root)
        model.params = {k: v.astype(np.float32, copy=False) for k, v in model.params.items()}
        model.dtype = np.dtype(np.float32)
        self.model = model
        self.tokenizer = tok
        self.meta = meta or {}
        self.loaded_at = time.time()
        self._loaded_name = name
        if self.verbose:
            print(f"[serve] model '{self.run_name}' (checkpoint '{name}') geladen: "
                  f"{model.num_params() / 1e6:.2f}M parameters, woordenschat {tok.vocab_size}, "
                  f"context {model.cfg.block_size} ({time.time() - t0:.1f}s)")

    def info(self) -> Dict[str, Any]:
        info = trainer.model_info(self.run_name, self.root)
        loaded = self.model is not None
        info.update({
            "loaded": loaded,
            "model_name": MODEL_NAME,
            "run_name": self.run_name,
            "params": (self.model.num_params() if loaded else info.get("params")) or 0,
            "vocab": (self.tokenizer.vocab_size if self.tokenizer else info.get("vocab")) or 0,
            "context": (self.model.cfg.block_size if loaded else info.get("context")) or 0,
            "layers": (self.model.cfg.n_layer if loaded else info.get("layers")),
            "width": (self.model.cfg.n_embd if loaded else info.get("width")),
            "train_status": self.train_status,
            "feedback": self.feedback.stats(),
        })
        return info

    # ------------------------------------------------------------------
    # Antwoord maken
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        gen: Optional[GenConfig] = None,
        seed: Optional[int] = None,
        system_suffix: str = "",
    ) -> Iterator[Tuple[str, str]]:
        """Stroomt (soort, tekst)-paren: soort is 'content' of 'tool_call' of 'done'."""
        gen = gen or GenConfig()
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None and self.tokenizer is not None
        tok = self.tokenizer

        prompt_text = render_prompt(messages, add_generation_prompt=True, tools=tools,
                                    system_suffix=system_suffix)
        ids = [tok.bos_id] + tok.encode(prompt_text)
        if len(ids) >= self.model.cfg.block_size:
            # prompt past niet in de context: houd het systeemdeel en het einde
            head = min(len(ids), self.model.cfg.block_size // 3)
            ids = ids[:head] + ids[-(self.model.cfg.block_size - head - 8) :]

        with self.lock:
            self._load_locked()  # ververst als er net getraind is
            generator = Generator(self.model, ids, cfg=gen, seed=seed)
            stop_ids = [tok.encode(t)[0] for t in (EOT, TURN, TOOL)]

            emitted = ""
            produced: List[int] = []
            in_tool_call = False
            tool_buffer = ""

            for token in generator.stream(extra_stop_ids=stop_ids):
                produced.append(token)
                text = tok.decode(produced)
                if not in_tool_call and TOOL_CALL in text:
                    # alles vóór de markering is gewoon antwoordtekst
                    before = text.split(TOOL_CALL, 1)[0]
                    if len(before) > len(emitted):
                        yield ("content", before[len(emitted) :])
                        emitted = before
                    in_tool_call = True
                    tool_buffer = text.split(TOOL_CALL, 1)[1]
                    continue
                if in_tool_call:
                    tool_buffer = text.split(TOOL_CALL, 1)[1] if TOOL_CALL in text else text
                    parsed, _ = _try_tool_call(tool_buffer)
                    if parsed:
                        yield ("tool_call", json.dumps(parsed, ensure_ascii=False))
                        tool_buffer = ""
                        in_tool_call = False
                        emitted = text
                    continue
                if len(text) > len(emitted):
                    yield ("content", text[len(emitted) :])
                    emitted = text

            if in_tool_call and tool_buffer:
                parsed, _ = _try_tool_call(tool_buffer)
                if parsed:
                    yield ("tool_call", json.dumps(parsed, ensure_ascii=False))

        full = tok.decode(produced)
        yield ("done", full)

    def complete_sync(self, **kwargs: Any) -> Dict[str, Any]:
        content = ""
        tool_calls: List[Dict[str, Any]] = []
        raw = ""
        for kind, payload in self.complete(**kwargs):
            if kind == "content":
                content += payload
            elif kind == "tool_call":
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "index": len(tool_calls),
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                    },
                })
            elif kind == "done":
                raw = payload
        return {
            "content": content.strip(),
            "tool_calls": tool_calls,
            "raw": raw,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }

    # ------------------------------------------------------------------
    # Trainen op de achtergrond
    # ------------------------------------------------------------------

    def start_training(self, preset: str = "tiny", steps: Optional[int] = None,
                       resume: bool = True, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.train_thread and self.train_thread.is_alive():
            return {"started": False, "reason": "Er draait al een training."}

        self.train_status = {
            "state": "starting", "preset": preset, "steps": steps,
            "started_at": time.time(), "step": 0, "loss": None, "message": "",
        }

        def run() -> None:
            try:
                overrides: Dict[str, Any] = {}
                if steps:
                    overrides["max_steps"] = int(steps)
                overrides["run_name"] = self.run_name
                overrides.update(extra or {})

                def on_progress(payload: Dict[str, Any]) -> None:
                    self.train_status.update({
                        "state": "training",
                        "step": payload.get("step"),
                        "loss": payload.get("loss"),
                        "eval_loss": payload.get("eval_loss"),
                        "tokens_per_sec": payload.get("tokens_per_sec"),
                    })

                overrides["progress"] = on_progress
                self.train_status["state"] = "training"
                result = trainer.train_from_preset(
                    preset, resume=resume, verbose=self.verbose, **overrides
                )
                self.train_status.update({
                    "state": "done",
                    "step": result.steps,
                    "loss": round(result.final_loss, 4),
                    "best_eval": None if result.best_eval is None else round(result.best_eval, 4),
                    "seconds": round(result.seconds, 1),
                    "finished_at": time.time(),
                    "message": f"Klaar: {result.steps} stappen, verlies {result.final_loss:.3f}",
                })
                self.load(force=True)
            except Exception as exc:  # noqa: BLE001 - moet zichtbaar zijn in de UI
                self.train_status.update({
                    "state": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "finished_at": time.time(),
                })
                traceback.print_exc()

        self.train_thread = threading.Thread(target=run, daemon=True, name="webai-train")
        self.train_thread.start()
        return {"started": True, "preset": preset}


def _try_tool_call(buffer: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Leest een tool-aanroep, maar pas als het JSON-object compleet is.

    Tijdens streamen komt de aanroep token voor token binnen. Een halve aanroep
    doorspelen zou betekenen dat de extensie een gereedschap uitvoert met
    onvolledige (of lege) argumenten.
    """
    parsed = parse_assistant_output(TOOL_CALL + buffer, require_complete=True)
    calls = parsed.get("tool_calls") or []
    if calls:
        return calls[0], buffer
    return None, buffer


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


PLAYGROUND_HTML = """<!doctype html>
<html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WebAI — je eigen model</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
      background:#0f1115;color:#e6e8ee}
 header{padding:16px 20px;border-bottom:1px solid #23262f;display:flex;gap:14px;
        align-items:center;flex-wrap:wrap}
 h1{font-size:16px;margin:0;font-weight:650}
 .pill{font-size:12px;padding:3px 9px;border-radius:999px;background:#1b1f27;
       border:1px solid #2b3040;color:#9aa3b5}
 main{max-width:820px;margin:0 auto;padding:20px}
 #log{display:flex;flex-direction:column;gap:12px;margin-bottom:18px}
 .msg{padding:11px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
 .user{background:#1c2b3f;align-self:flex-end;max-width:85%}
 .bot{background:#171a21;border:1px solid #23262f;align-self:flex-start;max-width:92%}
 .who{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#7d869b;margin-bottom:5px}
 form{display:flex;gap:8px}
 textarea{flex:1;resize:vertical;min-height:46px;padding:11px 13px;border-radius:11px;
          border:1px solid #2b3040;background:#12151b;color:#e6e8ee;font:inherit}
 button{padding:0 18px;border-radius:11px;border:0;background:#4f7cff;color:#fff;
        font:inherit;font-weight:600;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
 #stats{font-size:12.5px;color:#8b93a7;margin-top:16px;border-top:1px solid #23262f;padding-top:12px}
 code{background:#0b0d11;padding:1px 5px;border-radius:5px}
</style></head><body>
<header>
 <h1>WebAI</h1><span class="pill" id="p">laden…</span>
 <span class="pill" id="q">—</span>
</header>
<main>
 <div id="log"></div>
 <form id="f">
   <textarea id="i" placeholder="Stel een vraag aan je eigen model…" required></textarea>
   <button id="s" type="submit">Stuur</button>
 </form>
 <div id="stats"></div>
</main>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),
      i=document.getElementById('i'),s=document.getElementById('s');
let hist=[{role:'system',content:'Je bent een vriendelijke, behulpzame assistent. '+
  'Antwoord kort en duidelijk in het Nederlands.'}];
function add(cls,who){const d=document.createElement('div');d.className='msg '+cls;
  const w=document.createElement('div');w.className='who';w.textContent=who;
  d.appendChild(w);const t=document.createElement('div');d.appendChild(t);
  log.appendChild(t.parentNode);log.scrollTop=log.scrollHeight;return t;}
async function info(){try{const r=await fetch('/v1/info');const j=await r.json();
  document.getElementById('p').textContent=(j.params/1e6).toFixed(2)+'M parameters · '+
    j.layers+' lagen · context '+j.context;
  document.getElementById('q').textContent='verlies '+(j.eval_loss??j.loss??'—')+
    ' · '+j.steps+' stappen · '+j.vocab+' tokens';
  document.getElementById('stats').innerHTML=
    'Eigen model, getraind op jouw data. Run <code>'+j.run_name+'</code> in <code>'+j.path+
    '</code><br>Feedback: '+j.feedback.total+' berichten verzameld.';
}catch(e){document.getElementById('p').textContent='geen model';}}
f.onsubmit=async e=>{e.preventDefault();const q=i.value.trim();if(!q)return;
  i.value='';add('user','jij').textContent=q;hist.push({role:'user',content:q});
  const out=add('bot','webai');s.disabled=true;let text='';
  try{const r=await fetch('/v1/chat/completions',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:'webai',messages:hist,stream:true,temperature:0.7})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      let n;while((n=buf.indexOf('\\n'))>=0){const line=buf.slice(0,n).trim();buf=buf.slice(n+1);
        if(!line.startsWith('data:'))continue;const p=line.slice(5).trim();
        if(p==='[DONE]')continue;try{const c=JSON.parse(p);
          const d=c.choices?.[0]?.delta?.content;if(d){text+=d;out.textContent=text;}}catch(_){}}}}
  catch(err){text='[fout] '+err.message;out.textContent=text;}
  hist.push({role:'assistant',content:text});s.disabled=false;i.focus();};
info();i.focus();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    host: ModelHost = None  # type: ignore[assignment]
    server_version = "WebAI/0.2"
    protocol_version = "HTTP/1.1"

    # --- hulp ---
    def log_message(self, fmt: str, *args: Any) -> None:
        if self.host.verbose:
            sys.stderr.write("[serve] " + (fmt % args) + "\n")

    def _cors(self) -> None:
        # De extensie draait op chrome-extension://…; die moet toegang krijgen.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization, X-Requested-With")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "86400")

    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400, code: str = "invalid_request_error") -> None:
        self._json({"error": {"message": message, "type": code, "code": status}}, status)

    def _html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # --- routes ---
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/", "/index.html", "/playground"):
            return self._html(PLAYGROUND_HTML)
        if path == "/health":
            return self._json({"status": "ok", "model_loaded": self.host.model is not None})
        if path == "/v1/info" or path == "/info":
            return self._json(self.host.info())
        if path == "/v1/status":
            return self._json({"status": "ok", "train": self.host.train_status,
                               "model": self.host.info()})
        if path in ("/v1/models", "/models"):
            info = self.host.info()
            return self._json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(info.get("trained_at") or time.time()),
                    "owned_by": "jij",
                    "name": "WebAI (eigen, lokaal getraind model)",
                    "context_length": info.get("context") or 0,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "supported_parameters": ["temperature", "top_p", "max_tokens", "tools", "stop"],
                }],
            })
        if path in ("/v1/feedback", "/feedback"):
            return self._json(self.host.feedback.stats())
        return self._error(f"Onbekend pad: {path}", 404, "not_found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_body()
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self._chat(body)
            if path in ("/v1/completions", "/completions"):
                return self._completion(body)
            if path in ("/v1/feedback", "/feedback"):
                return self._feedback(body)
            if path in ("/v1/train", "/train"):
                return self._train(body)
            if path in ("/v1/reload", "/reload"):
                self.host.load(force=True)
                return self._json({"reloaded": True, "info": self.host.info()})
            return self._error(f"Onbekend pad: {path}", 404, "not_found")
        except FileNotFoundError as exc:
            return self._error(str(exc), 503, "model_not_trained")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500, "internal_error")

    # --- chat ---
    def _chat(self, body: Dict[str, Any]) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._error("Het veld 'messages' ontbreekt of is leeg.")
        tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        gen = sampling_params(body)
        if isinstance(body.get("top_k"), (int, float)):
            gen.top_k = int(body["top_k"])
        seed = body.get("seed")
        seed = int(seed) if isinstance(seed, (int, float)) else None
        want_stream = bool(body.get("stream"))
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created = int(time.time())
        model_name = str(body.get("model") or MODEL_NAME)
        t0 = time.time()

        # Zorg dat model + tokenizer geladen zijn vóór we tokens tellen.
        self.host.load()
        prompt_tokens = self._count_prompt_tokens(messages, tools)

        if not want_stream:
            result = self.host.complete_sync(messages=messages, tools=tools, gen=gen, seed=seed)
            message: Dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
            if result["tool_calls"]:
                message["tool_calls"] = [
                    {k: v for k, v in tc.items() if k != "index"} for tc in result["tool_calls"]
                ]
            completion_tokens = max(1, len(self.host.tokenizer.encode(result["raw"]))) if self.host.tokenizer else 0
            return self._json({
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": model_name, "choices": [{
                    "index": 0, "message": message,
                    "finish_reason": result["finish_reason"],
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })

        # --- streaming (SSE) ---
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(obj: Any) -> None:
            chunk = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

        def send_done() -> None:
            # Let op: [DONE] is een kale SSE-markering, géén JSON-string.
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        send({"id": completion_id, "object": "chat.completion.chunk", "created": created,
              "model": model_name,
              "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                           "finish_reason": None}]})

        finish = "stop"
        completion_tokens = 0
        tool_index = 0
        try:
            for kind, payload in self.host.complete(messages=messages, tools=tools,
                                                    gen=gen, seed=seed):
                if kind == "content":
                    completion_tokens += 1
                    send({"id": completion_id, "object": "chat.completion.chunk",
                          "created": created, "model": model_name,
                          "choices": [{"index": 0, "delta": {"content": payload},
                                       "finish_reason": None}]})
                elif kind == "tool_call":
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    finish = "tool_calls"
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                    send({"id": completion_id, "object": "chat.completion.chunk",
                          "created": created, "model": model_name,
                          "choices": [{"index": 0, "delta": {"tool_calls": [{
                              "index": tool_index, "id": call_id, "type": "function",
                              "function": {
                                  "name": data.get("name", ""),
                                  "arguments": json.dumps(data.get("arguments", {}),
                                                          ensure_ascii=False),
                              }}]}, "finish_reason": None}]})
                    tool_index += 1
                elif kind == "done":
                    completion_tokens = (len(self.host.tokenizer.encode(payload))
                                         if self.host.tokenizer else completion_tokens)
        except (BrokenPipeError, ConnectionResetError):
            return  # paneel gesloten
        finally:
            try:
                send({"id": completion_id, "object": "chat.completion.chunk",
                      "created": created, "model": model_name,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                      "usage": {"prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens}})
                send_done()
            except Exception:
                pass
        if self.host.verbose:
            print(f"[serve] chat klaar in {time.time() - t0:.1f}s "
                  f"({completion_tokens} tokens, {finish})")

    def _count_prompt_tokens(self, messages: List[Dict[str, Any]],
                             tools: Optional[List[Dict[str, Any]]]) -> int:
        if not self.host.tokenizer:
            return 0
        text = render_prompt(messages, add_generation_prompt=True, tools=tools)
        return len(self.host.tokenizer.encode(text))

    def _completion(self, body: Dict[str, Any]) -> None:
        """Klassieke /v1/completions (prompt -> tekst), handig om te testen."""
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        gen = sampling_params(body)
        result = self.host.complete_sync(
            messages=[{"role": "user", "content": str(prompt)}], gen=gen
        )
        self._json({
            "id": f"cmpl-{uuid.uuid4().hex[:16]}", "object": "text_completion",
            "created": int(time.time()), "model": MODEL_NAME,
            "choices": [{"index": 0, "text": result["content"], "finish_reason": "stop"}],
        })

    def _feedback(self, body: Dict[str, Any]) -> None:
        try:
            stored = self.host.feedback.add(body)
        except ValueError as exc:
            # slechte invoer van de kant van de gebruiker → 400, geen 500
            return self._error(str(exc), 400, "invalid_feedback")
        self._json({"ok": True, "saved": stored, "stats": self.host.feedback.stats()})

    def _train(self, body: Dict[str, Any]) -> None:
        # Eerst het feedback-materiaal omzetten in trainingsdata.
        written = self.host.feedback.export_training_data(self.host.run_name)
        result = self.host.start_training(
            preset=str(body.get("preset") or "tiny"),
            steps=body.get("steps"),
            resume=bool(body.get("resume", True)),
        )
        result["feedback_exported"] = written
        self._json(result)


# ---------------------------------------------------------------------------
# Starten
# ---------------------------------------------------------------------------


def make_server(host: ModelHost, http_host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"host": host})
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer((http_host, port), handler)


def serve(
    run_name: str = MODEL_NAME,
    http_host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    checkpoint: str = DEFAULT_CHECKPOINT,
    verbose: bool = True,
) -> None:
    host = ModelHost(run_name=run_name, checkpoint=checkpoint, verbose=verbose)
    try:
        host.load()
    except FileNotFoundError as exc:
        print(f"[serve] {exc}", file=sys.stderr)
        print("[serve] De server start toch; train het model en herlaad dan "
              "met POST /v1/reload.", file=sys.stderr)

    try:
        httpd = make_server(host, http_host, port)
    except OSError as exc:
        print(f"[serve] kon niet luisteren op {http_host}:{port} — {exc}", file=sys.stderr)
        raise SystemExit(1)

    shown = "localhost" if http_host in ("0.0.0.0", "") else http_host
    if verbose:
        print(f"[serve] WebAI draait op http://{shown}:{port}")
        print(f"[serve]   OpenAI-compatibel:  http://{shown}:{port}/v1/chat/completions")
        print(f"[serve]   Test-paneel:        http://{shown}:{port}/")
        print(f"[serve]   Stoppen met Ctrl+C")
        if http_host != "0.0.0.0":
            print(f"[serve]   (alleen lokaal bereikbaar; voor de extensie is dat prima)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if verbose:
            print("\n[serve] gestopt")
    finally:
        httpd.server_close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Start de WebAI-server (OpenAI-compatibel).")
    ap.add_argument("--run", default=MODEL_NAME, help="naam van de run (map in ai/runs)")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help="adres om op te luisteren (0.0.0.0 = ook vanaf andere apparaten)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--checkpoint", default="model",
                    help="'model' (laatst) of 'best' (laagste eval-verlies)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    serve(run_name=args.run, http_host=args.host, port=args.port,
          checkpoint=args.checkpoint, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
