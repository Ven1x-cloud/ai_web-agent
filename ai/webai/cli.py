"""Opdrachtregel voor WebAI: `python -m webai <commando> [opties]`.

Commando's:
  build     tokenizer (en dataset) bouwen uit ai/data
  train     het eigen model trainen of verder trainen
  serve     OpenAI-compatibele server starten voor de extensie
  chat      in de terminal praten met je eigen model
  ask       één vraag stellen en het antwoord tonen
  info      laten zien hoe ver je model is
  data      statistieken van je trainingsdata
  feedback  verzamelde feedback bekijken of omzetten in trainingsdata
  distill   trainingsdata laten maken door een cloudmodel (optioneel)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import train as trainer
from .config import (
    DATA_DIR,
    LEARNED_DIR,
    PRESETS,
    RUNS_DIR,
    SEED_DIR,
    ModelConfig,
    TokenizerConfig,
    TrainConfig,
    configs_from_preset,
    ensure_dirs,
)
from .dataset import build_dataset, dataset_tokens, iter_data_files
from .feedback import FeedbackStore
from .generate import GenConfig, Generator
from .model import TransformerLM
from .tokenizer import ByteLevelBPE

DEFAULT_RUN = "webai"

SYSTEM_PROMPT_CHAT = (
    "Je bent WebAI, een vriendelijke en behulpzame assistent. "
    "Antwoord kort, duidelijk en in de taal van de gebruiker (meestal Nederlands). "
    "Zeg eerlijk dat je iets niet weet als je het niet weet."
)


def _add_preset_args(ap: argparse.ArgumentParser) -> None:
    """Alleen wat elk commando nodig heeft: welke preset en welke run."""
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS),
                    help="modelgrootte: " + ", ".join(sorted(PRESETS)))
    ap.add_argument("--run", default=DEFAULT_RUN, help="naam van de run (map in ai/runs)")


def _add_train_args(ap: argparse.ArgumentParser) -> None:
    """Daarbovenop: alle knoppen om het trainen zelf bij te stellen."""
    _add_preset_args(ap)
    ap.add_argument("--steps", type=int, default=None, help="aantal trainingsstappen")
    ap.add_argument("--batch", type=int, default=None, help="batchgrootte")
    ap.add_argument("--lr", type=float, default=None, help="leersnelheid")
    ap.add_argument("--vocab", type=int, default=None, help="grootte van de woordenschat")
    ap.add_argument("--context", type=int, default=None, help="maximale contextlengte")
    ap.add_argument("--grad-accum", type=int, default=None, dest="grad_accum",
                    help="gradiënten over dit aantal stapjes optellen; laat je een "
                         "kleinere --batch gebruiken bij geheugentekort zonder dat "
                         "de effectieve batch verandert")
    ap.add_argument("--eval-every", type=int, default=None, dest="eval_every",
                    help="om de zoveel stappen evalueren op onbekende voorbeelden")
    ap.add_argument("--patience", type=int, default=None, dest="early_stop_patience",
                    help="stop zodra het eval-verlies dit aantal evaluaties op rij niet "
                         "meer verbetert. Laat je dit weg, dan beslist train zelf op "
                         "grond van hoeveel data en ronden je hebt")
    ap.add_argument("--no-early-stop", action="store_true", dest="no_early_stop",
                    help="train de opgegeven stappen af zonder bewaking op overfitten; "
                         "het beste moment wordt nog steeds apart bewaard als best.npz")
    ap.add_argument("--resume", action="store_true", help="verder gaan waar je bleef")
    ap.add_argument("--init-from", default=None,
                    help="start vanaf een andere run (verder trainen op nieuwe data)")


def cmd_build(args: argparse.Namespace) -> int:
    ensure_dirs()
    tok_cfg = TokenizerConfig(vocab_size=args.vocab or PRESETS[args.preset]["model"]["vocab_size"])
    tok = trainer.build_tokenizer(args.run, tok_cfg, verbose=not args.quiet)
    model_cfg, _ = configs_from_preset(args.preset)
    model_cfg.vocab_size = tok.vocab_size
    if args.context:
        model_cfg.block_size = args.context
    samples, stats = build_dataset(tok, model_cfg.block_size, verbose=not args.quiet)
    print(f"[build] {stats['files']} bestanden, {len(samples)} samples, "
          f"{dataset_tokens(samples)} tokens")
    print(f"[build] tokenizer: {trainer.tokenizer_path(args.run)}")
    print(f"[build] volgende stap:  python -m webai train --preset {args.preset} --run {args.run}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    ensure_dirs()
    model_cfg, train_cfg = configs_from_preset(args.preset)
    train_cfg.run_name = args.run
    if args.steps is not None:
        train_cfg.max_steps = args.steps
    if args.batch is not None:
        train_cfg.batch_size = args.batch
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.context is not None:
        model_cfg.block_size = args.context
    if args.grad_accum is not None:
        train_cfg.grad_accum = max(1, args.grad_accum)
    if args.eval_every is not None:
        train_cfg.eval_every = max(1, args.eval_every)
    if getattr(args, "no_early_stop", False):
        # -1 is het enige teken dat "bewust uit" onderscheidt van "niet opgegeven"
        # (0), want 0 laat train zelf beslissen.
        train_cfg.early_stop_patience = -1
    elif args.early_stop_patience is not None:
        train_cfg.early_stop_patience = max(1, args.early_stop_patience)
    if args.vocab is not None:
        model_cfg.vocab_size = args.vocab
    tok_cfg = TokenizerConfig(vocab_size=model_cfg.vocab_size)
    result = trainer.train(model_cfg, train_cfg, tok_cfg, resume=args.resume,
                           init_from=args.init_from, verbose=not args.quiet)
    print(f"\n[train] {result.steps} stappen in {result.seconds / 60:.1f} minuten, "
          f"{result.tokens_seen / 1e6:.2f}M tokens gezien")
    print(f"[train] verlies {_fmt_num(result.final_loss)}", end="")
    if result.best_eval is not None:
        print(f", beste eval {_fmt_num(result.best_eval)}")
    else:
        print()
    if (result.best_eval is not None and result.final_loss
            and result.best_eval > result.final_loss * 1.6):
        print("[train] let op: het eval-verlies ligt veel hoger dan het trainverlies.")
        print("        Het model leert uit zijn hoofd. Meer data helpt, langer trainen niet.")
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    print(f"\nVolgende stap:  python -m webai serve --run {args.run}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .serve import main as serve_main

    argv = ["--run", args.run, "--host", args.host, "--port", str(args.port)]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    if args.quiet:
        argv += ["--quiet"]
    return serve_main(argv)


def _load_for_chat(run: str, checkpoint: str = "auto"):
    """Laadt model + tokenizer voor gebruik in de terminal."""
    from .serve import pick_checkpoint

    directory = trainer.run_dir(run)
    if checkpoint == "auto":
        checkpoint = pick_checkpoint(directory, "auto")
        print(f"[chat] checkpoint '{checkpoint}' gebruikt", file=sys.stderr)
    model, meta = TransformerLM.load(directory, checkpoint)
    tok = trainer.load_tokenizer(run)
    model.params = {k: v.astype(np.float32, copy=False) for k, v in model.params.items()}
    model.dtype = np.dtype(np.float32)
    return model, tok, meta


def _stop_ids(tok: ByteLevelBPE) -> List[int]:
    from .tokenizer import EOT, TOOL, TURN

    return [tok.encode(t)[0] for t in (EOT, TURN, TOOL)]


def _stream_print(model, tok, messages: List[Dict[str, Any]], gen: GenConfig,
                  seed: Optional[int], label: str = "webai") -> Tuple[str, int, float]:
    """Genereert een antwoord, toont het terwijl het ontstaat, en geeft het terug."""
    from .chat import render_prompt

    prompt = render_prompt(messages, add_generation_prompt=True)
    ids = [tok.bos_id] + tok.encode(prompt)
    generator = Generator(model, ids, cfg=gen, seed=seed)
    if label:
        sys.stdout.write(f"{label} > ")
        sys.stdout.flush()

    shown = ""
    t0 = time.time()
    for _ in generator.stream(extra_stop_ids=_stop_ids(tok)):
        # telkens de hele reeks decoderen: zo knippen we nooit een UTF-8-teken doormidden
        text = tok.decode(generator.generated)
        if len(text) > len(shown):
            sys.stdout.write(text[len(shown):])
            sys.stdout.flush()
            shown = text
    elapsed = time.time() - t0
    n = len(generator.generated)
    sys.stdout.write("\n")
    if not shown.strip():
        # Lege of alleen-witruimte uitvoer ziet eruit als een bug, maar is bij een
        # nauwelijks getraind model het verwachte gedrag. Zeg dat eerlijk.
        if n > 0:
            sys.stdout.write(
                f"\n[{label or 'webai'}] Er kwam geen leesbare tekst uit "
                f"({n} tokens gegenereerd).\n"
                "         Dat hoort bij een model dat nog te weinig getraind is of\n"
                "         te weinig voorbeelden heeft gezien. Meer data helpt:\n"
                "           python -m webai distill --limit 200\n"
                "           python -m webai train --resume\n"
            )
        else:
            sys.stdout.write(f"\n[{label or 'webai'}] Geen tokens gegenereerd: "
                             "het model stopte meteen.\n")
        sys.stdout.flush()
    return shown, n, elapsed


def cmd_chat(args: argparse.Namespace) -> int:
    model, tok, meta = _load_for_chat(args.run, args.checkpoint)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]

    print(f"[chat] model '{args.run}': {meta.get('num_params', model.num_params()) / 1e6:.2f}M "
          f"parameters, context {model.cfg.block_size}, verlies {meta.get('loss', '?')}")
    print("[chat] 'nieuw' = leeg gesprek, 'stop' = wegwezen, Ctrl+C doet hetzelfde.\n")

    while True:
        try:
            text = input("jij > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("stop", "exit", "quit", "q"):
            break
        if text.lower() in ("nieuw", "reset", "clear"):
            messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
            print("[chat] gesprek geleegd\n")
            continue

        messages.append({"role": "user", "content": text})
        gen = GenConfig(temperature=args.temperature, max_new_tokens=args.max_tokens,
                        top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty)
        answer, n, elapsed = _stream_print(model, tok, messages, gen, args.seed)
        speed = n / max(0.01, elapsed)
        print(f"       [{n} tokens in {elapsed:.1f}s = {speed:.0f} tok/s]\n")
        messages.append({"role": "assistant", "content": answer})

        # gesprek inkorten als het te lang wordt voor de context
        while len(messages) > 2 and _too_long(tok, messages, model.cfg.block_size):
            del messages[1:3]
    return 0


def _too_long(tok: ByteLevelBPE, messages: List[Dict[str, Any]], block_size: int) -> bool:
    from .chat import render_prompt

    prompt = render_prompt(messages, add_generation_prompt=True)
    return len(tok.encode(prompt)) > block_size - 64


def cmd_ask(args: argparse.Namespace) -> int:
    model, tok, meta = _load_for_chat(args.run, args.checkpoint)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT},
                {"role": "user", "content": args.question}]
    gen = GenConfig(temperature=args.temperature, max_new_tokens=args.max_tokens,
                    top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty)
    answer, n, elapsed = _stream_print(model, tok, messages, gen, args.seed, label="")
    print(f"[{n} tokens in {elapsed:.1f}s = {n / max(0.01, elapsed):.0f} tok/s]",
          file=sys.stderr)
    return 0


def _fmt_num(value, digits: int = 4, fallback: str = "-") -> str:
    """Een getal netjes afronden, of een streepje als er niets te melden is."""
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback   # "nan" of "inf" zegt de lezer niets
    return f"{number:.{digits}f}"


def cmd_info(args: argparse.Namespace) -> int:
    info = trainer.model_info(args.run)
    if not info.get("exists"):
        print(f"Nog geen model '{args.run}'. Bouw en train er een:")
        print(f"  python -m webai build --run {args.run}")
        print(f"  python -m webai train --run {args.run} --preset tiny")
        return 1

    params = info.get("params") or 0
    print(f"Model '{info['run_name']}'")
    print(f"  map          {info.get('path')}")
    print(f"  parameters   {params / 1e6:.2f}M" if params else "  parameters   onbekend")
    print(f"  vorm         {info.get('layers') or '?'} lagen × {info.get('width') or '?'} breed, "
          f"context {info.get('context') or '?'}, woordenschat {info.get('vocab') or '?'}")

    if info.get("step"):
        print(f"  getraind     stap {info['step']}, "
              f"{(info.get('tokens_seen') or 0) / 1e6:.2f}M tokens gezien")
        seconds = info.get("seconds")
        if seconds:
            print(f"  duur         {seconds / 60:.1f} minuten")
        print(f"  verlies      {_fmt_num(info.get('loss'))}  (trainen)")
        print(f"               {_fmt_num(info.get('eval_loss') or info.get('best_eval'))}  "
              f"(evalueren — dit is het cijfer dat telt)")
    else:
        print("  getraind     nog niet")

    # Eerlijk advies: met weinig data is overfitten de normale uitkomst.
    loss = info.get("loss")
    evl = info.get("eval_loss") or info.get("best_eval")
    if loss and evl and evl > loss * 1.6:
        print("\n  Let op: het eval-verlies ligt flink hoger dan het trainingsverlies.")
        print("  Het model kent zijn voorbeelden uit zijn hoofd in plaats van taal te")
        print("  leren. Meer data helpt meer dan langer trainen:")
        print("    python -m webai distill --limit 200     (data laten maken)")
        print("    of zet eigen bestanden in ai/data/learned/")

    tail = info.get("log_tail") or []
    if tail:
        print("\n  Laatste logregels:")
        for row in tail[-8:]:
            ev = row.get("eval_loss")
            print(f"    stap {row.get('step'):>6}  verlies {_fmt_num(row.get('loss'), 3)}  "
                  f"eval {_fmt_num(ev, 3)}  {row.get('tokens_per_sec', 0):>6.0f} tok/s")

    print("\n  Volgende stap:")
    if not info.get("step"):
        print(f"    python -m webai train --run {args.run} --preset tiny")
    else:
        print(f"    python -m webai serve --run {args.run}   "
              f"(en open daarna de extensie)")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    files = iter_data_files()
    total_chars = 0
    rows = []
    for path in files:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        total_chars += size
        rows.append((path, size))
    print(f"{len(rows)} databestanden, {total_chars / 1024:.1f} kB totaal\n")
    for path, size in sorted(rows, key=lambda r: -r[1])[:40]:
        print(f"  {size / 1024:8.1f} kB  {os.path.relpath(path)}")
    tok_path = trainer.tokenizer_path(args.run)
    if os.path.exists(tok_path):
        tok = ByteLevelBPE.load(tok_path)
        model_cfg, _ = configs_from_preset(args.preset)
        samples, stats = build_dataset(tok, model_cfg.block_size)
        print(f"\nMet de tokenizer van '{args.run}': {len(samples)} samples, "
              f"{dataset_tokens(samples)} tokens")
        print(f"  gesprekken: {stats['chat']}   tekstblokken: {stats['text']}   "
              f"overgeslagen: {stats['skipped']}")
    else:
        print(f"\nNog geen tokenizer voor '{args.run}' — draai: python -m webai build")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    store = FeedbackStore()
    if args.export:
        n = store.export_training_data(args.run)
        print(f"{n} feedback-berichten omgezet naar trainingsdata in {LEARNED_DIR}")
        return 0
    stats = store.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    entries = store.all()[-args.last:] if args.last else []
    for e in entries:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("time", 0)))
        print(f"\n[{when}] {e.get('kind')}  (id {e.get('id')})")
        resp = (e.get("response") or "").replace("\n", " ")[:120]
        print(f"  antwoord: {resp}")
        if e.get("correction"):
            print(f"  verbetering: {e['correction'][:120]}")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from .distill import distill

    return distill(
        api_base=args.api_base,
        api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        model=args.model,
        questions_file=args.questions,
        out=args.out,
        limit=args.limit,
        system=args.system,
        verbose=not args.quiet,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="webai", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="tokenizer + dataset bouwen uit ai/data")
    _add_train_args(p)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("train", help="het eigen model trainen")
    _add_train_args(p)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="OpenAI-compatibele server starten")
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--checkpoint", default="auto", choices=("auto", "model", "best"),
                   help="auto = het checkpoint met het laagste eval-verlies")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_serve)

    for name, helptext in (("chat", "praten met je model in de terminal"),
                           ("ask", "één vraag stellen")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--run", default=DEFAULT_RUN)
        p.add_argument("--checkpoint", default="auto", choices=("auto", "model", "best"),
                   help="auto = het checkpoint met het laagste eval-verlies")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--top-p", type=float, default=0.9, dest="top_p")
        p.add_argument("--top-k", type=int, default=40, dest="top_k")
        p.add_argument("--max-tokens", type=int, default=400, dest="max_tokens")
        p.add_argument("--repetition-penalty", type=float, default=1.1,
                       dest="repetition_penalty")
        p.add_argument("--seed", type=int, default=None)
        if name == "ask":
            p.add_argument("question", help="de vraag")
        p.set_defaults(func=cmd_chat if name == "chat" else cmd_ask)

    p = sub.add_parser("info", help="status van je model")
    p.add_argument("--run", default=DEFAULT_RUN)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("data", help="statistieken van je trainingsdata")
    _add_preset_args(p)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("feedback", help="verzamelde feedback bekijken of exporteren")
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--export", action="store_true", help="omzetten naar trainingsdata")
    p.add_argument("--last", type=int, default=0, help="toon de laatste N berichten")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("distill", help="trainingsdata laten maken door een cloudmodel")
    p.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default="", help="of omgevingsvariabele OPENROUTER_API_KEY")
    p.add_argument("--model", default="qwen/qwen3.8-27b:free")
    p.add_argument("--questions", default=None,
                   help="bestand met vragen (één per regel, of JSONL met {'question': ...})")
    p.add_argument("--out", default=None, help="bestemming (standaard ai/data/learned/distill.jsonl)")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--system", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_distill)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    # SIGTERM (bijv. van een procesbeheerder of `kill`) omzetten in een
    # KeyboardInterrupt, zodat een lopende training netjes zijn checkpoint bewaart.
    import signal

    def _on_term(signum, frame):  # noqa: ANN001, ARG001
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        pass  # niet mogelijk buiten de hoofdthread

    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
