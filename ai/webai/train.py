"""Trainingslus: checkpoints, evaluatie, leerplanning, hervatten.

Typisch gebruik (via de CLI):

    python -m webai build --preset small     # tokenizer + dataset
    python -m webai train --preset small     # trainen
    python -m webai train --resume           # verder waar je gebleven was
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import nn
from .config import (
    RUNS_DIR,
    ModelConfig,
    TokenizerConfig,
    TrainConfig,
    configs_from_preset,
    ensure_dirs,
    json_safe,
    load_json,
    save_json,
)
from .dataset import Batcher, Sample, build_dataset, dataset_tokens, iter_data_files, split_samples
from .model import TransformerLM
from .tokenizer import ByteLevelBPE

F32 = np.float32


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------


def run_dir(name: str, root: str = RUNS_DIR) -> str:
    return os.path.join(root, name)


def tokenizer_path(name: str, root: str = RUNS_DIR) -> str:
    return os.path.join(run_dir(name, root), "tokenizer.json")


def state_path(name: str, root: str = RUNS_DIR) -> str:
    return os.path.join(run_dir(name, root), "train_state.json")


def log_path(name: str, root: str = RUNS_DIR) -> str:
    return os.path.join(run_dir(name, root), "train_log.jsonl")


def checkpoint_exists(name: str, root: str = RUNS_DIR) -> bool:
    return os.path.exists(os.path.join(run_dir(name, root), "model.npz"))


# ---------------------------------------------------------------------------
# Tekst verzamelen + tokenizer bouwen
# ---------------------------------------------------------------------------


def _strings_from_record(rec: Any) -> List[str]:
    out: List[str] = []
    if isinstance(rec, str):
        return [rec]
    if not isinstance(rec, dict):
        return out
    for key in ("text", "prompt", "completion", "instruction", "input", "output",
                "question", "answer", "content", "value"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value)
    msgs = rec.get("messages") or rec.get("conversations")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                for key in ("content", "value", "text"):
                    v = m.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v)
    return out


def collect_text(max_chars: int, files: Optional[Sequence[str]] = None) -> str:
    """Leest alle tekst uit ai/data (ook uit JSON-gesprekken) voor de tokenizer."""
    parts: List[str] = []
    total = 0

    def add(text: str) -> None:
        nonlocal total
        if text:
            parts.append(text)
            total += len(text)

    for path in files or iter_data_files():
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError as exc:
            print(f"  [data] kon {path} niet lezen: {exc}")
            continue
        if ext == ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            if data is None:
                add(raw)
            else:
                for rec in (data if isinstance(data, list) else [data]):
                    for t in _strings_from_record(rec):
                        add(t)
        elif ext in (".jsonl", ".ndjson"):
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    add(line)
                    continue
                for t in _strings_from_record(rec):
                    add(t)
        else:
            add(raw)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def build_tokenizer(
    name: str,
    cfg: TokenizerConfig,
    verbose: bool = True,
    files: Optional[Sequence[str]] = None,
    root: str = RUNS_DIR,
) -> ByteLevelBPE:
    ensure_dirs()
    all_files = list(files) if files else iter_data_files()
    if verbose:
        print(f"[tokenizer] tekst verzamelen uit {len(all_files)} bestanden...")
    text = collect_text(cfg.sample_chars, all_files)
    if len(text.strip()) < 200:
        raise SystemExit(
            "Te weinig tekst gevonden in ai/data. Zet er .txt/.md/.jsonl-bestanden in "
            "(zie ai/data/seed voor het formaat)."
        )
    if verbose:
        print(f"[tokenizer] {len(text)} tekens gevonden; woordenschat van "
              f"{cfg.vocab_size} tokens aan het leren...")
    t0 = time.time()
    tok = ByteLevelBPE.train(
        text, vocab_size=cfg.vocab_size, min_freq=cfg.min_freq, verbose=verbose
    )
    tok.save(tokenizer_path(name, root))
    if verbose:
        stats = tok.stats(text[:20000])
        print(f"[tokenizer] klaar in {time.time() - t0:.1f}s "
              f"-> {tokenizer_path(name, root)}")
        print(f"[tokenizer] {stats['chars_per_token']} tekens per token "
              f"({stats['bits_per_char']} bits/teken)")
    return tok


def load_tokenizer(name: str, root: str = RUNS_DIR) -> ByteLevelBPE:
    path = tokenizer_path(name, root)
    if not os.path.exists(path):
        raise SystemExit(
            f"Geen tokenizer gevonden op {path}.\n"
            f"Draai eerst: python -m webai build"
        )
    return ByteLevelBPE.load(path)


# ---------------------------------------------------------------------------
# Trainen
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    run_name: str
    steps: int
    final_loss: float
    best_eval: Optional[float]
    tokens_seen: int
    seconds: float
    params: int
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # json_safe: een NaN-verlies mag geen ongeldige JSON opleveren, want
        # JSON.parse in de browser (en veel andere parsers) weigert dat.
        return json_safe({
            "run_name": self.run_name,
            "steps": self.steps,
            "final_loss": round(self.final_loss, 4),
            "best_eval": None if self.best_eval is None else round(self.best_eval, 4),
            "tokens_seen": self.tokens_seen,
            "seconds": round(self.seconds, 1),
            "params": self.params,
        })


def accumulate(grads: Dict[str, np.ndarray], new: Dict[str, np.ndarray], scale: float) -> None:
    for k, v in new.items():
        if scale == 1.0:
            grads[k] = v.copy() if k not in grads else grads[k] + v
        else:
            grads[k] = v * scale if k not in grads else grads[k] + v * scale


def zero_grads(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.zeros_like(v) for k, v in params.items()}


def evaluate(model: TransformerLM, batches: Sequence[Tuple], eval_batches: int) -> float:
    if not batches:
        return float("nan")
    total, weight = 0.0, 0.0
    for ids, targets, mask in list(batches)[:eval_batches]:
        loss, _, _, cache = model.forward(ids, targets, mask)
        n = cache.get("ntok", 1)
        total += float(loss) * n
        weight += n
    return total / max(1.0, weight)


def rotate_checkpoints(directory: str, keep: int, prefix: str = "step") -> None:
    """Bewaart alleen de laatste `keep` tussen-checkpoints."""
    if keep <= 0:
        return
    found = []
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".npz"):
            try:
                step = int(name[len(prefix) : -len(".npz")])
            except ValueError:
                continue
            found.append((step, name))
    found.sort(reverse=True)
    for step, name in found[keep:]:
        for ext in (".npz", ".json"):
            path = os.path.join(directory, name[: -len(".npz")] + ext)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def planned_epochs(n_train_samples: int, batch_size: int, max_steps: int) -> float:
    """Hoe vaak het model zijn hele trainingsset te zien krijgt."""
    per_epoch = max(1, n_train_samples // max(1, batch_size))
    return max_steps / per_epoch


def auto_early_stop_patience(n_train_samples: int, batch_size: int, max_steps: int,
                             patience: int = 0, sample_floor: int = 300,
                             epoch_ceiling: float = 25.0) -> int:
    """Beslist of vroeg stoppen automatisch aan moet, en met welke patience.

    Een expliciet gekozen patience wint altijd; een negatief getal betekent
    "bewust uit" (dat is hoe `--no-early-stop` het doorgeeft). Anders zijn er twee redenen om
    het aan te zetten:

    1. **Weinig voorbeelden** (minder dan `sample_floor`). Dan is uit-het-hoofd
       leren bijna onvermijdelijk.
    2. **Veel ronden** (meer dan `epoch_ceiling`). Dit is de sterkere regel: bij
       een kleine dataset stapelen de ronden zich razendsnel op, ook als het
       absolute aantal samples best meevalt.

    Regel 2 alleen had niet gereikt. Een meting op de meegeleverde dataset
    (484 trainingsvoorbeelden, batch 16, 2500 stappen) liet het optimum zien bij
    stap 400 — 13 ronden — met een eval-verlies van 2,78. Daarna liep het op naar
    3,04 terwijl het trainverlies bleef dalen naar 2,36. Omdat 484 boven de oude
    grens van 300 lag, greep de bewaking niet in en trainde hij 450 stappen door
    in een richting die alleen maar slechter werd.
    """
    if patience > 0:
        return patience
    if patience < 0:
        return 0            # bewust uitgezet via --no-early-stop
    epochs = planned_epochs(n_train_samples, batch_size, max_steps)
    if n_train_samples < sample_floor or epochs > epoch_ceiling:
        return 4
    return 0


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    tok_cfg: Optional[TokenizerConfig] = None,
    samples: Optional[Sequence[Sample]] = None,
    tokenizer: Optional[ByteLevelBPE] = None,
    resume: bool = False,
    init_from: Optional[str] = None,
    verbose: bool = True,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    extra_files: Optional[Sequence[str]] = None,
    eval_fraction: float = 0.05,
    root: str = RUNS_DIR,
) -> TrainResult:
    """Traint (of vervolgt) een model. Geeft een TrainResult met de geschiedenis."""
    ensure_dirs()
    directory = run_dir(train_cfg.run_name, root)
    os.makedirs(directory, exist_ok=True)

    # --- tokenizer ---
    if tokenizer is None:
        if os.path.exists(tokenizer_path(train_cfg.run_name, root)):
            tokenizer = load_tokenizer(train_cfg.run_name, root)
        else:
            tok_cfg = tok_cfg or TokenizerConfig(vocab_size=model_cfg.vocab_size)
            tok_cfg.vocab_size = model_cfg.vocab_size
            tokenizer = build_tokenizer(train_cfg.run_name, tok_cfg, verbose=verbose,
                                        files=extra_files)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        if verbose:
            print(f"[train] vocab_size aangepast: {model_cfg.vocab_size} -> {tokenizer.vocab_size}")
        model_cfg.vocab_size = tokenizer.vocab_size

    # --- data ---
    if samples is None:
        if verbose:
            print("[train] dataset bouwen...")
        samples, stats = build_dataset(
            tokenizer, model_cfg.block_size, files=extra_files, repeat_small=1, verbose=verbose
        )
        if not samples:
            raise SystemExit("Geen trainingsdata gevonden. Zet bestanden in ai/data/.")
    else:
        samples = list(samples)
    train_samples, eval_samples = split_samples(samples, eval_fraction, seed=train_cfg.seed)
    if not eval_samples:
        eval_samples = train_samples[: max(1, len(train_samples) // 10)]
    if verbose:
        print(f"[train] {len(train_samples)} trainings- / {len(eval_samples)} evaluatiesamples, "
              f"{dataset_tokens(train_samples)} tokens")

    batcher = Batcher(train_samples, train_cfg.batch_size, model_cfg.block_size,
                      seed=train_cfg.seed)
    eval_set = batcher.eval_batches(train_cfg.eval_batches) if eval_samples else []
    if eval_samples and eval_samples is not train_samples:
        eval_batcher = Batcher(eval_samples, train_cfg.batch_size, model_cfg.block_size, seed=7)
        eval_set = eval_batcher.eval_batches(train_cfg.eval_batches)

    # --- model ---
    start_step = 0
    best_eval: Optional[float] = None
    resumed_best_step: Optional[int] = None
    saved_loss: Optional[float] = None
    saved_tokens: int = 0
    if resume and checkpoint_exists(train_cfg.run_name, root):
        model, meta = TransformerLM.load(directory)
        # De preset zegt bijvoorbeeld context 256, maar dit model is misschien
        # op 512 getraind. Neem de werkelijke vorm over, anders logt en toont
        # train een andere context dan het model echt heeft.
        model_cfg = model.cfg
        saved = load_json(state_path(train_cfg.run_name, root), {}) or {}
        start_step = int(saved.get("step", 0))
        best_eval = saved.get("best_eval")
        saved_loss = saved.get("loss")
        saved_tokens = int(saved.get("tokens_seen") or 0)
        # Zonder dit valt "beste checkpoint" bij hervatten terug op de startstap,
        # terwijl het optimum vaak veel eerder lag. Oudere statusbestanden hebben
        # het veld niet; die vallen terug op de metadata van best.npz.
        resumed_best_step = saved.get("best_step")
        if resumed_best_step is None:
            best_meta = load_json(os.path.join(run_dir(train_cfg.run_name, root),
                                               "best.json"), {}) or {}
            resumed_best_step = best_meta.get("step")
        if verbose:
            print(f"[train] hervat vanaf stap {start_step} "
                  f"(verlies {saved.get('loss', '?')})")
    elif init_from and checkpoint_exists(init_from, root):
        model, meta = TransformerLM.load(run_dir(init_from, root))
        if model.cfg.vocab_size != model_cfg.vocab_size:
            raise SystemExit(
                f"Kan {init_from} niet als start gebruiken: woordenschat "
                f"{model.cfg.vocab_size} != {model_cfg.vocab_size}. Bouw eerst een nieuwe tokenizer."
            )
        if verbose:
            print(f"[train] start vanaf bestaand model '{init_from}'")
    else:
        model = TransformerLM(model_cfg)
    model.params = {k: v.astype(F32, copy=False) for k, v in model.params.items()}
    model.dtype = np.dtype(F32)

    n_params = model.num_params()
    if verbose:
        print(f"[train] model: {model_cfg.n_layer} lagen, {model_cfg.n_embd} breed, "
              f"{model_cfg.n_head} koppen, context {model_cfg.block_size}, "
              f"{n_params / 1e6:.2f}M parameters")

    opt = nn.AdamW(
        model.params,
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )
    opt.t = start_step

    # --- aantal stappen ---
    steps_per_epoch = len(batcher)
    if train_cfg.epochs and train_cfg.epochs > 0:
        max_steps = start_step + int(train_cfg.epochs * max(1, steps_per_epoch))
    else:
        max_steps = train_cfg.max_steps

    # Hervatten met een stap-aantal dat al gepasseerd is zou nul stappen trainen
    # en dan toch het model en de status overschrijven — met een NaN-verlies erin.
    # Beter: duidelijk zeggen dat er niets te doen is en niets aanraken.
    if start_step > 0 and max_steps <= start_step:
        if verbose:
            print(f"[train] dit model staat al op stap {start_step}; er is niets te doen "
                  f"tot stap {max_steps}.")
            print(f"[train] wil je verder, geef dan meer stappen mee, bijvoorbeeld "
                  f"--steps {start_step + 500}.")
        return TrainResult(
            run_name=train_cfg.run_name, steps=start_step,
            final_loss=float(saved_loss) if saved_loss is not None else float("nan"),
            best_eval=best_eval, tokens_seen=saved_tokens, seconds=0.0,
            params=n_params, history=[],
        )

    train_cfg.max_steps = max(1, max_steps)  # voor het leerplannen
    if verbose:
        print(f"[train] {train_cfg.max_steps - start_step} stappen "
              f"({steps_per_epoch} batches per ronde), batch {train_cfg.batch_size}")

    # --- lus ---
    # Bij weinig data is overfitten bijna onvermijdelijk: zet vroeg stoppen dan
    # automatisch aan, zodat je niet 600 stappen wacht op een slechter model.
    # Bij weinig data óf bij veel geplande ronden is overfitten bijna
    # onvermijdelijk: zet vroeg stoppen dan automatisch aan, zodat je niet
    # honderden stappen wacht op een model dat alleen maar slechter wordt.
    patience = auto_early_stop_patience(
        len(train_samples), train_cfg.batch_size, train_cfg.max_steps,
        train_cfg.early_stop_patience,
    )
    if patience > 0 and train_cfg.early_stop_patience <= 0:
        epochs = planned_epochs(len(train_samples), train_cfg.batch_size,
                                train_cfg.max_steps)
        if verbose:
            reden = ("weinig data" if len(train_samples) < 300
                     else f"±{epochs:.0f} ronden over dezelfde data")
            print(f"[train] {reden} ({len(train_samples)} samples) → vroeg stoppen "
                  f"staat automatisch aan (patience {patience})")
    stale = 0
    # Bij een verse start is de beste stap gewoon de beginstap; bij hervatten
    # nemen we over waar het optimum werkelijk lag.
    best_step = resumed_best_step if resumed_best_step is not None else start_step
    overfit_warned = False
    early_stopped = False

    history: List[Dict[str, Any]] = []
    log_fh = open(log_path(train_cfg.run_name, root), "a", encoding="utf-8")
    batch_iter = batcher.batches()
    running = 0.0
    running_n = 0
    # Bij hervatten telt de eerdere stand mee, anders rapporteert elke hervatting
    # alleen zijn eigen tokens en lijkt het model veel minder getraind dan het is.
    tokens_seen = saved_tokens if resume else 0
    t0 = time.time()
    step = start_step
    accum = max(1, int(train_cfg.grad_accum or 1))

    def log_line(payload: Dict[str, Any]) -> None:
        history.append(payload)
        log_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        log_fh.flush()
        if progress:
            try:
                progress(payload)
            except Exception:
                pass

    try:
        while step < train_cfg.max_steps:
            opt_step_grads = zero_grads(model.params)
            for _ in range(accum):
                try:
                    ids, targets, mask = next(batch_iter)
                except StopIteration:
                    batch_iter = batcher.batches()
                    ids, targets, mask = next(batch_iter)
                loss, grads, _, cache = model.forward(ids, targets, mask)
                accumulate(opt_step_grads, grads, 1.0 / accum)
                ntok = int(cache.get("ntok", 0))
                running += float(loss) * ntok
                running_n += ntok
                tokens_seen += int((mask > 0).sum())

            norm = nn.clip_grad_norm(opt_step_grads, train_cfg.grad_clip)
            lr = nn.lr_at(step, train_cfg)
            opt.step(opt_step_grads, lr=lr)
            step += 1
            opt.t = step

            if step % train_cfg.log_every == 0 or step == train_cfg.max_steps:
                avg = running / max(1, running_n)
                elapsed = time.time() - t0
                done = max(1, step - start_step)
                payload = {
                    "step": step,
                    "loss": round(avg, 4),
                    "lr": round(lr, 6),
                    "grad_norm": round(norm, 3),
                    "tokens_seen": tokens_seen,
                    "sec": round(elapsed, 1),
                    "tokens_per_sec": round(tokens_seen / max(1e-6, elapsed), 1),
                    "eval_loss": None,
                }
                if verbose:
                    print(f"  stap {step:6d}/{train_cfg.max_steps}  verlies {avg:.4f}  "
                          f"lr {lr:.2e}  |g| {norm:.2f}  "
                          f"{payload['tokens_per_sec']:.0f} tok/s")
                running, running_n = 0.0, 0
                if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
                    ev = evaluate(model, eval_set, train_cfg.eval_batches)
                    payload["eval_loss"] = round(ev, 4)
                    improved = math.isfinite(ev) and (best_eval is None or ev < best_eval - 1e-4)
                    if improved:
                        best_eval = ev
                        best_step = step
                        stale = 0
                        model.save(directory, "best", extra={
                            "step": step, "eval_loss": best_eval,
                            "tokenizer_vocab": tokenizer.vocab_size,
                        })
                    elif math.isfinite(ev):
                        stale += 1
                    if verbose:
                        print(f"          eval-verlies {ev:.4f}"
                              + ("  ← nieuw beste" if improved else f"  (beste blijft {best_eval:.4f})"))
                    # Overfit-melding: het model leert de voorbeelden uit zijn hoofd.
                    if (math.isfinite(ev) and train_cfg.overfit_factor > 0
                            and ev > avg * train_cfg.overfit_factor and avg > 0):
                        payload["overfit"] = True
                        if verbose and not overfit_warned:
                            overfit_warned = True
                            print(f"          ⚠ eval-verlies {ev:.2f} ligt ver boven "
                                  f"trainingsverlies {avg:.2f}: het model onthoudt de "
                                  f"voorbeelden in plaats van de taal te leren.")
                            print("            Meer data is de enige echte oplossing: "
                                  "python -m webai distill  (of eigen bestanden in "
                                  "ai/data/learned/).")
                            print("            Het beste checkpoint blijft bewaard als "
                                  "'best' en wordt standaard geserveerd.")
                log_line(payload)
                if patience > 0 and stale >= patience and step < train_cfg.max_steps:
                    if verbose:
                        print(f"[train] vroeg gestopt op stap {step}: het eval-verlies "
                              f"verbeterde {patience} evaluaties op rij niet "
                              f"(beste {best_eval} op stap {best_step}).")
                    early_stopped = True
                    break

            if step % train_cfg.save_every == 0 and step < train_cfg.max_steps:
                model.save(directory, f"step{step}", extra={
                    "step": step, "tokenizer_vocab": tokenizer.vocab_size,
                })
                rotate_checkpoints(directory, train_cfg.keep_checkpoints)
                save_json(state_path(train_cfg.run_name, root), {
                    "step": step,
                    "loss": history[-1]["loss"] if history else None,
                    "eval_loss": history[-1].get("eval_loss") if history else None,
                    "best_eval": best_eval,
                    "best_step": best_step,
                    "tokens_seen": tokens_seen,
                    "run_name": train_cfg.run_name,
                    "model": model_cfg.to_dict(),
                    "train": train_cfg.to_dict(),
                    "saved_at": time.time(),
                })
    except KeyboardInterrupt:
        if verbose:
            print("\n[train] onderbroken — checkpoint wordt bewaard")
    finally:
        log_fh.close()

    # --- einde: definitief bewaren ---
    seconds = time.time() - t0
    final_loss = history[-1]["loss"] if history else float("nan")
    model.save(directory, "model", extra={
        "step": step,
        "loss": final_loss,
        "eval_loss": history[-1].get("eval_loss") if history else None,
        "best_eval": best_eval,
        "tokens_seen": tokens_seen,
        "seconds": round(seconds, 1),
        "tokenizer_vocab": tokenizer.vocab_size,
        "trained_at": time.time(),
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "early_stopped": early_stopped,
        "best_step": best_step,
        "overfit": bool(overfit_warned),
        "n_train_samples": len(train_samples),
    })
    save_json(state_path(train_cfg.run_name, root), {
        "step": step, "loss": final_loss, "best_eval": best_eval,
        "best_step": best_step,
        "tokens_seen": tokens_seen, "run_name": train_cfg.run_name,
        "model": model_cfg.to_dict(), "train": train_cfg.to_dict(),
        "saved_at": time.time(),
    })
    result = TrainResult(
        run_name=train_cfg.run_name, steps=step, final_loss=float(final_loss),
        best_eval=best_eval, tokens_seen=tokens_seen, seconds=seconds,
        params=n_params, history=history,
    )
    if verbose:
        best_text = "geen" if best_eval is None else f"{best_eval:.4f}"
        print(f"[train] klaar: {step} stappen in {seconds:.0f}s, "
              f"verlies {final_loss:.4f}, beste eval {best_text}")
        print(f"[train] checkpoint: {directory}")
    return result


def train_from_preset(
    preset_name: str = "small",
    run_name: Optional[str] = None,
    resume: bool = False,
    init_from: Optional[str] = None,
    verbose: bool = True,
    extra_files: Optional[Sequence[str]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    **overrides: Any,
) -> TrainResult:
    """Traint met een preset (tiny/small/medium/large); overrides mogen losse velden zijn."""
    model_cfg, train_cfg = configs_from_preset(preset_name, **overrides)
    if run_name:
        train_cfg.run_name = run_name
    tok_cfg = TokenizerConfig(vocab_size=model_cfg.vocab_size)
    return train(
        model_cfg, train_cfg, tok_cfg,
        resume=resume, init_from=init_from, verbose=verbose,
        extra_files=extra_files, progress=progress,
    )


def model_info(name: str, root: str = RUNS_DIR) -> Dict[str, Any]:
    """Korte samenvatting van een getraind model (voor de UI en `python -m webai info`)."""
    directory = run_dir(name, root)
    meta = load_json(os.path.join(directory, "model.json"), {}) or {}
    state = load_json(state_path(name, root), {}) or {}
    if not meta and not state:
        return {"exists": False, "run_name": name}
    log = []
    path = log_path(name, root)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        log.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    cfg = meta.get("model") or state.get("model") or {}
    # Soms staat het aantal parameters niet in de metadata (bijvoorbeeld wanneer
    # het trainen onderbroken werd vóór de eindopslag). Reken het dan uit de
    # vorm van het netwerk — dat is exact hetzelfde getal.
    params = meta.get("num_params") or 0
    if not params and cfg:
        try:
            params = ModelConfig.from_dict(cfg).estimate_params()
        except Exception:
            params = 0
    return {
        "exists": True,
        "run_name": name,
        "path": directory,
        "params": params,
        "step": meta.get("step") or state.get("step") or 0,
        "loss": meta.get("loss") if meta.get("loss") is not None else state.get("loss"),
        "eval_loss": meta.get("eval_loss") if meta.get("eval_loss") is not None else state.get("eval_loss"),
        "best_eval": meta.get("best_eval") or state.get("best_eval"),
        "tokens_seen": meta.get("tokens_seen") or state.get("tokens_seen") or 0,
        "seconds": meta.get("seconds"),
        "trained_at": meta.get("trained_at") or state.get("saved_at"),
        "config": cfg,
        "layers": cfg.get("n_layer"),
        "width": cfg.get("n_embd"),
        "context": cfg.get("block_size"),
        "vocab": cfg.get("vocab_size"),
        "log_tail": log[-40:],
    }
