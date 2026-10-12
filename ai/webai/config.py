"""Configuratie voor WebAI: modelgroetes, trainingsinstellingen en paden.

Alles staat in één plek zodat je makkelijk een groter of kleiner model kunt kiezen.
Een "preset" is gewoon een set getallen; je kunt ze ook los overschrijven via de CLI.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# ai/  (de map waar dit bestand in staat)
AI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(AI_ROOT, "data")
SEED_DIR = os.path.join(DATA_DIR, "seed")
LEARNED_DIR = os.path.join(DATA_DIR, "learned")
RUNS_DIR = os.path.join(AI_ROOT, "runs")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")


def ensure_dirs() -> None:
    """Zorgt dat de mappen voor data en checkpoints bestaan."""
    for path in (DATA_DIR, SEED_DIR, LEARNED_DIR, RUNS_DIR):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Vorm van het neurale netwerk.

    n_layer  : aantal transformer-blokken
    n_head   : aantal aandachtskoppen (moet n_embd delen)
    n_embd   : breedte van elke vector (de "grootte" van het model)
    block_size: maximale contextlengte in tokens
    vocab_size: aantal tokens (komt uit de tokenizer)
    dropout  : 0.0 = uit (numpy-implementatie houdt het simpel; regularisatie
               komt uit weight decay en de dataverdeling)
    """

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 512
    vocab_size: int = 4096
    dropout: float = 0.0
    tied_embeddings: bool = True
    # Initialisatie
    init_std: float = 0.02
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) moet deelbaar zijn door n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_dim(self) -> int:
        return 4 * self.n_embd

    def estimate_params(self) -> int:
        """Het exacte aantal parameters, zonder het netwerk te bouwen.

        Handig voor de UI en `webai info`: zo kun je de grootte van een model
        laten zien (of van een preset dat nog getraind moet worden) zonder
        tientallen megabytes aan gewichten in het geheugen te zetten.

        Per laag: 12·d² aan gewichten (qkv 3d², projectie d², MLP 4d² + 4d²)
        en 13·d aan bias- en layernorm-parameters. De uitvoerkop deelt de
        invoermatrix (tied), dus daar komt alleen de bias lm_b bij.
        """
        d = self.n_embd
        per_layer = 12 * d * d + 13 * d
        return (self.vocab_size * d            # wte (invoerembeddings)
                + self.block_size * d          # wpe (posities)
                + self.n_layer * per_layer     # transformer-blokken
                + 2 * d                        # lnf_g, lnf_b
                + self.vocab_size)             # lm_b (uitvoer-bias)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class TrainConfig:
    """Hoe er getraind wordt."""

    # Optimalisatie
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 60
    decay_style: str = "cosine"  # "cosine" | "linear" | "none"

    # Data
    batch_size: int = 16
    max_steps: int = 1500
    epochs: int = 0  # >0 wint van max_steps (handig voor kleine datasets)
    grad_accum: int = 1  # >1 = grotere effectieve batch zonder meer geheugen
    seed: int = 1337

    # Loggen / bewaren
    eval_every: int = 100
    eval_batches: int = 8
    log_every: int = 20
    save_every: int = 250
    run_name: str = "webai"
    keep_checkpoints: int = 3

    # Vroeg stoppen: stop als het eval-verlies dit aantal evaluaties op rij niet
    # meer verbetert. 0 = uit (wordt automatisch aangezet bij weinig data).
    early_stop_patience: int = 0
    # Waarschuw als het eval-verlies deze factor hoger ligt dan het trainingsverlies.
    overfit_factor: float = 1.6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    min_freq: int = 2
    sample_chars: int = 4_000_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizerConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ---------------------------------------------------------------------------
# Presets: van "draait op een oude laptop" tot "serieuze GPU-doos"
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    # ~0,6M parameters. Traint in een paar minuten op een gewone laptop-CPU.
    # Bedoeld om de hele keten (tokenizen -> trainen -> serveren) te proberen.
    "tiny": {
        "model": dict(n_layer=2, n_head=4, n_embd=128, block_size=256, vocab_size=2048),
        "train": dict(batch_size=16, max_steps=800, learning_rate=4e-3, warmup_steps=40),
    },
    # ~4M parameters. Standaard: nog steeds te doen op CPU (uren), mooi op GPU/colab.
    "small": {
        "model": dict(n_layer=4, n_head=4, n_embd=256, block_size=512, vocab_size=4096),
        "train": dict(batch_size=16, max_steps=2500, learning_rate=3e-3, warmup_steps=80),
    },
    # ~15M parameters. Aanrader als je een videokaart hebt of geduld.
    "medium": {
        "model": dict(n_layer=6, n_head=8, n_embd=384, block_size=512, vocab_size=8192),
        "train": dict(batch_size=16, max_steps=6000, learning_rate=2e-3, warmup_steps=150),
    },
    # ~45M parameters. Alleen zinvol met GPU en een flinke dataset.
    "large": {
        "model": dict(n_layer=8, n_head=8, n_embd=512, block_size=1024, vocab_size=16384),
        "train": dict(batch_size=16, max_steps=20000, learning_rate=1e-3, warmup_steps=400),
    },
}


def preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Onbekende preset '{name}'. Kies uit: {', '.join(PRESETS)}")
    return PRESETS[name]


def configs_from_preset(name: str = "small", **overrides: Any):
    """Geeft (ModelConfig, TrainConfig) voor een preset, met optionele overrides."""
    p = preset(name)
    model = dict(p["model"])
    train = dict(p["train"])
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in model:
            model[key] = value
        elif key in train:
            train[key] = value
    return ModelConfig(**model), TrainConfig(**train)


# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------


@dataclass
class GenConfig:
    """Standaardinstellingen bij het maken van tekst (per verzoek te overschrijven)."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stop"] = list(self.stop)
        return d


def json_safe(obj: Any) -> Any:
    """Vervangt NaN en oneindig door `None`, zodat de uitvoer geldige JSON is.

    Python schrijft `NaN` en `Infinity` gewoon weg, maar dat is géén geldige
    JSON: `JSON.parse` in de browser gooit het antwoord dan weg. Omdat verliezen
    tijdens trainen NaN kunnen zijn, zou één zo'n getal de hele
    instellingenpagina kunnen laten stranden. Vandaar deze omzetting op één plek.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(json_safe(obj), fh, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
