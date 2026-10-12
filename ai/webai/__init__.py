"""WebAI – een eigen, trainbaar taalmodel voor de AI Web Agent.

Deze map is bewust volledig los van de browserextensie: het is een normaal
Python-pakket met een tokenizer, een transformer, een trainingslus en een
OpenAI-compatibele server. De extensie praat er via HTTP mee.

Snelstart:
    python -m webai build      # tokenizer + dataset bouwen uit ai/data
    python -m webai train      # trainen (of: train --preset tiny)
    python -m webai serve      # server op http://localhost:8787/v1
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
