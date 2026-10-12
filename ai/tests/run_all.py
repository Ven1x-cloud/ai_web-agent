#!/usr/bin/env python3
"""Draait alle tests van het eigen AI-model.

    cd ai
    python tests/run_all.py            # alles
    python tests/run_all.py chat server  # alleen die twee

Sluit af met code 0 als alles groen is, anders 1. Zo kan CI erop vertrouwen.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.dirname(HERE)

# Volgorde = van laag naar hoog: eerst de wiskunde, dan de onderdelen, dan de server.
SUITES = [
    ("gradcheck", "test_gradcheck.py", "handgeschreven backprop klopt met numerieke afgeleiden"),
    ("tokenizer", "test_tokenizer.py", "byte-level BPE: coderen, ontcoderen, bewaren"),
    ("chat", "test_chat.py", "chatsjabloon en het teruglezen van tool-aanroepen"),
    ("dataset", "test_dataset.py", "datasetbouw en welke tokens meegeteld worden"),
    ("training", "test_training.py", "trainingslus: verlies, checkpoints, hervatten"),
    ("server", "test_server.py", "OpenAI-compatibele HTTP-server, echte verzoeken"),
    ("feedback", "test_feedback.py", "feedback bewaren en omzetten naar trainingsdata"),
]


def run(name: str, filename: str, python: str) -> int:
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        print(f"  ✗ {name}: {filename} bestaat niet")
        return 1
    t0 = time.time()
    proc = subprocess.run([python, path], cwd=AI_DIR, capture_output=True, text=True)
    seconds = time.time() - t0
    tail = [line for line in proc.stderr.splitlines() if line.strip()]
    # unittest schrijft naar stderr; pak de laatste regels met de samenvatting
    summary = next((line for line in reversed(tail)
                    if line.startswith(("OK", "FAILED", "Ran "))), "")
    counts = next((line for line in reversed(tail) if line.startswith("Ran ")), "")
    amount = counts.replace("Ran ", "").split(" in")[0].replace(" tests", "").replace(" test", "")
    if proc.returncode == 0:
        print(f"  ✓ {name:<11} {amount:>3} tests ({seconds:.1f}s)")
        return 0
    print(f"  ✗ {name:<11} {summary or 'mislukt'}")
    print(proc.stdout)
    print(proc.stderr)
    return 1


def main(argv: list[str]) -> int:
    only = set(argv)
    python = sys.executable
    selected = [(n, f) for (n, f, _d) in SUITES if not only or n in only]
    unknown = only - {n for (n, _f, _d) in SUITES}
    if unknown:
        print(f"Onbekende suites: {', '.join(sorted(unknown))}")
        print(f"Kies uit: {', '.join(n for (n, _f, _d) in SUITES)}")
        return 2
    if not selected:
        print("Niets te doen.")
        return 0

    print(f"Python {sys.version.split()[0]} — {len(selected)} suites")
    print()
    failures = [n for (n, f) in selected if run(n, f, python)]
    print()
    if failures:
        print(f"MISLUKT: {', '.join(failures)}")
        return 1
    print("Alles groen.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
