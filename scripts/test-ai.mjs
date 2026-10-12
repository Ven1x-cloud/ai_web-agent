#!/usr/bin/env node
// Draait de tests van het eigen AI-model (ai/tests/run_all.py).
//
//   npm run test:ai              alles
//   npm run test:ai -- server    één suite
//
// Zoekt zelf een bruikbare Python en zegt duidelijk wat er mist, in plaats van
// een raadselachtige "command not found" te geven.

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const AI = path.join(ROOT, "ai");
const RUNNER = path.join(AI, "tests", "run_all.py");

/** Kandidaten, van meest naar minst specifiek. */
function pythonCandidates() {
  const list = [
    path.join(ROOT, ".venv", "bin", "python"),        // macOS / Linux
    path.join(ROOT, ".venv", "Scripts", "python.exe"), // Windows
    path.join(ROOT, "ai", ".venv", "bin", "python"),
    path.join(ROOT, "ai", ".venv", "Scripts", "python.exe"),
    process.env.PYTHON || "",
    "python3",
    "python",
  ];
  return list.filter(Boolean);
}

function works(cmd) {
  try {
    execFileSync(cmd, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function hasNumpy(cmd) {
  try {
    execFileSync(cmd, ["-c", "import numpy"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function findPython() {
  const candidates = pythonCandidates();
  const usable = candidates.filter((c) => c.includes(path.sep) ? existsSync(c) : works(c));
  if (!usable.length) return { error: "python-missing" };
  // Een absolute pad bestaat al; bij 'python3'/'python' controleren we nog even.
  const found = usable.find((c) => c.includes(path.sep) || works(c));
  if (!found) return { error: "python-missing" };
  if (!hasNumpy(found)) return { error: "numpy-missing", python: found };
  return { python: found };
}

function main() {
  if (!existsSync(RUNNER)) {
    console.error(`✗ ${path.relative(ROOT, RUNNER)} bestaat niet.`);
    process.exit(1);
  }

  const { python, error } = findPython();
  if (error === "python-missing") {
    console.error("✗ Geen Python gevonden. Installeer Python 3.10 of nieuwer:");
    console.error("    https://www.python.org/downloads/");
    process.exit(1);
  }
  if (error === "numpy-missing") {
    console.error(`✗ Python gevonden (${python}), maar NumPy ontbreekt.`);
    console.error("  Maak een virtuele omgeving en installeer de afhankelijkheden:");
    console.error("    python3 -m venv .venv");
    console.error("    .venv/bin/pip install -r ai/requirements.txt");
    console.error("  (Windows: .venv\\Scripts\\pip install -r ai\\requirements.txt)");
    process.exit(1);
  }

  const args = process.argv.slice(2);
  console.log(`Python: ${path.relative(ROOT, python) || python}\n`);
  try {
    execFileSync(python, [RUNNER, ...args], { cwd: AI, stdio: "inherit" });
  } catch (e) {
    process.exit(e.status ?? 1);
  }
}

main();
