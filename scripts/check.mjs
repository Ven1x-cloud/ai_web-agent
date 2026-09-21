// Snelle controle: manifest geldig, alle genoemde bestanden bestaan, vendor aanwezig, eslint schoon.
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const ext = join(root, "extension");
let problems = 0;
const fail = (m) => { console.error("✗", m); problems++; };
const ok = (m) => console.log("✓", m);

const manifest = JSON.parse(readFileSync(join(ext, "manifest.json"), "utf8"));
ok(`manifest.json geldig (v${manifest.version}, MV${manifest.manifest_version})`);

const refs = [
  manifest.background?.service_worker,
  manifest.side_panel?.default_path,
  manifest.options_ui?.page,
  ...Object.values(manifest.icons || {}),
  ...Object.values(manifest.action?.default_icon || {}),
];
for (const r of refs) if (!existsSync(join(ext, r))) fail(`manifest verwijst naar ontbrekend bestand: ${r}`);

for (const v of ["marked.umd.js", "purify.min.js", "katex.min.js", "katex.min.css", "fonts/KaTeX_Main-Regular.woff2"]) {
  if (!existsSync(join(ext, "vendor", v))) fail(`vendor ontbreekt: ${v} (draai: npm run vendor)`);
}
if (!problems) ok("alle bestanden uit manifest en vendor aanwezig");

// HTML-pagina's: verwijzen ze naar bestaande scripts/styles?
for (const html of ["sidepanel/sidepanel.html", "options/options.html"]) {
  const src = readFileSync(join(ext, html), "utf8");
  const dir = dirname(join(ext, html));
  for (const m of src.matchAll(/(?:src|href)="([^"]+)"/g)) {
    const p = m[1];
    if (/^(https?:|#|mailto:)/.test(p)) continue;
    if (!existsSync(join(dir, p))) fail(`${html} verwijst naar ontbrekend bestand: ${p}`);
  }
}
ok("HTML-verwijzingen gecontroleerd");

try {
  execSync("npx eslint extension scripts test", { cwd: root, stdio: "inherit" });
  ok("eslint schoon");
} catch {
  fail("eslint meldde problemen");
}

if (problems) { console.error(`\n${problems} probleem/problemen gevonden.`); process.exit(1); }
console.log("\nAlles in orde.");
