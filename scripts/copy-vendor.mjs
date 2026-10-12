// Kopieert de benodigde third-party bestanden uit node_modules naar extension/vendor.
// Draai: npm run vendor   (alleen nodig na `npm install` / updaten van libs)
import { copyFileSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const nm = join(root, "node_modules");
const out = join(root, "extension", "vendor");
mkdirSync(join(out, "fonts"), { recursive: true });

const files = [
  ["marked/lib/marked.umd.js", "marked.umd.js"],
  ["dompurify/dist/purify.min.js", "purify.min.js"],
  ["katex/dist/katex.min.js", "katex.min.js"],
  ["katex/dist/katex.min.css", "katex.min.css"],
];
for (const [src, dst] of files) {
  const from = join(nm, src);
  if (!existsSync(from)) throw new Error("Ontbreekt: " + from + " (draai eerst npm install)");
  copyFileSync(from, join(out, dst));
  console.log("✓", dst);
}
// Alleen woff2-fonts zijn nodig voor moderne Chromium-browsers.
const fontDir = join(nm, "katex/dist/fonts");
let n = 0;
for (const f of readdirSync(fontDir)) {
  if (f.endsWith(".woff2")) { copyFileSync(join(fontDir, f), join(out, "fonts", f)); n++; }
}
console.log("✓", n, "KaTeX woff2 fonts");
