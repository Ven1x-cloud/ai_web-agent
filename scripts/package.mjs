// Maakt een ZIP van de map `extension/` zoals de Chrome Web Store en Microsoft Edge Add-ons die willen:
// manifest.json in de ROOT van de zip (niet in een submap). Zonder externe afhankelijkheden (werkt op Windows/Mac/Linux).
//
//   node scripts/package.mjs            → release/ai-web-agent.zip + release/ai-web-agent-v<versie>.zip
//   node scripts/package.mjs --out pad  → andere doelmap
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, copyFileSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateRawSync } from "node:zlib";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const srcDir = join(root, "extension");
const args = process.argv.slice(2);
const outDir = args.includes("--out") ? args[args.indexOf("--out") + 1] : join(root, "release");
const SKIP = /(^|\/)(\.DS_Store|Thumbs\.db|desktop\.ini|.*\.map)$/;

const manifest = JSON.parse(readFileSync(join(srcDir, "manifest.json"), "utf8"));
const version = manifest.version;

// ---------- Bestanden verzamelen ----------
function walk(dir, out = []) {
  for (const name of readdirSync(dir).sort()) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, out);
    else if (!SKIP.test(full.replace(/\\/g, "/"))) out.push({ full, rel: relative(srcDir, full).replace(/\\/g, "/"), mtime: st.mtime });
  }
  return out;
}

// ---------- Minimale ZIP-schrijver (deflate, UTF-8-namen) ----------
const CRC_TABLE = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function dosDateTime(d) {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}
function u16(n) { const b = Buffer.alloc(2); b.writeUInt16LE(n); return b; }
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n >>> 0); return b; }

function buildZip(files) {
  const parts = [];
  const central = [];
  let offset = 0;
  for (const f of files) {
    const data = readFileSync(f.full);
    const nameBuf = Buffer.from(f.rel, "utf8");
    const deflated = deflateRawSync(data, { level: 9 });
    const useDeflate = deflated.length < data.length;
    const payload = useDeflate ? deflated : data;
    const method = useDeflate ? 8 : 0;
    const crc = crc32(data);
    const { time, date } = dosDateTime(f.mtime);
    const local = Buffer.concat([
      u32(0x04034b50), u16(20), u16(0x0800), u16(method), u16(time), u16(date),
      u32(crc), u32(payload.length), u32(data.length), u16(nameBuf.length), u16(0), nameBuf,
    ]);
    parts.push(local, payload);
    central.push(Buffer.concat([
      u32(0x02014b50), u16(20), u16(20), u16(0x0800), u16(method), u16(time), u16(date),
      u32(crc), u32(payload.length), u32(data.length), u16(nameBuf.length), u16(0), u16(0),
      u16(0), u16(0), u32(0), u32(offset), nameBuf,
    ]));
    offset += local.length + payload.length;
  }
  const centralBuf = Buffer.concat(central);
  const eocd = Buffer.concat([
    u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
    u32(centralBuf.length), u32(offset), u16(0),
  ]);
  return Buffer.concat([...parts, centralBuf, eocd]);
}

// ---------- Uitvoeren ----------
const files = walk(srcDir);
if (!files.some((f) => f.rel === "manifest.json")) {
  console.error("manifest.json niet gevonden in extension/ – verkeerde map?");
  process.exit(1);
}
const zip = buildZip(files);
mkdirSync(outDir, { recursive: true });
const stable = join(outDir, "ai-web-agent.zip");
const versioned = join(outDir, `ai-web-agent-v${version}.zip`);
writeFileSync(stable, zip);
copyFileSync(stable, versioned);
const kb = (zip.length / 1024).toFixed(0);
console.log(`✓ ${files.length} bestanden, ${kb} KB → ${relative(root, stable)} en ${relative(root, versioned)}`);
console.log("  Upload dit bestand in de Chrome Web Store / Edge Add-ons, of hang het aan een GitHub-release.");
