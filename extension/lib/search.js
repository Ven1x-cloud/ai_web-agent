// Gratis webzoeken (DuckDuckGo, met Bing als reserve) en webpagina's ophalen als tekst.
// Draait in het zijpaneel (extensiepagina), dus DOMParser is beschikbaar en er is geen CORS-probleem
// dankzij host_permissions.

const UA_HEADERS = { Accept: "text/html,application/xhtml+xml", "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.8,de;q=0.7,fr;q=0.6" };

const clean = (s) => (s || "").replace(/\s+/g, " ").trim();

function decodeDdgHref(href) {
  try {
    const u = new URL(href, "https://duckduckgo.com");
    if (u.hostname.endsWith("duckduckgo.com") && u.searchParams.get("uddg")) return u.searchParams.get("uddg");
    return u.href;
  } catch (_) { return href; }
}

async function fetchText(url, init = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), init.timeout || 15000);
  try {
    const res = await fetch(url, { ...init, signal: ctrl.signal, headers: { ...UA_HEADERS, ...(init.headers || {}) } });
    const text = await res.text();
    return { ok: res.ok, status: res.status, text, contentType: res.headers.get("content-type") || "", url: res.url };
  } finally { clearTimeout(t); }
}

async function searchDuckDuckGo(query, { max = 6, region = "nl-nl" } = {}) {
  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}&kl=${region}`;
  const r = await fetchText(url);
  if (!r.ok) throw new Error(`DuckDuckGo gaf status ${r.status}`);
  const doc = new DOMParser().parseFromString(r.text, "text/html");
  if (doc.querySelector(".anomaly-modal, #challenge-form") || /captcha|anomaly/i.test(doc.title)) {
    throw new Error("DuckDuckGo vraagt om een captcha (te veel verzoeken).");
  }
  const results = [];
  for (const el of doc.querySelectorAll(".result")) {
    const a = el.querySelector("a.result__a");
    if (!a) continue;
    const link = decodeDdgHref(a.getAttribute("href") || "");
    if (!/^https?:/i.test(link)) continue;
    const snippet = clean(el.querySelector(".result__snippet")?.textContent);
    results.push({ title: clean(a.textContent), url: link, snippet });
    if (results.length >= max) break;
  }
  return results;
}

async function searchBing(query, { max = 6 } = {}) {
  const url = `https://www.bing.com/search?q=${encodeURIComponent(query)}&setlang=nl&cc=NL`;
  const r = await fetchText(url);
  if (!r.ok) throw new Error(`Bing gaf status ${r.status}`);
  const doc = new DOMParser().parseFromString(r.text, "text/html");
  const results = [];
  for (const li of doc.querySelectorAll("li.b_algo")) {
    const a = li.querySelector("h2 a");
    if (!a || !/^https?:/i.test(a.href || a.getAttribute("href") || "")) continue;
    const snippet = clean(li.querySelector(".b_caption p, .b_lineclamp2, .b_algoSlug")?.textContent);
    results.push({ title: clean(a.textContent), url: a.getAttribute("href"), snippet });
    if (results.length >= max) break;
  }
  return results;
}

/** Zoek op internet. Probeert DuckDuckGo, daarna Bing. */
export async function webSearch(query, opts = {}) {
  const errors = [];
  for (const [name, fn] of [["DuckDuckGo", searchDuckDuckGo], ["Bing", searchBing]]) {
    try {
      const results = await fn(query, opts);
      if (results.length) return { engine: name, query, results };
      errors.push(`${name}: geen resultaten`);
    } catch (e) {
      errors.push(`${name}: ${e.message}`);
    }
  }
  return { engine: null, query, results: [], error: errors.join(" | ") };
}

// ---------- HTML -> leesbare tekst ----------
const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "IFRAME", "SVG", "CANVAS", "HEAD", "NAV", "FOOTER", "ASIDE", "FORM", "BUTTON"]);
const BLOCK = new Set(["P", "DIV", "SECTION", "ARTICLE", "MAIN", "HEADER", "LI", "UL", "OL", "TABLE", "TR", "H1", "H2", "H3", "H4", "H5", "H6", "BR", "PRE", "BLOCKQUOTE", "FIGURE", "FIGCAPTION", "DL", "DT", "DD", "HR", "DETAILS", "SUMMARY"]);

export function htmlToText(doc, { maxChars = 120000 } = {}) {
  const out = [];
  let len = 0;
  const walk = (node) => {
    if (len > maxChars) return;
    if (node.nodeType === 3) {
      const t = node.nodeValue.replace(/\s+/g, " ");
      if (t.trim()) { out.push(t); len += t.length; }
      return;
    }
    if (node.nodeType !== 1) return;
    const tag = node.tagName.toUpperCase();
    if (SKIP.has(tag)) return;
    if (node.getAttribute("aria-hidden") === "true" || node.hidden) return;
    const role = node.getAttribute("role");
    if (role === "navigation" || role === "banner" || role === "contentinfo") return;
    if (tag === "MATH") {
      const ann = node.querySelector('annotation[encoding="application/x-tex"]');
      out.push(ann ? ` $${ann.textContent.trim()}$ ` : ` ${clean(node.textContent)} `);
      return;
    }
    if (tag === "IMG") { if (node.alt) out.push(` [afbeelding: ${clean(node.alt)}] `); return; }
    const block = BLOCK.has(tag);
    if (block) out.push("\n");
    if (/^H[1-6]$/.test(tag)) out.push("#".repeat(Number(tag[1])) + " ");
    if (tag === "LI") out.push("• ");
    if (tag === "TD" || tag === "TH") out.push(" | ");
    for (const c of node.childNodes) walk(c);
    if (tag === "TR") out.push("\n");
    if (block) out.push("\n");
  };
  const root = doc.querySelector("main, article, [role=main]") || doc.body || doc.documentElement;
  walk(root);
  let text = out.join("").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").replace(/ {2,}/g, " ").trim();
  if (root !== doc.body && text.length < 400 && doc.body) {
    // Hoofdinhoud te klein, val terug op de hele body.
    out.length = 0; len = 0;
    walk(doc.body);
    text = out.join("").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").replace(/ {2,}/g, " ").trim();
  }
  return text;
}

/** Haal een URL op en geef de leesbare tekst terug. */
export async function fetchUrlAsText(url, { offset = 0, maxChars = 8000 } = {}) {
  if (!/^https?:\/\//i.test(url)) throw new Error("Alleen http(s)-URL's kunnen worden opgehaald.");
  const r = await fetchText(url, { timeout: 20000 });
  if (!r.ok) throw new Error(`Pagina gaf status ${r.status}`);
  if (/application\/pdf/i.test(r.contentType) || /\.pdf($|\?)/i.test(url)) {
    throw new Error("Dit is een PDF; die kan ik (nog) niet als tekst lezen. Open de PDF in een tabblad en maak een screenshot.");
  }
  if (/application\/json|text\/plain/i.test(r.contentType)) {
    const text = r.text;
    return { url: r.url, title: "", text: text.slice(offset, offset + maxChars), totalChars: text.length, hasMore: offset + maxChars < text.length, offset };
  }
  const doc = new DOMParser().parseFromString(r.text, "text/html");
  const text = htmlToText(doc);
  return {
    url: r.url,
    title: clean(doc.title),
    text: text.slice(offset, offset + maxChars),
    totalChars: text.length,
    hasMore: offset + maxChars < text.length,
    offset,
  };
}
