// Markdown + wiskunde (KaTeX) veilig renderen in het zijpaneel.
// Verwacht de globals `marked`, `DOMPurify` en `katex` (vendor-scripts in sidepanel.html).
//
// Werkwijze: 1) code beschermen, 2) wiskunde beschermen, 3) markdown parsen + saneren,
// 4) wiskunde met KaTeX renderen op de plek van de placeholders. Zo wordt "$5 en $10" nooit
// per ongeluk als formule gezien en blijft wiskunde (underscores, backslashes) intact.

const MATH_PATTERNS = [
  { re: /\$\$([\s\S]+?)\$\$/g, display: true },
  { re: /\\\[([\s\S]+?)\\\]/g, display: true },
  { re: /\\\(([\s\S]+?)\\\)/g, display: false },
  { re: /(?<![\\\w$])\$(?!\s)([^$\n]{1,400}?)(?<!\s)\$(?![\w$])/g, display: false },
];
const CODE_PATTERNS = [/```[\s\S]*?```/g, /~~~[\s\S]*?~~~/g, /`[^`\n]+`/g];

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function protect(text) {
  const code = [];
  let out = text.replace(CODE_PATTERNS[0], (m) => `CODEPLACEHOLDER${code.push(m) - 1}X`)
    .replace(CODE_PATTERNS[1], (m) => `CODEPLACEHOLDER${code.push(m) - 1}X`)
    .replace(CODE_PATTERNS[2], (m) => `CODEPLACEHOLDER${code.push(m) - 1}X`);
  const math = [];
  for (const { re, display } of MATH_PATTERNS) {
    out = out.replace(re, (_m, inner) => `MATHPLACEHOLDER${math.push({ tex: inner, display }) - 1}X`);
  }
  out = out.replace(/CODEPLACEHOLDER(\d+)X/g, (_, i) => code[Number(i)]);
  return { text: out, math };
}

function renderTex({ tex, display }) {
  try {
    if (window.katex) return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
  } catch (_) {}
  return display ? `<pre class="math-fallback">${escapeHtml(tex)}</pre>` : `<code class="math-fallback">${escapeHtml(tex)}</code>`;
}

export function renderMarkdown(el, text) {
  const { text: protectedText, math } = protect(text || "");
  let html;
  try {
    html = window.marked.parse(protectedText, { gfm: true, breaks: true });
  } catch (_) {
    html = escapeHtml(protectedText).replace(/\n/g, "<br>");
  }
  html = window.DOMPurify.sanitize(html, { ADD_ATTR: ["target", "rel"] });
  // Placeholders (nu veilig gesaneerde tekst) vervangen door KaTeX-uitvoer.
  html = html.replace(/(<p>)?MATHPLACEHOLDER(\d+)X(<\/p>)?/g, (m, open, i, close) => {
    const item = math[Number(i)];
    if (!item) return m;
    const rendered = renderTex(item);
    // Display-wiskunde niet in een <p> laten staan (ongeldige HTML → rare marges).
    if (item.display && open && close) return rendered;
    return (open || "") + rendered + (close || "");
  });
  el.innerHTML = html;
  for (const a of el.querySelectorAll("a[href]")) {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener noreferrer");
  }
}

/** Kleine helper voor platte tekst met behoud van regeleinden. */
export function renderPlain(el, text) {
  el.textContent = text || "";
}
