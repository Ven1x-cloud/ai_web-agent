// Content-script van AI Web Agent.
// Wordt on-demand geïnjecteerd (chrome.scripting) in de actieve pagina en biedt het zijpaneel
// "ogen en handen": pagina lezen, afbeeldingen vinden, elementen aanklikken, typen, scrollen.
// Het script is idempotent: bij herhaald injecteren gebeurt er niets.
(() => {
  if (window.__aiWebAgentContent) return;
  window.__aiWebAgentContent = { version: 1 };

  const state = {
    els: [], // interactieve elementen, index = id
    imgs: [], // afbeeldingen, index = id
    imgIds: new Map(), // element -> id
    pageTextCache: null,
  };

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "IFRAME", "OBJECT", "EMBED", "VIDEO", "AUDIO",
    "HEAD", "META", "LINK", "TITLE", "CANVAS",
  ]);
  const NOISE_TAGS = new Set(["NAV", "FOOTER", "ASIDE"]);
  const BLOCK_TAGS = new Set([
    "P", "DIV", "SECTION", "ARTICLE", "MAIN", "HEADER", "FOOTER", "LI", "UL", "OL", "TABLE", "THEAD",
    "TBODY", "TR", "H1", "H2", "H3", "H4", "H5", "H6", "BR", "PRE", "BLOCKQUOTE", "FIGURE", "FIGCAPTION",
    "DL", "DT", "DD", "NAV", "ASIDE", "FORM", "FIELDSET", "HR", "DETAILS", "SUMMARY", "ADDRESS", "LABEL",
    "OPTION", "BUTTON",
  ]);
  const MAX_TEXT = 200_000;

  // ---------- Hulpfuncties ----------
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const trunc = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");

  function isVisible(el) {
    if (!(el instanceof Element)) return true;
    if (el.hidden) return false;
    if (el.getAttribute && el.getAttribute("aria-hidden") === "true") return false;
    try {
      if (typeof el.checkVisibility === "function") {
        return el.checkVisibility({ checkVisibilityCSS: true });
      }
    } catch (_) {}
    const cs = getComputedStyle(el);
    return cs.display !== "none" && cs.visibility !== "hidden";
  }

  function inViewport(rect) {
    return (
      rect.bottom > 0 && rect.right > 0 &&
      rect.top < (window.innerHeight || document.documentElement.clientHeight) &&
      rect.left < (window.innerWidth || document.documentElement.clientWidth)
    );
  }

  function absUrl(u) {
    try { return new URL(u, location.href).href; } catch (_) { return u; }
  }

  function labelFor(el) {
    const parts = [];
    const aria = el.getAttribute("aria-label");
    if (aria) parts.push(aria);
    if (el.labels && el.labels.length) parts.push(clean(el.labels[0].innerText));
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const t = labelledBy.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean)
        .map((n) => clean(n.innerText)).join(" ");
      if (t) parts.push(t);
    }
    if (el.placeholder) parts.push("placeholder: " + el.placeholder);
    if (el.title) parts.push(el.title);
    const txt = clean(el.innerText || el.textContent);
    if (txt) parts.push(txt);
    if (el.tagName === "INPUT" && (el.type === "submit" || el.type === "button") && el.value) parts.push(el.value);
    if (el.tagName === "IMG" && el.alt) parts.push(el.alt);
    const img = el.querySelector && el.querySelector("img[alt]");
    if (img && img.alt && !txt) parts.push(img.alt);
    if (el.name && parts.length === 0) parts.push("name: " + el.name);
    return trunc(clean([...new Set(parts)].join(" · ")), 120);
  }

  // ---------- Afbeeldingen ----------
  function collectImages() {
    state.imgs = [];
    state.imgIds = new Map();
    const push = (el, info) => {
      const id = state.imgs.length;
      state.imgIds.set(el, id);
      state.imgs.push({ id, ...info });
    };
    const seen = new Set();
    for (const img of Array.from(document.images)) {
      if (seen.has(img)) continue;
      seen.add(img);
      const rect = img.getBoundingClientRect();
      const w = Math.round(rect.width || img.naturalWidth || 0);
      const h = Math.round(rect.height || img.naturalHeight || 0);
      const src = img.currentSrc || img.src || img.getAttribute("data-src") || "";
      if (!src) continue;
      if (w < 40 || h < 40) continue; // iconen/trackingpixels overslaan
      push(img, {
        kind: "img", src: absUrl(src), alt: clean(img.alt), title: clean(img.title),
        width: w, height: h, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight,
        visible: isVisible(img) && inViewport(rect),
      });
    }
    for (const svg of Array.from(document.querySelectorAll("svg"))) {
      const rect = svg.getBoundingClientRect();
      if (rect.width < 80 || rect.height < 80) continue; // alleen diagrammen/grafieken, geen iconen
      if (svg.closest("button, a, nav, header, footer")) continue;
      push(svg, {
        kind: "svg", src: "", alt: clean(svg.getAttribute("aria-label") || svg.querySelector("title")?.textContent),
        width: Math.round(rect.width), height: Math.round(rect.height), visible: isVisible(svg) && inViewport(rect),
      });
    }
    for (const c of Array.from(document.querySelectorAll("canvas"))) {
      const rect = c.getBoundingClientRect();
      if (rect.width < 80 || rect.height < 80) continue;
      push(c, {
        kind: "canvas", src: "", alt: clean(c.getAttribute("aria-label")),
        width: Math.round(rect.width), height: Math.round(rect.height), visible: isVisible(c) && inViewport(rect),
      });
    }
    return state.imgs;
  }

  function mathToText(el) {
    // MathML / KaTeX / MathJax: haal liefst de LaTeX-bron op.
    const ann = el.querySelector && el.querySelector('annotation[encoding="application/x-tex"]');
    if (ann && ann.textContent.trim()) return "$" + ann.textContent.trim() + "$";
    if (el.tagName === "MJX-CONTAINER") {
      const al = el.getAttribute("aria-label");
      if (al) return "[formule: " + clean(al) + "]";
      const mml = el.querySelector("mjx-assistive-mml");
      if (mml) return "[formule: " + clean(mml.textContent) + "]";
    }
    return "[formule: " + clean(el.textContent) + "]";
  }

  // ---------- Tekst uitlezen ----------
  function extractText(root) {
    const out = [];
    let len = 0;
    let stopped = false;
    const pushText = (t) => {
      if (!t) return;
      out.push(t);
      len += t.length;
      if (len > MAX_TEXT) stopped = true;
    };
    const newline = () => {
      if (out.length && out[out.length - 1] !== "\n") out.push("\n");
    };

    function walk(node) {
      if (stopped) return;
      if (node.nodeType === Node.TEXT_NODE) {
        const t = node.nodeValue.replace(/\s+/g, " ");
        if (t.trim()) pushText(t);
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      const el = node;
      const tag = el.tagName;
      if (SKIP_TAGS.has(tag)) return;
      if (tag === "SVG" || tag === "svg") return;
      if (!isVisible(el)) return;
      if (NOISE_TAGS.has(tag) || ["navigation", "banner", "contentinfo"].includes(el.getAttribute("role"))) {
        if (root !== el && !el.contains(root)) return;
      }
      if (tag === "MATH" || tag === "math" || tag === "MJX-CONTAINER") {
        pushText(" " + mathToText(el) + " ");
        return;
      }
      if (el.classList.contains("katex")) {
        const ann = el.querySelector('annotation[encoding="application/x-tex"]');
        pushText(ann ? " $" + ann.textContent.trim() + "$ " : " " + clean(el.textContent) + " ");
        return;
      }
      if (tag === "IMG") {
        const id = state.imgIds.get(el);
        const alt = clean(el.alt);
        if (id !== undefined) pushText(` [afbeelding #${id}${alt ? ": " + alt : ""}] `);
        else if (alt) pushText(` [afbeelding: ${alt}] `);
        return;
      }
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        if (tag === "INPUT" && ["hidden", "submit", "button", "image", "reset"].includes(el.type)) return;
        let v = "";
        if (tag === "SELECT") v = el.selectedOptions?.[0]?.text || "";
        else if (el.type === "checkbox" || el.type === "radio") v = el.checked ? "aangevinkt" : "niet aangevinkt";
        else v = el.value || "";
        const lbl = labelFor(el);
        pushText(` [invoerveld${lbl ? " “" + lbl + "”" : ""}${v ? ": " + trunc(clean(v), 200) : " (leeg)"}] `);
        return;
      }
      const isBlock = BLOCK_TAGS.has(tag);
      if (isBlock) newline();
      if (/^H[1-6]$/.test(tag)) pushText("#".repeat(Number(tag[1])) + " ");
      if (tag === "LI") pushText("• ");
      if (tag === "TD" || tag === "TH") pushText(" | ");
      for (const child of el.childNodes) walk(child);
      if (tag === "TR") newline();
      if (tag === "TD" || tag === "TH") pushText(" ");
      if (isBlock) newline();
    }
    walk(root);
    return out.join("").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").replace(/ {2,}/g, " ").trim();
  }

  function getMainRoot() {
    const candidates = ["main", "[role=main]", "article", "#content", "#main", ".content", ".main"];
    for (const sel of candidates) {
      const el = document.querySelector(sel);
      if (el && isVisible(el)) return el;
    }
    return null;
  }

  function getPageText() {
    collectImages();
    const bodyText = extractText(document.body || document.documentElement);
    const main = getMainRoot();
    if (main) {
      const mainText = extractText(main);
      // Alleen de hoofdinhoud gebruiken als die het grootste deel van de tekst bevat.
      if (mainText.length > 500 && mainText.length >= bodyText.length * 0.5) return mainText;
    }
    return bodyText;
  }

  function getSelectionText() {
    const sel = window.getSelection && window.getSelection();
    let t = sel ? sel.toString() : "";
    if (!t) {
      const a = document.activeElement;
      if (a && (a.tagName === "TEXTAREA" || a.tagName === "INPUT") && a.selectionStart !== a.selectionEnd) {
        t = a.value.slice(a.selectionStart, a.selectionEnd);
      }
    }
    return clean(t);
  }

  function pageInfo({ offset = 0, maxChars = 6000 } = {}) {
    const text = getPageText();
    state.pageTextCache = text;
    const headings = Array.from(document.querySelectorAll("h1, h2, h3")).filter(isVisible)
      .map((h) => trunc(clean(h.innerText), 100)).filter(Boolean).slice(0, 40);
    const frames = Array.from(document.querySelectorAll("iframe")).filter(isVisible).map((f, i) => {
      const r = f.getBoundingClientRect();
      return { index: i, src: absUrl(f.src || ""), title: clean(f.title), width: Math.round(r.width), height: Math.round(r.height) };
    }).filter((f) => f.width > 100 && f.height > 100).slice(0, 10);
    const slice = text.slice(offset, offset + maxChars);
    return {
      url: location.href,
      title: document.title,
      lang: document.documentElement.lang || "",
      description: clean(document.querySelector('meta[name="description"]')?.content || ""),
      selection: trunc(getSelectionText(), 8000),
      headings,
      images: state.imgs.length,
      frames,
      text: slice,
      offset,
      totalChars: text.length,
      hasMore: offset + maxChars < text.length,
      scroll: { y: Math.round(window.scrollY), height: Math.round(document.documentElement.scrollHeight), viewport: window.innerHeight },
    };
  }

  function findText({ query, context = 300, max = 8 }) {
    const text = state.pageTextCache || getPageText();
    const q = (query || "").toLowerCase();
    if (!q) return { matches: [] };
    const lower = text.toLowerCase();
    const matches = [];
    let i = 0;
    while (matches.length < max) {
      const at = lower.indexOf(q, i);
      if (at < 0) break;
      const start = Math.max(0, at - context);
      const end = Math.min(text.length, at + q.length + context);
      matches.push({ offset: at, snippet: (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "") });
      i = at + q.length;
    }
    return { matches, total: matches.length };
  }

  // ---------- Interactieve elementen ----------
  const INTERACTIVE_SELECTOR = [
    "a[href]", "button", "input", "textarea", "select", "summary", "label",
    "[role=button]", "[role=link]", "[role=tab]", "[role=menuitem]", "[role=checkbox]", "[role=radio]",
    "[role=option]", "[role=textbox]", "[role=switch]", "[role=combobox]",
    "[contenteditable='']", "[contenteditable='true']", "[onclick]", "[tabindex]:not([tabindex='-1'])",
  ].join(",");

  function listInteractive({ limit = 150, viewportOnly = false } = {}) {
    const nodes = Array.from(document.querySelectorAll(INTERACTIVE_SELECTOR));
    state.els = [];
    const items = [];
    for (const el of nodes) {
      if (!isVisible(el)) continue;
      if (el.tagName === "INPUT" && el.type === "hidden") continue;
      const rect = el.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) continue;
      const vp = inViewport(rect);
      if (viewportOnly && !vp) continue;
      // Geneste interactieve elementen (bijv. <a><button>) niet dubbel opnemen.
      if (el.parentElement && el.parentElement.closest(INTERACTIVE_SELECTOR) && el.tagName !== "INPUT" && el.tagName !== "SELECT" && el.tagName !== "TEXTAREA") {
        const parentInteractive = el.parentElement.closest(INTERACTIVE_SELECTOR);
        if (parentInteractive && clean(parentInteractive.innerText) === clean(el.innerText)) continue;
      }
      const id = state.els.length;
      state.els.push(el);
      const item = {
        id,
        tag: el.tagName.toLowerCase(),
        label: labelFor(el),
        inViewport: vp,
      };
      if (el.tagName === "INPUT") {
        item.type = el.type;
        if (["checkbox", "radio"].includes(el.type)) item.checked = !!el.checked;
        else if (!["password"].includes(el.type)) item.value = trunc(el.value || "", 120);
      } else if (el.tagName === "TEXTAREA") {
        item.type = "textarea";
        item.value = trunc(el.value || "", 200);
      } else if (el.tagName === "SELECT") {
        item.type = "select";
        item.value = el.selectedOptions?.[0]?.text || "";
        item.options = Array.from(el.options).slice(0, 40).map((o) => o.text);
      } else if (el.isContentEditable) {
        item.type = "contenteditable";
        item.value = trunc(clean(el.innerText), 200);
      } else if (el.tagName === "A") {
        item.href = trunc(el.href, 200);
      }
      if (el.getAttribute("role")) item.role = el.getAttribute("role");
      if (el.disabled) item.disabled = true;
      items.push(item);
      if (items.length >= limit) break;
    }
    return { elements: items, total: nodes.length, truncated: items.length >= limit };
  }

  function getEl(id) {
    const el = state.els[id];
    if (!el || !el.isConnected) {
      throw new Error(`Element #${id} bestaat niet (meer). Vraag eerst opnieuw de interactieve elementen op.`);
    }
    return el;
  }

  function flash(el, color = "#6366f1") {
    try {
      const r = el.getBoundingClientRect();
      const d = document.createElement("div");
      Object.assign(d.style, {
        position: "fixed", left: r.left - 4 + "px", top: r.top - 4 + "px",
        width: r.width + 8 + "px", height: r.height + 8 + "px",
        border: `3px solid ${color}`, borderRadius: "8px", boxShadow: `0 0 0 4px ${color}44`,
        pointerEvents: "none", zIndex: "2147483647", transition: "opacity .4s", opacity: "1",
      });
      document.documentElement.appendChild(d);
      setTimeout(() => { d.style.opacity = "0"; }, 600);
      setTimeout(() => d.remove(), 1100);
    } catch (_) {}
  }

  function mouseSeq(el) {
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const opts = { bubbles: true, cancelable: true, composed: true, view: window, clientX: x, clientY: y, button: 0 };
    for (const type of ["pointerover", "mouseover", "pointerdown", "mousedown"]) {
      el.dispatchEvent(type.startsWith("pointer") ? new PointerEvent(type, { ...opts, pointerId: 1, pointerType: "mouse", isPrimary: true }) : new MouseEvent(type, opts));
    }
    try { el.focus({ preventScroll: true }); } catch (_) {}
    for (const type of ["pointerup", "mouseup", "click"]) {
      el.dispatchEvent(type.startsWith("pointer") ? new PointerEvent(type, { ...opts, pointerId: 1, pointerType: "mouse", isPrimary: true }) : new MouseEvent(type, opts));
    }
  }

  function clickEl({ id }) {
    const el = getEl(id);
    el.scrollIntoView({ block: "center", inline: "nearest" });
    flash(el);
    mouseSeq(el);
    return { ok: true, clicked: labelFor(el), tag: el.tagName.toLowerCase(), url: location.href };
  }

  function setNativeValue(el, value) {
    const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) setter.call(el, value); else el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function typeText({ id, text = "", clear = true, submit = false }) {
    const el = getEl(id);
    el.scrollIntoView({ block: "center", inline: "nearest" });
    flash(el, "#10b981");
    el.focus();
    if (el.isContentEditable) {
      if (clear) {
        const range = document.createRange();
        range.selectNodeContents(el);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
      const ok = document.execCommand("insertText", false, text);
      if (!ok) { if (clear) el.textContent = ""; el.append(document.createTextNode(text)); el.dispatchEvent(new Event("input", { bubbles: true })); }
    } else if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
      const before = clear ? "" : el.value;
      if (clear) { el.select?.(); }
      let ok = false;
      try { ok = document.execCommand("insertText", false, text); } catch (_) { ok = false; }
      if (!ok || el.value !== before + text) setNativeValue(el, before + text);
    } else if (el.tagName === "SELECT") {
      return selectOption({ id, value: text });
    } else {
      throw new Error("Dit element is geen tekstveld.");
    }
    let submitted = false;
    if (submit) {
      const kopts = { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true };
      const notPrevented = el.dispatchEvent(new KeyboardEvent("keydown", kopts));
      el.dispatchEvent(new KeyboardEvent("keypress", kopts));
      el.dispatchEvent(new KeyboardEvent("keyup", kopts));
      if (notPrevented && el.form) {
        if (typeof el.form.requestSubmit === "function") el.form.requestSubmit(); else el.form.submit();
        submitted = true;
      } else if (notPrevented) {
        submitted = false;
      } else submitted = true;
    }
    return { ok: true, field: labelFor(el), value: trunc(el.isContentEditable ? el.innerText : el.value, 200), submitted };
  }

  function selectOption({ id, value }) {
    const el = getEl(id);
    if (el.tagName !== "SELECT") throw new Error("Element is geen keuzelijst (select).");
    const v = String(value).toLowerCase();
    const opt = Array.from(el.options).find((o) => o.value.toLowerCase() === v || o.text.toLowerCase() === v)
      || Array.from(el.options).find((o) => o.text.toLowerCase().includes(v));
    if (!opt) throw new Error("Optie niet gevonden. Beschikbaar: " + Array.from(el.options).map((o) => o.text).join(", "));
    el.value = opt.value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    flash(el, "#10b981");
    return { ok: true, selected: opt.text };
  }

  function pressKey({ id, key = "Enter" }) {
    const el = id !== undefined && id !== null ? getEl(id) : document.activeElement || document.body;
    const map = { Enter: 13, Escape: 27, Tab: 9, ArrowDown: 40, ArrowUp: 38, ArrowLeft: 37, ArrowRight: 39, Backspace: 8, " ": 32 };
    const kopts = { key, code: key === " " ? "Space" : key, keyCode: map[key] || 0, which: map[key] || 0, bubbles: true, cancelable: true };
    const notPrevented = el.dispatchEvent(new KeyboardEvent("keydown", kopts));
    el.dispatchEvent(new KeyboardEvent("keypress", kopts));
    el.dispatchEvent(new KeyboardEvent("keyup", kopts));
    if (key === "Enter" && notPrevented && el.form && el.tagName === "INPUT") el.form.requestSubmit?.();
    return { ok: true };
  }

  function scrollPage({ direction = "down", amount, id } = {}) {
    if (id !== undefined && id !== null) {
      getEl(id).scrollIntoView({ block: "center", behavior: "instant" });
    } else if (direction !== "none") {
      const step = amount || Math.round(window.innerHeight * 0.85);
      if (direction === "top") window.scrollTo({ top: 0, behavior: "instant" });
      else if (direction === "bottom") window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" });
      else window.scrollBy({ top: direction === "up" ? -step : step, behavior: "instant" });
    }
    return {
      ok: true,
      scrollY: Math.round(window.scrollY),
      pageHeight: Math.round(document.documentElement.scrollHeight),
      viewportHeight: window.innerHeight,
      atBottom: window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2,
    };
  }

  // ---------- Afbeeldingsdata ----------
  function imgRect(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, width: r.width, height: r.height, dpr: window.devicePixelRatio || 1, inViewport: inViewport(r) };
  }

  async function imageData({ id, maxSize = 1600 }) {
    if (!state.imgs.length) collectImages();
    const info = state.imgs[id];
    if (!info) throw new Error(`Afbeelding #${id} bestaat niet. Er zijn ${state.imgs.length} afbeeldingen (0 t/m ${state.imgs.length - 1}).`);
    let el = null;
    for (const [k, v] of state.imgIds) if (v === id) el = k;
    if (!el || !el.isConnected) throw new Error("Afbeelding is niet meer op de pagina.");
    el.scrollIntoView({ block: "center", behavior: "instant" });
    flash(el, "#f59e0b");
    const rect = imgRect(el);
    const result = { id, kind: info.kind, src: info.src, alt: info.alt, rect };
    try {
      if (info.kind === "img") {
        if (!el.complete || !el.naturalWidth) throw new Error("not loaded");
        const scale = Math.min(1, maxSize / Math.max(el.naturalWidth, el.naturalHeight));
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(el.naturalWidth * scale));
        c.height = Math.max(1, Math.round(el.naturalHeight * scale));
        c.getContext("2d").drawImage(el, 0, 0, c.width, c.height);
        result.dataUrl = c.toDataURL("image/png"); // gooit SecurityError bij cross-origin ("tainted")
      } else if (info.kind === "canvas") {
        result.dataUrl = el.toDataURL("image/png");
      } else if (info.kind === "svg") {
        result.dataUrl = await svgToPng(el, maxSize);
      }
    } catch (e) {
      result.error = String(e && e.message || e);
    }
    return result;
  }

  function svgToPng(svg, maxSize) {
    return new Promise((resolve, reject) => {
      const clone = svg.cloneNode(true);
      const r = svg.getBoundingClientRect();
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      if (!clone.getAttribute("width")) clone.setAttribute("width", r.width);
      if (!clone.getAttribute("height")) clone.setAttribute("height", r.height);
      // Inline computed kleuren voor de meest voorkomende eigenschappen (CSS gaat anders verloren).
      const src = svg.querySelectorAll("*"), dst = clone.querySelectorAll("*");
      for (let i = 0; i < src.length && i < dst.length; i++) {
        const cs = getComputedStyle(src[i]);
        for (const p of ["fill", "stroke", "stroke-width", "font-size", "font-family", "opacity"]) {
          const v = cs.getPropertyValue(p);
          if (v) dst[i].style.setProperty(p, v);
        }
      }
      const xml = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        try {
          const scale = Math.min(1, maxSize / Math.max(r.width, r.height)) * 2;
          const c = document.createElement("canvas");
          c.width = Math.max(1, Math.round(r.width * scale));
          c.height = Math.max(1, Math.round(r.height * scale));
          const ctx = c.getContext("2d");
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(0, 0, c.width, c.height);
          ctx.drawImage(img, 0, 0, c.width, c.height);
          resolve(c.toDataURL("image/png"));
        } catch (e) { reject(e); } finally { URL.revokeObjectURL(url); }
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("SVG kon niet gerenderd worden")); };
      img.src = url;
    });
  }

  function elementRect({ id }) {
    const el = getEl(id);
    el.scrollIntoView({ block: "center", behavior: "instant" });
    return imgRect(el);
  }

  // ---------- Berichten ----------
  const handlers = {
    ping: () => ({ ok: true, url: location.href, title: document.title }),
    page_info: pageInfo,
    find_text: findText,
    images: ({ limit = 60 } = {}) => {
      const imgs = collectImages();
      return { images: imgs.slice(0, limit), total: imgs.length };
    },
    image_data: imageData,
    interactive: listInteractive,
    click: clickEl,
    type: typeText,
    select: selectOption,
    key: pressKey,
    scroll: scrollPage,
    element_rect: elementRect,
    selection: () => ({ selection: getSelectionText() }),
  };

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || msg.__aiWebAgent !== true) return false;
    const fn = handlers[msg.action];
    if (!fn) { sendResponse({ ok: false, error: "Onbekende actie: " + msg.action }); return false; }
    Promise.resolve()
      .then(() => fn(msg.args || {}))
      .then((result) => sendResponse({ ok: true, result }))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true; // asynchroon antwoord
  });
})();
