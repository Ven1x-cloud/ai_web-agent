// Tool-definities (voor het model) en de uitvoering ervan (in het zijpaneel).
import * as page from "./page.js";
import { webSearch, fetchUrlAsText } from "../lib/search.js";

const MAX_RESULT_CHARS = 14000;

const frameParam = { frame_id: { type: "integer", description: "Optional frame id from list_frames (default: main page)." } };

const READ_TOOLS = [
  {
    name: "read_page",
    description: "Read the text of the current page (more of it, or after the page changed). Returns URL, title, selection, headings and a slice of the text. Use offset to page through long texts.",
    parameters: { type: "object", properties: { offset: { type: "integer", description: "Character offset to start from (default 0)." }, max_chars: { type: "integer", description: "Max characters to return (default 8000, max 20000)." }, ...frameParam }, required: [] },
  },
  {
    name: "find_text",
    description: "Find a word or phrase in the current page text and return the surrounding snippets (handy on long pages).",
    parameters: { type: "object", properties: { query: { type: "string" }, ...frameParam }, required: ["query"] },
  },
  {
    name: "get_images",
    description: "List images, SVG diagrams and canvases on the page with their id, alt text and size. Then call view_image to actually look at one.",
    parameters: { type: "object", properties: { ...frameParam }, required: [] },
  },
  {
    name: "view_image",
    description: "Look at an image so you can read/describe/analyse it (photos, diagrams, graphs, maps, geometry, formulas). Give either the image id from get_images/[afbeelding #N], or a direct image URL.",
    parameters: { type: "object", properties: { id: { type: "integer", description: "Image id on the page." }, url: { type: "string", description: "Direct URL of an image (http/https)." }, ...frameParam }, required: [] },
  },
  {
    name: "take_screenshot",
    description: "Take a screenshot of the visible part of the current tab so you can see layout, shapes, colours, graphs, rendered formulas or anything text extraction misses. Optionally crop to an interactive element id.",
    parameters: { type: "object", properties: { element_id: { type: "integer", description: "Optional id from get_interactive to crop to that element." }, ...frameParam }, required: [] },
  },
  {
    name: "fetch_url",
    description: "Fetch a web page (http/https) in the background and return its readable text. Use after web_search to read a result, or to read a link from the page.",
    parameters: { type: "object", properties: { url: { type: "string" }, offset: { type: "integer" }, max_chars: { type: "integer", description: "Default 8000, max 20000." } }, required: ["url"] },
  },
  {
    name: "list_tabs",
    description: "List the open tabs in this window (id, title, url).",
    parameters: { type: "object", properties: {}, required: [] },
  },
  {
    name: "list_frames",
    description: "List the iframes inside the current page (frame_id, url). Some sites (exercise platforms, embedded viewers) keep their content in an iframe; pass frame_id to other tools to read/operate it.",
    parameters: { type: "object", properties: {}, required: [] },
  },
];

const SEARCH_TOOL = {
  name: "web_search",
  description: "Search the internet (DuckDuckGo). Returns titles, URLs and snippets. Use fetch_url to read a result in full. Write the query in the most useful language (Dutch for Dutch topics, English for international ones).",
  parameters: { type: "object", properties: { query: { type: "string" }, max_results: { type: "integer", description: "1-10, default 6." } }, required: ["query"] },
};

const ACTION_TOOLS = [
  {
    name: "get_interactive",
    description: "List clickable/typable elements on the page (links, buttons, inputs, selects, checkboxes) with ids to use in click/type_text/select_option. Call this before interacting; ids change after the page changes.",
    parameters: { type: "object", properties: { viewport_only: { type: "boolean", description: "Only elements currently visible in the viewport (default false)." }, ...frameParam }, required: [] },
  },
  {
    name: "click",
    description: "Click an element by id (from get_interactive).",
    parameters: { type: "object", properties: { id: { type: "integer" }, ...frameParam }, required: ["id"] },
  },
  {
    name: "type_text",
    description: "Type text into an input, textarea or editable field by id. Clears the field first unless clear=false. Set submit=true to press Enter/submit afterwards.",
    parameters: { type: "object", properties: { id: { type: "integer" }, text: { type: "string" }, clear: { type: "boolean" }, submit: { type: "boolean" }, ...frameParam }, required: ["id", "text"] },
  },
  {
    name: "select_option",
    description: "Choose an option in a <select> dropdown by id; value may be the option text or value.",
    parameters: { type: "object", properties: { id: { type: "integer" }, value: { type: "string" }, ...frameParam }, required: ["id", "value"] },
  },
  {
    name: "press_key",
    description: "Press a key (Enter, Escape, Tab, ArrowDown, ...) on an element id or the focused element.",
    parameters: { type: "object", properties: { key: { type: "string" }, id: { type: "integer" }, ...frameParam }, required: ["key"] },
  },
  {
    name: "scroll",
    description: "Scroll the page (down/up/top/bottom) or scroll an element id into view. Useful before take_screenshot.",
    parameters: { type: "object", properties: { direction: { type: "string", enum: ["down", "up", "top", "bottom"] }, id: { type: "integer" }, ...frameParam }, required: [] },
  },
  {
    name: "navigate",
    description: "Open a URL in the current tab and wait for it to load.",
    parameters: { type: "object", properties: { url: { type: "string" } }, required: ["url"] },
  },
  {
    name: "go_back",
    description: "Go back to the previous page in the current tab.",
    parameters: { type: "object", properties: {}, required: [] },
  },
  {
    name: "switch_tab",
    description: "Make another open tab the active one (id from list_tabs). Subsequent tools work on that tab.",
    parameters: { type: "object", properties: { tab_id: { type: "integer" } }, required: ["tab_id"] },
  },
  {
    name: "wait",
    description: "Wait a few seconds (max 10), e.g. for a page to finish loading.",
    parameters: { type: "object", properties: { seconds: { type: "number" } }, required: ["seconds"] },
  },
];

/** Bouwt de tools-array voor het API-verzoek, afhankelijk van de instellingen. */
export function buildTools(settings) {
  const tools = [...READ_TOOLS];
  if (settings.searchProvider === "duckduckgo") tools.push(SEARCH_TOOL);
  if (settings.allowActions) tools.push(...ACTION_TOOLS);
  const out = tools.map((t) => ({ type: "function", function: t }));
  if (settings.searchProvider === "openrouter" && /openrouter\.ai/.test(settings.baseUrl)) {
    out.unshift({ type: "openrouter:web_search", parameters: { max_results: 5, max_uses: 4 } });
  }
  return out;
}

// Nederlandse statusregels voor in de UI
export function describeToolCall(name, args = {}) {
  const q = (v) => (v == null || v === "" ? "…" : String(v));
  args = { ...args };
  for (const k of ["query", "url", "text", "value", "key", "id", "tab_id", "seconds"]) if (args[k] == null) args[k] = k === "id" || k === "tab_id" ? "?" : q(args[k]);
  switch (name) {
    case "read_page": return "📄 Leest de pagina" + (args.offset ? ` (vanaf teken ${args.offset})` : "");
    case "find_text": return `🔎 Zoekt op de pagina naar “${args.query}”`;
    case "get_images": return "🖼️ Bekijkt welke afbeeldingen er zijn";
    case "view_image": return args.url ? "🖼️ Bekijkt afbeelding via URL" : `🖼️ Bekijkt afbeelding #${args.id}`;
    case "take_screenshot": return "📷 Maakt een screenshot";
    case "web_search": return `🌐 Zoekt op internet: “${args.query}”`;
    case "fetch_url": return `🌐 Leest ${shortUrl(args.url)}`;
    case "list_tabs": return "🗂️ Bekijkt de open tabbladen";
    case "list_frames": return "🧩 Bekijkt de frames op de pagina";
    case "get_interactive": return "🖱️ Bekijkt knoppen en invoervelden";
    case "click": return `🖱️ Klikt op element #${args.id}`;
    case "type_text": return `⌨️ Typt “${truncate(args.text, 40)}”`;
    case "select_option": return `☑️ Kiest “${args.value}”`;
    case "press_key": return `⌨️ Drukt op ${args.key}`;
    case "scroll": return `↕️ Scrolt ${args.direction || "naar element"}`;
    case "navigate": return `➡️ Gaat naar ${shortUrl(args.url)}`;
    case "go_back": return "⬅️ Gaat terug";
    case "switch_tab": return `🗂️ Wisselt naar tabblad ${args.tab_id}`;
    case "wait": return `⏳ Wacht ${args.seconds}s`;
    default: return `🔧 ${name}`;
  }
}

const truncate = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");
function shortUrl(u) {
  try { const x = new URL(u); return x.hostname + (x.pathname.length > 1 ? truncate(x.pathname, 30) : ""); } catch (_) { return truncate(u, 40); }
}

function clampInt(v, lo, hi, def) {
  const n = Number.isFinite(Number(v)) ? Math.round(Number(v)) : def;
  return Math.max(lo, Math.min(hi, n));
}

function toResultText(obj) {
  const s = typeof obj === "string" ? obj : JSON.stringify(obj);
  return s.length > MAX_RESULT_CHARS ? s.slice(0, MAX_RESULT_CHARS) + `…[afgekapt, ${s.length} tekens totaal]` : s;
}

/**
 * Voert een tool uit. `ctx` = { getTabId(), setTabId(id), windowId, settings }.
 * Geeft { text, images?: [{dataUrl, label}] } terug. Fouten worden als tekst teruggegeven zodat het model kan herstellen.
 */
export async function executeTool(name, args, ctx) {
  try {
    const r = await run(name, args || {}, ctx);
    return r;
  } catch (e) {
    return { text: JSON.stringify({ error: String((e && e.message) || e) }) };
  }
}

async function run(name, args, ctx) {
  const tabId = ctx.getTabId();
  const frame = args.frame_id != null ? Number(args.frame_id) : null;
  switch (name) {
    case "read_page": {
      const r = await page.pageAction(tabId, "page_info", { offset: clampInt(args.offset, 0, 5e6, 0), maxChars: clampInt(args.max_chars, 500, 20000, 8000) }, frame);
      return { text: toResultText(r) };
    }
    case "find_text": {
      const r = await page.pageAction(tabId, "find_text", { query: String(args.query || "") }, frame);
      return { text: toResultText(r) };
    }
    case "get_images": {
      const r = await page.pageAction(tabId, "images", { limit: 80 }, frame);
      const images = r.images.map((i) => ({ id: i.id, kind: i.kind, alt: i.alt || undefined, size: `${i.width}x${i.height}`, visible: i.visible, src: i.kind === "img" ? truncate(i.src, 120) : undefined }));
      return { text: toResultText({ total: r.total, images }) };
    }
    case "view_image": {
      if (args.url) {
        const img = await page.normalizeImage(String(args.url), { maxDim: 1400 });
        return { text: JSON.stringify({ ok: true, url: args.url, note: "Image attached below." }), images: [{ dataUrl: img.dataUrl, label: `Afbeelding: ${shortUrl(args.url)}` }] };
      }
      if (args.id == null) throw new Error("Geef een id of url op.");
      const r = await page.pageAction(tabId, "image_data", { id: Number(args.id) }, frame);
      let img = null;
      if (r.dataUrl) {
        img = await page.normalizeImage(r.dataUrl, { maxDim: 1400 });
      } else if (r.src && /^https?:/i.test(r.src)) {
        try { img = await page.normalizeImage(r.src, { maxDim: 1400 }); } catch (_) { img = null; }
      }
      if (!img && r.rect) {
        // Laatste redmiddel: bijgesneden screenshot van het element (dat net in beeld gescrold is).
        await page.sleep(250);
        img = await page.captureVisible(ctx.windowId, { maxDim: 1400, crop: r.rect });
      }
      if (!img) throw new Error("Afbeelding kon niet worden geladen: " + (r.error || "onbekend"));
      return { text: JSON.stringify({ ok: true, id: r.id, kind: r.kind, alt: r.alt, note: "Image attached below." }), images: [{ dataUrl: img.dataUrl, label: `Afbeelding #${r.id}${r.alt ? " – " + truncate(r.alt, 60) : ""}` }] };
    }
    case "take_screenshot": {
      let crop = null;
      if (args.element_id != null) {
        crop = await page.pageAction(tabId, "element_rect", { id: Number(args.element_id) }, frame);
        await page.sleep(250);
      } else {
        await page.ensureInjected(tabId);
      }
      const shot = await page.captureVisible(ctx.windowId, { maxDim: 1400, crop });
      const info = await page.pageAction(tabId, "scroll", { direction: "none", amount: 0 }, frame).catch(() => null);
      return {
        text: JSON.stringify({ ok: true, note: "Screenshot attached below.", width: shot.width, height: shot.height, scroll: info ? { y: info.scrollY, pageHeight: info.pageHeight, viewportHeight: info.viewportHeight } : undefined }),
        images: [{ dataUrl: shot.dataUrl, label: crop ? `Screenshot van element #${args.element_id}` : "Screenshot" }],
      };
    }
    case "web_search": {
      const r = await webSearch(String(args.query || ""), { max: clampInt(args.max_results, 1, 10, 6) });
      return { text: toResultText(r) };
    }
    case "fetch_url": {
      const r = await fetchUrlAsText(String(args.url || ""), { offset: clampInt(args.offset, 0, 5e6, 0), maxChars: clampInt(args.max_chars, 500, 20000, 8000) });
      return { text: toResultText(r) };
    }
    case "list_tabs": {
      return { text: toResultText({ tabs: await page.listTabs() }) };
    }
    case "list_frames": {
      return { text: toResultText({ frames: await page.listFrames(tabId) }) };
    }
    case "get_interactive": {
      const r = await page.pageAction(tabId, "interactive", { limit: 150, viewportOnly: !!args.viewport_only }, frame);
      return { text: toResultText(r) };
    }
    case "click": {
      const before = await page.getTab(tabId);
      const r = await page.pageAction(tabId, "click", { id: Number(args.id) }, frame);
      await page.sleep(700);
      let after = await page.getTab(tabId);
      if (after && (after.status === "loading" || after.url !== before?.url)) after = await page.waitForLoad(tabId, 15000);
      return { text: toResultText({ ...r, navigated: after?.url !== before?.url, url: after?.url, title: after?.title, hint: "Page may have changed; call read_page or get_interactive again if needed." }) };
    }
    case "type_text": {
      const r = await page.pageAction(tabId, "type", { id: Number(args.id), text: String(args.text ?? ""), clear: args.clear !== false, submit: !!args.submit }, frame);
      if (args.submit) { await page.sleep(800); const t = await page.getTab(tabId); if (t?.status === "loading") await page.waitForLoad(tabId, 15000); }
      return { text: toResultText(r) };
    }
    case "select_option": {
      const r = await page.pageAction(tabId, "select", { id: Number(args.id), value: String(args.value ?? "") }, frame);
      return { text: toResultText(r) };
    }
    case "press_key": {
      const r = await page.pageAction(tabId, "key", { id: args.id != null ? Number(args.id) : undefined, key: String(args.key || "Enter") }, frame);
      await page.sleep(500);
      return { text: toResultText(r) };
    }
    case "scroll": {
      const r = await page.pageAction(tabId, "scroll", { direction: args.direction || "down", id: args.id != null ? Number(args.id) : undefined }, frame);
      await page.sleep(300);
      return { text: toResultText(r) };
    }
    case "navigate": {
      const r = await page.navigate(tabId, String(args.url || ""));
      return { text: toResultText({ ok: true, ...r, hint: "Call read_page to read the new page." }) };
    }
    case "go_back": {
      const r = await page.goBack(tabId);
      return { text: toResultText({ ok: true, ...r }) };
    }
    case "switch_tab": {
      const r = await page.switchTab(Number(args.tab_id));
      ctx.setTabId(r.id);
      return { text: toResultText({ ok: true, ...r, hint: "Tools now operate on this tab. Call read_page to read it." }) };
    }
    case "wait": {
      const s = Math.max(0, Math.min(10, Number(args.seconds) || 1));
      await page.sleep(s * 1000);
      return { text: JSON.stringify({ ok: true, waited: s }) };
    }
    default:
      throw new Error(`Onbekende tool: ${name}`);
  }
}
