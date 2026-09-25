// End-to-end test zonder echte browser:
//  - jsdom speelt de webpagina + het content-script
//  - een nep-OpenRouter (SSE-stream met tool calls) speelt het model
//  - de echte agent-lus, tools en content-script-code worden gebruikt
// Draai: npm test
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const ext = join(root, "extension");
let passed = 0;
const test = async (name, fn) => {
  try { await fn(); passed++; console.log("✓", name); }
  catch (e) { console.error("✗", name); console.error(e); process.exitCode = 1; }
};

// ---------- Nep-webpagina + content-script ----------
const PAGE_HTML = `<!doctype html><html lang="nl"><head><title>Wiskunde – Opgave 3</title>
<meta name="description" content="Oefenen met de stelling van Pythagoras"></head>
<body>
<nav><a href="/home">Home</a><a href="/hoofdstukken">Hoofdstukken</a></nav>
<main>
  <h1>Opgave 3: Pythagoras</h1>
  <p>Een rechthoekige driehoek heeft rechthoekszijden van 6 cm en 8 cm. Bereken de schuine zijde.</p>
  <img src="https://example.com/driehoek.png" alt="Driehoek ABC met rechte hoek bij C" width="320" height="240">
  <img src="https://example.com/pixel.gif" width="1" height="1">
  <table><tr><th>zijde</th><th>lengte</th></tr><tr><td>a</td><td>6 cm</td></tr><tr><td>b</td><td>8 cm</td></tr></table>
  <span class="katex"><span class="katex-mathml"><math><semantics><mrow></mrow><annotation encoding="application/x-tex">c^2 = a^2 + b^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true">c2=a2+b2</span></span>
  <form id="f"><label for="antwoord">Jouw antwoord (cm)</label><input id="antwoord" name="antwoord" type="text" placeholder="bijv. 12"><button type="button" id="check">Controleer</button></form>
  <p id="feedback"></p>
  <div style="display:none">VERBORGEN TEKST</div>
  <script>document.getElementById('check').addEventListener('click', () => { document.getElementById('feedback').textContent = 'Gecontroleerd: ' + document.getElementById('antwoord').value; });</script>
</main>
<footer>© School</footer>
</body></html>`;

function makePage(html = PAGE_HTML, url = "https://school.example.nl/wiskunde/opgave-3") {
  const dom = new JSDOM(html, { url, runScripts: "dangerously", pretendToBeVisual: true });
  const w = dom.window;
  // jsdom heeft geen layout: geef elementen nep-afmetingen zodat zichtbaarheidsfilters werken.
  w.Element.prototype.getBoundingClientRect = function () {
    const width = Number(this.getAttribute?.("width")) || 200;
    const height = Number(this.getAttribute?.("height")) || 24;
    return { x: 10, y: 20, left: 10, top: 20, width, height, right: 10 + width, bottom: 20 + height };
  };
  Object.defineProperty(w.HTMLElement.prototype, "innerText", { get() { return this.textContent; }, configurable: true });
  w.HTMLElement.prototype.scrollIntoView = function () {};
  w.scrollTo = () => {}; w.scrollBy = () => {};
  let listener = null;
  w.chrome = { runtime: { onMessage: { addListener: (fn) => { listener = fn; } } } };
  w.eval(readFileSync(join(ext, "content/content.js"), "utf8"));
  assert.ok(listener, "content-script registreerde geen listener");
  const send = (action, args = {}) => new Promise((resolve) => listener({ __aiWebAgent: true, action, args }, {}, resolve));
  return { dom, window: w, send };
}

// ---------- Nep-OpenRouter ----------
function sse(events) {
  // events: array van objecten (chunks) of strings (ruwe regels). Wordt in kleine, willekeurig geknipte stukken gestreamd.
  const text = events.map((e) => (typeof e === "string" ? e : `data: ${JSON.stringify(e)}\n\n`)).join("") + "data: [DONE]\n\n";
  const enc = new TextEncoder();
  let i = 0;
  return new ReadableStream({
    pull(controller) {
      if (i >= text.length) { controller.close(); return; }
      const n = 7 + Math.floor(Math.random() * 20);
      controller.enqueue(enc.encode(text.slice(i, i + n)));
      i += n;
    },
  });
}
const chunk = (delta, finish = null, extra = {}) => ({ id: "x", choices: [{ index: 0, delta, finish_reason: finish }], ...extra });
function toolCallChunks(calls) {
  // Splits argumenten in stukjes, zoals echte providers doen.
  const out = [];
  calls.forEach((c, index) => {
    const args = JSON.stringify(c.args);
    out.push(chunk({ tool_calls: [{ index, id: c.id, type: "function", function: { name: c.name, arguments: "" } }] }));
    for (let i = 0; i < args.length; i += 5) out.push(chunk({ tool_calls: [{ index, function: { arguments: args.slice(i, i + 5) } }] }));
  });
  out.push(chunk({}, "tool_calls", { usage: { prompt_tokens: 100, completion_tokens: 20, cost: 0 } }));
  return out;
}
function textChunks(text, usage = { cost: 0.00042 }) {
  const out = [": OPENROUTER PROCESSING\n\n"];
  for (let i = 0; i < text.length; i += 9) out.push(chunk({ content: text.slice(i, i + 9) }));
  out.push(chunk({}, "stop", { usage: { prompt_tokens: 100, completion_tokens: 50, ...usage }, model: "test/model" }));
  return out;
}

const DDG_HTML = `<html><body><div class="results">
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnl.wikipedia.org%2Fwiki%2FStelling_van_Pythagoras&rut=abc">Stelling van Pythagoras - Wikipedia</a><a class="result__snippet">De stelling van Pythagoras is een wiskundige stelling…</a></div>
<div class="result"><a class="result__a" href="https://www.wiskundeacademie.nl/pythagoras">Wiskunde Academie</a><a class="result__snippet">Uitleg met voorbeelden.</a></div>
</div></body></html>`;
const WIKI_HTML = `<html><head><title>Stelling van Pythagoras</title></head><body><nav>menu</nav><main><h1>Stelling van Pythagoras</h1><p>In een rechthoekige driehoek geldt a² + b² = c².</p><script>x=1</script></main></body></html>`;

/** Installeert globale stubs (chrome, document, fetch) zodat de extensiemodules in Node draaien. */
function installGlobals(pageCtx, script) {
  const w = pageCtx.window;
  for (const k of ["document", "Node", "Element", "HTMLElement", "Image", "Blob", "XMLSerializer", "DOMParser", "Event", "KeyboardEvent", "MouseEvent", "PointerEvent", "getComputedStyle"]) {
    if (w[k] !== undefined) globalThis[k] = w[k];
  }
  globalThis.window = w;
  if (!globalThis.DOMException) globalThis.DOMException = w.DOMException;
  globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
  const tab = { id: 1, windowId: 1, url: w.location.href, title: w.document.title, status: "complete", active: true };
  globalThis.chrome = {
    tabs: {
      get: async () => ({ ...tab, url: w.location.href, title: w.document.title }),
      query: async () => [tab],
      sendMessage: async (_id, msg) => pageCtx.send(msg.action, msg.args),
      update: async (id, props) => ({ ...tab, ...props }),
      onUpdated: { addListener() {}, removeListener() {} },
      captureVisibleTab: async () => { throw new Error("geen screenshot in test"); },
    },
    scripting: { executeScript: async () => [{ frameId: 0, result: { url: w.location.href, title: w.document.title, top: true } }] },
    storage: { local: { get: async () => ({}), set: async () => {} }, onChanged: { addListener() {} } },
    runtime: { openOptionsPage() {} },
  };
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    const u = String(url);
    if (u.endsWith("/chat/completions")) {
      const body = JSON.parse(init.body);
      const step = calls.filter((c) => c.url.endsWith("/chat/completions")).length;
      const events = script(step, body);
      if (events.status) return new Response(JSON.stringify(events.body), { status: events.status, headers: events.headers || {} });
      return new Response(sse(events), { status: 200, headers: { "Content-Type": "text/event-stream" } });
    }
    if (u.startsWith("https://html.duckduckgo.com/")) return new Response(DDG_HTML, { status: 200, headers: { "Content-Type": "text/html" } });
    if (u.startsWith("https://nl.wikipedia.org/")) return new Response(WIKI_HTML, { status: 200, headers: { "Content-Type": "text/html" } });
    return new Response("not found", { status: 404 });
  };
  return { calls };
}

// ============================================================
await test("content-script: page_info leest hoofdtekst, tabel, formule en invoervelden; verbergt ruis", async () => {
  const p = makePage();
  const r = await p.send("page_info", { maxChars: 6000 });
  assert.equal(r.ok, true, r.error);
  const info = r.result;
  assert.equal(info.title, "Wiskunde – Opgave 3");
  assert.match(info.text, /# Opgave 3: Pythagoras/);
  assert.match(info.text, /rechthoekszijden van 6 cm en 8 cm/);
  assert.match(info.text, /\| zijde \| lengte/);
  assert.match(info.text, /\$c\^2 = a\^2 \+ b\^2\$/, "KaTeX-formule moet als LaTeX verschijnen");
  assert.match(info.text, /\[afbeelding #0: Driehoek ABC/);
  assert.match(info.text, /\[invoerveld “[^”]*Jouw antwoord/);
  assert.doesNotMatch(info.text, /VERBORGEN TEKST/);
  assert.doesNotMatch(info.text, /Hoofdstukken/, "navigatie moet weggelaten worden");
  assert.equal(info.images, 1, "1x1-pixel moet genegeerd worden");
  assert.equal(JSON.stringify(info.headings), JSON.stringify(["Opgave 3: Pythagoras"]));
});

await test("content-script: find_text, images, interactive, type + click werken", async () => {
  const p = makePage();
  await p.send("page_info", {});
  const f = (await p.send("find_text", { query: "schuine zijde" })).result;
  assert.equal(f.matches.length, 1);
  const imgs = (await p.send("images", {})).result;
  assert.equal(imgs.images[0].alt, "Driehoek ABC met rechte hoek bij C");
  const inter = (await p.send("interactive", {})).result;
  const input = inter.elements.find((e) => e.tag === "input");
  const btn = inter.elements.find((e) => e.tag === "button");
  assert.ok(input && btn, "invoerveld en knop moeten gevonden worden");
  assert.match(input.label, /Jouw antwoord/);
  const t = (await p.send("type", { id: input.id, text: "10" }));
  assert.equal(t.ok, true, t.error);
  assert.equal(p.window.document.getElementById("antwoord").value, "10");
  const c = await p.send("click", { id: btn.id });
  assert.equal(c.ok, true, c.error);
  assert.equal(p.window.document.getElementById("feedback").textContent, "Gecontroleerd: 10");
  const bad = await p.send("click", { id: 999 });
  assert.equal(bad.ok, false);
  assert.match(bad.error, /bestaat niet/);
});

await test("openrouter: SSE-parser plakt tekst en tool-argumenten correct aan elkaar", async () => {
  const p = makePage();
  installGlobals(p, (step) => step === 1
    ? toolCallChunks([{ id: "call_1", name: "read_page", args: { offset: 0, max_chars: 1234 } }, { id: "call_2", name: "find_text", args: { query: "zijde" } }])
    : textChunks("Hallo **wereld**"));
  const { streamChat } = await import("../extension/lib/openrouter.js");
  const r1 = await streamChat({ baseUrl: "https://openrouter.ai/api/v1", apiKey: "k", body: { model: "m", messages: [] } });
  assert.equal(r1.finish_reason, "tool_calls");
  assert.equal(r1.tool_calls.length, 2);
  assert.deepEqual(JSON.parse(r1.tool_calls[0].function.arguments), { offset: 0, max_chars: 1234 });
  assert.equal(r1.tool_calls[1].function.name, "find_text");
  let streamed = "";
  const r2 = await streamChat({ baseUrl: "https://openrouter.ai/api/v1", apiKey: "k", body: { model: "m", messages: [] } }, { onText: (d) => { streamed += d; } });
  assert.equal(r2.content, "Hallo **wereld**");
  assert.equal(streamed, r2.content);
  assert.equal(r2.usage.cost, 0.00042);
  assert.equal(r2.model, "test/model");
});

await test("openrouter: HTTP-fouten worden vertaald naar begrijpelijke meldingen", async () => {
  const p = makePage();
  installGlobals(p, (step) => (step === 1
    ? { status: 429, body: { error: { code: 429, message: "Rate limit exceeded: free-models-per-day" } }, headers: { "X-RateLimit-Reset": String(Date.now() + 3600e3) } }
    : { status: 401, body: { error: { code: 401, message: "No auth credentials found" } } }));
  const { streamChat, ApiError } = await import("../extension/lib/openrouter.js");
  await assert.rejects(streamChat({ baseUrl: "https://openrouter.ai/api/v1", apiKey: "k", body: {} }), (e) => e instanceof ApiError && e.status === 429 && /50 vragen per dag/.test(e.message) && /niet om tokens/.test(e.message) && /1\.000 vragen per dag/.test(e.message));
  await assert.rejects(streamChat({ baseUrl: "https://openrouter.ai/api/v1", apiKey: "k", body: {} }), (e) => e.status === 401 && /API-sleutel/.test(e.message));
});

await test("search: DuckDuckGo-resultaten en fetch_url naar leesbare tekst", async () => {
  const p = makePage();
  installGlobals(p, () => []);
  const { webSearch, fetchUrlAsText } = await import("../extension/lib/search.js");
  const s = await webSearch("stelling van pythagoras");
  assert.equal(s.engine, "DuckDuckGo");
  assert.equal(s.results[0].url, "https://nl.wikipedia.org/wiki/Stelling_van_Pythagoras", "redirect-link moet gedecodeerd worden");
  assert.match(s.results[0].snippet, /wiskundige stelling/);
  const page = await fetchUrlAsText(s.results[0].url);
  assert.equal(page.title, "Stelling van Pythagoras");
  assert.match(page.text, /a² \+ b² = c²/);
  assert.doesNotMatch(page.text, /menu|x=1/);
});

await test("agent: volledige lus — pagina lezen, zoeken, typen, klikken, eindantwoord met kosten", async () => {
  const p = makePage();
  const seenBodies = [];
  const { calls } = installGlobals(p, (step, body) => {
    seenBodies.push(body);
    if (step === 1) return toolCallChunks([{ id: "c1", name: "web_search", args: { query: "stelling van pythagoras" } }, { id: "c2", name: "get_interactive", args: {} }]);
    if (step === 2) {
      // Het model gebruikt de element-ids uit het vorige toolresultaat.
      const toolMsg = body.messages.filter((m) => m.role === "tool").find((m) => m.name === "get_interactive");
      const els = JSON.parse(toolMsg.content).elements;
      const input = els.find((e) => e.tag === "input"), btn = els.find((e) => e.tag === "button");
      return toolCallChunks([{ id: "c3", name: "type_text", args: { id: input.id, text: "10" } }, { id: "c4", name: "click", args: { id: btn.id } }, { id: "c5", name: "fetch_url", args: { url: "https://nl.wikipedia.org/wiki/Stelling_van_Pythagoras" } }]);
    }
    return textChunks("De schuine zijde is $c = \\sqrt{6^2+8^2} = 10$ cm.");
  });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { buildUserContent } = await import("../extension/lib/prompts.js");
  const { DEFAULTS } = await import("../extension/lib/settings.js");
  const settings = { ...DEFAULTS, apiKey: "sk-test", model: "qwen/qwen3.8-27b:free" };
  const info = (await p.send("page_info", { maxChars: 6000 })).result;
  const userContent = buildUserContent({ text: "Bereken de schuine zijde en vul het antwoord in.", page: info, includePage: true });
  const events = [];
  const result = await runAgent({
    settings, history: [], userContent,
    ctx: { getTabId: () => 1, setTabId() {}, windowId: 1, tabTitle: info.title, tabUrl: info.url },
    ui: { onToolStart: (label) => events.push("start:" + label), onToolEnd: (label, call, ok) => events.push((ok ? "ok:" : "fail:") + call.function.name) },
  });
  assert.equal(result.steps, 3);
  assert.equal(result.messages[0].role, "assistant", "user-bericht mag niet dubbel terugkomen");
  assert.match(result.content, /10\$ cm/);
  assert.equal(result.cost, 0.00042);
  assert.equal(result.model, "test/model");
  assert.equal(p.window.document.getElementById("feedback").textContent, "Gecontroleerd: 10", "agent moet echt getypt en geklikt hebben");
  assert.ok(events.includes("ok:web_search") && events.includes("ok:click") && events.includes("ok:fetch_url"), events.join(","));
  // Verzoek-inhoud controleren
  const first = seenBodies[0];
  assert.equal(first.model, "qwen/qwen3.8-27b:free");
  assert.deepEqual(first.models, ["qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free", "thinkingmachines/inkling:free"]);
  assert.equal(first.messages[0].role, "system");
  assert.match(first.messages[0].content, /Answer in the language the user writes in/);
  assert.match(first.messages[1].content[0].text, /rechthoekszijden van 6 cm en 8 cm/, "paginatekst moet in het eerste bericht zitten");
  assert.ok(first.tools.some((t) => t.function?.name === "web_search"), "DuckDuckGo-tool aanwezig");
  assert.ok(!first.tools.some((t) => t.type === "openrouter:web_search"));
  const third = seenBodies[2];
  const toolNames = third.messages.filter((m) => m.role === "tool").map((m) => m.name);
  assert.deepEqual(toolNames, ["web_search", "get_interactive", "type_text", "click", "fetch_url"]);
  const searchResult = JSON.parse(third.messages.find((m) => m.role === "tool" && m.name === "web_search").content);
  assert.equal(searchResult.results[0].url, "https://nl.wikipedia.org/wiki/Stelling_van_Pythagoras");
  assert.equal(calls.filter((c) => c.url.endsWith("/chat/completions")).length, 3);
  // Gespreksgeschiedenis: het volgende verzoek moet compact zijn (geen oude paginatekst)
  const { compactHistory } = await import("../extension/sidepanel/agent.js");
  const compact = compactHistory([{ role: "user", content: userContent }, ...result.messages]);
  assert.match(compact[0].content[0].text, /Page context omitted/);
  assert.match(compact[0].content[0].text, /Bereken de schuine zijde/);
});

await test("agent: instellingen sturen tools aan (acties uit, OpenRouter-zoeken aan, taal vast)", async () => {
  const p = makePage();
  let body0 = null;
  installGlobals(p, (step, body) => { body0 = body0 || body; return textChunks("ok"); });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS } = await import("../extension/lib/settings.js");
  const settings = { ...DEFAULTS, apiKey: "k", allowActions: false, searchProvider: "openrouter", answerLanguage: "fr", reasoning: "low", fallbackModels: "" };
  await runAgent({ settings, history: [], userContent: [{ type: "text", text: "hoi" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 } });
  const names = body0.tools.map((t) => t.function?.name || t.type);
  assert.ok(names.includes("openrouter:web_search"));
  assert.ok(!names.includes("click") && !names.includes("web_search"));
  assert.ok(names.includes("read_page") && names.includes("view_image"));
  assert.match(body0.messages[0].content, /ALWAYS answer in French/);
  assert.deepEqual(body0.reasoning, { effort: "low" });
  assert.equal(body0.models, undefined);
});

await test("agent: bij 400 op extra parameters wordt zonder reasoning/models opnieuw geprobeerd", async () => {
  const p = makePage();
  const bodies = [];
  installGlobals(p, (step, body) => { bodies.push(body); return step === 1 ? { status: 400, body: { error: { code: 400, message: "reasoning not supported" } } } : textChunks("prima"); });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS } = await import("../extension/lib/settings.js");
  const r = await runAgent({ settings: { ...DEFAULTS, apiKey: "k", reasoning: "high" }, history: [], userContent: [{ type: "text", text: "hoi" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 } });
  assert.equal(r.content, "prima");
  assert.ok(bodies[0].reasoning && !bodies[1].reasoning);
});

await test("openrouter: 429 wordt onderscheiden in daglimiet / minuutlimiet / overbelaste aanbieder", async () => {
  const { classifyError, friendlyError } = await import("../extension/lib/openrouter.js");
  const busy = { error: { code: 429, message: "Provider returned error", metadata: { raw: "Rate limit exceeded", provider_name: "Chutes" } } };
  assert.equal(classifyError(429, busy), "provider_busy");
  const m = friendlyError(429, busy, null, { model: "qwen/qwen3.8-27b:free" });
  assert.match(m, /overbelast/); assert.match(m, /qwen3.8-27b/); assert.match(m, /Chutes/); assert.doesNotMatch(m, /daglimiet is op/);
  assert.equal(classifyError(429, { error: { code: 429, message: "Rate limit exceeded: free-models-per-day" } }), "daily_limit");
  assert.equal(classifyError(429, { error: { code: 429, message: "Rate limit exceeded: free-models-per-min" } }), "minute_limit");
  assert.match(friendlyError(429, { error: { message: "Rate limit exceeded: free-models-per-min" } }), /20 verzoeken per minuut/);
  assert.equal(classifyError(503, { error: { message: "upstream down" } }), "provider_down");
  assert.equal(classifyError(402, {}), "credits");
});

await test("agent: bij drukte (429 Provider returned error) schakelt hij over naar het volgende gratis model", async () => {
  const p = makePage();
  const bodies = [];
  installGlobals(p, (step, body) => {
    bodies.push(body);
    if (step === 1) return { status: 429, body: { error: { code: 429, message: "Provider returned error", metadata: { provider_name: "Chutes", raw: "429 Too Many Requests" } } } };
    if (step === 2) return { status: 503, body: { error: { code: 503, message: "Provider unavailable" } } };
    return textChunks("gelukt via reserve", { cost: 0 });
  });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS } = await import("../extension/lib/settings.js");
  const retries = [];
  const r = await runAgent({
    settings: { ...DEFAULTS, apiKey: "k" }, history: [], userContent: [{ type: "text", text: "hoi" }],
    ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 }, retryDelays: [0, 0],
    ui: { onRetry: (info) => retries.push(info) },
  });
  assert.equal(r.content, "gelukt via reserve");
  assert.equal(bodies.length, 3);
  assert.equal(bodies[0].model, "qwen/qwen3.8-27b:free");
  assert.equal(bodies[1].model, "google/gemma-4-31b-it:free", "na 429 moet het volgende reserve-model voorop staan");
  assert.equal(bodies[1].models[0], "google/gemma-4-31b-it:free");
  assert.equal(bodies[2].model, "thinkingmachines/inkling:free");
  assert.equal(retries.length, 2);
  assert.equal(retries[0].error.kind, "provider_busy");
  assert.equal(retries[0].to, "google/gemma-4-31b-it:free");
  assert.equal(retries[1].error.kind, "provider_down");
});

await test("agent: daglimiet (free-models-per-day) → meteen duidelijke fout, geen extra verzoeken", async () => {
  const p = makePage();
  const bodies = [];
  installGlobals(p, (step, body) => { bodies.push(body); return { status: 429, body: { error: { code: 429, message: "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day" } } }; });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS } = await import("../extension/lib/settings.js");
  const { ApiError } = await import("../extension/lib/openrouter.js");
  await assert.rejects(
    runAgent({ settings: { ...DEFAULTS, apiKey: "k" }, history: [], userContent: [{ type: "text", text: "hoi" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 }, retryDelays: [0, 0] }),
    (e) => e instanceof ApiError && e.kind === "daily_limit" && !e.transient && /daglimiet is op/.test(e.message),
  );
  assert.equal(bodies.length, 1, "bij de daglimiet heeft opnieuw proberen geen zin");
});

await test("page: pagina lezen richt zich op het hoofdkader (frame 0) en neemt iframes met inhoud mee", async () => {
  const p = makePage();
  installGlobals(p, () => []);
  const sent = [];
  const realSend = globalThis.chrome.tabs.sendMessage;
  globalThis.chrome.tabs.sendMessage = async (id, msg, opts) => {
    sent.push({ action: msg.action, frameId: opts?.frameId });
    if (opts?.frameId === 5 && msg.action === "page_info") {
      return { ok: true, result: { url: "about:blank", title: "", text: "1 f_nny  2 h_ppy  3 s_d [invoerveld (leeg)]", images: 0, frames: [], totalChars: 40 } };
    }
    if (opts?.frameId === 5) return { ok: true }; // ping
    return realSend(id, msg, opts);
  };
  globalThis.chrome.scripting.executeScript = async ({ func }) => (func
    ? [{ frameId: 0, result: { url: p.window.location.href, title: "top", top: true, textLength: 900, inputs: 1 } },
       { frameId: 3, result: { url: "about:blank", title: "", top: false, textLength: 0, inputs: 0 } },
       { frameId: 5, result: { url: "about:blank", title: "", top: false, textLength: 40, inputs: 9 } }]
    : [{ frameId: 0 }]);
  const page = await import("../extension/sidepanel/page.js");
  const info = await page.pageInfoWithFrames(1, { maxChars: 6000 });
  assert.match(info.text, /# Opgave 3: Pythagoras/, "hoofdkader eerst");
  assert.match(info.text, /Embedded frame \(frame_id 5\)[\s\S]*f_nny/, "inhoud van het iframe met invoervelden moet meekomen");
  assert.doesNotMatch(info.text, /frame_id 3/, "leeg iframe wordt overgeslagen");
  assert.equal(info.frameTexts, 1);
  assert.ok(sent.filter((x) => x.action === "page_info").every((x) => typeof x.frameId === "number"), "page_info altijd naar een specifiek kader sturen");
  assert.equal(sent.find((x) => x.action === "page_info").frameId, 0, "standaard het hoofdkader");
  const frames = await page.listFrames(1);
  assert.deepEqual(frames.map((f) => f.frameId), [0, 5], "kaders zonder tekst/velden weglaten, hoofdkader eerst");
});

const GEMINI = { id: "p_g", preset: "google", name: "Google AI Studio (Gemini)", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", apiKey: "AIza-test", model: "gemini-3.8-flash", enabled: true };

await test("agent: reserve-aanbieder neemt het over als de daglimiet op is – gesprek blijft bewaard", async () => {
  const p = makePage();
  const bodies = [];
  const { calls } = installGlobals(p, (step, body) => {
    bodies.push(body);
    if (step === 1) return { status: 429, body: { error: { code: 429, message: "Rate limit exceeded: free-models-per-day" } }, headers: { "X-RateLimit-Reset": String(Date.now() + 3 * 3600e3) } };
    return textChunks("hallo vanuit gemini", {});
  });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS, providerKey } = await import("../extension/lib/settings.js");
  const events = { provider: [], exhausted: [] };
  const providerState = {};
  const settings = { ...DEFAULTS, apiKey: "k", searchProvider: "openrouter", providers: [GEMINI, { ...GEMINI, id: "leeg", apiKey: "" }, { ...GEMINI, id: "uit", enabled: false }] };
  const r = await runAgent({
    settings, history: [{ role: "user", content: "eerdere vraag over pythagoras" }, { role: "assistant", content: "eerder antwoord" }],
    userContent: [{ type: "text", text: "en nu de volgende opgave" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 }, retryDelays: [0, 0], providerState,
    ui: { onProvider: (i) => events.provider.push(i), onExhausted: (prov, until) => events.exhausted.push({ prov, until }) },
  });
  assert.equal(r.content, "hallo vanuit gemini");
  assert.equal(r.provider, "Google AI Studio (Gemini)");
  assert.equal(r.providerId, "p_g");
  const urls = calls.filter((c) => c.url.endsWith("/chat/completions")).map((c) => c.url);
  assert.equal(urls.length, 2, "één poging bij OpenRouter, dan meteen door naar Gemini");
  assert.match(urls[0], /^https:\/\/openrouter\.ai\//);
  assert.equal(urls[1], "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions");
  assert.equal(calls.find((c) => c.url === urls[1]).init.headers.Authorization, "Bearer AIza-test");
  const g = bodies[1];
  assert.equal(g.model, "gemini-3.8-flash");
  assert.ok(!g.usage && !g.models && !g.reasoning, "OpenRouter-specifieke velden niet naar andere aanbieders sturen");
  assert.ok(!g.tools.some((t) => t.type === "openrouter:web_search"), "server-side zoektool alleen bij OpenRouter");
  assert.ok(g.tools.some((t) => t.function?.name === "web_search"), "elders valt hij terug op DuckDuckGo-zoeken");
  assert.ok(g.messages.some((m) => m.role === "user" && /pythagoras/.test(JSON.stringify(m.content))), "eerdere gespreksgeschiedenis gaat mee naar de nieuwe aanbieder");
  assert.ok(g.messages.some((m) => /volgende opgave/.test(JSON.stringify(m.content))));
  assert.equal(events.exhausted.length, 1);
  assert.equal(events.exhausted[0].prov.id, "primary");
  assert.ok(events.exhausted[0].until > Date.now() + 2 * 3600e3, "resettijd uit de X-RateLimit-Reset-header");
  assert.ok(providerState[providerKey({ baseUrl: DEFAULTS.baseUrl, apiKey: "k" })]?.until > Date.now());
  assert.equal(events.provider.length, 1);
  assert.equal(events.provider[0].reason, "daily_limit");
  assert.equal(events.provider[0].to.id, "p_g");
});

await test("agent: aanbieder die vandaag al op is wordt overgeslagen; zonder reserve blijft de fout duidelijk", async () => {
  const p = makePage();
  const { calls } = installGlobals(p, () => textChunks("direct via gemini", {}));
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS, providerKey, providerChain, nextResetMs, isExhausted } = await import("../extension/lib/settings.js");
  const settings = { ...DEFAULTS, apiKey: "k", providers: [GEMINI] };
  const providerState = { [providerKey({ baseUrl: DEFAULTS.baseUrl, apiKey: "k" })]: { until: Date.now() + 3600e3, reason: "daily_limit" } };
  const events = [];
  const r = await runAgent({ settings, history: [], userContent: [{ type: "text", text: "hoi" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 }, retryDelays: [0, 0], providerState, ui: { onProvider: (i) => events.push(i) } });
  assert.equal(r.content, "direct via gemini");
  assert.equal(calls.filter((c) => c.url.endsWith("/chat/completions")).length, 1);
  assert.match(calls[0].url, /googleapis\.com/);
  assert.equal(events[0]?.reason, "exhausted");

  // Zonder reserve-aanbieders: de daglimiet-fout komt gewoon terug (geen eindeloze pogingen).
  const p2 = makePage();
  const { calls: calls2 } = installGlobals(p2, () => ({ status: 429, body: { error: { code: 429, message: "Rate limit exceeded: free-models-per-day" } } }));
  await assert.rejects(runAgent({ settings: { ...DEFAULTS, apiKey: "k" }, history: [], userContent: [{ type: "text", text: "hoi" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 }, retryDelays: [0, 0], providerState: {} }), (e) => e.kind === "daily_limit");
  assert.equal(calls2.filter((c) => c.url.endsWith("/chat/completions")).length, 1);

  // Hulpfuncties
  assert.deepEqual(providerChain({ ...DEFAULTS, apiKey: "k", providers: [GEMINI, { ...GEMINI, apiKey: "" }, { ...GEMINI, enabled: false }, { ...GEMINI, model: "" }] }).map((x) => x.id), ["primary", "p_g"]);
  assert.equal(providerChain({ ...DEFAULTS, apiKey: "k", providers: [{ id: "o", name: "Ollama", baseUrl: "http://localhost:11434/v1", model: "qwen2.5vl:7b", apiKey: "" }] }).length, 2, "lokale server heeft geen sleutel nodig");
  const t0 = Date.UTC(2026, 8, 21, 14, 0); // 21 sep 2026 14:00 UTC
  assert.equal(nextResetMs(DEFAULTS.baseUrl, t0), Date.UTC(2026, 8, 22, 0, 0), "OpenRouter reset om 00:00 UTC");
  assert.equal(nextResetMs(GEMINI.baseUrl, t0), Date.UTC(2026, 8, 22, 8, 0), "Google reset om middernacht Pacific");
  assert.ok(isExhausted({ x: 1, [providerKey(GEMINI)]: { until: t0 + 1 } }, GEMINI, t0) && !isExhausted({ [providerKey(GEMINI)]: { until: t0 - 1 } }, GEMINI, t0));
});

await test("agent: tool-budget op → laatste stap dwingt een antwoord af", async () => {
  const p = makePage();
  const bodies = [];
  installGlobals(p, (step, body) => { bodies.push(body); return step < 3 ? toolCallChunks([{ id: "c" + step, name: "read_page", args: {} }]) : textChunks("klaar"); });
  const { runAgent } = await import("../extension/sidepanel/agent.js");
  const { DEFAULTS } = await import("../extension/lib/settings.js");
  const r = await runAgent({ settings: { ...DEFAULTS, apiKey: "k", maxSteps: 3 }, history: [], userContent: [{ type: "text", text: "hoi" }], ctx: { getTabId: () => 1, setTabId() {}, windowId: 1 } });
  assert.equal(r.steps, 3);
  assert.equal(bodies[2].tool_choice, "none");
  assert.equal(r.content, "klaar");
});

await test("render: markdown + wiskunde worden veilig gerenderd", async () => {
  const p = makePage("<!doctype html><html><body><div id='out'></div></body></html>");
  installGlobals(p, () => []);
  const w = p.window;
  const { marked } = await import("marked");
  const createDOMPurify = (await import("dompurify")).default;
  w.marked = marked; w.DOMPurify = createDOMPurify(w);
  const katex = (await import("katex")).default;
  w.katex = katex;
  const { renderMarkdown } = await import("../extension/sidepanel/render.js");
  const el = w.document.getElementById("out");
  renderMarkdown(el, "**Vet** en $x_1 + x_2$ en\n\n$$\\frac{a}{b}$$\n\nPrijs: $5 en $10.\n\n<img src=x onerror=alert(1)> [link](https://example.com)");
  assert.ok(el.querySelector("strong"));
  assert.ok(el.querySelector(".katex"), "KaTeX moet gerenderd zijn");
  assert.ok(el.querySelector(".katex-display"), "display-math moet gerenderd zijn");
  assert.ok(!el.querySelector("img[onerror]"), "onerror moet gesaneerd zijn");
  assert.equal(el.querySelector("a").getAttribute("target"), "_blank");
  assert.match(el.textContent, /Prijs: \$5 en \$10/, "valuta mag niet als wiskunde gezien worden");
  assert.equal(el.querySelectorAll(".katex").length, 2, "precies twee formules verwacht");
  assert.ok(!el.querySelector("em"), "underscore in wiskunde mag geen cursief worden");
  // Code met dollartekens blijft code
  renderMarkdown(el, "Typ `$ npm install` en dan:\n\n```sh\necho $HOME $USER\n```\n\nEn $a_1$ blijft wiskunde.");
  assert.equal(el.querySelectorAll(".katex").length, 1);
  assert.match(el.querySelector("pre code").textContent, /\$HOME \$USER/);
  assert.match(el.querySelector("p code").textContent, /^\$ npm install$/);
});

console.log(`\n${passed} test(s) geslaagd${process.exitCode ? ", maar er zijn fouten" : ""}.`);
