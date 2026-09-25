// UI van het zijpaneel: chat, bijlagen, tabblad-info, geschiedenis, quota.
import { loadSettings, saveSettings, onSettingsChanged, CURATED_MODELS, isFreeModel, providerChain, providerKey, isExhausted } from "../lib/settings.js";
import { getKeyInfo, getCredits, formatCost, ApiError } from "../lib/openrouter.js";
import { buildUserContent, PAGE_CTX_END } from "../lib/prompts.js";
import { runAgent } from "./agent.js";
import { describeToolCall } from "./tools.js";
import { renderMarkdown } from "./render.js";
import * as page from "./page.js";

// ---------- DOM helpers ----------
const $ = (sel) => document.querySelector(sel);
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}

const ui = {
  messages: $("#messages"), welcome: $("#welcome"), input: $("#input"), send: $("#btn-send"), stop: $("#btn-stop"),
  attach: $("#btn-attach"), shot: $("#btn-shot"), includePage: $("#include-page"), modelSelect: $("#model-select"),
  attachments: $("#attachments"), status: $("#status"), fileInput: $("#file-input"), composerBox: $("#composer-box"),
  tabTitle: $("#tab-title"), tabFavicon: $("#tab-favicon"), tabInfo: $("#tab-info"), quota: $("#quota"),
  setup: $("#setup"), setupKey: $("#setup-key"), setupSave: $("#setup-save"), setupStatus: $("#setup-status"), setupOptions: $("#setup-options"),
  historyPanel: $("#history-panel"), historyList: $("#history-list"),
};

// ---------- State ----------
let settings = null;
let currentTab = null;
let windowId = null;
let attachments = []; // [{dataUrl, name, thumb}]
let conversation = newConversation();
let abortController = null;
let saveTimer = null;
let quotaCache = { at: 0, text: "", cls: "" };

function newConversation() {
  return { id: `c_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`, title: "", messages: [], createdAt: Date.now(), updatedAt: Date.now() };
}

// ---------- Init ----------
async function init() {
  settings = await loadSettings();
  const win = await chrome.windows.getCurrent().catch(() => null);
  windowId = win?.id ?? null;
  populateModelSelect();
  ui.includePage.checked = settings.includePage !== false;
  if (!settings.apiKey) showSetup(true);
  await restoreLastConversation();
  await refreshTab();
  refreshQuota();
  checkPendingRequest();
  ui.input.focus();
}

onSettingsChanged((s) => {
  settings = s;
  populateModelSelect();
  if (settings.apiKey) showSetup(false);
  refreshQuota(true);
});

// ---------- Setup-scherm ----------
function showSetup(show) {
  ui.setup.hidden = !show;
}
ui.setupSave.addEventListener("click", async () => {
  const key = ui.setupKey.value.trim();
  if (!key) { setSetupStatus("Plak eerst je sleutel.", "err"); return; }
  setSetupStatus("Sleutel controleren…", "");
  try {
    const info = await getKeyInfo({ baseUrl: settings.baseUrl, apiKey: key });
    settings = await saveSettings({ apiKey: key, onboarded: true });
    const free = info.free_model_daily_requests;
    setSetupStatus(`Gelukt! ${free ? `Gratis vragen vandaag: ${free.remaining}/${free.limit}.` : ""}`, "ok");
    setTimeout(() => showSetup(false), 900);
    refreshQuota(true);
  } catch (e) {
    setSetupStatus(e.message || String(e), "err");
  }
});
ui.setupKey.addEventListener("keydown", (e) => { if (e.key === "Enter") ui.setupSave.click(); });
ui.setupOptions.addEventListener("click", () => chrome.runtime.openOptionsPage());
function setSetupStatus(t, cls) { ui.setupStatus.textContent = t; ui.setupStatus.className = "setup-status " + cls; }

// ---------- Model-keuze ----------
function populateModelSelect() {
  const sel = ui.modelSelect;
  sel.innerHTML = "";
  const free = h("optgroup", { label: "Gratis" });
  const paid = h("optgroup", { label: "Met tegoed" });
  const ids = new Set();
  for (const m of CURATED_MODELS) {
    ids.add(m.id);
    (m.free ? free : paid).append(h("option", { value: m.id, text: shortModelName(m) }));
  }
  if (settings.model && !ids.has(settings.model)) {
    (isFreeModel(settings.model) ? free : paid).append(h("option", { value: settings.model, text: settings.model.split("/").pop() }));
  }
  sel.append(free, paid);
  sel.value = settings.model;
}
function shortModelName(m) {
  return m.name.replace(/ — .*$/, "").replace(/\s*\(gratis\)/, " (gratis)");
}
ui.modelSelect.addEventListener("change", async () => {
  settings = await saveSettings({ model: ui.modelSelect.value });
  refreshQuota(true);
});
ui.includePage.addEventListener("change", () => saveSettings({ includePage: ui.includePage.checked }));

// ---------- Tabblad-info ----------
async function refreshTab() {
  const tab = await page.getActiveTab();
  currentTab = tab;
  if (!tab) { ui.tabTitle.textContent = "Geen tabblad"; ui.tabFavicon.hidden = true; return; }
  const restricted = page.isRestrictedUrl(tab.url);
  ui.tabTitle.textContent = restricted ? "(browserpagina – niet leesbaar)" : (tab.title || tab.url || "");
  ui.tabInfo.title = tab.url || "";
  if (tab.favIconUrl && !restricted) { ui.tabFavicon.src = tab.favIconUrl; ui.tabFavicon.hidden = false; } else ui.tabFavicon.hidden = true;
}
chrome.tabs.onActivated.addListener((info) => { if (windowId == null || info.windowId === windowId) refreshTab(); });
chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (tab.active && (windowId == null || tab.windowId === windowId) && (info.title || info.url || info.favIconUrl || info.status === "complete")) refreshTab();
});
chrome.windows.onFocusChanged?.addListener(() => refreshTab());

// ---------- Quota / tegoed ----------
async function refreshQuota(force = false) {
  if (!settings?.apiKey || !/openrouter\.ai/.test(settings.baseUrl || "")) { ui.quota.hidden = true; return; }
  if (!force && Date.now() - quotaCache.at < 60_000 && quotaCache.text) { showQuota(quotaCache.text, quotaCache.cls); return; }
  try {
    if (isFreeModel(settings.model)) {
      const info = await getKeyInfo({ baseUrl: settings.baseUrl, apiKey: settings.apiKey });
      const f = info.free_model_daily_requests;
      if (f && typeof f.remaining === "number") {
        const cls = f.remaining <= 0 ? "empty" : f.remaining <= 10 ? "low" : "";
        quotaCache = { at: Date.now(), text: `gratis: ${f.remaining}/${f.limit} vandaag`, cls };
      } else quotaCache = { at: Date.now(), text: "gratis model", cls: "" };
    } else {
      const c = await getCredits({ baseUrl: settings.baseUrl, apiKey: settings.apiKey });
      const balance = (Number(c.total_credits) || 0) - (Number(c.total_usage) || 0);
      quotaCache = { at: Date.now(), text: `tegoed: $${balance.toFixed(2)}`, cls: balance <= 0 ? "empty" : balance < 1 ? "low" : "" };
    }
    showQuota(quotaCache.text, quotaCache.cls);
  } catch (_) {
    ui.quota.hidden = true;
  }
}
function showQuota(text, cls) {
  const reserves = providerChain(settings).slice(1);
  const now = Date.now();
  const left = reserves.filter((p) => !isExhausted(providerState, p, now)).length;
  ui.quota.textContent = reserves.length ? `${text} · ${left}/${reserves.length} reserve` : text;
  ui.quota.className = "quota " + (cls || "");
  const base = isFreeModel(settings.model)
    ? "Gratis modellen: 50 verzoeken per dag (1000 na eenmalig $10 tegoed), max 20 per minuut. Elke stap van de agent telt als 1 verzoek."
    : "Resterend tegoed op OpenRouter";
  ui.quota.title = reserves.length
    ? `${base}\nReserve-aanbieders (automatisch als de hoofdaanbieder op is):\n` + reserves.map((p) => `• ${p.name}${isExhausted(providerState, p, now) ? ` – op tot ${fmtTime(providerState[providerKey(p)].until)}` : ""}`).join("\n")
    : base;
  ui.quota.hidden = false;
}

// ---------- Bijlagen ----------
async function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    try {
      const img = await page.fileToImage(f);
      attachments.push({ dataUrl: img.dataUrl, name: img.name, label: img.name });
    } catch (e) { setStatus("Afbeelding kon niet worden geladen: " + e.message); }
  }
  renderAttachments();
}
function renderAttachments() {
  ui.attachments.innerHTML = "";
  attachments.forEach((a, i) => {
    ui.attachments.append(h("div", { class: "attachment", title: a.label || "" },
      h("img", { src: a.dataUrl, alt: a.name || "" }),
      h("button", { title: "Verwijderen", onclick: () => { attachments.splice(i, 1); renderAttachments(); } }, "✕")));
  });
}
ui.attach.addEventListener("click", () => ui.fileInput.click());
ui.fileInput.addEventListener("change", () => { addFiles([...ui.fileInput.files]); ui.fileInput.value = ""; });
ui.input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});
for (const evt of ["dragenter", "dragover"]) document.addEventListener(evt, (e) => { e.preventDefault(); ui.composerBox.classList.add("dragover"); });
for (const evt of ["dragleave", "drop"]) document.addEventListener(evt, (e) => { e.preventDefault(); ui.composerBox.classList.remove("dragover"); });
document.addEventListener("drop", (e) => {
  const files = [...(e.dataTransfer?.files || [])];
  if (files.length) addFiles(files);
  else {
    const url = e.dataTransfer?.getData("text/uri-list") || e.dataTransfer?.getData("text/plain");
    if (url && /^https?:\/\//i.test(url) && /\.(png|jpe?g|gif|webp|bmp|svg)(\?|$)/i.test(url)) attachImageUrl(url);
  }
});
async function attachImageUrl(url) {
  try {
    const img = await page.normalizeImage(url, { maxDim: 1600 });
    attachments.push({ dataUrl: img.dataUrl, name: "afbeelding", label: url });
    renderAttachments();
  } catch (e) { setStatus("Afbeelding kon niet worden opgehaald: " + e.message); }
}
ui.shot.addEventListener("click", async () => {
  try {
    if (!currentTab) await refreshTab();
    const shot = await page.captureVisible(currentTab?.windowId ?? windowId, { maxDim: 1600 });
    attachments.push({ dataUrl: shot.dataUrl, name: "screenshot", label: "Screenshot van de pagina" });
    renderAttachments();
    setStatus("Screenshot toegevoegd.");
  } catch (e) { setStatus(e.message || String(e)); }
});

// ---------- Invoer ----------
ui.input.addEventListener("input", autosize);
function autosize() { ui.input.style.height = "auto"; ui.input.style.height = Math.min(160, ui.input.scrollHeight) + "px"; }
ui.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
ui.send.addEventListener("click", () => send());
ui.stop.addEventListener("click", () => abortController?.abort());
document.querySelectorAll(".quick-actions .chip").forEach((b) => b.addEventListener("click", () => { ui.input.value = b.dataset.prompt; send(); }));
$("#btn-settings").addEventListener("click", () => chrome.runtime.openOptionsPage());
$("#btn-new").addEventListener("click", () => startNewConversation());

function setStatus(text, spinner = false) {
  ui.status.innerHTML = "";
  if (spinner) ui.status.append(h("span", { class: "spinner" }));
  ui.status.append(document.createTextNode(text || ""));
}

// ---------- Versturen / agent draaien ----------
async function send() {
  const text = ui.input.value.trim();
  if (!text && !attachments.length) return;
  if (abortController) return; // er loopt al iets
  if (!settings.apiKey) { showSetup(true); return; }

  const myAttachments = attachments.slice();
  attachments = []; renderAttachments();
  ui.input.value = ""; autosize();
  ui.welcome.hidden = true;

  // Pagina-context ophalen
  await refreshTab();
  let pageInfo = null, pageNotice = "";
  const wantPage = ui.includePage.checked;
  if (currentTab && !page.isRestrictedUrl(currentTab.url)) {
    try {
      pageInfo = wantPage
        ? await page.pageInfoWithFrames(currentTab.id, { maxChars: Number(settings.pageChars) || 6000 })
        : await page.pageAction(currentTab.id, "page_info", { maxChars: 0 });
    } catch (e) {
      pageNotice = "Pagina kon niet gelezen worden: " + (e.message || e);
    }
  } else if (currentTab) {
    pageNotice = "Dit is een browserpagina; ik kan die niet lezen. Open een gewone website als je vragen over een pagina hebt.";
  }

  const userContent = buildUserContent({ text, page: pageInfo, images: myAttachments, includePage: wantPage });
  const userMsg = {
    role: "user", content: userContent,
    _meta: { question: text, thumbs: myAttachments.map((a) => a.dataUrl), url: pageInfo?.url || currentTab?.url, title: pageInfo?.title || currentTab?.title, page: !!(wantPage && pageInfo) },
  };
  conversation.messages.push(userMsg);
  if (!conversation.title) conversation.title = (text || "Afbeelding").slice(0, 60);
  const userEl = appendUserBubble(userMsg);
  if (pageNotice) ui.messages.append(h("div", { class: "notice", text: pageNotice }));

  // Assistant-beurt
  const turn = createTurnView();
  abortController = new AbortController();
  ui.send.hidden = true; ui.stop.hidden = false;
  setStatus("Denkt na…", true);
  const ctx = {
    getTabId: () => currentTab?.id,
    setTabId: async (id) => { currentTab = await page.getTab(id); refreshTab(); },
    windowId: currentTab?.windowId ?? windowId,
    tabTitle: currentTab?.title, tabUrl: currentTab?.url,
  };
  const history = conversation.messages.slice(0, -1);
  try {
    const result = await runAgent({
      settings, history, userContent, ctx, signal: abortController.signal, providerState,
      ui: {
        onStep: (n, max) => setStatus(n === 1 ? "Denkt na…" : `Stap ${n}/${max}…`, true),
        onExhausted: (p, until, err) => {
          saveProviderState();
          const why = err?.kind === "credits" ? "geen tegoed meer" : "daglimiet bereikt";
          turn.note(`🚫 ${providerLabel(p)}: ${why} (weer beschikbaar rond ${fmtTime(until)}).`);
        },
        onProvider: ({ from, to, reason }) => {
          const why = reason === "busy" ? "alle modellen overbelast" : reason === "auth" ? "sleutel werkt niet" : "vandaag op";
          const msg = `🔁 ${providerLabel(from)} ${why} → verder met ${providerLabel(to)} (${to.model.split(",")[0].trim()}). Het gesprek blijft bewaard.`;
          setStatus(msg, true);
          turn.note(msg);
          refreshQuota(true);
        },
        onRetry: ({ from, to, waitMs, error: err }) => {
          const secs = Math.round(waitMs / 1000);
          const why = err.kind === "minute_limit" ? "minuutlimiet bereikt" : err.kind === "provider_down" ? "aanbieder niet beschikbaar" : "overbelast bij de aanbieder";
          const msg = from === to
            ? `«${from.split("/").pop()}» ${why} — over ${secs} s nog een keer…`
            : `«${from.split("/").pop()}» ${why} — over ${secs} s verder met «${to.split("/").pop()}»…`;
          setStatus(msg, true);
          turn.note("⏳ " + msg);
        },
        onText: (full) => turn.text(full),
        onReasoning: (full) => turn.reasoning(full),
        onToolPending: (name) => setStatus(describeToolCall(name, {}) + "…", true),
        onToolStart: (label, call) => turn.toolStart(label, call),
        onToolEnd: (label, call, ok, out) => turn.toolEnd(call, ok, out),
      },
    });
    for (const m of result.messages) conversation.messages.push(m);
    const last = conversation.messages[conversation.messages.length - 1];
    if (last?.role === "assistant") last._meta = { ...(last._meta || {}), cost: result.cost, model: result.model, steps: result.steps };
    const shownModel = result.providerId && result.providerId !== "primary" ? `${result.provider} · ${result.model || ""}` : result.model;
    turn.finish({ content: result.content, cost: result.cost, model: shownModel, steps: result.steps, truncated: result.truncated });
    setStatus("");
  } catch (e) {
    const aborted = e?.name === "AbortError";
    let message = aborted ? "Gestopt." : (e?.message || String(e));
    if (!aborted && e instanceof ApiError && e.status === 429) message += await quotaSentence();
    const retry = () => { userEl.remove(); turn.remove(); send(); };
    turn.error(message, !aborted, e, retry, {
      alternatives: !aborted && e instanceof ApiError && e.transient ? alternativeModels() : [],
      onSwitch: async (id) => { settings = await saveSettings({ model: id }); populateModelSelect(); refreshQuota(true); retry(); },
    });
    if (!aborted) {
      // Vraag terugzetten zodat opnieuw proberen makkelijk is.
      conversation.messages.pop();
      ui.input.value = text; autosize();
      attachments = myAttachments; renderAttachments();
      setStatus("");
    } else {
      conversation.messages.push({ role: "assistant", content: "(gestopt door gebruiker)" });
      setStatus("");
    }
  } finally {
    abortController = null;
    ui.send.hidden = false; ui.stop.hidden = true;
    conversation.updatedAt = Date.now();
    scheduleSave();
    refreshQuota(true);
    ui.input.focus();
  }
}

/** "Gratis vragen vandaag: 47/50 over." — zodat duidelijk is of een 429 aan de daglimiet ligt of aan drukte. */
async function quotaSentence() {
  if (!settings.apiKey || !/openrouter\.ai/.test(settings.baseUrl || "")) return "";
  try {
    const info = await Promise.race([
      getKeyInfo({ baseUrl: settings.baseUrl, apiKey: settings.apiKey }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 4000)),
    ]);
    const f = info.free_model_daily_requests;
    if (!f || typeof f.remaining !== "number") return "";
    return f.remaining <= 0
      ? ` Je hebt vandaag alle ${f.limit} gratis vragen gebruikt.`
      : ` (Gratis vragen vandaag: nog ${f.remaining} van ${f.limit} over — je daglimiet is dus niet het probleem.)`;
  } catch (_) { return ""; }
}

// ---------- Reserve-aanbieders: onthouden wie vandaag "op" is ----------
let providerState = {};
chrome.storage.local.get("providerState").then((r) => { providerState = r.providerState || {}; }).catch(() => {});
async function saveProviderState() {
  const now = Date.now();
  for (const k of Object.keys(providerState)) if (!(providerState[k]?.until > now)) delete providerState[k];
  try { await chrome.storage.local.set({ providerState }); } catch (_) {}
}
const providerLabel = (p) => p?.name || "aanbieder";
function fmtTime(ms) {
  try { return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); } catch (_) { return ""; }
}

/** Andere gratis modellen dan het huidige (voor de knoppen "Probeer met …"). */
function alternativeModels() {
  const want = isFreeModel(settings.model) ? (m) => m.free : () => true;
  return CURATED_MODELS.filter((m) => m.id !== settings.model && want(m)).slice(0, 2).map((m) => ({ id: m.id, label: shortModelName(m).replace(/ \(gratis\)$/, "") }));
}

// ---------- Weergave ----------
function nearBottom() {
  const m = ui.messages;
  return m.scrollHeight - m.scrollTop - m.clientHeight < 120;
}
function scrollToBottom(force = false) {
  if (force || nearBottom()) ui.messages.scrollTop = ui.messages.scrollHeight;
}

function appendUserBubble(msg) {
  const meta = msg._meta || {};
  const wrap = h("div", { class: "msg user" });
  if (meta.thumbs?.length) wrap.append(h("div", { class: "thumbs" }, meta.thumbs.map((src) => h("img", { src, alt: "" }))));
  const text = meta.question ?? extractQuestion(msg.content);
  if (text) wrap.append(h("div", { class: "bubble", text }));
  if (meta.page && meta.title) wrap.append(h("div", { class: "ctx", text: "📄 " + truncate(meta.title, 50), title: meta.url || "" }));
  ui.messages.append(wrap);
  scrollToBottom(true);
  return wrap;
}

function extractQuestion(content) {
  if (typeof content === "string") return content;
  const t = (content || []).filter((p) => p.type === "text").map((p) => p.text).join("\n");
  const i = t.indexOf(PAGE_CTX_END);
  return i >= 0 ? t.slice(i + PAGE_CTX_END.length) : t;
}

/** Maakt de weergave voor één assistant-beurt (tekst, stappen, meta). */
function createTurnView() {
  const wrap = h("div", { class: "msg assistant" });
  const steps = h("div", { class: "steps" });
  wrap.append(steps);
  ui.messages.append(wrap);
  let bubble = null; // huidige tekstballon
  let reasoningEl = null;
  let raf = 0, pendingText = null;
  const stepEls = new Map();

  const ensureBubble = () => {
    if (!bubble) {
      bubble = h("div", { class: "bubble md" });
      wrap.append(bubble);
    }
    return bubble;
  };
  const flush = () => {
    raf = 0;
    if (pendingText != null) { renderMarkdown(ensureBubble(), pendingText); pendingText = null; scrollToBottom(); }
  };
  return {
    text(full) {
      pendingText = full;
      if (!raf) raf = requestAnimationFrame(flush);
    },
    reasoning(full) {
      if (!settings.showReasoning) return;
      if (!reasoningEl) {
        reasoningEl = h("details", { class: "reasoning" }, h("summary", { text: "Denkproces" }), h("pre"));
        wrap.insertBefore(reasoningEl, steps);
      }
      reasoningEl.querySelector("pre").textContent = full.slice(-4000);
    },
    toolStart(label, call) {
      flush();
      bubble = null; // volgende tekst komt ná de stappen
      const el = h("div", { class: "step" }, h("span", { class: "spinner st" }), h("span", { class: "lbl", text: label, title: label }));
      steps.append(el);
      wrap.append(steps); // stappen onderaan houden (na eerdere tekst)
      stepEls.set(call.id, el);
      setStatus(label + "…", true);
      scrollToBottom();
    },
    toolEnd(call, ok, out) {
      const el = stepEls.get(call.id);
      if (!el) return;
      el.classList.add(ok ? "ok" : "fail");
      el.querySelector(".st").replaceWith(h("span", { class: "st", text: ok ? "✓" : "✕" }));
      if (!ok) {
        try { const err = JSON.parse(out.text).error; if (err) el.title = err; } catch (_) {}
      }
      for (const img of out?.images || []) el.append(h("img", { src: img.dataUrl, alt: img.label || "", title: img.label || "" }));
      scrollToBottom();
    },
    finish({ content, cost, model, steps: n, truncated }) {
      flush();
      if (content) renderMarkdown(ensureBubble(), content);
      else if (!bubble) wrap.append(h("div", { class: "notice", text: "Het model gaf geen tekst terug. Probeer het opnieuw of kies een ander model." }));
      if (truncated) wrap.append(h("div", { class: "notice", text: "Het antwoord is afgekapt (maximale lengte bereikt)." }));
      const meta = h("div", { class: "meta" },
        h("span", { text: (model || settings.model).split("/").pop() }),
        h("span", { text: "· " + formatCost(cost) }),
        n > 1 ? h("span", { text: `· ${n} stappen` }) : null,
        h("button", { text: "Kopieer", onclick: () => navigator.clipboard.writeText(content || "").then(() => setStatus("Gekopieerd.")) }),
      );
      wrap.append(meta);
      scrollToBottom();
    },
    note(text) {
      steps.append(h("div", { class: "step note", text }));
      scrollToBottom();
    },
    error(message, retry, err, onRetry, { alternatives = [], onSwitch = null } = {}) {
      flush();
      const box = h("div", { class: "error" }, h("div", { text: message }));
      const actions = h("div", { class: "error-actions" });
      if (err instanceof ApiError && err.status === 401) actions.append(h("button", { text: "Instellingen", onclick: () => chrome.runtime.openOptionsPage() }));
      if (retry) actions.append(h("button", { text: "Opnieuw", onclick: () => (onRetry ? onRetry() : send()) }));
      for (const alt of alternatives) {
        actions.append(h("button", { text: `Probeer met ${alt.label}`, title: alt.id, onclick: () => onSwitch?.(alt.id) }));
      }
      if (actions.childElementCount) box.append(actions);
      if (err instanceof ApiError && (err.details || err.status)) {
        const lines = [
          err.status ? `HTTP ${err.status}${err.kind ? ` · ${err.kind}` : ""}` : "",
          err.model ? `model: ${err.model}` : "",
          err.provider ? `aanbieder: ${err.provider}` : "",
          err.details || "",
        ].filter(Boolean).join("\n");
        box.append(h("details", { class: "error-details" }, h("summary", { text: "Technische details" }), h("pre", { text: lines })));
      }
      wrap.append(box);
      scrollToBottom(true);
    },
    remove() { wrap.remove(); },
  };
}

/** Rendert een opgeslagen gesprek opnieuw. */
function renderConversation() {
  ui.messages.querySelectorAll(".msg, .notice").forEach((n) => n.remove());
  ui.welcome.hidden = conversation.messages.length > 0;
  const msgs = conversation.messages;
  const toolResults = new Map();
  msgs.forEach((m) => { if (m.role === "tool") toolResults.set(m.tool_call_id, m); });
  let turn = null;
  for (const m of msgs) {
    if (m.role === "user" && !m._toolImage) { appendUserBubble(m); turn = null; continue; }
    if (m.role === "assistant") {
      if (!turn) { turn = h("div", { class: "msg assistant" }); ui.messages.append(turn); }
      if (m.content) { const b = h("div", { class: "bubble md" }); renderMarkdown(b, m.content); turn.append(b); }
      if (m.tool_calls?.length) {
        const steps = h("div", { class: "steps" });
        for (const c of m.tool_calls) {
          let args = {}; try { args = JSON.parse(c.function.arguments || "{}"); } catch (_) {}
          const res = toolResults.get(c.id);
          const failed = res && /^\{"error"/.test(res.content || "");
          steps.append(h("div", { class: "step " + (failed ? "fail" : "ok") }, h("span", { class: "st", text: failed ? "✕" : "✓" }), h("span", { class: "lbl", text: describeToolCall(c.function.name, args) })));
        }
        turn.append(steps);
      }
      if (m._meta?.model || m._meta?.cost != null) {
        turn.append(h("div", { class: "meta" }, h("span", { text: (m._meta.model || "").split("/").pop() }), h("span", { text: "· " + formatCost(m._meta.cost) })));
      }
    }
  }
  scrollToBottom(true);
}

// ---------- Opslag van gesprekken ----------
function stripForStorage(msgs) {
  // Afbeeldingen uit oudere beurten weglaten (scheelt veel opslag); compactHistory doet dit toch al bij het versturen.
  let lastUser = -1;
  msgs.forEach((m, i) => { if (m.role === "user" && !m._toolImage) lastUser = i; });
  return msgs.map((m, i) => {
    if (i >= lastUser && !m._toolImage) return m;
    if (Array.isArray(m.content) && m.content.some((p) => p.type === "image_url")) {
      return { ...m, content: m.content.map((p) => (p.type === "image_url" ? { type: "text", text: "[afbeelding eerder gedeeld]" } : p)) };
    }
    return m;
  }).map((m) => (m._meta?.thumbs ? { ...m, _meta: { ...m._meta, thumbs: m._meta.thumbs.slice(0, 4) } } : m));
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveConversation, 400);
}
async function saveConversation() {
  if (!conversation.messages.length) return;
  const { conversations = [] } = await chrome.storage.local.get("conversations");
  const idx = conversations.findIndex((c) => c.id === conversation.id);
  const record = { ...conversation, messages: stripForStorage(conversation.messages) };
  if (idx >= 0) conversations[idx] = record; else conversations.unshift(record);
  conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  while (conversations.length > 30) conversations.pop();
  try {
    await chrome.storage.local.set({ conversations, lastConversationId: conversation.id });
  } catch (e) {
    // Opslag vol: oudste gesprekken weggooien en opnieuw proberen.
    await chrome.storage.local.set({ conversations: conversations.slice(0, 5), lastConversationId: conversation.id }).catch(() => {});
  }
}
async function restoreLastConversation() {
  const { conversations = [], lastConversationId } = await chrome.storage.local.get(["conversations", "lastConversationId"]);
  const last = conversations.find((c) => c.id === lastConversationId);
  if (last && Date.now() - last.updatedAt < 6 * 3600_000) {
    conversation = last;
    renderConversation();
  }
}
function startNewConversation() {
  if (abortController) abortController.abort();
  conversation = newConversation();
  renderConversation();
  ui.historyPanel.hidden = true;
  ui.input.focus();
}

// Geschiedenis-paneel
$("#btn-history").addEventListener("click", async () => {
  if (!ui.historyPanel.hidden) { ui.historyPanel.hidden = true; return; }
  const { conversations = [] } = await chrome.storage.local.get("conversations");
  ui.historyList.innerHTML = "";
  if (!conversations.length) ui.historyList.append(h("li", { class: "empty", text: "Nog geen gesprekken." }));
  for (const c of conversations) {
    const li = h("li", { class: c.id === conversation.id ? "active" : "" },
      h("span", { class: "t", text: c.title || "(zonder titel)" }),
      h("span", { class: "d", text: new Date(c.updatedAt).toLocaleDateString("nl-NL", { day: "numeric", month: "short" }) }),
      h("button", { class: "icon-btn", title: "Verwijderen", onclick: async (e) => {
        e.stopPropagation();
        const rest = conversations.filter((x) => x.id !== c.id);
        await chrome.storage.local.set({ conversations: rest });
        li.remove();
        if (c.id === conversation.id) startNewConversation();
      } }, "🗑"));
    li.addEventListener("click", () => { conversation = c; renderConversation(); ui.historyPanel.hidden = true; });
    ui.historyList.append(li);
  }
  ui.historyPanel.hidden = false;
});
$("#btn-history-close").addEventListener("click", () => { ui.historyPanel.hidden = true; });

// ---------- Verzoeken vanuit het contextmenu ----------
async function checkPendingRequest() {
  const { pendingRequest } = await chrome.storage.session.get("pendingRequest");
  if (pendingRequest && Date.now() - pendingRequest.ts < 60_000) {
    await chrome.storage.session.remove("pendingRequest");
    await handlePending(pendingRequest);
  }
}
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "session" && changes.pendingRequest?.newValue) {
    chrome.storage.session.remove("pendingRequest");
    handlePending(changes.pendingRequest.newValue);
  }
});
async function handlePending(req) {
  if (req.kind === "selection") {
    ui.input.value = `Over deze geselecteerde tekst:\n"${req.text.trim().slice(0, 2000)}"\n\nLeg dit uit.`;
    autosize();
    ui.input.focus();
    ui.input.setSelectionRange(ui.input.value.length - "Leg dit uit.".length, ui.input.value.length);
  } else if (req.kind === "image") {
    await attachImageUrl(req.src);
    if (!ui.input.value) ui.input.value = "Wat zie je op deze afbeelding? Leg het uit.";
    autosize();
    ui.input.focus();
  }
}

const truncate = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");

window.addEventListener("error", (e) => setStatus("Fout: " + (e.message || "onbekend")));
window.addEventListener("unhandledrejection", (e) => setStatus("Fout: " + (e.reason?.message || e.reason || "onbekend")));

init();
