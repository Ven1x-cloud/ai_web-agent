import { loadSettings, saveSettings, CURATED_MODELS, DEFAULTS, isFreeModel } from "../lib/settings.js";
import { getKeyInfo, getCredits, listModels } from "../lib/openrouter.js";

const $ = (s) => document.querySelector(s);
let settings = null;
let loadedModels = null; // volledige lijst van de API (na klikken)

const fields = ["apiKey", "baseUrl", "fallbackModels", "answerLanguage", "reasoning", "userName", "schoolLevel", "pageChars", "maxSteps"];
const checks = ["allowActions", "includePage", "showReasoning"];

function flashSaved() {
  const el = $("#saved");
  el.classList.add("show");
  clearTimeout(flashSaved.t);
  flashSaved.t = setTimeout(() => el.classList.remove("show"), 1200);
}

async function save(patch) {
  settings = await saveSettings(patch);
  flashSaved();
}

function fillModelSelect() {
  const sel = $("#model");
  const onlyFree = $("#onlyFree").checked;
  sel.innerHTML = "";
  const list = loadedModels || CURATED_MODELS;
  const groups = { free: document.createElement("optgroup"), paid: document.createElement("optgroup") };
  groups.free.label = "Gratis";
  groups.paid.label = "Met tegoed";
  let hasCurrent = false;
  for (const m of list) {
    if (onlyFree && !m.free) continue;
    const opt = document.createElement("option");
    opt.value = m.id;
    let label = m.name || m.id;
    if (loadedModels && !m.free) label += ` — $${fmt(m.promptPerM)} / $${fmt(m.completionPerM)} per 1M tokens`;
    opt.textContent = label;
    if (m.id === settings.model) hasCurrent = true;
    (m.free ? groups.free : groups.paid).append(opt);
  }
  if (!hasCurrent && settings.model) {
    const opt = document.createElement("option");
    opt.value = settings.model; opt.textContent = settings.model + " (huidig)";
    (isFreeModel(settings.model) ? groups.free : groups.paid).prepend(opt);
  }
  sel.append(groups.free, groups.paid);
  sel.value = settings.model;
}
const fmt = (n) => (n >= 1 ? n.toFixed(2) : n.toFixed(3)).replace(/\.?0+$/, "");

async function init() {
  settings = await loadSettings();
  for (const f of fields) $("#" + f).value = settings[f] ?? "";
  for (const c of checks) $("#" + c).checked = !!settings[c];
  const radio = document.querySelector(`input[name=searchProvider][value="${settings.searchProvider}"]`);
  if (radio) radio.checked = true;
  fillModelSelect();
  const curated = CURATED_MODELS.some((m) => m.id === settings.model);
  $("#customModel").value = curated ? "" : settings.model;

  for (const f of fields) {
    $("#" + f).addEventListener("change", (e) => {
      let v = e.target.value.trim();
      if (f === "baseUrl" && !v) v = DEFAULTS.baseUrl;
      if (f === "pageChars") v = Math.max(1000, Math.min(30000, Number(v) || DEFAULTS.pageChars));
      if (f === "maxSteps") v = Math.max(1, Math.min(25, Number(v) || DEFAULTS.maxSteps));
      save({ [f]: v });
    });
  }
  for (const c of checks) $("#" + c).addEventListener("change", (e) => save({ [c]: e.target.checked }));
  document.querySelectorAll("input[name=searchProvider]").forEach((r) => r.addEventListener("change", (e) => { if (e.target.checked) save({ searchProvider: e.target.value }); }));
  $("#model").addEventListener("change", (e) => { $("#customModel").value = ""; save({ model: e.target.value }); });
  $("#customModel").addEventListener("change", (e) => {
    const v = e.target.value.trim();
    if (v) save({ model: v }).then(fillModelSelect);
  });
  $("#onlyFree").addEventListener("change", fillModelSelect);
  $("#toggleKey").addEventListener("click", () => { const i = $("#apiKey"); i.type = i.type === "password" ? "text" : "password"; });

  $("#testKey").addEventListener("click", async () => {
    const st = $("#keyStatus");
    st.className = "status"; st.textContent = "Bezig met testen…";
    const apiKey = $("#apiKey").value.trim();
    const baseUrl = $("#baseUrl").value.trim() || DEFAULTS.baseUrl;
    if (apiKey !== settings.apiKey || baseUrl !== settings.baseUrl) await save({ apiKey, baseUrl });
    try {
      if (/openrouter\.ai/.test(baseUrl)) {
        const info = await getKeyInfo({ baseUrl, apiKey });
        const parts = [`Sleutel “${info.label || "zonder naam"}” werkt.`];
        if (info.free_model_daily_requests) parts.push(`Gratis vragen vandaag: ${info.free_model_daily_requests.remaining}/${info.free_model_daily_requests.limit}.`);
        try {
          const c = await getCredits({ baseUrl, apiKey });
          const bal = (Number(c.total_credits) || 0) - (Number(c.total_usage) || 0);
          parts.push(`Tegoed: $${bal.toFixed(2)}.`);
        } catch (_) {}
        st.textContent = parts.join(" ");
      } else {
        const models = await listModels({ baseUrl, apiKey });
        st.textContent = `Server bereikbaar. ${models.length} model(len) gevonden: ${models.slice(0, 5).map((m) => m.id).join(", ")}${models.length > 5 ? "…" : ""}`;
      }
      st.className = "status ok";
    } catch (e) {
      st.className = "status err";
      st.textContent = e.message || String(e);
    }
  });

  $("#loadModels").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true; btn.textContent = "Laden…";
    try {
      loadedModels = await listModels({ baseUrl: settings.baseUrl, apiKey: settings.apiKey });
      fillModelSelect();
      btn.textContent = `${loadedModels.length} modellen geladen`;
    } catch (err) {
      btn.textContent = "Laden mislukt: " + (err.message || err);
    } finally {
      btn.disabled = false;
    }
  });

  $("#clearHistory").addEventListener("click", async () => {
    await chrome.storage.local.remove(["conversations", "lastConversationId"]);
    const st = $("#clearStatus");
    st.textContent = "Alle gesprekken zijn gewist.";
    st.className = "status ok";
  });
}

init();
