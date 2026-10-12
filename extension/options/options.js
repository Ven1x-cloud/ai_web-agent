import { loadSettings, saveSettings, CURATED_MODELS, DEFAULTS, isFreeModel, PROVIDER_PRESETS, usesOwnModel } from "../lib/settings.js";
import { getKeyInfo, getCredits, listModels } from "../lib/openrouter.js";
import { trainStatus, startTraining, describeModel, trainingAdvice, ping, LOCAL_AI_BASE } from "../lib/localai.js";

const $ = (s) => document.querySelector(s);
let settings = null;
let loadedModels = null; // volledige lijst van de API (na klikken)
let aiTimer = 0;

const fields = ["apiKey", "baseUrl", "fallbackModels", "answerLanguage", "reasoning", "userName", "schoolLevel", "pageChars", "maxSteps", "localPageChars"];
const checks = ["allowActions", "includePage", "showReasoning", "sendFeedback", "localUseTools"];

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

// ---------- Reserve-aanbieders ----------
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

function providerRows() { return Array.isArray(settings.providers) ? settings.providers : []; }

async function saveProviders(list) {
  await save({ providers: list });
  renderProviders();
}

function renderProviders() {
  const box = $("#providers");
  box.innerHTML = "";
  const list = providerRows();
  if (!list.length) {
    box.append(el("p", { class: "muted small", text: "Nog geen reserve-aanbieders. Kies er hieronder een en klik op Toevoegen." }));
    return;
  }
  list.forEach((p, i) => {
    const preset = PROVIDER_PRESETS.find((x) => x.id === p.preset) || PROVIDER_PRESETS[PROVIDER_PRESETS.length - 1];
    const upd = (patch) => { const next = list.map((x, j) => (j === i ? { ...x, ...patch } : x)); return saveProviders(next); };
    const field = (label, key, type = "text", placeholder = "") => el("label", {}, label,
      el("input", { type, value: p[key] || "", placeholder, autocomplete: "off", spellcheck: "false", onchange: (e) => upd({ [key]: e.target.value.trim() }) }));
    const status = el("span", { class: "status", style: "margin:0" });
    const card = el("div", { class: "provider" + (p.enabled === false ? " off" : "") },
      el("div", { class: "top" },
        el("label", { class: "inline", style: "margin:0;flex-direction:row;align-items:center" },
          el("input", { type: "checkbox", checked: p.enabled !== false, onchange: (e) => upd({ enabled: e.target.checked }) }), " aan"),
        el("strong", { text: `${i + 1}. ${p.name || preset.name}` }),
        el("button", { type: "button", class: "ghost", title: "Omhoog", text: "↑", disabled: i === 0, onclick: () => { const n = list.slice(); [n[i - 1], n[i]] = [n[i], n[i - 1]]; saveProviders(n); } }),
        el("button", { type: "button", class: "ghost", title: "Omlaag", text: "↓", disabled: i === list.length - 1, onclick: () => { const n = list.slice(); [n[i + 1], n[i]] = [n[i], n[i + 1]]; saveProviders(n); } }),
        el("button", { type: "button", text: "Test", onclick: async () => {
          status.className = "status"; status.textContent = "Testen…";
          try {
            const models = await listModels({ baseUrl: p.baseUrl, apiKey: p.apiKey, onlyCapable: false });
            const want = (p.model || "").split(",")[0].trim().replace(/^models\//, "");
            const has = models.some((m) => String(m.id).replace(/^models\//, "") === want);
            status.className = "status " + (has ? "ok" : "err");
            status.textContent = has ? `Werkt ✓ (${models.length} modellen)` : `Sleutel werkt, maar model “${p.model}” staat niet in de lijst. Beschikbaar o.a.: ${models.slice(0, 6).map((m) => m.id).join(", ")}`;
          } catch (e) { status.className = "status err"; status.textContent = e.message || String(e); }
        } }),
        el("button", { type: "button", class: "danger", text: "✕", title: "Verwijderen", onclick: () => saveProviders(list.filter((_, j) => j !== i)) }),
      ),
      field("Naam", "name", "text", preset.name),
      field("Model (eventueel meerdere, komma-gescheiden)", "model", "text", preset.model),
      el("label", { class: "full" }, "Server (base URL)", el("input", { type: "text", value: p.baseUrl || "", placeholder: preset.baseUrl, onchange: (e) => upd({ baseUrl: e.target.value.trim().replace(/\/+$/, "") }) })),
      el("label", { class: "full" }, "API-sleutel", el("input", { type: "password", value: p.apiKey || "", placeholder: preset.id === "ollama" ? "(niet nodig)" : "sleutel van deze dienst", autocomplete: "off", onchange: (e) => upd({ apiKey: e.target.value.trim() }) })),
      el("div", { class: "note" }, preset.note ? preset.note + " " : "", preset.keysUrl ? el("a", { href: preset.keysUrl, target: "_blank", rel: "noopener", text: "Sleutel aanmaken ↗" }) : null, " ", status),
    );
    box.append(card);
  });
}

function initProviders() {
  const sel = $("#providerPreset");
  for (const p of PROVIDER_PRESETS) sel.append(el("option", { value: p.id, text: p.name }));
  renderProviders();
  $("#addProvider").addEventListener("click", () => {
    const preset = PROVIDER_PRESETS.find((x) => x.id === sel.value) || PROVIDER_PRESETS[0];
    const row = { id: `p_${Date.now().toString(36)}`, preset: preset.id, name: preset.name, baseUrl: preset.baseUrl, model: preset.model, apiKey: preset.apiKey || "", enabled: true };
    saveProviders([...providerRows(), row]);
    $("#providerStatus").textContent = preset.keysUrl ? `Toegevoegd. Maak een sleutel aan via ${preset.keysUrl} en plak die hierboven.` : "Toegevoegd. Vul server, model en sleutel in.";
  });
}

// ---------- Je eigen AI (map ai/) ----------

const AI_PRESETS = [
  { id: "tiny", label: "tiny (snel, ~0,7M params)" },
  { id: "small", label: "small (~4M params)" },
  { id: "medium", label: "medium (~15M params)" },
];

function aiStateText(state) {
  switch (state) {
    case "training": return "Bezig met trainen…";
    case "starting": return "Training starten…";
    case "done": return "Training klaar ✓";
    case "error": return "Training mislukt";
    default: return "";
  }
}

async function refreshAi() {
  const stateEl = $("#aiState"), statusEl = $("#aiStatus"), modelEl = $("#aiModel"), adviceEl = $("#aiAdvice");
  const base = usesOwnModel(settings) ? settings.baseUrl : LOCAL_AI_BASE;
  $("#aiPlayground").href = String(base).replace(/\/v1\/?$/, "/");
  try {
    const { info } = await ping(base);
    stateEl.textContent = info?.loaded ? "Eigen AI draait ✓" : "Eigen AI draait, maar geen model geladen";
    statusEl.className = "status " + (info?.loaded ? "ok" : "err");
    statusEl.textContent = info?.loaded
      ? `Server bereikbaar op ${base}.`
      : "Server bereikbaar, maar er is nog geen getraind model. Train er een (knop hierboven of in de terminal).";
    modelEl.textContent = describeModel(info);
    adviceEl.textContent = trainingAdvice(info);
    const fb = info?.feedback;
    if (fb?.total) {
      modelEl.textContent += ` · ${fb.total} feedback-berichten (${fb.pending} nog niet verwerkt)`;
    }
    updateTrainButton(info);
  } catch (e) {
    stateEl.textContent = "Eigen AI niet bereikbaar";
    statusEl.className = "status err";
    statusEl.textContent = (e.message || String(e)).split("\n")[0];
    modelEl.textContent = `Start de server met: cd ai && python -m webai serve`;
    adviceEl.textContent = "";
    updateTrainButton(null);
  }
}

function updateTrainButton(info) {
  const btn = $("#aiTrain");
  const training = info?.train_status?.state === "training" || info?.train_status?.state === "starting";
  btn.disabled = training;
  btn.textContent = training ? "Traint… (even geduld)" : "Train verder";
}

async function pollTraining() {
  clearTimeout(aiTimer);
  const base = usesOwnModel(settings) ? settings.baseUrl : LOCAL_AI_BASE;
  try {
    const st = await trainStatus(base);
    const statusEl = $("#aiStatus");
    if (st.state === "training" || st.state === "starting") {
      statusEl.className = "status";
      statusEl.textContent = `${aiStateText(st.state)} stap ${st.step ?? "?"}${st.loss != null ? `, verlies ${Number(st.loss).toFixed(3)}` : ""}${st.tokens_per_sec ? `, ${Math.round(st.tokens_per_sec)} tok/s` : ""}`;
      updateTrainButton({ train_status: st });
      aiTimer = setTimeout(pollTraining, 2000);
    } else if (st.state === "done") {
      statusEl.className = "status ok";
      statusEl.textContent = st.message || "Training klaar.";
      await refreshAi();
    } else if (st.state === "error") {
      statusEl.className = "status err";
      statusEl.textContent = "Training mislukt: " + (st.message || "onbekende fout");
      updateTrainButton(null);
    } else {
      updateTrainButton(null);
    }
  } catch (_) {
    updateTrainButton(null);
  }
}

async function init() {
  settings = await loadSettings();
  initProviders();
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
      if (f === "localPageChars") v = Math.max(300, Math.min(8000, Number(v) || DEFAULTS.localPageChars));
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

  // --- eigen AI ---
  $("#aiRefresh").addEventListener("click", refreshAi);
  $("#aiTrain").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const base = usesOwnModel(settings) ? settings.baseUrl : LOCAL_AI_BASE;
    const preset = window.prompt(
      "Hoe groot moet de training zijn?\n\n"
      + AI_PRESETS.map((p) => `  ${p.id} – ${p.label}`).join("\n")
      + "\n\nTyp de naam (tiny is genoeg om te proberen).",
      "tiny",
    );
    if (preset == null) return;
    const name = String(preset).trim().toLowerCase();
    if (!AI_PRESETS.some((p) => p.id === name)) {
      $("#aiStatus").className = "status err";
      $("#aiStatus").textContent = `Onbekende grootte “${name}”. Kies uit: ${AI_PRESETS.map((p) => p.id).join(", ")}.`;
      return;
    }
    btn.disabled = true;
    btn.textContent = "Starten…";
    try {
      const res = await startTraining({ preset: name, resume: true }, base);
      if (res.started === false) throw new Error(res.reason || "Niet gestart");
      $("#aiStatus").className = "status";
      $("#aiStatus").textContent = `Training gestart (${name})${res.feedback_exported ? `, ${res.feedback_exported} feedback-voorbeelden toegevoegd` : ""}. De server blijft ondertussen gewoon antwoorden.`;
      pollTraining();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Train verder";
      $("#aiStatus").className = "status err";
      $("#aiStatus").textContent = (err.message || String(err)).split("\n")[0];
    }
  });

  // --- snel wisselen tussen brein en cloud ---
  const switchTo = async (presetId) => {
    const preset = PROVIDER_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    await save({
      baseUrl: preset.baseUrl,
      model: preset.model,
      apiKey: preset.id === "webai" ? "" : settings.apiKey,
      fallbackModels: preset.id === "webai" ? "" : settings.fallbackModels,
    });
    $("#baseUrl").value = settings.baseUrl;
    $("#apiKey").value = settings.apiKey || "";
    $("#fallbackModels").value = settings.fallbackModels || "";
    fillModelSelect();
    refreshAi();
  };
  $("#useOwn").addEventListener("click", () => switchTo("webai"));
  $("#useOpenRouter").addEventListener("click", () => switchTo("openrouter"));
  $("#useOllama").addEventListener("click", () => switchTo("ollama"));

  await refreshAi();
  pollTraining();
}

init();
