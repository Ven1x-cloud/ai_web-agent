// Instellingen (chrome.storage.local) + curated modellenlijst.
//
// Standaard is het brein het EIGEN model uit de map ai/ (los van de extensie,
// zelf te trainen). Cloud-aanbieders staan klaar als reserve: die kun je
// aanzetten zodra je meer rekenkracht of betere antwoorden wilt.

import { LOCAL_AI_BASE, LOCAL_AI_MODEL, isLocalAi } from "./localai.js";

export const DEFAULT_BASE_URL = LOCAL_AI_BASE;
export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

// Modellen die zowel afbeeldingen (vision) als tools ondersteunen op OpenRouter.
// "free" = gratis variant (met daglimiet: 50 verzoeken/dag zonder tegoed, 1000/dag na eenmalig $10 tegoed).
export const CURATED_MODELS = [
  { id: "qwen/qwen3.8-27b:free", name: "Qwen 3.8 27B (gratis) — aanbevolen zonder tegoed", free: true },
  { id: "google/gemma-4-31b-it:free", name: "Google Gemma 4 31B (gratis)", free: true },
  { id: "thinkingmachines/inkling:free", name: "Thinking Machines Inkling (gratis)", free: true },
  { id: "nex-agi/nex-n2.5-pro:free", name: "Nex N2.5 Pro (gratis)", free: true },
  { id: "inclusionai/ling-3.0-flash-vl:free", name: "Ling 3.0 Flash VL (gratis)", free: true },
  { id: "google/gemini-3.5-flash", name: "Google Gemini 3.5 Flash — aanbevolen met tegoed", free: false },
  { id: "google/gemini-3.1-flash-lite", name: "Google Gemini 3.1 Flash Lite (heel goedkoop)", free: false },
  { id: "openai/gpt-5.2", name: "OpenAI GPT-5.2", free: false },
  { id: "anthropic/claude-sonnet-4.5", name: "Anthropic Claude Sonnet 4.5", free: false },
];

// Reserve-aanbieders: andere diensten met een EIGEN gratis limiet (OpenAI-compatibele API met vision + tools).
// Let op: meerdere OpenRouter-accounts is verboden in hun voorwaarden en extra sleutels binnen één account delen
// dezelfde 50/dag. Andere diensten hebben hun eigen limiet – dat is wél toegestaan.
export const PROVIDER_PRESETS = [
  { id: "webai", name: "Eigen AI (lokaal, zelf te trainen)", baseUrl: LOCAL_AI_BASE, model: LOCAL_AI_MODEL, apiKey: "", keysUrl: "", note: "Jouw eigen model uit de map ai/. Start de server met `cd ai && python -m webai serve`. Geen sleutel nodig, geen daglimiet, alles blijft op je eigen computer." },
  { id: "ollama", name: "Ollama (open model op je eigen pc)", baseUrl: "http://localhost:11434/v1", model: "qwen2.5vl:7b", apiKey: "ollama", keysUrl: "https://ollama.com/download", note: "Onbeperkt en gratis. Start Ollama met OLLAMA_ORIGINS=chrome-extension://* en haal een model met `ollama pull qwen2.5vl:7b`." },
  { id: "openrouter", name: "OpenRouter (cloud, gratis modellen)", baseUrl: OPENROUTER_BASE_URL, model: "qwen/qwen3.8-27b:free", apiKey: "", keysUrl: "https://openrouter.ai/settings/keys", note: "Veel slimmer dan een klein eigen model, maar 50 vragen per dag gratis en je hebt een account nodig (18+)." },
  { id: "google", name: "Google AI Studio (Gemini)", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-3.8-flash", keysUrl: "https://aistudio.google.com/apikey", note: "Gratis tier met eigen daglimiet (reset om 09:00 NL-tijd). Controleer de modelnaam op aistudio.google.com." },
  { id: "groq", name: "Groq", baseUrl: "https://api.groq.com/openai/v1", model: "qwen/qwen3.8-27b", keysUrl: "https://console.groq.com/keys", note: "Gratis tier met eigen limieten; kies een model met afbeeldingen + tools uit hun lijst." },
  { id: "mistral", name: "Mistral", baseUrl: "https://api.mistral.ai/v1", model: "mistral-small-latest", keysUrl: "https://console.mistral.ai/api-keys", note: "Gratis 'Experiment'-tier (telefoonverificatie nodig)." },
  { id: "custom", name: "Anders (OpenAI-compatibel)", baseUrl: "", model: "", keysUrl: "", note: "Elke server met een OpenAI-compatibele /chat/completions die afbeeldingen en tools ondersteunt." },
];

export const DEFAULTS = {
  apiKey: "",
  baseUrl: DEFAULT_BASE_URL,
  model: LOCAL_AI_MODEL,
  // Het hoofdmodel zelf wordt overgeslagen; dit is de wolk-terugval als je die aanzet.
  fallbackModels: "",
  searchProvider: "duckduckgo", // "duckduckgo" | "openrouter" | "off"
  answerLanguage: "auto", // "auto" | "nl" | "en" | "fr" | "de"
  reasoning: "low", // "auto" | "low" | "medium" | "high" — laag = snel; hoog voor moeilijke wiskunde
  allowActions: true, // klikken/typen/navigeren toestaan
  includePage: true, // paginatekst automatisch meesturen bij elke vraag
  pageChars: 6000,
  maxSteps: 10,
  temperature: 0.4,
  showReasoning: false, // "Denkproces" van het model tonen (indien het model dat meestuurt)
  userName: "",
  schoolLevel: "", // bijv. "3 havo", "5 vwo" — helpt bij het niveau van de uitleg
  onboarded: false,
  providers: [], // reserve-aanbieders: [{ id, preset, name, baseUrl, apiKey, model, enabled }]
  // Feedback op antwoorden naar de eigen AI-server sturen, zodat het model leert.
  sendFeedback: true,
  // Pagina-inhoud beperken voor het eigen model: een klein model heeft een korte context.
  localPageChars: 1800,
  // Gereedschappen (pagina lezen, zoeken, klikken) ook aan je eigen model geven.
  // Uit is het verstandige begin: de gereedschapslijst kost veel van de korte
  // context en een klein model kan nog niet betrouwbaar JSON-aanroepen maken.
  localUseTools: false,
};

export async function loadSettings() {
  const stored = await chrome.storage.local.get("settings");
  return { ...DEFAULTS, ...(stored.settings || {}) };
}

export async function saveSettings(patch) {
  const current = await loadSettings();
  const next = { ...current, ...patch };
  await chrome.storage.local.set({ settings: next });
  return next;
}

export function onSettingsChanged(cb) {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.settings) cb({ ...DEFAULTS, ...(changes.settings.newValue || {}) });
  });
}

export function isFreeModel(id) {
  return /:free$/i.test(id || "");
}

export function parseModelList(str) {
  return String(str || "")
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Sleutel om per aanbieder te onthouden dat de daglimiet op is (zonder de hele API-sleutel op te slaan). */
export function providerKey(p) {
  return `${p.baseUrl || ""}|${(p.apiKey || "").slice(-6)}`;
}

/** De hoofdaanbieder (instellingen bovenaan de pagina) in dezelfde vorm als een reserve-aanbieder. */
export function primaryProvider(settings) {
  return {
    id: "primary",
    name: providerName(settings.baseUrl),
    baseUrl: settings.baseUrl,
    apiKey: settings.apiKey,
    model: settings.model,
    fallbackModels: settings.fallbackModels,
    enabled: true,
    local: isLocalAi(settings.baseUrl),
  };
}

/** Leesbare naam bij een API-adres. */
export function providerName(baseUrl) {
  const url = String(baseUrl || "");
  if (/openrouter\.ai/.test(url)) return "OpenRouter";
  if (/googleapis\.com/.test(url)) return "Google AI Studio";
  if (/groq\.com/.test(url)) return "Groq";
  if (/mistral\.ai/.test(url)) return "Mistral";
  if (/:11434/.test(url)) return "Ollama";
  if (isLocalAi(url)) return "Eigen AI (lokaal)";
  return url ? "Hoofdaanbieder" : "Geen aanbieder";
}

/**
 * Hoeveel tekens van de pagina er meegaan in de vraag.
 *
 * Een klein, zelf getraind model heeft een korte context (vaak 256–512 tokens).
 * Stuur je daar 6000 tekens pagina naartoe, dan is de context vol voordat het
 * kan antwoorden. Daarom geldt voor de eigen AI een lagere grens.
 */
export function effectivePageChars(settings) {
  const base = Number(settings.pageChars) || 6000;
  if (usesOwnModel(settings)) {
    const local = Number(settings.localPageChars) || 1800;
    return Math.max(300, Math.min(base, local));
  }
  return base;
}

/**
 * Model-ID's van het eigen model. De server in ai/ meldt zich altijd als
 * `webai`, ongeacht de run-naam — dat is het kenmerk.
 */
export const OWN_MODEL_IDS = ["webai"];

/**
 * Is deze aanbieder je eigen, lokaal getrainde model?
 *
 * Bewust eng gehouden: een willekeurige lokale server (Ollama, LM Studio, een
 * test-double) is níét je eigen model. Alleen lokaal + model-ID `webai` telt.
 * Dat verschil is belangrijk, want het eigen model krijgt een korte
 * systeemprompt en standaard geen gereedschapslijst.
 */
export function isOwnModelProvider(p) {
  const url = String(p?.baseUrl || "");
  if (!isLocalAi(url) || /:11434/.test(url)) return false;
  const first = String(p?.model || "").split(",")[0].trim().toLowerCase();
  if (!first) return false;
  return OWN_MODEL_IDS.some((id) => first === id || first.startsWith(id + "/") || first.startsWith(id + ":"));
}

/** Draait het brein nu op je eigen, lokale model? */
export function usesOwnModel(settings) {
  return isOwnModelProvider({ baseUrl: settings?.baseUrl, model: settings?.model });
}

function providerUsable(p) {
  if (!p || p.enabled === false || !p.baseUrl || !p.model) return false;
  // Een lokale server (eigen AI, Ollama) heeft geen sleutel nodig.
  return !!p.apiKey || isLocalAi(p.baseUrl);
}

/** Hoofdaanbieder + bruikbare reserve-aanbieders, in volgorde. */
export function providerChain(settings) {
  return [primaryProvider(settings), ...(settings.providers || []).filter(providerUsable)];
}

/** Tot wanneer een aanbieder "op" is: eigen resetmoment van de dienst, anders 6 uur. */
export function nextResetMs(baseUrl, now = Date.now()) {
  const d = new Date(now);
  const utcMidnight = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1);
  if (/openrouter\.ai/.test(baseUrl || "")) return utcMidnight; // OpenRouter: 00:00 UTC
  if (/googleapis\.com/.test(baseUrl || "")) { // Google: middernacht Pacific ≈ 07:00–08:00 UTC
    let t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 8);
    if (t <= now) t += 86_400_000;
    return t;
  }
  return now + 6 * 3_600_000;
}

export function isExhausted(state, p, now = Date.now()) {
  const e = state && state[providerKey(p)];
  return !!(e && e.until > now);
}
