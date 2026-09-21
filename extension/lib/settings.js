// Instellingen (chrome.storage.local) + curated modellenlijst.

export const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";

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

export const DEFAULTS = {
  apiKey: "",
  baseUrl: DEFAULT_BASE_URL,
  model: "qwen/qwen3.8-27b:free",
  fallbackModels: "google/gemma-4-31b-it:free, thinkingmachines/inkling:free",
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
