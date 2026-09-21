// Kleine client voor de OpenRouter API (OpenAI-compatibel, met streaming en tool calls).
// Werkt ook met andere OpenAI-compatibele servers (Ollama, LM Studio) via een andere baseUrl.

export class ApiError extends Error {
  constructor(message, { status = 0, code = "", data = null, retryAfter = null, kind = "", model = "", provider = "", details = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.data = data;
    this.retryAfter = retryAfter;
    this.kind = kind; // zie classifyError()
    this.model = model;
    this.provider = provider;
    this.details = details; // ruwe tekst van de aanbieder (voor "technische details")
  }
  /** Heeft het zin om (met een ander model of na een korte pauze) opnieuw te proberen? */
  get transient() { return TRANSIENT_KINDS.has(this.kind); }
}

const TRANSIENT_KINDS = new Set(["provider_busy", "provider_down", "minute_limit"]);

function errorText(data) {
  const msg = String(data?.error?.message || data?.message || "");
  const meta = data?.error?.metadata || {};
  let raw = meta.raw;
  if (raw && typeof raw !== "string") { try { raw = JSON.stringify(raw); } catch (_) { raw = String(raw); } }
  return { msg, meta, raw: raw ? String(raw) : "", provider: meta.provider_name || "" };
}

/**
 * Bepaalt wat voor soort fout dit is:
 *  - daily_limit   : jouw gratis daglimiet (50/dag) is op → wachten tot morgen (of Ollama/tegoed)
 *  - minute_limit  : te veel verzoeken per minuut → even wachten
 *  - provider_busy : de aanbieder van het model is overbelast (429 "Provider returned error") → ander model/later
 *  - provider_down : aanbieder tijdelijk niet beschikbaar (5xx/408) → ander model/later
 *  - credits / auth / "" (overig)
 */
export function classifyError(status, data) {
  const { msg, raw } = errorText(data);
  const all = `${msg} ${raw}`;
  if (status === 429) {
    if (/per[-_\s]?day|daily|per dag/i.test(all)) return "daily_limit";
    if (/per[-_\s]?min|per minute|per minuut/i.test(all)) return "minute_limit";
    return "provider_busy";
  }
  if (status === 402) return "credits";
  if (status === 401) return "auth";
  if ([408, 500, 502, 503, 504, 520, 521, 522, 523, 524].includes(status)) return "provider_down";
  return "";
}

const APP_HEADERS = {
  "HTTP-Referer": "https://github.com/Ven1x-cloud/ai_web-agent",
  "X-Title": "AI Web Agent",
};

function headers(apiKey, extra = {}) {
  const h = { "Content-Type": "application/json", ...APP_HEADERS, ...extra };
  if (apiKey) h.Authorization = `Bearer ${apiKey}`;
  return h;
}

function joinUrl(base, path) {
  return String(base || "").replace(/\/+$/, "") + path;
}

export function friendlyError(status, data, resetHeader, { model = "" } = {}) {
  const { msg, meta, provider } = errorText(data);
  const kind = classifyError(status, data);
  const modelName = model ? `«${model.split("/").pop()}»` : "dit model";
  const via = provider ? ` (${provider})` : "";
  switch (status) {
    case 401:
      return "De API-sleutel is ongeldig of ontbreekt. Controleer je OpenRouter-sleutel bij Instellingen.";
    case 402:
      return "Geen tegoed voor dit model (402). Kies een gratis model (eindigt op :free) of voeg tegoed toe op openrouter.ai. " + (meta.remedy_hint || "");
    case 403:
      return "Toegang geweigerd (403): " + (msg || "dit model of deze functie is niet beschikbaar voor jouw account.");
    case 404:
      return "Model niet gevonden (404). Controleer de model-ID bij Instellingen. " + msg;
    case 408:
      return `Time-out bij de aanbieder van ${modelName}${via} (408). Probeer het nog eens of kies een ander model.`;
    case 429: {
      let when = "";
      if (resetHeader) {
        const ms = Number(resetHeader);
        if (ms > 1e12) when = ` Reset om ${new Date(ms).toLocaleTimeString()}.`;
      }
      if (kind === "daily_limit") {
        return "Je gratis daglimiet is op: 50 per dag zonder tegoed (1000 per dag na eenmalig $10 tegoed). De teller reset om 00:00 UTC (01:00/02:00 Nederlandse tijd)." + when
          + " Tip: met een lokaal Ollama-model (zie Instellingen) kun je nu wél verder.";
      }
      if (kind === "minute_limit") {
        return "Te snel achter elkaar: bij gratis modellen mag je maximaal 20 verzoeken per minuut doen. Wacht een halve minuut en probeer het dan opnieuw." + when;
      }
      return `Het model ${modelName} is op dit moment overbelast bij de aanbieder${via} (429). Dat ligt niet aan jouw daglimiet: gratis modellen delen hun capaciteit met alle OpenRouter-gebruikers. Probeer een ander gratis model of probeer het over een minuut opnieuw.`;
    }
    case 500:
    case 502:
    case 503:
    case 504:
      return `De aanbieder van ${modelName}${via} is (tijdelijk) niet beschikbaar (${status}). Probeer het opnieuw of kies een ander model.`;
    default:
      return msg || `Onbekende fout (${status}).`;
  }
}

function makeApiError(status, data, { resetHeader = null, retryAfter = null, model = "" } = {}) {
  const { msg, raw, provider } = errorText(data);
  const details = [msg, raw].filter(Boolean).join(" — ").slice(0, 400);
  return new ApiError(friendlyError(status, data, resetHeader, { model }), {
    status, code: data?.error?.code || "", data, retryAfter, kind: classifyError(status, data), model, provider, details,
  });
}

async function readError(res, { model = "" } = {}) {
  let data = null;
  try { data = await res.json(); } catch (_) { try { data = { message: await res.text() }; } catch (__) {} }
  return makeApiError(res.status, data, {
    retryAfter: res.headers.get("Retry-After"),
    resetHeader: res.headers.get("X-RateLimit-Reset"),
    model,
  });
}

/**
 * Streamt een chat completion. `handlers` krijgt tussentijdse updates; het resultaat is het complete bericht.
 * @returns {Promise<{content:string, reasoning:string, tool_calls:Array, finish_reason:string, usage:object|null, model:string, annotations:Array}>}
 */
export async function streamChat({ baseUrl, apiKey, body, signal }, handlers = {}) {
  const res = await fetch(joinUrl(baseUrl, "/chat/completions"), {
    method: "POST",
    headers: headers(apiKey, { Accept: "text/event-stream" }),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok) throw await readError(res, { model: body.model });
  if (!res.body) throw new ApiError("Geen antwoordstroom ontvangen.", { status: 0, kind: "provider_down", model: body.model });

  const result = { content: "", reasoning: "", reasoning_details: [], tool_calls: [], finish_reason: "", usage: null, model: "", annotations: [] };
  const calls = new Map(); // index -> tool call
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleChunk = (chunk) => {
    if (chunk.error) {
      throw makeApiError(Number(chunk.error.code) || 500, { error: chunk.error }, { model: body.model });
    }
    if (chunk.model) result.model = chunk.model;
    if (chunk.usage) result.usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (!choice) return;
    const delta = choice.delta || {};
    if (typeof delta.content === "string" && delta.content) {
      result.content += delta.content;
      handlers.onText?.(delta.content, result.content);
    }
    if (Array.isArray(delta.reasoning_details)) result.reasoning_details.push(...delta.reasoning_details);
    const reasoningDelta = typeof delta.reasoning === "string" ? delta.reasoning
      : Array.isArray(delta.reasoning_details) ? delta.reasoning_details.map((d) => d.text || d.summary || "").join("") : "";
    if (reasoningDelta) {
      result.reasoning += reasoningDelta;
      handlers.onReasoning?.(reasoningDelta, result.reasoning);
    }
    if (Array.isArray(delta.annotations)) result.annotations.push(...delta.annotations);
    if (Array.isArray(choice.message?.annotations)) result.annotations.push(...choice.message.annotations);
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        const idx = tc.index ?? calls.size;
        let call = calls.get(idx);
        if (!call) {
          call = { id: tc.id || null, type: "function", function: { name: "", arguments: "" } };
          calls.set(idx, call);
          if (tc.function?.name) handlers.onToolCallStart?.(tc.function.name);
        }
        if (tc.id && !call.id) call.id = tc.id;
        if (tc.function?.name && tc.function.name !== call.function.name) {
          // Sommige providers sturen de naam in één keer, andere in stukjes.
          call.function.name = call.function.name && tc.function.name.startsWith(call.function.name)
            ? tc.function.name : call.function.name + tc.function.name;
        }
        if (tc.function?.arguments) call.function.arguments += tc.function.arguments;
      }
    }
    if (choice.finish_reason) result.finish_reason = choice.finish_reason;
    if (choice.finish_reason === "error") {
      throw new ApiError("De aanbieder brak het antwoord af (stream error).", { status: 500, data: chunk, kind: "provider_down", model: body.model });
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).replace(/\r$/, "");
      buffer = buffer.slice(nl + 1);
      if (!line || line.startsWith(":")) continue; // keep-alive/commentaar
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") continue;
      let chunk;
      try { chunk = JSON.parse(payload); } catch (_) { continue; }
      handleChunk(chunk);
    }
  }
  if (buffer.trim().startsWith("data:")) {
    try { handleChunk(JSON.parse(buffer.trim().slice(5).trim())); } catch (_) {}
  }

  result.tool_calls = [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([i, c]) => {
    if (!c.id) c.id = `call_${Date.now().toString(36)}_${i}`;
    return c;
  }).filter((c) => c.function.name);
  if (!result.finish_reason) result.finish_reason = result.tool_calls.length ? "tool_calls" : "stop";
  return result;
}

/** Info over de API-sleutel (tegoed, gratis-limiet). Alleen OpenRouter. */
export async function getKeyInfo({ baseUrl, apiKey }) {
  const res = await fetch(joinUrl(baseUrl, "/key"), { headers: headers(apiKey) });
  if (!res.ok) throw await readError(res);
  const json = await res.json();
  return json.data || json;
}

/** Tegoed-overzicht (alleen OpenRouter): { total_credits, total_usage }. */
export async function getCredits({ baseUrl, apiKey }) {
  const res = await fetch(joinUrl(baseUrl, "/credits"), { headers: headers(apiKey) });
  if (!res.ok) throw await readError(res);
  const json = await res.json();
  return json.data || json;
}

/** Lijst modellen met vision + tools. Werkt met OpenRouter en (beperkt) met andere OpenAI-compatibele servers. */
export async function listModels({ baseUrl, apiKey, onlyCapable = true }) {
  const isOpenRouter = /openrouter\.ai/.test(baseUrl);
  const url = isOpenRouter && onlyCapable
    ? joinUrl(baseUrl, "/models?supported_parameters=tools&input_modalities=image")
    : joinUrl(baseUrl, "/models");
  const res = await fetch(url, { headers: headers(apiKey) });
  if (!res.ok) throw await readError(res);
  const json = await res.json();
  const arr = json.data || json.models || [];
  return arr.map((m) => {
    const prompt = Number(m.pricing?.prompt ?? 0);
    const completion = Number(m.pricing?.completion ?? 0);
    return {
      id: m.id || m.name,
      name: m.name || m.id,
      free: /:free$/.test(m.id || "") || (m.pricing && prompt === 0 && completion === 0),
      promptPerM: prompt * 1e6,
      completionPerM: completion * 1e6,
      context: m.context_length || m.top_provider?.context_length || null,
      vision: m.architecture?.input_modalities?.includes("image") ?? true,
      tools: m.supported_parameters?.includes("tools") ?? true,
    };
  }).sort((a, b) => (a.free === b.free ? a.name.localeCompare(b.name) : a.free ? -1 : 1));
}

export function formatCost(usd) {
  if (!usd || usd <= 0) return "gratis";
  if (usd < 0.001) return `$${usd.toFixed(5)}`;
  if (usd < 0.1) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}
