// Kleine client voor de OpenRouter API (OpenAI-compatibel, met streaming en tool calls).
// Werkt ook met andere OpenAI-compatibele servers (Ollama, LM Studio) via een andere baseUrl.

export class ApiError extends Error {
  constructor(message, { status = 0, code = "", data = null, retryAfter = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.data = data;
    this.retryAfter = retryAfter;
  }
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

export function friendlyError(status, data, resetHeader) {
  const msg = data?.error?.message || data?.message || "";
  const meta = data?.error?.metadata || {};
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
      return "Time-out bij de aanbieder (408). Probeer het nog eens.";
    case 429: {
      let when = "";
      if (resetHeader) {
        const ms = Number(resetHeader);
        if (ms > 1e12) when = ` Reset om ${new Date(ms).toLocaleTimeString()}.`;
      }
      return "Te veel verzoeken (429). Bij gratis modellen geldt 20 per minuut en 50 per dag (1000/dag na eenmalig $10 tegoed)." + when + (msg ? ` (${msg})` : "");
    }
    case 502:
    case 503:
      return "De aanbieder van dit model is (tijdelijk) niet beschikbaar. Probeer het opnieuw of kies een ander model. " + msg;
    default:
      return msg || `Onbekende fout (${status}).`;
  }
}

async function readError(res) {
  let data = null;
  try { data = await res.json(); } catch (_) { try { data = { message: await res.text() }; } catch (__) {} }
  const retryAfter = res.headers.get("Retry-After");
  const reset = res.headers.get("X-RateLimit-Reset");
  return new ApiError(friendlyError(res.status, data, reset), {
    status: res.status, code: data?.error?.code || "", data, retryAfter,
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
  if (!res.ok) throw await readError(res);
  if (!res.body) throw new ApiError("Geen antwoordstroom ontvangen.");

  const result = { content: "", reasoning: "", reasoning_details: [], tool_calls: [], finish_reason: "", usage: null, model: "", annotations: [] };
  const calls = new Map(); // index -> tool call
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleChunk = (chunk) => {
    if (chunk.error) {
      throw new ApiError(friendlyError(chunk.error.code || 500, { error: chunk.error }), { status: chunk.error.code || 500, data: chunk });
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
      throw new ApiError("De aanbieder brak het antwoord af (stream error).", { status: 500, data: chunk });
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
