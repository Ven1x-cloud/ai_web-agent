// Client voor de EIGEN AI: de losse Python-server in ai/ (map `webai`).
//
// De server is OpenAI-compatibel, dus chatten gaat via lib/openrouter.js (streamChat).
// Dit bestand doet alleen de extra's die een cloud-API niet heeft:
//   - hoe ver is mijn model getraind?            GET  /v1/info
//   - 👍/👎/verbetering doorgeven                 POST /v1/feedback
//   - een training starten zonder terminal       POST /v1/train
//   - hoe ver is die training?                   GET  /v1/status
//   - model opnieuw laden na een training        POST /v1/reload
//
// Alles gebeurt op localhost; er gaat niets naar buiten.

export const LOCAL_AI_BASE = "http://localhost:8787/v1";
export const LOCAL_AI_MODEL = "webai";

/** Is dit een eigen, lokaal model (en geen cloud-aanbieder)? */
export function isLocalAi(baseUrl) {
  return /^(https?:\/\/)?(localhost|127\.0\.0\.1|\[::1\])(:\d+)?/i.test(String(baseUrl || "").trim());
}

function join(base, path) {
  return String(base || LOCAL_AI_BASE).replace(/\/+$/, "") + path;
}

async function request(base, path, { method = "GET", body = null, timeoutMs = 8000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(join(base, path), {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = { message: text }; }
    if (!res.ok) {
      const err = new Error(data?.error?.message || `Server gaf ${res.status} terug`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  } catch (e) {
    // fetch faalt met TypeError als de server niet draait → duidelijke melding
    if (e.name === "AbortError") {
      const err = new Error("De eigen AI-server reageerde niet op tijd.");
      err.unreachable = true;
      throw err;
    }
    if (e instanceof TypeError || /fetch|network|failed/i.test(String(e.message))) {
      const err = new Error(
        "De eigen AI-server is niet bereikbaar op " + join(base, "") + ".\n"
        + "Start hem in een terminal met:\n"
        + "    cd ai\n    python -m webai serve\n"
        + "Of kies bij Instellingen een aanbieder in de cloud."
      );
      err.unreachable = true;
      throw err;
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/** Draait de server en is er een model geladen? */
export async function ping(base = LOCAL_AI_BASE) {
  const info = await request(base, "/info", { timeoutMs: 4000 });
  return {
    ok: true,
    loaded: !!info?.loaded,
    info,
  };
}

/** Statistieken van het eigen model (parameters, verlies, aantal stappen…). */
export async function modelInfo(base = LOCAL_AI_BASE) {
  return request(base, "/info", { timeoutMs: 5000 });
}

/** Hoe ver is een lopende training? */
export async function trainStatus(base = LOCAL_AI_BASE) {
  const data = await request(base, "/status", { timeoutMs: 5000 });
  return data?.train || { state: "idle" };
}

/**
 * Geef feedback op een antwoord. De server bewaart het in ai/data/feedback.jsonl
 * en maakt er bij de volgende training leermateriaal van.
 * @param {object} p
 * @param {"thumbs_up"|"thumbs_down"|"correction"|"note"} p.kind
 * @param {Array} p.messages  het gesprek tot en met de vraag van de gebruiker
 * @param {string} p.response het antwoord van het model
 * @param {string} [p.correction] jouw verbeterde antwoord
 * @param {string} [p.note] vrije tekst
 */
export async function sendFeedback({ kind, messages, response, correction = "", note = "", model = "" }, base = LOCAL_AI_BASE) {
  return request(base, "/feedback", {
    method: "POST",
    body: { kind, messages, response, correction, note, model },
    timeoutMs: 6000,
  });
}

/** Start een training op de achtergrond (de server blijft ondertussen antwoorden). */
export async function startTraining({ preset = "tiny", steps = null, resume = true } = {}, base = LOCAL_AI_BASE) {
  return request(base, "/train", { method: "POST", body: { preset, steps, resume }, timeoutMs: 15000 });
}

/** Laat de server het zojuist getrainde model in het geheugen vervangen. */
export async function reloadModel(base = LOCAL_AI_BASE) {
  return request(base, "/reload", { method: "POST", timeoutMs: 20000 });
}

/** Leesbare samenvatting van het model, voor in de instellingen. */
export function describeModel(info) {
  if (!info || !info.exists) return "Nog geen model getraind.";
  const bits = [];
  if (info.params) bits.push(`${(info.params / 1e6).toFixed(2)}M parameters`);
  if (info.layers) bits.push(`${info.layers} lagen`);
  if (info.width) bits.push(`${info.width} breed`);
  if (info.vocab) bits.push(`${info.vocab} tokens`);
  if (info.context) bits.push(`context ${info.context}`);
  const loss = info.eval_loss ?? info.best_eval ?? info.loss;
  if (loss != null) bits.push(`verlies ${Number(loss).toFixed(2)}`);
  if (info.steps) bits.push(`${info.steps} stappen`);
  if (info.tokens_seen) bits.push(`${Math.round(info.tokens_seen / 1000)}k tokens gezien`);
  return bits.join(" · ") || "Model geladen.";
}

/** Kort advies op basis van de statistieken (overfit, te weinig data, …). */
export function trainingAdvice(info) {
  if (!info || !info.exists) {
    return "Train je eerste model: python -m webai build && python -m webai train --preset tiny";
  }
  const train = Number(info.loss);
  const evl = Number(info.eval_loss ?? info.best_eval ?? NaN);
  const tokens = Number(info.tokens_seen || 0);
  if (Number.isFinite(train) && Number.isFinite(evl) && evl > train * 1.6) {
    return "Let op: het eval-verlies ligt ver boven het trainingsverlies — het model leert de "
      + "voorbeeldantwoorden uit zijn hoofd in plaats van de taal. Meer data helpt het meest: "
      + "python -m webai distill (antwoorden laten maken door een cloudmodel) of zet eigen "
      + "bestanden in ai/data/learned/.";
  }
  if (tokens < 500_000) {
    return "Je model heeft nog weinig tekst gezien. Voeg data toe in ai/data/learned/ "
      + "(.txt, .md of .jsonl) en train verder met --init-from.";
  }
  return "Model ziet er redelijk uit. Voeg nieuwe voorbeelden toe en train verder om het te verbeteren.";
}
