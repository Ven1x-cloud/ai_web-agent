// De agent-lus: model aanroepen -> tool calls uitvoeren -> resultaat teruggeven -> ... -> eindantwoord.
import { streamChat, ApiError } from "../lib/openrouter.js";
import { buildSystemPrompt, PAGE_CTX_START, PAGE_CTX_END } from "../lib/prompts.js";
import { buildTools, executeTool, describeToolCall } from "./tools.js";
import { parseModelList, providerChain, providerKey, isExhausted, nextResetMs } from "../lib/settings.js";

/** Maakt oudere berichten compact (geen oude pagina-snapshots/afbeeldingen meesturen). */
export function compactHistory(messages, { keepRecentUsers = 0, maxMessages = 40 } = {}) {
  let msgs = messages.slice(-maxMessages);
  // Niet midden in een tool-ronde beginnen.
  while (msgs.length && msgs[0].role !== "user") msgs.shift();
  const userIdx = [];
  msgs.forEach((m, i) => { if (m.role === "user" && !m._toolImage) userIdx.push(i); });
  const recent = new Set(keepRecentUsers > 0 ? userIdx.slice(-keepRecentUsers) : []); // let op: slice(-0) = alles
  return msgs.map((m, i) => {
    if (m.role === "user" && !recent.has(i)) {
      if (Array.isArray(m.content)) {
        const parts = m.content.map((p) => {
          if (p.type === "image_url") return { type: "text", text: "[afbeelding eerder gedeeld]" };
          if (p.type === "text") return { type: "text", text: stripPageContext(p.text) };
          return p;
        });
        return { ...m, content: parts };
      }
      if (typeof m.content === "string") return { ...m, content: stripPageContext(m.content) };
    }
    if (m.role === "tool" && !recent.has(lastUserBefore(userIdx, i)) && typeof m.content === "string" && m.content.length > 1500) {
      return { ...m, content: m.content.slice(0, 1500) + "…[ingekort]" };
    }
    return m;
  });
}

function lastUserBefore(userIdx, i) {
  let last = -1;
  for (const u of userIdx) { if (u < i) last = u; else break; }
  return last;
}

function stripPageContext(text) {
  if (!text || !text.startsWith(PAGE_CTX_START)) return text;
  const end = text.indexOf(PAGE_CTX_END);
  const urlLine = (text.match(/^URL: (.*)$/m) || [])[1] || "";
  const rest = end >= 0 ? text.slice(end + PAGE_CTX_END.length) : text;
  return `[Page context omitted (older turn) — was: ${urlLine}]\n` + rest;
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Gestopt", "AbortError"));
    const t = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve(); }, ms);
    const onAbort = () => { clearTimeout(t); reject(new DOMException("Gestopt", "AbortError")); };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Hoe lang wachten voor een nieuwe poging (ms). Bij een minuutlimiet langer, en Retry-After respecteren. */
function retryWait(err, base) {
  const ra = Number(err.retryAfter);
  if (Number.isFinite(ra) && ra > 0) return Math.min(ra * 1000, 30_000);
  if (err.kind === "minute_limit") return Math.max(base, 15_000);
  return base;
}

export const DEFAULT_RETRY_DELAYS = [2000, 5000];

/** Verwijdert interne velden voordat berichten naar de API gaan. */
function toApiMessages(messages) {
  return messages.map(({ _toolImage, _meta, ...m }) => m);
}

/**
 * Draait één beurt van de agent.
 * @param {object} p
 * @param {object} p.settings
 * @param {Array} p.history  eerdere berichten (API-formaat, zonder system)
 * @param {Array} p.userContent  content-array voor het nieuwe user-bericht
 * @param {object} p.ctx  { getTabId, setTabId, windowId, tabTitle, tabUrl }
 * @param {object} p.ui  callbacks: onText, onReasoning, onToolStart, onToolEnd, onStep, onRetry, onProvider, onExhausted
 * @param {AbortSignal} p.signal
 * @param {number[]} [p.retryDelays]  wachttijden (ms) voor automatische nieuwe pogingen bij drukte; lengte = max. aantal extra pogingen
 * @param {object} [p.providerState]  { [providerKey]: { until, reason } } – aanbieders waarvan de daglimiet op is
 * @returns {Promise<{messages:Array, content:string, cost:number, model:string, provider:string, providerId:string, steps:number, truncated:boolean}>}  messages = nieuwe assistant/tool-berichten (zonder het user-bericht)
 */
export async function runAgent({ settings, history, userContent, ctx, ui = {}, signal, retryDelays = DEFAULT_RETRY_DELAYS, providerState = {} }) {
  const system = { role: "system", content: buildSystemPrompt(settings, { tabTitle: ctx.tabTitle, tabUrl: ctx.tabUrl }) };
  const newMessages = [{ role: "user", content: userContent }];
  let cost = 0, usedModel = "", steps = 0, finalContent = "", truncated = false;
  const maxSteps = Math.max(1, Math.min(25, Number(settings.maxSteps) || 10));

  // Aanbieders: hoofdaanbieder eerst, dan reserve-aanbieders; wie vandaag "op" is wordt overgeslagen.
  // Het gesprek zit in `messages` (in de extensie), dus bij een wissel vergeet de AI niets.
  const chain = providerChain(settings);
  let available = chain.filter((p) => !isExhausted(providerState, p));
  if (!available.length) available = chain.slice(); // alles gemarkeerd als op → toch proberen (status kan verouderd zijn)
  let provider = available.shift();
  let isOpenRouter, candidates, tools;
  const useProvider = (p) => {
    provider = p;
    isOpenRouter = /openrouter\.ai/.test(p.baseUrl || "");
    // Kandidaat-modellen: het gekozen model eerst, dan de reserve-modellen. Bij drukte (429/5xx) schuift de lijst door.
    candidates = [...new Set([...parseModelList(p.model), ...parseModelList(p.fallbackModels)].filter(Boolean))];
    // De server-side zoektool bestaat alleen bij OpenRouter; elders valt hij terug op DuckDuckGo.
    const searchProvider = settings.searchProvider === "openrouter" && !isOpenRouter ? "duckduckgo" : settings.searchProvider;
    tools = buildTools({ ...settings, baseUrl: p.baseUrl, searchProvider });
  };
  useProvider(provider);
  if (provider.id !== "primary") ui.onProvider?.({ from: chain[0], to: provider, reason: "exhausted", error: null });

  const markExhausted = (p, err) => {
    const until = err?.resetAt || nextResetMs(p.baseUrl);
    providerState[providerKey(p)] = { until, reason: err?.kind || "" };
    ui.onExhausted?.(p, until, err);
  };
  /** Naar de volgende aanbieder; false als er geen meer is. */
  const switchProvider = (err, reason) => {
    const next = available.shift();
    if (!next) return false;
    const from = provider;
    useProvider(next);
    ui.onProvider?.({ from, to: next, reason, error: err });
    return true;
  };

  const call = (body) => streamChat({ baseUrl: provider.baseUrl, apiKey: provider.apiKey, body, signal }, {
    onText: (_d, full) => ui.onText?.(full),
    onReasoning: (_d, full) => ui.onReasoning?.(full),
    onToolCallStart: (name) => ui.onToolPending?.(name),
  });

  while (steps < maxSteps) {
    if (signal?.aborted) throw new DOMException("Gestopt", "AbortError");
    steps++;
    ui.onStep?.(steps, maxSteps);
    const messages = [system, ...toApiMessages([...compactHistory(history), ...newMessages])];
    if (steps === maxSteps) {
      // Laatste stap: dwing een eindantwoord af.
      messages.push({ role: "system", content: "Tool budget exhausted. Answer the user now with what you have, and say what you could not verify." });
    }

    let result;
    let attempt = 0;
    let minimalBody = false; // na een 400: zonder reasoning/models/usage/temperature
    while (true) {
      const body = { model: candidates[0], messages, tools, tool_choice: steps === maxSteps ? "none" : "auto" };
      if (!minimalBody) {
        body.temperature = Number.isFinite(Number(settings.temperature)) ? Number(settings.temperature) : 0.4;
        if (isOpenRouter) {
          body.usage = { include: true };
          if (candidates.length > 1) body.models = candidates; // server-side reserve-modellen
          if (settings.reasoning && settings.reasoning !== "auto") body.reasoning = { effort: settings.reasoning };
        }
      }
      try {
        result = await call(body);
        break;
      } catch (e) {
        if (!(e instanceof ApiError) || signal?.aborted) throw e;
        // Sommige providers weigeren `reasoning`/`models`/`usage`/`temperature`; probeer één keer met een minimale body.
        if (e.status === 400 && !minimalBody) { minimalBody = true; continue; }
        // Daglimiet op, geen tegoed of sleutel ongeldig: deze aanbieder is vandaag klaar → volgende aanbieder.
        if (e.kind === "daily_limit" || e.kind === "credits" || e.kind === "auth") {
          if (e.kind !== "auth") markExhausted(provider, e);
          if (switchProvider(e, e.kind)) { attempt = 0; minimalBody = false; continue; }
          throw e;
        }
        // Drukte bij de aanbieder (429 "Provider returned error", 5xx) of minuutlimiet: even wachten en
        // met het volgende model verder; daarna eventueel naar de volgende aanbieder.
        if (e.transient) {
          if (attempt < retryDelays.length) {
            const waitMs = retryWait(e, retryDelays[attempt]);
            attempt++;
            const from = candidates[0];
            if (e.kind !== "minute_limit" && candidates.length > 1) candidates = [...candidates.slice(1), from];
            ui.onRetry?.({ attempt, max: retryDelays.length, from, to: candidates[0], waitMs, error: e });
            await sleep(waitMs, signal);
            continue;
          }
          if (switchProvider(e, "busy")) { attempt = 0; minimalBody = false; continue; }
        }
        throw e;
      }
    }
    if (result.usage?.cost) cost += Number(result.usage.cost) || 0;
    if (result.model) usedModel = result.model;

    const assistantMsg = { role: "assistant", content: result.content || "" };
    if (result.tool_calls.length) assistantMsg.tool_calls = result.tool_calls;
    if (result.reasoning_details.length) assistantMsg.reasoning_details = result.reasoning_details;
    if (result.annotations.length) assistantMsg._meta = { annotations: result.annotations };
    newMessages.push(assistantMsg);

    if (!result.tool_calls.length) {
      finalContent = result.content || "";
      truncated = result.finish_reason === "length";
      break;
    }

    // Tools uitvoeren
    const imageMsgs = [];
    for (const call of result.tool_calls) {
      if (signal?.aborted) throw new DOMException("Gestopt", "AbortError");
      let args = {};
      let parseError = null;
      try { args = call.function.arguments ? JSON.parse(call.function.arguments) : {}; }
      catch (e) { parseError = e; args = {}; }
      const label = describeToolCall(call.function.name, args);
      ui.onToolStart?.(label, call);
      let out;
      if (parseError) {
        out = { text: JSON.stringify({ error: "Invalid JSON in tool arguments: " + parseError.message }) };
      } else {
        out = await executeTool(call.function.name, args, ctx);
      }
      const failed = /^\{"error"/.test(out.text || "");
      ui.onToolEnd?.(label, call, !failed, out);
      newMessages.push({ role: "tool", tool_call_id: call.id, name: call.function.name, content: out.text || "" });
      for (const img of out.images || []) {
        imageMsgs.push({
          role: "user",
          _toolImage: true,
          content: [
            { type: "text", text: `[Image returned by tool ${call.function.name}: ${img.label || ""}]` },
            { type: "image_url", image_url: { url: img.dataUrl } },
          ],
        });
      }
    }
    newMessages.push(...imageMsgs);
  }

  // Het user-bericht zelf zit al bij de aanroeper; alleen de nieuwe assistant/tool-berichten teruggeven.
  return { messages: newMessages.slice(1), content: finalContent, cost, model: usedModel, provider: provider.name, providerId: provider.id, steps, truncated };
}
