// De agent-lus: model aanroepen -> tool calls uitvoeren -> resultaat teruggeven -> ... -> eindantwoord.
import { streamChat, ApiError } from "../lib/openrouter.js";
import { buildSystemPrompt, PAGE_CTX_START, PAGE_CTX_END } from "../lib/prompts.js";
import { buildTools, executeTool, describeToolCall } from "./tools.js";
import { parseModelList } from "../lib/settings.js";

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

function stripInternal(body) {
  const { _retried, ...rest } = body;
  return rest;
}

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
 * @param {object} p.ui  callbacks: onText, onReasoning, onToolStart, onToolEnd, onStep
 * @param {AbortSignal} p.signal
 * @returns {Promise<{messages:Array, content:string, cost:number, model:string, steps:number, truncated:boolean}>}  messages = nieuwe assistant/tool-berichten (zonder het user-bericht)
 */
export async function runAgent({ settings, history, userContent, ctx, ui = {}, signal }) {
  const system = { role: "system", content: buildSystemPrompt(settings, { tabTitle: ctx.tabTitle, tabUrl: ctx.tabUrl }) };
  const newMessages = [{ role: "user", content: userContent }];
  const tools = buildTools(settings);
  const fallbacks = parseModelList(settings.fallbackModels).filter((m) => m !== settings.model);
  let cost = 0, usedModel = "", steps = 0, finalContent = "", truncated = false;
  const maxSteps = Math.max(1, Math.min(25, Number(settings.maxSteps) || 10));

  while (steps < maxSteps) {
    if (signal?.aborted) throw new DOMException("Gestopt", "AbortError");
    steps++;
    ui.onStep?.(steps, maxSteps);
    const messages = [system, ...toApiMessages([...compactHistory(history), ...newMessages])];
    const body = {
      model: settings.model,
      messages,
      tools,
      tool_choice: "auto",
      temperature: Number.isFinite(Number(settings.temperature)) ? Number(settings.temperature) : 0.4,
      usage: { include: true },
    };
    if (fallbacks.length && /openrouter\.ai/.test(settings.baseUrl)) body.models = [settings.model, ...fallbacks];
    if (settings.reasoning && settings.reasoning !== "auto") body.reasoning = { effort: settings.reasoning };
    if (steps === maxSteps) {
      // Laatste stap: dwing een eindantwoord af.
      body.tool_choice = "none";
      messages.push({ role: "system", content: "Tool budget exhausted. Answer the user now with what you have, and say what you could not verify." });
    }

    let result;
    try {
      result = await streamChat({ baseUrl: settings.baseUrl, apiKey: settings.apiKey, body, signal }, {
        onText: (_d, full) => ui.onText?.(full),
        onReasoning: (_d, full) => ui.onReasoning?.(full),
        onToolCallStart: (name) => ui.onToolPending?.(name),
      });
    } catch (e) {
      // Sommige providers weigeren `reasoning`/`models`/`usage`/`temperature`; probeer één keer met een minimale body.
      if (e instanceof ApiError && e.status === 400 && !body._retried) {
        for (const k of ["reasoning", "models", "usage", "temperature"]) delete body[k];
        body._retried = true;
        result = await streamChat({ baseUrl: settings.baseUrl, apiKey: settings.apiKey, body: stripInternal(body), signal }, {
          onText: (_d, full) => ui.onText?.(full),
          onReasoning: (_d, full) => ui.onReasoning?.(full),
          onToolCallStart: (name) => ui.onToolPending?.(name),
        });
      } else throw e;
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
  return { messages: newMessages.slice(1), content: finalContent, cost, model: usedModel, steps, truncated };
}
