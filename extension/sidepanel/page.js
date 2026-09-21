// Brug tussen het zijpaneel en de actieve tab: content-script injecteren, berichten sturen,
// screenshots maken, afbeeldingen ophalen, tabs beheren.

export class PageError extends Error {}

const RESTRICTED = /^(chrome|edge|brave|opera|vivaldi|about|devtools|view-source|chrome-extension|moz-extension):|^https:\/\/(chrome\.google\.com\/webstore|chromewebstore\.google\.com|microsoftedge\.microsoft\.com\/addons)/i;

export function isRestrictedUrl(url) {
  return !url || RESTRICTED.test(url);
}

export async function getActiveTab() {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab || null;
}

export async function getTab(tabId) {
  try { return await chrome.tabs.get(tabId); } catch (_) { return null; }
}

async function ping(tabId, frameId = 0) {
  try {
    const r = await chrome.tabs.sendMessage(tabId, { __aiWebAgent: true, action: "ping" }, { frameId });
    return !!(r && r.ok);
  } catch (_) {
    return false;
  }
}

/** Zorgt dat het content-script in de tab aanwezig is. */
export async function ensureInjected(tabId) {
  const tab = await getTab(tabId);
  if (!tab) throw new PageError("Tabblad niet gevonden.");
  if (isRestrictedUrl(tab.url)) {
    throw new PageError("Op deze pagina (browser-eigen pagina of webwinkel) mag een extensie niets lezen. Open een gewone website.");
  }
  if (await ping(tabId)) return tab;
  try {
    await chrome.scripting.executeScript({ target: { tabId, allFrames: true }, files: ["content/content.js"] });
  } catch (e) {
    throw new PageError("Kon het hulpscript niet in de pagina laden: " + (e.message || e));
  }
  // Korte wacht zodat het script zijn listener registreert.
  for (let i = 0; i < 10; i++) {
    if (await ping(tabId)) return tab;
    await sleep(100);
  }
  throw new PageError("Het hulpscript reageert niet op deze pagina. Herlaad de pagina en probeer opnieuw.");
}

/**
 * Stuurt een actie naar het content-script en geeft het resultaat terug (of gooit een fout).
 * Zonder frameId gaat de actie naar het hoofdkader (frame 0). Zonder expliciet frame zou Chrome het bericht
 * naar álle kaders sturen en het eerste antwoord teruggeven – dan "wint" soms een leeg about:blank-iframe.
 */
export async function pageAction(tabId, action, args = {}, frameId = null) {
  await ensureInjected(tabId);
  let res;
  try {
    res = await chrome.tabs.sendMessage(tabId, { __aiWebAgent: true, action, args }, { frameId: frameId != null ? Number(frameId) : 0 });
  } catch (e) {
    throw new PageError("Geen verbinding met de pagina (" + (e.message || e) + "). Is de pagina nog aan het laden?");
  }
  if (!res) throw new PageError("Geen antwoord van de pagina.");
  if (!res.ok) throw new PageError(res.error || "Onbekende fout in de pagina.");
  return res.result;
}

/** Alle kaders van de tab (hoofdpagina + iframes) met hoeveel tekst erin staat. Lege kaders worden weggelaten. */
export async function listFrames(tabId) {
  const results = await chrome.scripting.executeScript({
    target: { tabId, allFrames: true },
    func: () => ({
      url: location.href, title: document.title, top: window === window.top,
      textLength: ((document.body && document.body.innerText) || "").trim().length,
      inputs: document.querySelectorAll("input:not([type=hidden]), textarea, select, [contenteditable=true]").length,
    }),
  });
  return results
    .map((r) => ({ frameId: r.frameId, ...(r.result || {}) }))
    .filter((f) => f.top || f.textLength > 0 || f.inputs > 0)
    .sort((a, b) => (a.top ? -1 : b.top ? 1 : b.textLength - a.textLength));
}

/**
 * page_info van het hoofdkader, aangevuld met de inhoud van ingebedde kaders (iframes) die tekst of invoervelden
 * bevatten. Oefenplatforms zetten de opgave vaak in zo'n kader; zonder dit zag de agent alleen de kop van de pagina.
 */
export async function pageInfoWithFrames(tabId, { maxChars = 6000, offset = 0 } = {}) {
  const info = await pageAction(tabId, "page_info", { offset, maxChars }, 0);
  if (offset > 0) return info;
  let frames = [];
  try { frames = await listFrames(tabId); } catch (_) { return info; }
  const children = frames.filter((f) => !f.top && (f.textLength > 30 || f.inputs > 0)).slice(0, 5);
  if (!children.length) return info;
  let budget = Math.max(1500, maxChars - (info.text || "").length);
  const sections = [];
  for (const f of children) {
    if (budget < 200) break;
    try {
      const fi = await pageAction(tabId, "page_info", { maxChars: Math.min(budget, maxChars) }, f.frameId);
      if (!fi.text || fi.text.trim().length < 20) continue;
      const src = fi.url && !/^about:/.test(fi.url) ? `, ${fi.url}` : "";
      sections.push(`\n\n=== Embedded frame (frame_id ${f.frameId}${src}) — for elements inside it, call get_interactive/click/type_text with frame_id ${f.frameId} ===\n${fi.text}`);
      budget -= fi.text.length;
    } catch (_) { /* kader niet bereikbaar (bijv. nog aan het laden) */ }
  }
  if (sections.length) {
    info.text = (info.text || "") + sections.join("");
    info.frameTexts = sections.length;
    info.totalChars = (info.totalChars || 0) + sections.reduce((n, t) => n + t.length, 0);
  }
  return info;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Wacht tot de tab klaar is met laden (met time-out). */
export function waitForLoad(tabId, timeout = 20000) {
  return new Promise((resolve) => {
    let done = false;
    const finish = async () => {
      if (done) return;
      done = true;
      chrome.tabs.onUpdated.removeListener(listener);
      clearTimeout(timer);
      await sleep(400); // scripts op de pagina even tijd geven
      resolve(await getTab(tabId));
    };
    const listener = (id, info) => { if (id === tabId && info.status === "complete") finish(); };
    chrome.tabs.onUpdated.addListener(listener);
    const timer = setTimeout(finish, timeout);
    sleep(500).then(async () => {
      const t = await getTab(tabId);
      if (t && t.status === "complete") finish();
    });
  });
}

export async function navigate(tabId, url) {
  if (!/^https?:\/\//i.test(url)) url = "https://" + url.replace(/^\/+/, "");
  await chrome.tabs.update(tabId, { url });
  const tab = await waitForLoad(tabId);
  return { url: tab?.url || url, title: tab?.title || "" };
}

export async function goBack(tabId) {
  try { await chrome.tabs.goBack(tabId); } catch (e) { throw new PageError("Kan niet terug: " + (e.message || e)); }
  const tab = await waitForLoad(tabId, 10000);
  return { url: tab?.url, title: tab?.title };
}

export async function listTabs() {
  const tabs = await chrome.tabs.query({ currentWindow: true });
  return tabs.map((t) => ({ id: t.id, title: t.title, url: t.url, active: t.active, restricted: isRestrictedUrl(t.url) }));
}

export async function switchTab(tabId) {
  const tab = await chrome.tabs.update(tabId, { active: true });
  await sleep(300);
  return { id: tab.id, title: tab.title, url: tab.url };
}

// ---------- Afbeeldingen ----------

/** Maakt een afbeelding (dataURL/Blob/URL) kleiner en levert een JPEG-dataURL op. */
export async function normalizeImage(input, { maxDim = 1400, quality = 0.85, crop = null, mime = "image/jpeg" } = {}) {
  let blob;
  if (input instanceof Blob) blob = input;
  else if (typeof input === "string" && input.startsWith("data:")) blob = await (await fetch(input)).blob();
  else {
    const res = await fetch(input, { credentials: "include" });
    if (!res.ok) throw new PageError(`Afbeelding kon niet worden opgehaald (status ${res.status}).`);
    blob = await res.blob();
  }
  if (blob.type === "image/svg+xml" || blob.type === "image/gif" && blob.size > 4_000_000) {
    // createImageBitmap kan SVG zonder afmeting niet aan; via <img> tekenen.
    return await viaImageElement(blob, { maxDim, quality, crop, mime });
  }
  let bmp;
  try {
    bmp = await createImageBitmap(blob);
  } catch (_) {
    return await viaImageElement(blob, { maxDim, quality, crop, mime });
  }
  return drawToDataUrl(bmp, bmp.width, bmp.height, { maxDim, quality, crop, mime });
}

function viaImageElement(blob, opts) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      try { resolve(drawToDataUrl(img, img.naturalWidth || img.width || 800, img.naturalHeight || img.height || 600, opts)); }
      catch (e) { reject(e); } finally { URL.revokeObjectURL(url); }
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new PageError("Afbeelding kon niet worden gedecodeerd.")); };
    img.src = url;
  });
}

function drawToDataUrl(source, srcW, srcH, { maxDim, quality, crop, mime }) {
  let sx = 0, sy = 0, sw = srcW, sh = srcH;
  if (crop) {
    sx = Math.max(0, Math.round(crop.x)); sy = Math.max(0, Math.round(crop.y));
    sw = Math.max(1, Math.min(srcW - sx, Math.round(crop.width)));
    sh = Math.max(1, Math.min(srcH - sy, Math.round(crop.height)));
  }
  const scale = Math.min(1, maxDim / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * scale)), h = Math.max(1, Math.round(sh * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, w, h);
  return { dataUrl: canvas.toDataURL(mime, quality), width: w, height: h };
}

/** Screenshot van het zichtbare deel van de actieve tab (optioneel bijgesneden op een element). */
export async function captureVisible(windowId, { maxDim = 1400, quality = 0.85, crop = null } = {}) {
  let dataUrl;
  try {
    dataUrl = windowId != null
      ? await chrome.tabs.captureVisibleTab(windowId, { format: "png" })
      : await chrome.tabs.captureVisibleTab({ format: "png" });
  } catch (e) {
    throw new PageError("Screenshot mislukt: " + (e.message || e) + ". Zorg dat het tabblad zichtbaar is.");
  }
  let cropPx = null;
  if (crop) {
    const dpr = crop.dpr || 1;
    const pad = 8 * dpr;
    cropPx = { x: crop.x * dpr - pad, y: crop.y * dpr - pad, width: crop.width * dpr + pad * 2, height: crop.height * dpr + pad * 2 };
  }
  return normalizeImage(dataUrl, { maxDim, quality, crop: cropPx });
}

/** Leest een File (upload/plakken) in als geschaalde dataURL. */
export async function fileToImage(file, opts = {}) {
  const out = await normalizeImage(file, { maxDim: 1600, quality: 0.88, ...opts });
  return { ...out, name: file.name || "afbeelding" };
}
