// Achtergrond-serviceworker (Manifest V3).
// Doet bewust weinig: het zijpaneel openen, contextmenu's en kleine hulpjes.
// De eigenlijke agent draait in het zijpaneel (sidepanel/), zodat streaming en DOM-API's
// (DOMParser, canvas) gewoon beschikbaar zijn.

const MENU = {
  ASK_SELECTION: "ai-web-agent-ask-selection",
  IMAGE: "ai-web-agent-image",
  PAGE: "ai-web-agent-page",
};

function setupMenus() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU.ASK_SELECTION,
      title: "Vraag AI Web Agent over “%s”",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: MENU.IMAGE,
      title: "Afbeelding bekijken met AI Web Agent",
      contexts: ["image"],
    });
    chrome.contextMenus.create({
      id: MENU.PAGE,
      title: "AI Web Agent openen",
      contexts: ["page", "frame", "link"],
    });
  });
}

chrome.runtime.onInstalled.addListener(async (details) => {
  try {
    await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  } catch (e) {
    console.warn("sidePanel.setPanelBehavior mislukt:", e);
  }
  setupMenus();
  if (details.reason === "install") {
    // Eerste keer: meteen de instellingen openen zodat de API-sleutel ingevuld kan worden.
    chrome.runtime.openOptionsPage();
  }
});

chrome.runtime.onStartup?.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  // Het paneel openen mag alleen direct na een gebruikersactie; doe dat dus als eerste.
  try {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  } catch (e) {
    console.warn("Kon zijpaneel niet openen:", e);
  }

  let pending = null;
  if (info.menuItemId === MENU.ASK_SELECTION && info.selectionText) {
    pending = { kind: "selection", text: info.selectionText, tabId: tab.id, url: tab.url };
  } else if (info.menuItemId === MENU.IMAGE && info.srcUrl) {
    pending = { kind: "image", src: info.srcUrl, tabId: tab.id, url: tab.url };
  }
  if (pending) {
    pending.ts = Date.now();
    await chrome.storage.session.set({ pendingRequest: pending });
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "openOptions") {
    chrome.runtime.openOptionsPage();
    sendResponse({ ok: true });
  }
  return false;
});
