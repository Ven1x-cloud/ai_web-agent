# Winkelteksten (kopieer-plak voor Chrome Web Store en Microsoft Edge Add-ons)

Alle velden die de winkels vragen staan hier klaar. De Engelse teksten heb je nodig voor de
formulieren over permissies/privacy (die zijn in het Engels); de Nederlandse voor de winkelpagina.

## Basis

| Veld | Waarde |
|---|---|
| Naam | `AI Web Agent` |
| Categorie | Education (alternatief: Productivity) |
| Taal | Nederlands (primaire taal), Engels als tweede |
| Website / homepage | `https://ven1x-cloud.github.io/ai_web-agent/` |
| Support-URL | `https://github.com/Ven1x-cloud/ai_web-agent/issues` |
| Privacy policy URL | `https://ven1x-cloud.github.io/ai_web-agent/privacy.html` |
| Prijs | Gratis |
| Leeftijd | Alle leeftijden (geen volwassen inhoud) |

## Korte beschrijving (max. 132 tekens)

**NL** (130 tekens):

```
AI-hulp naast elke website: leest de pagina, bekijkt afbeeldingen, zoekt op internet en helpt met huiswerk. Gratis via OpenRouter.
```

**EN** (126 tekens):

```
AI helper next to any website: reads the page, looks at images, searches the web and helps with homework. Free via OpenRouter.
```

## Uitgebreide beschrijving

**NL**

```
AI Web Agent zet een slimme assistent in het zijpaneel van je browser. Hij leest de website waar je op zit, bekijkt afbeeldingen en screenshots, zoekt op internet als dat nodig is en legt uit – in het Nederlands, Engels, Frans of Duits.

WAT KAN HIJ?
• Vragen beantwoorden over de pagina die open staat: teksten, opgaven, tabellen, formules (ook in ingebedde kaders).
• Afbeeldingen bekijken: figuren, grafieken, kaarten, foto's van sommen die je plakt of uploadt.
• Op internet zoeken en bronnen noemen.
• Huiswerkhulp voor wiskunde (nette formules), scheikunde, natuurkunde, aardrijkskunde, geschiedenis, talen en meer – met uitleg en stappen, niet alleen het antwoord.
• Op jouw verzoek klikken, typen en scrollen op de pagina, bijvoorbeeld om antwoorden in te vullen. Hij vraagt eerst voordat hij iets verstuurt.

GRATIS
De extensie is gratis en open source. Als "brein" gebruikt hij OpenRouter: maak daar een gratis account, maak een sleutel aan en plak die in het zijpaneel. Met een gratis model heb je 50 vragen per dag zonder creditcard of tegoed. Liever helemaal lokaal? Dat kan met Ollama op je eigen computer.

PRIVACY
Geen account bij ons, geen tracking, geen eigen servers. De tekst van een pagina wordt alleen naar het AI-model gestuurd als jij een vraag stelt. Gesprekken en instellingen blijven in je eigen browser. Volledige privacyverklaring: https://ven1x-cloud.github.io/ai_web-agent/privacy.html

TIPS
• Selecteer tekst en klik met de rechtermuisknop op "Vraag AI Web Agent".
• Rechtermuisknop op een afbeelding → "Afbeelding bekijken met AI Web Agent".
• Sneltoets: Ctrl+Shift+Y opent het zijpaneel.

Werkt op vrijwel alle gewone websites (niet op browserpagina's zoals chrome://). Bedoeld om te leren en te controleren – niet voor toetsen. Controleer antwoorden altijd zelf.
```

**EN**

```
AI Web Agent puts a smart assistant in your browser's side panel. It reads the website you are on, looks at images and screenshots, searches the web when needed and explains things – in Dutch, English, French or German.

WHAT IT DOES
• Answers questions about the open page: texts, exercises, tables, formulas (including embedded frames).
• Looks at images: figures, charts, maps, photos of exercises you paste or upload.
• Searches the web and cites sources.
• Homework help for maths (proper formulas), chemistry, physics, geography, history, languages and more – with explanations and steps, not just the answer.
• On your request it clicks, types and scrolls on the page, e.g. to fill in answers. It asks before submitting anything.

FREE
The extension is free and open source. It uses OpenRouter as its "brain": create a free account, create a key and paste it into the side panel. With a free model you get 50 questions a day without a credit card. Prefer fully local? Use Ollama on your own computer.

PRIVACY
No account with us, no tracking, no servers of our own. Page text is only sent to the AI model when you ask a question. Conversations and settings stay in your browser. Full privacy policy: https://ven1x-cloud.github.io/ai_web-agent/privacy.html

Works on virtually all regular websites (not on browser pages such as chrome://). Meant for learning and checking – not for exams. Always verify answers yourself.
```

## Single purpose (Chrome Web Store, "Privacy practices")

```
An AI assistant in the browser side panel that, on the user's request, reads and acts on the currently open web page (text, images, form fields) and answers questions about it.
```

## Permissiemotivatie (Engels, kopieer per permissie)

| Permissie | Justification |
|---|---|
| `sidePanel` | The entire user interface (chat with the assistant) lives in the browser side panel. |
| `storage` / `unlimitedStorage` | Stores the user's settings (API key, chosen model, preferences) and conversation history locally. Conversations can contain screenshots, which is why unlimited storage is requested. Nothing is synced to a server. |
| `tabs` | Read the title and URL of the active tab, list/switch tabs when the user asks (“open the tab with the article”), and detect when a page has finished loading after a navigation the user requested. |
| `activeTab` | Access the page the user is currently looking at when they ask a question about it, and capture a screenshot of the visible tab when the user asks the assistant to look at the page. |
| `scripting` | Inject the content script on demand (only when the user asks a question) to read page text/images and to perform the actions the user requested (click, type, scroll). No script runs in the background. |
| `contextMenus` | Right-click menu items: “Ask AI Web Agent” on selected text and “Look at this image” on images. |
| Host permission `<all_urls>` | Users decide themselves on which website they use the assistant (school platforms, Wikipedia, news sites…), so it must work on any site. Host access is also needed to fetch images from the page for the vision model, to run the client-side web search (DuckDuckGo/Bing) and to read a linked page when the assistant needs its content. The extension only accesses a page when the user asks a question. |
| Remote code | **No.** All code ships inside the package (including the bundled libraries marked, DOMPurify and KaTeX). The AI model's responses are data, not executable code. |

## Gegevensgebruik (Chrome Web Store "Data usage" / Edge "Privacy")

Aanvinken:

- **Website content** – page text, images and screenshots are sent to the AI provider the user configured (OpenRouter by default) only when the user asks a question.
- **Authentication information** – the user's own OpenRouter API key, stored locally and sent only to that provider.
- **User activity: no. Personal communications: no. Location: no. Financial/health/PII: no.**

Verklaringen (alle drie aanvinken):
- I do not sell or transfer user data to third parties, outside of the approved use cases.
- I do not use or transfer user data for purposes that are unrelated to my item's single purpose.
- I do not use or transfer user data to determine creditworthiness or for lending purposes.

## Afbeeldingen (in deze map)

| Bestand | Waarvoor |
|---|---|
| `logo-300x300.png` | Edge Add-ons: winkellogo (300×300) |
| `promo-440x280.png` | Chrome Web Store: small promo tile (440×280) |
| `marquee-1400x560.png` | Chrome Web Store: marquee (1400×560, optioneel) |
| `../extension/icons/icon128.png` | Chrome Web Store: store icon (128×128) |

**Screenshots moet je zelf maken** (minstens 1, liefst 3–5): 1280×800 of 640×400 (Chrome); 1280×800 of 640×480 (Edge).
Open een oefenwebsite, open het zijpaneel met een mooi antwoord in beeld en maak een schermafbeelding
(Windows: Win+Shift+S; Mac: Cmd+Shift+4). Snijd bij op 1280×800 (bijv. met Paint of Foto's).
Zorg dat er geen naam, e-mailadres of API-sleutel in beeld staat.
