# AI Web Agent 🧠🖱️

Een browserextensie (Chrome / Edge / Brave) die als **AI-agent naast elke website** meekijkt en meehelpt:

- 📄 **Leest de pagina** die je open hebt (tekst, tabellen, formules, invoervelden) en beantwoordt vragen erover.
- 🖼️ **Bekijkt afbeeldingen** en maakt screenshots: foto's, grafieken, kaarten, meetkundige figuren, scheikundige structuren – hij herkent vormen, leest tekst uit plaatjes en legt uit wat hij ziet.
- 🌐 **Zoekt op internet** (gratis via DuckDuckGo, of via OpenRouter) en leest bronnen om antwoorden te controleren.
- 🖱️ **Kan de pagina bedienen**: klikken, typen, kiezen, scrollen, navigeren, van tabblad wisselen (uit te zetten in de instellingen; bij "gevaarlijke" acties vraagt hij eerst).
- 🎓 **Schoolvakken**: Nederlands, Engels, Frans, Duits, wiskunde (met nette formules), natuurkunde, scheikunde, biologie, aardrijkskunde, geschiedenis, economie… en algemene kennis.
- 🗣️ Werkt in het **Nederlands, Engels, Frans en Duits** (antwoordt in de taal waarin je schrijft, of in een vaste taal naar keuze).
- 💸 **Gratis te gebruiken** met de gratis modellen van [OpenRouter](https://openrouter.ai) (50 vragen per dag zonder ook maar iets te betalen).

Het "brein" is een taalmodel via OpenRouter (of, als je wilt, een lokaal model via Ollama). De agent zelf – ogen, handen, zoeken, geheugen – zit in deze extensie.

---

## Installatie (5 minuten)

1. **Download de code**: klik op GitHub op *Code → Download ZIP* en pak het uit (of `git clone` deze repo).
2. Open in Chrome/Edge/Brave: `chrome://extensions` (Edge: `edge://extensions`).
3. Zet rechtsboven **Ontwikkelaarsmodus** aan.
4. Klik **Uitgepakte extensie laden** en kies de map **`extension`** uit de download.
5. De instellingenpagina opent vanzelf. Pin het icoon in je werkbalk (puzzelstukje → speld).

### OpenRouter-sleutel (gratis)

1. Maak een account op [openrouter.ai](https://openrouter.ai) (e-mail of Google).
2. Ga naar [Keys](https://openrouter.ai/settings/keys) → *Create key* → kopieer de sleutel (`sk-or-v1-…`).
3. Plak hem in het zijpaneel of op de instellingenpagina van de extensie. Klaar.

> Zonder tegoed: **50 vragen per dag** (max 20 per minuut) met een gratis model.
> Als er ooit éénmalig **$10 tegoed** wordt gekocht (bankpas/creditcard, Apple Pay of crypto – geen iDEAL of paysafecard, en prepaid kaarten zijn in NL niet meer in winkels te koop): **1000 gratis vragen per dag** én $10 voor de sterke betaalde modellen (Gemini 3.5 Flash, GPT-5.2, Claude…). Een gemiddelde vraag met paginatekst kost bij Gemini 3.5 Flash minder dan een cent.

Let op: elke *stap* van de agent (een tool-aanroep zoals "pagina lezen" of "zoeken") telt als één verzoek. Het paneel toont hoeveel gratis vragen je vandaag nog hebt.

---

## Gebruik

- Klik op het icoon (of **Ctrl+Shift+Y**) → het zijpaneel opent naast de website. Het paneel volgt automatisch het actieve tabblad.
- Typ je vraag. De tekst van de pagina gaat standaard mee (uit te zetten met het vinkje "Pagina").
- Voorbeelden:
  - *"Leg opgave 4 stap voor stap uit."*
  - *"Wat staat er in de grafiek? Wat is de helling?"*
  - *"Vertaal de geselecteerde tekst naar het Nederlands en leg de grammatica uit."*
  - *"Overhoor mij over deze pagina."*
  - *"Zoek op wanneer dit gebeurd is en geef bronnen."*
  - *"Vul bij het formulier mijn naam in en kies 'havo 4'."* (acties aan in instellingen)
- Rechtermuisknop op geselecteerde tekst → **Vraag AI Web Agent**; rechtermuisknop op een afbeelding → **Afbeelding bekijken met AI Web Agent**.
- Plak (Ctrl+V) of sleep een foto van je huiswerk in het paneel, of klik 📷 voor een screenshot van de pagina.
- ➕ start een nieuw gesprek, 🕘 toont eerdere gesprekken (alleen lokaal opgeslagen).

## Instellingen

| Instelling | Uitleg |
| --- | --- |
| Model | Standaard `qwen/qwen3.8-27b:free` (gratis, kan plaatjes zien en tools gebruiken). Met tegoed is `google/gemini-3.5-flash` aanbevolen. "Alle geschikte modellen laden" haalt de actuele lijst op. |
| Reserve-modellen | Worden automatisch geprobeerd als het eerste model vol of offline is. |
| Zoeken op internet | **DuckDuckGo** (gratis, in je browser) · **OpenRouter web search** (beter, ±$0,007 per zoekopdracht) · Uit |
| Antwoordtaal | Automatisch (taal van je vraag) of vast NL/EN/FR/DE |
| Denk-niveau | Laag = snel; Hoog voor pittige wiskunde/natuurkunde |
| Acties toestaan | Klikken/typen/navigeren aan of uit |
| Paginatekst / max. stappen | Minder = sneller en goedkoper |
| Andere server | Bijv. Ollama op je eigen pc: `http://localhost:11434/v1` (zie instellingenpagina) |

## Privacy & veiligheid

- Je API-sleutel en gesprekken staan alleen lokaal in je browser (`chrome.storage`).
- Paginatekst, geselecteerde tekst en (alleen als de agent dat nodig heeft) afbeeldingen/screenshots worden naar het gekozen model gestuurd via OpenRouter. Stuur dus geen pagina's met wachtwoorden of bankgegevens mee.
- De agent typt nooit wachtwoorden of betaalgegevens en vraagt om bevestiging voor onomkeerbare acties (versturen van toetsen/formulieren, kopen, verwijderen).
- Browser-eigen pagina's (`chrome://…`, de webwinkel) zijn voor extensies afgeschermd; daar werkt het paneel niet.

## Eigen AI op je eigen pc (optioneel)

Een taalmodel zelf trainen is niet haalbaar (dat kost miljoenen aan rekenkracht), maar je kunt wél een **open model lokaal draaien** met [Ollama](https://ollama.com):

```bash
ollama pull gemma3:12b            # of qwen2.5vl:7b (beide kunnen afbeeldingen zien)
OLLAMA_ORIGINS="chrome-extension://*" ollama serve
```

Zet in de instellingen de API-basis-URL op `http://localhost:11434/v1`, vul als eigen model-ID `gemma3:12b` in en zet een willekeurige API-sleutel (bijv. `ollama`). Dat is 100% gratis en offline, maar langzamer en minder slim dan de online modellen; internet zoeken werkt dan via DuckDuckGo.

---

## Voor ontwikkelaars

```
extension/
  manifest.json          Manifest V3 (zijpaneel, contextmenu's, scripting, <all_urls>)
  background.js          Service worker: paneel openen, contextmenu's
  content/content.js     Content-script: pagina lezen, afbeeldingen, elementen, klikken/typen/scrollen
  sidepanel/             Chat-UI, agent-lus (agent.js), tools (tools.js), tab-brug (page.js), markdown+KaTeX (render.js)
  options/               Instellingenpagina
  lib/                   OpenRouter-client (streaming + tool calls), instellingen, prompts, zoeken (DuckDuckGo/Bing, URL → tekst)
  vendor/                marked, DOMPurify, KaTeX (gekopieerd uit node_modules met `npm run vendor`)
scripts/                 copy-vendor.mjs, check.mjs, make-icons.py
test/e2e.mjs             End-to-end test zonder browser: jsdom speelt de pagina + het content-script, een nep-OpenRouter speelt het model
```

- De agent draait in het zijpaneel (extensiepagina): daar zijn `fetch`-streaming, `DOMParser`, canvas en alle `chrome.*`-API's beschikbaar.
- Tools worden als OpenAI-style *function calls* aan het model gegeven; `openrouter:web_search` (server tool) wordt toegevoegd als je die zoekoptie kiest.
- Afbeeldingen die een tool oplevert (screenshot, `view_image`) gaan als `image_url` terug naar het model.
- Oude beurten worden compact gemaakt (geen oude paginatekst/afbeeldingen) om tokens en gratis verzoeken te sparen.

```bash
npm install          # dev-dependencies (eslint, jsdom, libs voor vendor)
npm run vendor       # vendor-bestanden verversen na een update van marked/dompurify/katex
npm run check        # manifest + verwijzingen + eslint
npm test             # e2e-tests (jsdom + nep-OpenRouter)
python3 scripts/make-icons.py   # iconen opnieuw genereren (Pillow)
```

## Foutmeldingen: wat betekent wat?

| Melding | Oorzaak | Wat doe je |
|---|---|---|
| **… overbelast bij de aanbieder (429)** | Het gratis model wordt door te veel mensen tegelijk gebruikt (OpenRouter stuurt dan “Provider returned error”). Dit gaat **niet** van jouw 50 vragen per dag af. | De agent wacht een paar seconden en probeert automatisch de reserve-modellen. Blijft het misgaan: klik *Probeer met …*, kies een ander gratis model in het menu, of probeer het over een minuut opnieuw. |
| **Je gratis daglimiet is op** | 50 gratis verzoeken per dag verbruikt (elke stap van de agent = 1 verzoek). | Wachten tot 00:00 UTC (01:00/02:00 NL-tijd), of tijdelijk een lokaal Ollama-model gebruiken. |
| **Te snel achter elkaar (20 per minuut)** | Minuutlimiet van gratis modellen. | Halve minuut wachten. |
| **Dit is een browserpagina; ik kan die niet lezen** | Je zat op een `chrome://`-pagina, de Chrome Web Store of een nieuw leeg tabblad; daar mag geen enkele extensie bij. | Open een gewone website en stel je vraag opnieuw. |
| **API-sleutel ongeldig (401)** | Sleutel verkeerd geplakt of ingetrokken. | Nieuwe sleutel maken op openrouter.ai/keys en opnieuw invullen. |

Onder elke foutmelding zit *Technische details* met de ruwe melding van OpenRouter – handig om mee te sturen als je hulp vraagt.

## Bekende beperkingen / ideeën

- PDF's in de browser kunnen (nog) niet als tekst gelezen worden – maak een screenshot (📷) van het stuk dat je bedoelt.
- Firefox wordt nog niet ondersteund (zijpaneel-API verschilt).
- DuckDuckGo kan bij heel veel zoekopdrachten om een captcha vragen; dan valt de agent terug op Bing.
- Gratis modellen zijn soms druk of traag (vooral ’s avonds); de agent schakelt dan automatisch over op de reserve-modellen, maar soms zijn die óók druk – dan even later opnieuw proberen.
