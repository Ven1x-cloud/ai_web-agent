# AI Web Agent 🧠🖱️

Een browserextensie (Chrome / Edge / Brave) die als **AI-agent naast elke website** meekijkt en meehelpt:

- 📄 **Leest de pagina** die je open hebt (tekst, tabellen, formules, invoervelden) en beantwoordt vragen erover.
- 🖼️ **Bekijkt afbeeldingen** en maakt screenshots: foto's, grafieken, kaarten, meetkundige figuren, scheikundige structuren – hij herkent vormen, leest tekst uit plaatjes en legt uit wat hij ziet.
- 🌐 **Zoekt op internet** (gratis via DuckDuckGo, of via OpenRouter) en leest bronnen om antwoorden te controleren.
- 🖱️ **Kan de pagina bedienen**: klikken, typen, kiezen, scrollen, navigeren, van tabblad wisselen (uit te zetten in de instellingen; bij "gevaarlijke" acties vraagt hij eerst).
- 🎓 **Schoolvakken**: Nederlands, Engels, Frans, Duits, wiskunde (met nette formules), natuurkunde, scheikunde, biologie, aardrijkskunde, geschiedenis, economie… en algemene kennis.
- 🗣️ Werkt in het **Nederlands, Engels, Frans en Duits** (antwoordt in de taal waarin je schrijft, of in een vaste taal naar keuze).
- 💸 **Gratis te gebruiken** met de gratis modellen van [OpenRouter](https://openrouter.ai) (50 vragen per dag zonder ook maar iets te betalen).
- 🧠 **Je eigen AI die je kunt trainen**: een zelf getraind taalmodel in de map [`ai/`](ai/), geschreven in Python met alleen NumPy. Geen account, geen sleutel, geen daglimiet; je vraag verlaat je computer niet. Onder elk antwoord geef je 👍 👎 of ✏️, en die feedback wordt trainingsdata voor de volgende ronde.

Het "brein" is uitwisselbaar. Standaard is dat **je eigen, zelf getrainde AI** die in de map [`ai/`](ai/) staat en volledig op je eigen pc draait. Liever meteen een slim model? Met één druk op de knop schakel je over op OpenRouter (gratis, 50 vragen per dag) of op een lokaal model via Ollama. De agent zelf – ogen, handen, zoeken, geheugen – zit in deze extensie en werkt met alle drie precies hetzelfde.

> **Eerlijk over je eigen AI:** het model dat hier meegeleverd wordt is écht en zelf getraind, maar klein (0,95 miljoen parameters, 145 kB voorbeelddata met 510 vraag-antwoordparen). Cloudmodellen zijn miljoenen keren groter. Daarmee geeft het nu **nog geen lopende antwoorden** – dat is geen storing maar gebrek aan data, en het is de reden waarom je het kunt blijven trainen. Het is privé, offline en zonder daglimiet, en het wordt beter naarmate jij het gebruikt. Voor huiswerk is het cloudmodel voorlopig verstandiger. Zie [ai/README.md](ai/README.md) voor de volledige uitleg en hoe je het laat groeien.

---

🌐 **Website met uitleg en download:** <https://ven1x-cloud.github.io/ai_web-agent/>

## Installatie (5 minuten)

1. **Download** [`ai-web-agent.zip`](https://github.com/Ven1x-cloud/ai_web-agent/releases/latest/download/ai-web-agent.zip) (nieuwste release) en pak het uit. *(Ontwikkelaars: `git clone` en gebruik de map `extension/`.)*
2. Open in Chrome/Edge/Brave: `chrome://extensions` (Edge: `edge://extensions`).
3. Zet rechtsboven **Ontwikkelaarsmodus** aan.
4. Klik **Uitgepakte extensie laden** en kies de uitgepakte map (waar `manifest.json` in staat).
5. De instellingenpagina opent vanzelf. Pin het icoon in je werkbalk (puzzelstukje → speld).

### OpenRouter-sleutel (gratis)

1. Maak een account op [openrouter.ai](https://openrouter.ai) (e-mail of Google).
2. Ga naar [Keys](https://openrouter.ai/settings/keys) → *Create key* → kopieer de sleutel (`sk-or-v1-…`).
3. Plak hem in het zijpaneel of op de instellingenpagina van de extensie. Klaar.

> **Jonger dan 18?** OpenRouter eist in zijn voorwaarden dat de accounthouder 18+ is. Laat dan een ouder/verzorger het account en de sleutel aanmaken (gratis, 5 minuten) – zie de [uitleg voor ouders](OUDERS.md). Zonder account werken kan ook, met [Ollama](#eigen-ai-op-je-eigen-pc-optioneel) op je eigen pc.
>
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
| Reserve-aanbieders | Andere diensten (Gemini, Groq, Mistral, Ollama…) met een eigen gratis limiet; nemen het over als de daglimiet van de hoofdaanbieder op is. |
| Zoeken op internet | **DuckDuckGo** (gratis, in je browser) · **OpenRouter web search** (beter, ±$0,007 per zoekopdracht) · Uit |
| Antwoordtaal | Automatisch (taal van je vraag) of vast NL/EN/FR/DE |
| Denk-niveau | Laag = snel; Hoog voor pittige wiskunde/natuurkunde |
| Acties toestaan | Klikken/typen/navigeren aan of uit |
| Paginatekst / max. stappen | Minder = sneller en goedkoper |
| Andere server | Bijv. Ollama op je eigen pc: `http://localhost:11434/v1` (zie instellingenpagina) |

## Meer dan 50 vragen per dag? Reserve-aanbieders

**Wat níet werkt:** extra sleutels van hetzelfde OpenRouter-account (de 50/dag geldt per account, niet per sleutel) of meerdere OpenRouter-accounts – dat laatste is uitdrukkelijk verboden in hun [voorwaarden](https://openrouter.ai/terms) (§7.3) en kan alle accounts kosten.

**Wat wél werkt:** andere diensten met een *eigen* gratis limiet toevoegen als **reserve-aanbieder** (Instellingen → 6). Is de daglimiet van de hoofdaanbieder op, dan schakelt de agent automatisch door naar de volgende – midden in een gesprek, zonder dat de AI iets vergeet (de gespreksgeschiedenis zit in de extensie, niet bij de sleutel). In het paneel zie je `· 2/2 reserve` bij de teller en een melding `🔁 OpenRouter vandaag op → verder met Google AI Studio`.

| Aanbieder | Gratis | Sleutel | Opmerking |
|---|---|---|---|
| Google AI Studio (Gemini) | gratis tier met eigen daglimiet | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Model bijv. `gemini-3.8-flash`; reset rond 09:00 NL-tijd |
| Groq | gratis tier | [console.groq.com/keys](https://console.groq.com/keys) | Kies een model met afbeeldingen + tools uit hun lijst |
| Mistral | gratis "Experiment"-tier | [console.mistral.ai](https://console.mistral.ai/api-keys) | `mistral-small-latest`; telefoonverificatie |
| Ollama (eigen pc) | onbeperkt | geen | Zie hieronder; traag zonder goede videokaart |

Voor al deze accounts geldt (net als bij OpenRouter) dat de accounthouder 18+ moet zijn – zie [OUDERS.md](OUDERS.md). Met de knop **Test** bij elke aanbieder controleer je of sleutel, server en modelnaam kloppen.

## Privacy & veiligheid

- Je API-sleutel en gesprekken staan alleen lokaal in je browser (`chrome.storage`).
- Paginatekst, geselecteerde tekst en (alleen als de agent dat nodig heeft) afbeeldingen/screenshots worden naar het gekozen model gestuurd via OpenRouter. Stuur dus geen pagina's met wachtwoorden of bankgegevens mee.
- De agent typt nooit wachtwoorden of betaalgegevens en vraagt om bevestiging voor onomkeerbare acties (versturen van toetsen/formulieren, kopen, verwijderen).
- Browser-eigen pagina's (`chrome://…`, de webwinkel) zijn voor extensies afgeschermd; daar werkt het paneel niet.

## Je eigen AI trainen 🧠

In de map [`ai/`](ai/) staat een **zelf getraind taalmodel**, helemaal los van de
extensie. De extensie praat er tegenaan via dezelfde API als OpenRouter, dus
wisselen tussen je eigen model en een cloudmodel kost geen code.

Alles is geschreven in **Python met alleen NumPy**: de transformer, de
backpropagation, de tokenizer en de optimalisator zijn met de hand geschreven. Geen
installatie van meerdere gigabytes en niets dat op je videokaart moet passen.

```bash
python3 -m venv .venv                       # één keer
.venv/bin/pip install -r ai/requirements.txt

cd ai
python -m webai build --preset tiny         # tokenizer + dataset bouwen   (~5 s)
python -m webai train --preset tiny         # trainen                     (~2-10 min)
python -m webai serve                       # server op poort 8787
```

Laat dat laatste venster open staan en open dan de extensie: die staat er al op
ingesteld. Zonder extensie proberen kan ook op <http://localhost:8787/>, waar een
klein speelterrein staat.

### Zo train je het verder

Onder elk antwoord in het zijpaneel staan drie knopjes:

| Knop | Wat er gebeurt | Wordt het trainingsdata? |
| --- | --- | --- |
| 👍 | Het antwoord was goed | **Ja** |
| 👎 | Het antwoord was fout | Nee – anders zou het model precies dat foute antwoord nadoen |
| ✏️ | Jij schrijft het goede antwoord | **Ja**, en dit is het waardevolst |

Daarna zet je de rondte voort:

```bash
python -m webai feedback --export    # feedback → trainingsdata
python -m webai train --resume       # verder trainen op het nieuwe materiaal
```

De server merkt een nieuw model zelf op en laadt het in; herstarten hoeft niet.

Wil je sneller groeien, laat dan een sterk cloudmodel voorbeelden schrijven die jouw
eigen model daarna overneemt:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python -m webai distill --limit 200
```

Of zet eigen tekst (`.txt`, `.md`, `.jsonl`) in `ai/data/learned/`.

### Verwacht dit er (nog) niet van

| Trainingsdata | Wat je dan krijgt |
| --- | --- |
| ±12.000 tokens | Kent de voorbeelden uit zijn hoofd; daarbuiten **woordensoep** |
| ±98.000 tokens *(nu)* | Overfit niet meer en pakt patronen, maar nog geen vloeiende zinnen |
| ±1 miljoen tokens | Echte zinnen, redelijke grammatica, geen betrouwbare feiten |
| ±50 miljoen tokens | Leesbaar Nederlands over bekende onderwerpen |

`python -m webai info` laat zien waar je staat, en waarschuwt vanzelf als het model
aan het overfitten is (dan kent het zijn voorbeelden uit zijn hoofd in plaats van
taal te leren). Het getal dat telt is het **eval-verlies**: hoe goed het is op
voorbeelden die het nooit gezien heeft.

Alles staat in **[ai/README.md](ai/README.md)**: installatie, presets, alle
commando's, de HTTP-API, hoe het gebouwd is, en wat er nodig is om het echt goed te
maken.

### Alternatief: een open model via Ollama

Liever een kant-en-klaar lokaal model dat wél slim is? Dat kan met
[Ollama](https://ollama.com):

```bash
ollama pull gemma3:12b            # of qwen2.5vl:7b (beide kunnen afbeeldingen zien)
OLLAMA_ORIGINS="chrome-extension://*" ollama serve
```

Kies in de instellingen **Ollama** (of zet de API-basis-URL op
`http://localhost:11434/v1` met model-ID `gemma3:12b` en een willekeurige sleutel
zoals `ollama`). Ook 100% gratis en offline, en vele malen slimmer dan het eigen
model – maar je traint het niet zelf.

---

## Voor ontwikkelaars

```
ai/                      ← JE EIGEN AI, los van de extensie (Python, alleen NumPy)
  webai/                 model.py (transformer), nn.py (handgeschreven backprop),
                         tokenizer.py (byte-level BPE), dataset.py, train.py,
                         generate.py (KV-cache), chat.py (gespreksformaat + tool calls),
                         serve.py (OpenAI-compatibele HTTP-server + speelterrein),
                         feedback.py (leerlus), distill.py, cli.py
  data/seed/             meegeleverde voorbeelden (.jsonl)
  data/learned/          wat jij toevoegt of laat maken (niet in Git)
  runs/                  getrainde modellen (niet in Git)
  tests/                 165 tests: gradcheck, echte HTTP-server, en de seeddata zelf
extension/
  manifest.json          Manifest V3 (zijpaneel, contextmenu's, scripting, <all_urls>)
  background.js          Service worker: paneel openen, contextmenu's
  content/content.js     Content-script: pagina lezen, afbeeldingen, elementen, klikken/typen/scrollen
  sidepanel/             Chat-UI, agent-lus (agent.js), tools (tools.js), tab-brug (page.js), markdown+KaTeX (render.js)
  options/               Instellingenpagina, incl. status van je eigen AI en trainen vanuit de browser
  lib/                   OpenAI-client (streaming + tool calls), localai.js (brug naar ai/), instellingen, prompts, zoeken (DuckDuckGo/Bing, URL → tekst)
  vendor/                marked, DOMPurify, KaTeX (gekopieerd uit node_modules met `npm run vendor`)
scripts/                 copy-vendor.mjs, check.mjs, package.mjs (release-ZIP), make-icons.py
test/e2e.mjs             End-to-end test zonder browser: jsdom speelt de pagina + het content-script, een nep-OpenRouter speelt het model
docs/                    Website (GitHub Pages) + privacyverklaring
store/                   Winkelteksten, permissiemotivaties en promo-afbeeldingen voor Chrome Web Store / Edge Add-ons
.github/workflows/       release.yml: bij een tag `v*` tests draaien, ZIP bouwen en GitHub-release maken
```

De twee delen zijn bewust gescheiden: de extensie kent `ai/` alleen via HTTP op
`http://localhost:8787/v1`. Daardoor kun je het model los trainen, vervangen of
weglaten zonder de extensie aan te raken — en andersom.

- De agent draait in het zijpaneel (extensiepagina): daar zijn `fetch`-streaming, `DOMParser`, canvas en alle `chrome.*`-API's beschikbaar.
- Tools worden als OpenAI-style *function calls* aan het model gegeven; `openrouter:web_search` (server tool) wordt toegevoegd als je die zoekoptie kiest.
- Afbeeldingen die een tool oplevert (screenshot, `view_image`) gaan als `image_url` terug naar het model.
- Oude beurten worden compact gemaakt (geen oude paginatekst/afbeeldingen) om tokens en gratis verzoeken te sparen.

```bash
npm install          # dev-dependencies (eslint, jsdom, libs voor vendor)
npm run vendor       # vendor-bestanden verversen na een update van marked/dompurify/katex
npm run check        # manifest + verwijzingen + eslint
npm test             # e2e-tests (jsdom + nep-OpenRouter), 20 tests
npm run test:ai      # tests van het eigen AI-model (Python), 165 tests
npm run test:all     # beide suites achter elkaar
npm run package      # release/ai-web-agent.zip bouwen (voor winkels of een GitHub-release)
python3 scripts/make-icons.py   # iconen opnieuw genereren (Pillow)
```

Voor het eigen AI-model geldt daarnaast:

```bash
python3 -m venv .venv && .venv/bin/pip install -r ai/requirements.txt   # één keer
cd ai && python tests/run_all.py            # alle Python-tests
cd ai && python tests/run_all.py server     # alleen één suite
```

De `test:ai`-opdracht zoekt zelf een werkende Python (`.venv` heeft voorrang).

> Twee dingen die de tests bewijzen en die de moeite van het lezen waard zijn:
> `test_gradcheck.py` vergelijkt elke handgeschreven afgeleide met een numerieke
> benadering (87 parameters, grootste afwijking 5,4·10⁻⁶) en `test_server.py` start
> een echte HTTP-server en controleert dat de antwoorden exact de vorm hebben die de
> extensie verwacht.

### Publiceren

Zie [PUBLICEREN.md](PUBLICEREN.md): gratis via de website + GitHub-releases (staat klaar), gratis via **Microsoft Edge Add-ons**, of via de **Chrome Web Store** (eenmalig $5, alleen per betaalkaart).

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
