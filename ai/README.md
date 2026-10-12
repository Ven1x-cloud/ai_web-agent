# Je eigen AI 🧠

Een **zelf getraind taalmodel** dat in deze map (`ai/`) leeft, helemaal los van de
browserextensie. De extensie praat er tegenaan over HTTP met dezelfde API die
OpenRouter gebruikt, dus je kunt jederzeit wisselen tussen je eigen model en een
cloudmodel — zonder één regel code in de extensie aan te passen.

Alles is geschreven in **Python met alleen NumPy**: geen PyTorch, geen TensorFlow,
geen installatie van meerdere gigabytes. De transformer, de backpropagation, de
tokenizer en de optimalisator zijn met de hand geschreven, zodat je kunt zien wat
er gebeurt en het kunt aanpassen.

```
ai/
├── webai/          het programma (model, tokenizer, trainen, serveren)
├── data/           trainingsdata
│   ├── seed/       voorbeelden die meekomen in de broncode
│   └── learned/    wat jíj toevoegt of laat maken (staat niet in Git)
├── runs/           getrainde modellen (staat niet in Git)
└── tests/          165 tests die alles controleren
```

---

## ⚠️ Eerlijk over wat dit nu kan

Dit is een **echt, werkend, zelfgetraind model** — en tegelijk is het klein. Dat
twee dingen tegelijk waar zijn, is belangrijk om te begrijpen voordat je er tijd
insteekt.

Wat er nu meegeleverd wordt: **145 kB voorbeelddata** (510 vraag-antwoordparen over
negen onderwerpen, ±98.000 tokens) en een model van **0,95 miljoen parameters**. Ter vergelijking: de gratis cloudmodellen waar de
extensie standaard mee werkt hebben miljarden parameters en zijn getraind op
biljoenen tokens. Het verschil is een factor miljoen of meer.

| Trainingsdata | Wat je dan krijgt |
| --- | --- |
| ±12.000 tokens | Het model kent de voorbeelden uit zijn hoofd. Buiten die voorbeelden komt er **onsamenhangende tekst** uit. |
| ±98.000 tokens *(nu)* | Het model overfit niet meer: train- en eval-verlies liggen vlak bij elkaar. Het begint patronen te pakken, maar vloeiende zinnen komen er nog niet uit. |
| ±1 miljoen tokens | Echte zinnen, redelijke Nederlandse grammatica. Nog geen betrouwbare feitenkennis. |
| ±50 miljoen tokens | Leesbaar Nederlands over bekende onderwerpen. Fouten blijven vaak zelfverzekerd. |
| ±1 miljard tokens | Pas dan wordt het een bruikbare gesprekspartner voor eenvoudige vragen. |

**Concreet betekent dat:** met de meegeleverde data geeft je eigen model antwoorden
die nog niet lopen. Dat is geen bug en het is niet op te lossen door aan de
instellingen te draaien — het model heeft gewoon te weinig gezien. Een bruikbaar
model zit onder een eval-verlies van 2,0; de voorbeeldtraining komt tot nu toe niet
lager dan ongeveer 3,0.

Wel is er een duidelijke grens zichtbaar. Bij de eerste 12.000 tokens liep het
eval-verlies op van 3,3 naar 7,8 terwijl het trainverlies naar 0,004 zakte: puur
uit-het-hoofd leren. Bij de huidige 98.000 tokens blijven train en eval binnen een
paar honderdsten van elkaar — het model leert dan wél iets over taal in plaats van
alleen zijn voorbeelden. Meer data is dus niet zomaar méér van hetzelfde; het verlegt
wat het model doet.

Wat wél meteen werkt, en waar dit voor bedoeld is:

- 🔒 **Privé**: je vraag verlaat je computer niet. Geen account, geen sleutel, geen log bij een ander bedrijf.
- 🔌 **Offline**: geen internet nodig, geen daglimiet van 50 vragen, geen "provider is druk".
- 🎓 **Leren**: elke laag is leesbaar. Je ziet letterlijk hoe een taalmodel werkt, en kunt het zelf veranderen.
- 🌱 **Groeien**: het model wordt beter naarmate jíj het gebruikt. Elke 👍 en elke verbetering die je in de extensie geeft, wordt trainingsdata.

Voor huiswerk blijft het cloudmodel dus voorlopig verstandiger. Zie
[§ Hoe maak ik dit echt goed](#hoe-maak-ik-dit-echt-goed) voor de weg naar een
model dat wél meekan.

---

## Installatie

Je hebt Python 3.10 of nieuwer nodig en NumPy.

```bash
python3 -m venv .venv                      # één keer
.venv/bin/pip install -r ai/requirements.txt
```

Op Windows:

```bat
python -m venv .venv
.venv\Scripts\pip install -r ai\requirements.txt
```

> Werkt `pip install` niet vanwege een "externally-managed-environment"-melding?
> Gebruik dan de virtuele omgeving hierboven; dat is precies waarvoor hij bedoeld is.

---

## Snelstart

Vier commando's, vanuit de map `ai/`:

```bash
cd ai

python -m webai build --preset tiny        # 1. tokenizer + dataset bouwen  (~5 s)
python -m webai train --preset tiny        # 2. trainen                    (~2-10 min)
python -m webai serve                      # 3. server starten op poort 8787
python -m webai chat                       # 4. proberen in de terminal
```

De server blijft draaien zolang dat venster open staat. Daarna:

1. Open in Chrome/Edge/Brave `chrome://extensions` → **Uitgepakte extensie laden** → kies de map `extension/`.
2. Klik op het icoon van de extensie (of **Ctrl+Shift+Y**).
3. De instellingenpagina wijst zichzelf al naar je eigen model: basis-URL
   `http://localhost:8787/v1`, model `webai`.

Liever eerst kijken zonder extensie? Open <http://localhost:8787/> in je browser:
daar staat een klein donker **speelterrein** waarin je direct met je model kunt
praten en het opnieuw kunt trainen.

### Voortgang bekijken

```bash
python -m webai info --run webai
```

```
Model 'webai'
  parameters   0.68M
  vorm         2 lagen × 128 breed, context 512, woordenschat 1652
  getraind     stap 250, 0.68M tokens gezien
  verlies      0.7617  (trainen)
               3.3012  (evalueren — dit is het cijfer dat telt)
```

Twee verliezen, en het verschil daartussen is het belangrijkste getal dat je
krijgt:

- **trainverlies** — hoe goed het model zijn eigen voorbeelden nog kan nadoen. Dit daalt altijd.
- **eval-verlies** — hoe goed het is op voorbeelden die het **nooit gezien** heeft. Alleen dit cijfer zegt iets over hoe slim het model is.

Loopt het trainverlies richting 0 en blijft het eval-verlies hoog, dan is het model
aan het **overfitten**: het leert de voorbeelden uit zijn hoofd in plaats van taal.
`info` waarschuwt daar zelf voor. De oplossing is nooit "minder lang trainen" maar
**meer data**.

---

## De leerlus: zo train je het met je eigen gebruik

Dit is het deel dat "een eigen AI die ik kan trainen" concreet maakt. Het is een
rondje dat je steeds opnieuw kunt draaien:

```
   ┌──────────────────────────────────────────────────────┐
   │                                                      │
   ▼                                                      │
 praten in de extensie ──► 👍 👎 ✏️ geven ──► feedback.jsonl
                                               │
                                               ▼
                              python -m webai feedback --export
                                               │
                                               ▼
                                     data/learned/feedback_*.jsonl
                                               │
                                               ▼
                                  python -m webai train --resume
                                               │
                                               └── beter model ──┘
```

Onder elk antwoord in het zijpaneel staan drie knopjes:

| Knop | Wat er gebeurt | Wordt het trainingsdata? |
| --- | --- | --- |
| 👍 | Het antwoord was goed | **Ja** — vraag én antwoord worden een voorbeeld |
| 👎 | Het antwoord was fout | Nee — het model leert alleen wat het *niet* moet zeggen |
| ✏️ | Jij schrijft het goede antwoord | **Ja** — en dit is het waardevolst: jouw tekst vervangt het foute antwoord |

Een 👎 zonder verbetering wordt dus bewust **niet** als voorbeeld gebruikt: anders
zou het model precies dat foute antwoord gaan nadoen. Een ✏️ leert het model jouw
formulering, niet de oude.

```bash
python -m webai feedback              # kijken wat er verzameld is
python -m webai feedback --export     # omzetten naar trainingsdata
python -m webai train --resume        # verder trainen, inclusief het nieuwe materiaal
```

De server hoeft daarvoor niet gestopt te worden: hij merkt een nieuw opgeslagen
model aan het bestand zelf en laadt het automatisch in. Je ziet het in het
zijpaneel verschijnen zonder herstart.

### Sneller groeien: data laten maken

Feedback verzamelen gaat zo snel als je de extensie gebruikt. Wil je meteen een
grotere stap, laat dan een sterk cloudmodel trainingsmateriaal schrijven dat jouw
eigen model daarna overneemt:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python -m webai distill --limit 200
python -m webai train --resume
```

`distill` stuurt schoolvragen naar een cloudmodel en bewaart de antwoorden als
voorbeelden in `data/learned/distill.jsonl`. Zonder `--questions` gebruikt het 55
ingebouwde Nederlandse vragen over rekenen, taal, natuurkunde, geschiedenis en
aardrijkskunde. Met `--questions mijn_vragen.txt` (één vraag per regel) bepaal je
zelf de onderwerpen.

> Let op: zo'n model leert dan wat het cloudmodel weet, inclusief diens fouten.
> Controleer steekproeven voordat je het als bron van waarheid gaat gebruiken.

---

## Trainingsdata: wat mag erin

Alles in `ai/data/` wordt meegenomen, zowel `seed/` als `learned/`. Deze formaten
worden herkend:

**`.jsonl`** — één JSON-object per regel. Al deze varianten werken:

```jsonl
{"messages": [{"role": "user", "content": "Wat is 7 × 8?"}, {"role": "assistant", "content": "56."}]}
{"prompt": "Wat is 7 × 8?", "completion": "56."}
{"instruction": "Los op", "input": "7 × 8", "output": "56."}
{"question": "Wat is 7 × 8?", "answer": "56."}
{"conversations": [{"from": "human", "value": "Wat is 7 × 8?"}, {"from": "gpt", "value": "56."}]}
```

**`.txt` / `.md`** — platte tekst. Wordt in blokken van de contextlengte gehakt en
volledig meegetraind. Handig voor een samenvatting, een woordenlijst of een
hoofdstuk uit een boek dat je zelf mag gebruiken.

**`.json`** — een lijst met zulke objecten.

Kijk in `data/seed/` voor echte voorbeelden. Een kapotte regel wordt overgeslagen en
geteld in `python -m webai data`, zodat één typefout niet stilletjes je hele
training verpest.

```bash
python -m webai data
```

```
9 databestanden, 145.1 kB totaal

      30.5 kB  data/seed/09_maatschappij_nl.jsonl
      23.2 kB  data/seed/07_natuurwetenschappen_nl.jsonl
      20.5 kB  data/seed/06_taal_nl.jsonl
      15.8 kB  data/seed/02_school_nl.jsonl
      14.6 kB  data/seed/05_rekenen_nl.jsonl
      ...

Met de tokenizer van 'webai': 510 samples, 86918 tokens
  gesprekken: 510   tekstblokken: 0   overgeslagen: 0
```

De negen meegeleverde bestanden behandelen: wie de agent is, schoolvakken algemeen,
gereedschapsgebruik, feitenkennis (hoofdsteden, Nederland, wereld), rekenen en
wiskunde, Nederlands en grammatica, natuurkunde/scheikunde/biologie, Engels, en
maatschappij (economie, informatica, privacy, klimaat).

**Alleen het antwoord telt mee voor het verlies.** De vraag, de systeemprompt en
de resultaten van gereedschap staan wel in de context (het model leest ze), maar
worden niet als iets nadoenswaardigs behandeld. Daardoor leert het model antwoorden
geven in plaats van vragen verzinnen.

> ⚠️ Zet nooit iets in `data/` wat je niet kwijt wilt: het model kan gaan
> reproduceren wat het geleerd heeft. Geen wachtwoorden, geen persoonsgegevens van
> anderen, geen teksten waarvan je het auteursrecht niet mag gebruiken.
> `data/learned/` en `runs/` staan daarom niet in Git.

---

## Presets

| Preset | Parameters | Lagen × breedte | Context | Woordenschat | Piekgeheugen | CPU (2 kernen) |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | 0,69M | 2 × 128 | 256 | 2048 | ±0,4 GB | minuten |
| `small` | 4,3M | 4 × 256 | 512 | 4096 | ±1,8 GB | uren |
| `medium` | 14,0M | 6 × 384 | 512 | 8192 | ±1,9 GB bij batch 8 | dagen |
| `large` | 34,1M | 8 × 512 | 1024 | 16384 | ±8 GB | alleen met GPU |

De geheugencijfers zijn **gemeten** op deze machine (batch 16, behalve bij `medium`:
daar paste batch 16 niet in 4 GB). Ze zeggen iets belangrijks: het geheugen wordt
niet opgegeten door de gewichten maar door de tussenresultaten van de berekening.
De modeltoestand zelf (gewichten + gradiënten + de twee AdamW-geheugens, allemaal
float32) is slechts 11 MB bij `tiny` en 546 MB bij `large`.

Dat betekent ook dat je geheugentekort eenvoudig oplost: verlaag `--batch` en
compenseer met `grad_accum` als je dezelfde effectieve batch wilt houden.

Het aantal parameters is uitgerekend uit de vorm van het netwerk
(`ModelConfig.estimate_params()`), dus je kunt het al zien vóór je begint. De
formule is exact gecontroleerd tegen het werkelijke netwerk voor alle vier presets.

Kies je preset op basis van je **data**, niet van je geduld: een `medium` model op
12.000 tokens leert alleen maar sneller uit zijn hoofd. Vuistregel: reken op
minstens 100 tokens per parameter. Voor `tiny` (0,69M) is dat ±70 miljoen tokens —
waarbij geldt dat ook dát nog weinig is.

```bash
python -m webai train --preset small --steps 2500 --batch 16
python -m webai train --preset tiny --lr 0.002 --context 384
```

Trainen mag onderbroken worden met **Ctrl+C**: er wordt dan netjes een checkpoint
weggeschreven en je gaat later verder met `--resume`.

---

## Commando's

```bash
python -m webai <commando> [opties]
```

| Commando | Wat het doet |
| --- | --- |
| `build` | Tokenizer en dataset bouwen uit `ai/data/` |
| `train` | Het model trainen (of verder trainen met `--resume`) |
| `serve` | OpenAI-compatibele server starten op poort 8787 |
| `chat` | Praten met je model in de terminal |
| `ask` | Eén vraag stellen en het antwoord tonen |
| `info` | Status, verliezen en advies over je model |
| `data` | Statistieken van je trainingsdata |
| `feedback` | Verzamelde feedback bekijken of omzetten in trainingsdata |
| `distill` | Trainingsdata laten maken door een cloudmodel |

Vaak gebruikte opties bij `train`:

| Optie | Betekenis |
| --- | --- |
| `--preset` | `tiny`, `small`, `medium` of `large` |
| `--run` | Naam van het model (een map in `ai/runs/`); standaard `webai` |
| `--steps` | Aantal trainingsstappen |
| `--batch` | Batchgrootte (hoger = stabieler, meer geheugen) |
| `--lr` | Leersnelheid |
| `--context` | Maximale contextlengte in tokens |
| `--grad-accum` | Gradiënten over dit aantal stapjes optellen: kleiner `--batch` bij geheugentekort, zonder de effectieve batch te veranderen |
| `--eval-every` | Om de zoveel stappen evalueren op onbekende voorbeelden |
| `--resume` | Verder waar je gebleven was |
| `--init-from` | Starten vanaf een ander model (bijv. verder op nieuwe data) |

Bij `serve`:

| Optie | Betekenis |
| --- | --- |
| `--host` | Standaard `127.0.0.1`. Gebruik `0.0.0.0` alleen als je vanaf een ander apparaat wilt verbinden — dan staat je model open voor je hele netwerk |
| `--port` | Standaard `8787` |
| `--checkpoint` | `auto` (het beste eval-verlies), `model` (het laatste) of `best` |
| `--run` | Welk model je serveert |

---

## De HTTP-API

De server praat dezelfde taal als OpenRouter en OpenAI. Daardoor werkt de
bestaande client van de extensie (`extension/lib/openrouter.js`) zonder
aanpassing, en kun je elk ander programma dat OpenAI-compatibel is erop aansluiten.

| Methode | Pad | Wat |
| --- | --- | --- |
| `GET` | `/` | Speelterrein in de browser |
| `GET` | `/health` | Levenscheck |
| `GET` | `/v1/models` | Lijst met modellen |
| `GET` | `/v1/info` | Uitgebreide modelstatus (ook gebruikt door de instellingenpagina) |
| `GET` | `/v1/status` | Kort: is het model geladen, traint het |
| `GET` | `/v1/feedback` | Verzamelde feedback en statistieken |
| `POST` | `/v1/chat/completions` | Vragen stellen (met `stream: true` voor SSE) |
| `POST` | `/v1/completions` | Tekst aanvullen |
| `POST` | `/v1/feedback` | Feedback opslaan |
| `POST` | `/v1/train` | Training starten vanuit de browser |
| `POST` | `/v1/reload` | Model opnieuw inlezen |

Een voorbeeld met `curl`:

```bash
curl http://localhost:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"webai",
       "messages":[{"role":"user","content":"Wat is de hoofdstad van Nederland?"}],
       "temperature":0.3, "max_tokens":120}'
```

Bij `stream: true` komen de antwoorden als *server-sent events* binnen, in
losse `delta.content`-stukjes, gevolgd door een eindbericht met `usage` en
`data: [DONE]`. Gereedschapsaanroepen komen als `delta.tool_calls`. Dat is precies
de vorm die de extensie al verstond voor OpenRouter.

CORS staat open (`Access-Control-Allow-Origin: *`) omdat een browserextensie vanaf
`chrome-extension://…` komt en anders niet mag verbinden. Bind daarom op
`127.0.0.1` en niet op `0.0.0.0` als anderen op jouw netwerk zitten.

---

## Hoe het gebouwd is

| Module | Wat erin zit |
| --- | --- |
| `config.py` | Model-, trainings- en generatie-instellingen; de vier presets |
| `nn.py` | Losse bouwstenen met handgeschreven backpropagation: matmul, layernorm, softmax, GELU, kruisentropie, AdamW |
| `model.py` | De transformer: `forward()` voor verlies + gradiënten, `step()` met KV-cache voor genereren, `save()`/`load()` |
| `tokenizer.py` | Byte-level BPE. Elk teken dat bestaat kan gecodeerd worden, dus ook emoji en Frans zonder woordenboek |
| `chat.py` | Het gespreksformaat met `<\|im_start\|>`/`<\|im_end\|>`-achtige markeringen, gereedschapsbeschrijvingen, en het teruglezen van aanroepen |
| `dataset.py` | Bestanden inlezen, omzetten naar samples, bepalen welke tokens meetellen, batches maken |
| `generate.py` | Token voor token tekst maken: temperatuur, top-k, top-p, herhalingsstraf, KV-cache |
| `train.py` | De trainingslus: leerplanning, gradiëntaccumulatie, evalueren, checkpoints, vroeg stoppen |
| `serve.py` | De HTTP-server, het speelterrein, en het automatisch herladen van nieuwe modellen |
| `feedback.py` | Feedback bewaren, beoordelen op bruikbaarheid, omzetten naar trainingsdata |
| `distill.py` | Trainingsdata laten schrijven door een cloudmodel |
| `cli.py` | Alle commando's hierboven |

Twee dingen die de moeite waard zijn als je wilt begrijpen hoe dit werkt:

**De backpropagation is handgeschreven en bewezen correct.** In `tests/test_gradcheck.py`
wordt elke afgeleide vergeleken met een numerieke benadering (het verlies een
klein beetje verstoren en kijken wat er gebeurt). Voor 87 parameters komt de
relatieve afwijking uit op hooguit 5,4·10⁻⁶. Drie echte rekenfouten zijn daar door
aan het licht gekomen en hersteld — onder andere een ontbrekend causaal masker,
waarbij het model stiekem naar latere tokens in dezelfde stap kon kijken.

**De generatie gebruikt een KV-cache.** Bij elk nieuw token hoeft het model niet de
hele tekst opnieuw te verwerken; de sleutels en waarden van eerdere tokens worden
bewaard. `tests/test_gradcheck.py` controleert dat `step()` (met cache) op
10⁻⁵ nauwkeurig hetzelfde oplevert als `forward()` (alles opnieuw).

---

## Tests

```bash
cd ai
python tests/run_all.py                  # alles (±5 seconden)
python tests/run_all.py chat server      # alleen die twee
```

```
Python 3.11.2 — 8 suites

  ✓ gradcheck     6 tests (0.3s)
  ✓ tokenizer    10 tests (0.1s)
  ✓ chat         22 tests (0.0s)
  ✓ dataset      28 tests (0.2s)
  ✓ seeddata     24 tests (1.5s)
  ✓ training     18 tests (4.0s)
  ✓ server       21 tests (0.8s)
  ✓ feedback     36 tests (0.2s)

Alles groen.
```

**165 tests.** De servertest start een echte HTTP-server op een vrije poort en doet
echte verzoeken, zodat de vorm van de antwoorden gecontroleerd wordt tegen wat de
extensie verwacht — inclusief streaming en gereedschapsaanroepen.

`test_seed_data.py` controleert niet de code maar **de meegeleverde data zelf**: elke
regel geldige JSON, elk gesprek eindigt op een antwoord, geen lege of dubbele vragen,
en de hele set verwerkt zonder overslagen records. Die test heeft meteen drie dubbele
vragen gevonden in de handgeschreven bestanden.

De tests van de extensie zelf (20 stuks) draai je met `npm test` vanuit de
hoofdmap; die controleren ook dat de extensie correct omschakelt naar je eigen
model.

---

## Problemen oplossen

**"Geen model gevonden in ai/runs/webai"**
Er is nog niets getraind. Draai `python -m webai build --preset tiny` en daarna
`python -m webai train --preset tiny`.

**De extensie zegt "server niet bereikbaar"**
De server draait niet, of staat op een andere poort. Start hem met
`cd ai && python -m webai serve` en laat dat venster open. Controleer met
`curl http://localhost:8787/health`.

**Het model geeft onsamenhangende tekst**
Bij de meegeleverde data is dat het verwachte gedrag, geen storing. Meer data is de
enige oplossing die werkt: `python -m webai distill --limit 200` of eigen bestanden
in `data/learned/`, en dan `train --resume`. Zie
[⚠️ Eerlijk over wat dit nu kan](#️-eerlijk-over-wat-dit-nu-kan).

**Poort 8787 is al in gebruik**
`python -m webai serve --port 8888`, en zet in de instellingenpagina van de
extensie de basis-URL op `http://localhost:8888/v1`.

**Trainen is traag**
Op 2 CPU-kernen haalt `tiny` ongeveer 2.500 tokens per seconde. Een grotere preset
kost al snel uren. Zet `--steps` lager om eerst te proberen of alles werkt, of
train op een machine met een videokaart.

**`MemoryError` tijdens trainen**
Verlaag `--batch` (bijvoorbeeld naar 4 of 2). Het geheugengebruik groeit met
batchgrootte × contextlengte.

**Ctrl+C tijdens trainen**
Dat mag: er wordt een checkpoint bewaard en je kunt verder met `--resume`. Wordt
het proces met `kill` gestopt, dan wordt die opslag overgeslagen — je laatste
`step*.npz` blijft dan bruikbaar via `--checkpoint model`.

---

## Hoe maak ik dit echt goed

Eerlijk: dit model wordt pas een bruikbare gesprekspartner met veel meer data en
rekenkracht dan hier beschikbaar is. De stappen die het meeste opleveren, in volgorde:

1. **Meer data** — veruit de grootste factor. `distill` met een paar duizend vragen, plus eigen samenvattingen in `data/learned/`. Reken op miljoenen tokens, niet duizenden.
2. **Een groter model** — pas zinvol ná stap 1. Anders overfit het alleen maar sneller.
3. **Langer trainen** — met genoeg data en een vroeg-stopbewaking.
4. **Een videokaart** — de NumPy-implementatie is bewust eenvoudig gehouden, niet snel. Voor serieuze training is een GPU-backend de volgende stap; `nn.py` is zo opgezet dat die verwisselbaar is.

Wat ook helpt, en minder werk is:

- **De woordenschat vergroten** (`--vocab 8192`): kortere teksten per voorbeeld, dus meer inhoud in dezelfde context. Kost wel geheugen.
- **Feedback blijven geven**: de leerlus hierboven werkt vanaf de eerste dag en kost alleen de moeite van een klik.

---

## Vergunning

Deze code hoort bij het project in de hoofdmap en valt onder dezelfde voorwaarden.
De meegeleverde voorbeelddata in `data/seed/` is zelf geschreven.
