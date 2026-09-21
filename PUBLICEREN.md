# AI Web Agent online zetten

Er zijn drie manieren om de extensie "online" te krijgen. Ze sluiten elkaar niet uit – begin bij 1,
dat is gratis en staat al klaar.

| | Kosten | Wie kan installeren | Moeite |
|---|---|---|---|
| **1. Eigen website + GitHub-release** | € 0 | Iedereen, via ZIP + ontwikkelaarsmodus | Staat klaar ✅ |
| **2. Microsoft Edge Add-ons** | € 0 (registratie is gratis) | Edge-gebruikers, met één klik | ± 1 uur + wachten op review |
| **3. Chrome Web Store** | eenmalig $ 5 (≈ € 4,50), alleen per bankpas/creditcard, 18+ | Chrome-, Edge-, Brave-gebruikers | ± 1 uur + wachten op review |

> **Over de $5 van Google:** die kun je niet met contant geld of een cadeaukaart betalen; Google Payments
> accepteert alleen een betaalkaart en het account moet van een volwassene zijn. De praktische route is
> dat een ouder/verzorger het developer-account op zijn/haar naam aanmaakt en de $5 betaalt – het is eenmalig,
> daarna kun je tot 20 extensies publiceren. Tot die tijd zijn 1 en 2 volledig gratis alternatieven.

---

## 1. Eigen website + GitHub-release (gratis, staat klaar)

**Website:** `https://ven1x-cloud.github.io/ai_web-agent/` – de bestanden staan in de map `docs/`.
Daar staan de uitleg, de downloadknop en de privacyverklaring (die heb je ook nodig voor de winkels).

**Downloadlink die altijd naar de nieuwste versie wijst:**
`https://github.com/Ven1x-cloud/ai_web-agent/releases/latest/download/ai-web-agent.zip`

### Nieuwe versie uitbrengen (3 stappen)

1. Zet het nieuwe versienummer in `extension/manifest.json` (bijv. `"version": "0.2.0"`) en commit dat.
2. Maak een tag met hetzelfde nummer en push die:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
3. GitHub Actions (`.github/workflows/release.yml`) draait de tests, bouwt `ai-web-agent.zip` en zet een release
   klaar onder **Releases**. De downloadknop op de website wijst automatisch naar deze nieuwe versie.

Zonder GitHub Actions kan het ook met de hand: `npm run package` maakt `release/ai-web-agent.zip`;
die upload je bij **Releases → Draft a new release**.

### Website aanpassen

- Teksten: `docs/index.html` (gewone HTML, geen build-stap).
- Zodra de extensie in een winkel staat: haal in `docs/index.html` het commentaar rond de winkelknoppen weg en vul de link in.
- GitHub Pages instellen (eenmalig, als dat nog niet gedaan is): repository → **Settings → Pages → Build and deployment →
  Source: Deploy from a branch → Branch: `main`, map `/docs` → Save**. Na een minuut is de site live.

---

## 2. Microsoft Edge Add-ons (gratis officiële winkel)

Edge is ook Chromium, dus dezelfde ZIP werkt. Registreren en publiceren is gratis; je kunt zelfs inloggen met je GitHub-account.

1. Maak de ZIP: `npm run package` (of download `ai-web-agent.zip` van de laatste release).
2. Ga naar <https://partner.microsoft.com/dashboard/microsoftedge/overview> en registreer je voor het
   *Microsoft Edge program* (gratis; kies "Individual").
3. **Create new extension** → upload de ZIP.
4. Vul de winkelpagina in met de teksten uit [`store/listing.md`](store/listing.md):
   - Beschrijving (NL en EN), categorie *Education*, privacy policy URL, support-URL.
   - Logo: `store/logo-300x300.png`. Screenshots: zelf maken (1280×800), zie onderaan `store/listing.md`.
5. **Notes for certification**: schrijf kort hoe de reviewer het kan testen, bijvoorbeeld:
   > The extension needs an OpenRouter API key (free at openrouter.ai/keys). Open any website, click the toolbar icon to open the side panel, paste the key and ask e.g. "Summarize this page". Free models work without credits.
6. **Publish**. De review duurt meestal 1–7 dagen. Daarna krijg je een link als
   `https://microsoftedge.microsoft.com/addons/detail/...` – zet die op de website.

Updates: verhoog de versie in `manifest.json`, maak een nieuwe ZIP en upload die in het dashboard onder *Packages*.

---

## 3. Chrome Web Store

1. Iemand van 18+ met een betaalkaart gaat naar <https://chrome.google.com/webstore/devconsole>, accepteert de
   voorwaarden en betaalt de eenmalige registratie van $5.
2. **New item** → upload `ai-web-agent.zip`.
3. Tabblad **Store listing**: teksten uit [`store/listing.md`](store/listing.md), icoon `extension/icons/icon128.png`,
   promo-tegel `store/promo-440x280.png`, eventueel `store/marquee-1400x560.png`, plus je eigen screenshots (1280×800).
4. Tabblad **Privacy practices**: single purpose, permissiemotivaties en gegevensgebruik staan letterlijk in `store/listing.md`
   (in het Engels). Privacy policy URL: `https://ven1x-cloud.github.io/ai_web-agent/privacy.html`.
5. Tabblad **Distribution**: gratis, alle landen (of alleen Nederland/België), zichtbaarheid *Public* – of *Unlisted* als je
   hem eerst alleen met vrienden wilt delen via de link.
6. **Submit for review**. Omdat de extensie op alle websites mag werken (`<all_urls>`), kijkt Google wat strenger;
   reken op enkele dagen tot een paar weken. Leg in het veld voor de reviewer uit dat een gratis OpenRouter-sleutel nodig is
   en hoe hij kan testen (zie punt 5 bij Edge).

---

## Checklist vóór elke publicatie

- [ ] `npm run check` en `npm test` slagen.
- [ ] Versie in `extension/manifest.json` verhoogd (winkels weigeren dezelfde versie twee keer).
- [ ] Geen API-sleutel, e-mailadres of naam in screenshots of code.
- [ ] Privacyverklaring nog kloppend (`docs/privacy.html`) als er nieuwe functies zijn die data versturen.
- [ ] Even getest in een schone browserprofiel: installeren → sleutel invullen → vraag stellen op een website.
