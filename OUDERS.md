# Voor ouders en verzorgers

*Uw kind heeft een browserextensie gemaakt en wil die online zetten. Deze pagina legt in vijf minuten uit wat dat inhoudt, wat er van u wordt gevraagd en waarom. (Dezelfde tekst staat als webpagina in `docs/ouders.html`.)*

## Wat is het?

**AI Web Agent** is een uitbreiding voor Chrome/Edge die een AI-assistent in het zijpaneel van de browser zet: hij leest de website die open staat, bekijkt afbeeldingen en helpt met uitleg bij schoolwerk (talen, wiskunde, scheikunde, aardrijkskunde …). De AI zelf draait niet in de extensie maar bij een externe dienst (standaard [OpenRouter](https://openrouter.ai), die ook gratis modellen aanbiedt). De volledige broncode is openbaar en door iedereen te controleren: deze GitHub-repository.

De extensie is gratis, bevat geen advertenties en geen aankopen, en verzamelt zelf geen gegevens. Wat er met gegevens gebeurt staat in de privacyverklaring (`docs/privacy.html`).

## Waarom wordt uw hulp gevraagd?

Voor een paar stappen is een account nodig waarvoor je volgens de voorwaarden **volwassen** moet zijn. Uw kind kan dat dus niet zelf, en het is verstandig dat ook niet te proberen met verzonnen gegevens.

| Account | Waarvoor | Kosten | Leeftijdseis |
|---|---|---|---|
| [OpenRouter](https://openrouter.ai) | De AI-dienst die de extensie gebruikt. U maakt een account en één "API-sleutel" aan; die sleutel plakt uw kind in de extensie. Zonder tegoed: 50 gratis vragen per dag. Er wordt niets automatisch afgeschreven; er hoeft geen betaalmethode gekoppeld te worden. | € 0 | 18+ volgens de [voorwaarden](https://openrouter.ai/terms) |
| [Microsoft Partner Center](https://partner.microsoft.com/dashboard/microsoftedge/overview) | Om de extensie in de officiële winkel van Microsoft Edge te zetten (Edge Add-ons). | € 0 | Contract; volwassene als accounthouder |
| [Chrome Web Store](https://chrome.google.com/webstore/devconsole) | Om de extensie in de winkel van Google Chrome te zetten. Optioneel; kan later. | Eenmalig $ 5 (≈ € 4,50), alleen per bankpas/creditcard | 18+ (Google Payments) |

## Welke gegevens vragen zij, en wie ziet die?

- **Naam, e-mailadres, telefoonnummer (voor een sms-code) en postadres.** Dit is voor de administratie van Microsoft/Google, net als bij een gewoon Microsoft- of Google-account. Zij sturen geen post; communicatie gaat per e-mail.
- **Openbaar op de winkelpagina** staan alleen de gekozen publisher-naam (bijvoorbeeld "Ven1x"), de website en een support-contact (bijvoorbeeld de GitHub-pagina). Het adres wordt alleen openbaar als u zich opgeeft als "handelaar" (*trader*) — dat is niet van toepassing op een gratis extensie zonder verdienmodel; kies dan **"not a trader"**.
- Voor OpenRouter is alleen een e-mailadres (of Google-login) nodig.

## Waar tekent u voor?

De accounthouder is formeel verantwoordelijk voor wat er gepubliceerd wordt. In dit geval: een gratis, open-source hulpmiddel dat pagina's alleen leest als de gebruiker daarom vraagt, geen wachtwoorden of betaalgegevens invult en eerst om bevestiging vraagt voordat het iets verstuurt. De winkels controleren elke inzending vóór publicatie. U kunt het account op elk moment weer sluiten.

## Checklist (± 20 minuten)

1. **OpenRouter (5 min):** account maken op [openrouter.ai](https://openrouter.ai) → *Keys* → *Create key* → de sleutel (`sk-or-v1-…`) aan uw kind geven; die plakt hem in de instellingen van de extensie. Bewaar de inloggegevens zelf. Een eventueel eerder door uw kind gemaakt account kan bij *Settings* worden verwijderd.
2. **Edge Add-ons (15 min + wachten op controle):** inloggen op [Partner Center](https://partner.microsoft.com/dashboard/microsoftedge/overview) met een Microsoft-account, registreren voor het *Microsoft Edge program* (kies **Individual**), gegevens invullen, akkoord met de *App Developer Agreement*. Het uploaden en invullen van de winkelpagina kan uw kind daarna zelf doen; de teksten liggen klaar in [`store/listing.md`](store/listing.md) en de stappen in [`PUBLICEREN.md`](PUBLICEREN.md).
3. **Chrome Web Store (optioneel):** zelfde idee, met de eenmalige $ 5.
4. **Website aanzetten (1 min, mag uw kind zelf):** in deze GitHub-repository → *Settings → Pages → Deploy from a branch → map `/docs`*. Daarna staat de uitleg op `https://ven1x-cloud.github.io/ai_web-agent/`.

## Liever helemaal geen accounts?

Dat kan ook. De extensie staat al gratis online (downloaden en zelf installeren), en de AI kan volledig op de eigen computer draaien met [Ollama](https://ollama.com) — dan is er geen enkel account nodig en verlaat er niets de computer. Dat vraagt wel een redelijk moderne pc (minimaal 8 GB werkgeheugen) en is trager.

## Vragen?

Alle code, teksten en deze uitleg zijn openbaar in deze repository. Vragen kunt u stellen via *Issues*.
