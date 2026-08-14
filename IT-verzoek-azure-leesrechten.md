# Verzoek aan IT — leesrechten op twee Azure OpenAI-deployments

**Aanvrager:** Ruud Kooiman
**Datum:** 14 augustus 2026
**Betreft:** Entra ID-leesrechten (Reader) op twee Cognitive Services-accounts

---

## Aanleiding

Voor een analysepijplijn die open antwoorden uit vragenlijsten codeert, worden
grote aantallen aanroepen naar Azure OpenAI gedaan. Om die aanroepen netjes te
doseren leest de pijplijn per deployment uit hoeveel verkeer is toegestaan — het
zogeheten rate-limit-quotum (tokens en requests per minuut).

Tot voor kort kwam die informatie mee in de response-headers van de API zelf.
Microsoft levert die headers niet meer betrouwbaar op de Responses API en
verwijst voor het uitlezen van quota naar de Azure Resource Manager. Zie
[Manage Azure OpenAI quota](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota).

De pijplijn draait op dit moment gewoon door op een terugvalmechanisme, dus dit
is **niet urgent**. Het verzoek maakt de oplossing duurzaam in plaats van
afhankelijk van een route die Microsoft aan het uitfaseren is.

---

## Het verzoek

Een **app-registratie (service principal)** in Entra ID met de rol **Reader**,
gescoped op deze twee Cognitive Services-accounts:

| resource | rol |
|---|---|
| `mot-azure-open-ai` | Reader |
| `mot-azure-openai-dev-resource` | Reader |

**Niet subscription-breed** — alleen deze twee resources.

Het gaat uitsluitend om lezen. Er wordt niets aangemaakt, gewijzigd of
verwijderd. De aanroep die de pijplijn doet is één `GET` per run:

```
GET https://management.azure.com/subscriptions/{sub}/resourceGroups/{rg}
    /providers/Microsoft.CognitiveServices/accounts/{account}
    /deployments/{deployment}?api-version=2025-06-01
```

Wil de beveiliging het strakker dan de ingebouwde Reader-rol, dan is dit de
enige benodigde actie:

```
Microsoft.CognitiveServices/accounts/deployments/read
```

---

## Wat ik terug nodig heb

| waarde | waarvoor |
|---|---|
| Tenant ID | authenticatie |
| Client ID (application ID) | authenticatie |
| Client secret | authenticatie |
| **Vervaldatum van de secret** | om tijdige vernieuwing in te plannen |
| Subscription ID | om de resource-URL samen te stellen |

De resource groups hoeven niet meegeleverd te worden; die zijn na
authenticatie zelf op te vragen.

De secret wordt opgeslagen in een lokaal `.env`-bestand dat buiten versiebeheer
valt, en wordt niet gedeeld of naar een server gestuurd.

---

## Waarom een service principal en niet mijn eigen account

Rechten op mijn persoonlijke account werken ook, maar met twee bezwaren:

1. **Een persoonlijke login verloopt.** Als dat gebeurt valt de pijplijn
   stilzwijgend terug op conservatieve standaardwaarden. Dat is geen foutmelding
   maar een stille vertraging — vandaag kostte precies dat scenario een halve
   werkochtend aan zoeken.
2. **Het werkt alleen op mijn laptop.** Een service principal werkt ook
   onbeheerd en vanaf een andere machine.

Is een app-registratie bezwaarlijk, dan is **Reader op mijn eigen account** voor
diezelfde twee resources een bruikbare tussenstap. Daarmee kan ik aantonen dat
de route werkt; het lost de twee punten hierboven niet op.

---

## Samengevat

- **Wat:** service principal met Reader op twee Cognitive Services-accounts
- **Waarom:** quotum uitlezen zoals Microsoft het nu voorschrijft
- **Risico:** nihil, alleen-lezen op twee resources
- **Urgentie:** laag — er draait een werkende terugval
- **Terug nodig:** tenant ID, client ID, client secret + vervaldatum, subscription ID
