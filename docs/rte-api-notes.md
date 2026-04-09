# RTE Data API — Notes d'exploration

**Date du test** : 2026-04-09
**Contexte** : évaluation de la viabilité de l'API RTE pour remplacer l'ingestion fichier (éCO2mix annuel) par un flux live, dans le cadre de la démo WeatherNews.
**Statut** : exploratoire — **pas intégré au framework**, les fichiers locaux `data/ecomix-rte/*.zip` restent la source primaire pour la démo WN.

---

## Résumé exécutif

L'API RTE Data est **productionisable**, avec des contraintes claires et documentées. Le portail développeur est catastrophique à naviguer, mais une fois les abonnements activés, l'API elle-même est propre, stable, et documentée. Pour un déploiement live de notre framework demand, il suffit de remplacer le lecteur de fichiers par un client OAuth2 : ~30 lignes de code, pattern déjà existant dans le projet voisin `wattcast`.

**Verdict** : swap fichier → API = 15 minutes de dev si le pattern wattcast est porté en sync. Le reste du pipeline (schema, QC, features, training, MLflow) ne change pas.

---

## Credentials

Credentials de test stockés hors repo (dans `.env` local ou vault) :

```
WINDCAST_RTE_CLIENT_ID=<uuid v4>
WINDCAST_RTE_CLIENT_SECRET=<uuid v4>
```

**Ne jamais commiter** ces valeurs. Obtenues via inscription sur https://data.rte-france.com/ → création d'une "Application" → souscription manuelle aux API individuelles (une souscription = un endpoint).

---

## Endpoints testés

### OAuth2 Token — `POST /token/oauth/`

```
URL   : https://digital.iservices.rte-france.com/token/oauth/
Auth  : Basic base64(client_id:client_secret)
Réponse : { access_token, token_type=Bearer, expires_in=3600, scope }
```

- **TTL** : 3600 s (1 heure)
- **Pas de scope** requis
- **Zero friction** : 1 appel, tout fonctionne

### ✅ `consumption/v1/short_term` (PROD)

```
URL    : https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term
Auth   : Bearer <token>
Params : start_date, end_date (ISO 8601 avec offset), type=REALISED,D-1
```

**Ce qui marche :**
- **Résolution native : 15 min** (plus fin que les fichiers éCO2mix qui sont en 30 min pour la consommation)
- **REALISED + D-1 en un seul appel** via `type=REALISED,D-1` (comma-separated, pas d'espace)
- **Historique profond** : testé OK jusqu'en 2016-04-10 ; 2011-04-12 retourne 0 valeurs (cap empirique ~2014-2015)
- **Forecast D-1 historique disponible** — pas seulement pour demain, donc on peut backtester notre modèle vs la prévision officielle RTE sur toute la période validation
- **Données temps réel** : dernière valeur à quelques minutes près (testé : `updated_date` = 23:50:03 pour la journée en cours)
- **Structure de réponse** :
  ```json
  {
    "short_term": [
      {
        "type": "REALISED",
        "start_date": "...", "end_date": "...",
        "values": [
          {"start_date": "...", "end_date": "...", "updated_date": "...", "value": 44778},
          ...
        ]
      },
      { "type": "D-1", "values": [...] }
    ]
  }
  ```

**Limite critique — chunks 150 jours max** :
- 30 jours → HTTP 200, 2 780 valeurs
- 150 jours → HTTP 200, 14 240 valeurs
- 200 jours → **HTTP 400** `CONSUMPTION_SHORTTERM_F03` : _"The API does not provide feedback on such a long period in one call. To retrieve all the..."_
- 365 jours → même erreur

Pour backfiller 2014 → 2024 (~3 650 jours), il faut ~25 chunks, ~25 appels, temps total estimé **2-3 minutes** avec 0.5 s de courtoisie entre chunks.

### ✅ `wholesale_market/v3/france_power_exchanges` (PROD)

```
URL : https://digital.iservices.rte-france.com/open_api/wholesale_market/v3/france_power_exchanges
```

Bonus non prévu qui **marche** avec nos credentials :
- Prix spot marché (€/MWh) + volumes (MWh) en 15 min
- Contient déjà les valeurs du **lendemain** (publication marché J-1 midi)
- Exemple réel : `{"value": 20068.3, "price": 35.43}` pour 2026-04-10 00:00
- Structure : `{ "france_power_exchanges": [ { "start_date", "end_date", "updated_date", "values": [...] } ] }`

Utilisable en l'état comme feature exogène `price_eur_mwh` pour un modèle demand price-aware — **non nécessaire pour Pass 7** mais noté comme polish future.

### ⚠️ `consumption/v1/sandbox/short_term` — inutile

Le sandbox renvoie **toujours les mêmes données figées du 2016-02-01** (échantillon statique pour tester le format sans consommer le quota PROD). Contient des types exotiques (`ID`, `D-2`) qui n'existent probablement pas tous en PROD. **Zéro valeur** pour notre cas d'usage : utiliser directement PROD.

### ❌ Sous-endpoints wholesale non autorisés (HTTP 403)

Ces endpoints existent mais notre souscription actuelle ne les couvre pas :

| Endpoint | HTTP | Note |
|---|---|---|
| `wholesale_market/v3` (root) | 403 | Root non exposé |
| `wholesale_market/v3/epex_france_power_auction` | 403 | Abonnement séparé requis |
| `wholesale_market/v3/eod_france_power_exchanges` | 403 | Abonnement séparé requis |
| `wholesale_market/v3/clearing_prices` | 403 | Abonnement séparé requis |

**Pattern RTE** : chaque API est un abonnement distinct à activer manuellement dans le catalogue. Pour ajouter un endpoint, retourner sur le portail, chercher l'API dans le catalogue, cliquer "Souscrire". Pas de wildcard, pas de groupe.

---

## Pattern de code de référence

Le client OAuth2 complet existe déjà dans le projet voisin : **`../wattcast/src/wattcast/data/rte.py`** (lignes 31-134). Wattcast utilise `httpx.AsyncClient`. Pour porter en sync vers windcast :

- Remplacer `httpx.AsyncClient` → `httpx.Client`
- Retirer `async`/`await` et `asyncio.Lock`
- Garder la logique de cache token (expiration check avec `TOKEN_TTL_BUFFER = 60`)
- Garder le pattern `_chunk_dates(start, end, chunk_days=150)` pour le backfill
- Remplacer la sortie pandas par Polars (`pl.DataFrame` from list of dicts)

Estimation porting : **~80 lignes, 30 minutes de dev**.

---

## Comparaison fichiers éCO2mix vs API

| Critère | Fichiers `data/ecomix-rte/*.zip` | API `/consumption/v1/short_term` |
|---|---|---|
| **Résolution consommation** | 30 min | **15 min** |
| **Résolution forecast D-1** | 15 min | 15 min |
| **Historique** | 2014-2024 (11 ans dispo localement) | 2014/15 → aujourd'hui |
| **Temps réel** | Non (annuel définitif, J+1 an) | **Oui, ~H-1 min** |
| **Données consolidées** | Définitif = meilleure qualité | REALISED (révisions possibles) |
| **Credentials requis** | Non | Oui (OAuth2) |
| **Réseau requis** | Non | Oui |
| **Reproductibilité** | **Totale** (ZIP versionnés) | Dépend des révisions serveur |
| **Volume par backfill 10 ans** | ~50 MB × 11 fichiers | ~25 appels × 150 jours |
| **Temps backfill complet** | Instantané (lecture disque) | ~2-3 min |
| **Risque production** | Aucun (offline) | Network, quotas, credentials |
| **Refresh quotidien** | Re-télécharger 1× par an | **1 appel API/jour** |

**Conclusion** : pour la démo WN (déterministe, offline, une seule fois), les fichiers sont supérieurs. Pour une exécution en production live, l'API est supérieure et triviale à brancher.

---

## Verdict productionisation

| Critère | Verdict |
|---|---|
| **Authentification** | OAuth2 standard, 1h TTL, auto-refresh trivial |
| **Documentation** | Documentée en ligne, code de référence wattcast, messages d'erreur explicites |
| **Stabilité** | Existe depuis années, utilisée en production par wattcast |
| **Historique suffisant** | ~10 ans — largement assez pour ML |
| **Temps réel** | Latence quelques minutes — OK pour day-ahead forecasting |
| **Chunking** | 150 jours max par appel (documenté empiriquement) |
| **Rate limits** | Non observés, wattcast sleep 0.5 s par courtoisie |
| **Fiabilité réseau** | Standard httpx, retries à implémenter basiquement |
| **Coût dev (port sync)** | ~30 minutes (pattern existe) |
| **Coût opérationnel** | Nul (API gratuite pour usage standard) |
| **Seul pain point** | Portail développeur RTE très peu ergonomique |

**Productionisable : oui.** Pour intégrer en mode live dans windcast, il suffit d'ajouter :
1. Les credentials `WINDCAST_RTE_CLIENT_ID` / `WINDCAST_RTE_CLIENT_SECRET` dans `.env`
2. Un client `src/windcast/data/rte_france_api.py` (port sync du pattern wattcast)
3. Un ingest script `scripts/ingest_rte_france_live.py` qui fetch + concat à la parquet existante
4. Un scheduler (cron/hook) pour le refresh quotidien

Le reste du pipeline (schema, QC, features, training, MLflow) est **strictement inchangé** — c'est tout l'intérêt du pattern "parser + canonical schema".

---

## Utilité pour le pitch WeatherNews

Cette analyse nourrit directement une slide **"Production Roadmap"** de la présentation :

> *"Le parser de la démo lit les fichiers annuels RTE. Pour une exécution en production, on remplace la lecture fichier par un client OAuth2 vers `/open_api/consumption/v1/short_term` — 30 minutes de dev, pattern déjà validé dans un projet voisin. Refresh quotidien = 1 appel API. Le reste du pipeline (schema, QC, features, training, MLflow) reste strictement identique. C'est le point clé du framework : l'ingestion est pluggable, tout le reste est figé."*

Message WN : **migration incrémentale, composants remplaçables, pas de big-bang**. Exactement ce qu'ils cherchent pour sortir de MARS (cf. `docs/WNchallenge/CR WeatherNews...`).

---

## Références

- Portail développeur : https://data.rte-france.com/
- Token endpoint : `POST https://digital.iservices.rte-france.com/token/oauth/`
- Base API : `https://digital.iservices.rte-france.com/open_api`
- Client de référence complet : `../wattcast/src/wattcast/data/rte.py` (fetch_load, fetch_generation, fetch_nuclear_unavailability, _chunk_dates, RTEClient)
- Tests de référence : `../wattcast/tests/test_data/test_rte.py` (mocks httpx, token caching, date chunking)
- Licence RTE data : Etalab 2.0 — https://www.etalab.gouv.fr/licence-ouverte-open-licence
