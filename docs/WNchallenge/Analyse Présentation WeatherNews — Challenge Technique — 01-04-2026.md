---
type: "entretien"
date: "2026-04-01"
opportunite: "[[Senior Data Scientist Energy — WeatherNews]]"
interlocuteurs:
  - "Yoel Chetboun (Resp. Énergie)"
  - "Michel Kolasinski (Head of Ops)"
type_entretien: "Challenge technique — Analyse document"
categorie: "Analyse"
---

# WeatherNews — Onshore Energy Services: Current Landscape and Challenges

**Source :** Présentation PDF, Weathernews Inc., Mars 2026
**Contexte :** Document remis lors du briefing challenge technique du 01/04/2026
**CR associé :** [[CR WeatherNews — Yoel Chetboun Michel Kolasinski — 01-04-2026]]

---

## Agenda de la présentation

1. Our approach for data science projects
2. Wind & solar generation forecasts
3. Gas & Power demand forecasts
4. Specific use cases
5. Next steps

---

## 1. Approche Data Science WNI

### Trois piliers

| Pilier | Description |
|--------|-------------|
| **WNI Experiences** | Expérience cross-sectorielle (hors énergie) qui enrichit les méthodes |
| **Multi-skilled team** | Équipes R&D solides → génération d'idées nouvelles |
| **WNI approach** | Méthodologie propre développée au fil des projets data-driven |

### Trois phases de projet

| Phase | Description |
|-------|-------------|
| **1. Understanding** | Compréhension business et données |
| **2. Modelling** | Construction d'algorithmes prédictifs |
| **3. Simulating** | Système opérationnel + tests de précision |

### Pipeline en 5 étapes

| # | Étape | Description |
|---|-------|-------------|
| 1 | **Data Cleaning** | Scripts automatisés et/ou algorithmes ML |
| 2 | **Business Data** | Analyse de toutes les données disponibles + Feature Engineering |
| 3 | **Wx Sources** | Données météo issues de modèles traditionnels ET innovants |
| 4 | **Modelling** | Modélisation statistique, innovation + améliorations post-process |
| 5 | **Deployment** | Déploiement des modèles, industrialisation et scaling |

---

## 2. Wind & Solar Generation Forecasts

### Pipeline et stack technologique

| Étape | Activité | Technologies identifiées |
|-------|----------|------------------------|
| **Data analysis / Data cleaning** | Nettoyage et analyse exploratoire | **R** |
| **Wx inputs extraction / Model tuning** | Extraction features météo + tuning modèles | **AWS**, **scikit-learn**, **AutoGluon**, **Python** |
| **Model deployment / Customer delivery** | Déploiement + livraison client | **AWS**, **scikit-learn**, **AutoGluon**, **Python** |
| **Reporting / Continuous improvements / Retraining** | Reporting, amélioration continue, ré-entraînement | **R**, **Shiny** |

### Situation actuelle

- **Sources météo :** ECMWF, ICON-EU, AROME...
- **Features :** wind speed, solar radiation, temperature...
- **Algorithmes :** catboost, xgboost, bagging / stacking

### Challenges identifiés

| Challenge | Détail |
|-----------|--------|
| **Accuracy** | Principalement sur le **solaire** |
| **Productivity** | Data cleaning, système de retraining, reporting |

### Observations clés (analyse Renaud)

- Stack **hybride R + Python** : R pour l'analyse/reporting, Python/AWS pour le ML et le déploiement
- **AutoGluon** (Amazon) utilisé = AutoML pour benchmark rapide de multiples modèles. Indique une approche pragmatique (essayer beaucoup de modèles, garder le meilleur)
- **scikit-learn** présent = modèles classiques ML (pas de deep learning apparent)
- **catboost + xgboost + bagging/stacking** = approche ensemble classique, gradient boosting dominant
- **Pas de deep learning** mentionné (pas de PyTorch, TensorFlow) → opportunité d'amélioration ?
- **AWS** pour le compute et le déploiement
- Le reporting est en **R Shiny** = dashboards interactifs mais technologie vieillissante pour du reporting moderne
- Le problème de précision solaire est **classique** : le solaire est plus difficile que l'éolien à cause de la variabilité des nuages (nébulosité), de l'albédo, et des effets de température sur les panneaux

---

## 3. Gas & Power Demand Forecasts

### Pipeline et stack technologique

| Étape | Activité | Technologies identifiées |
|-------|----------|------------------------|
| **Data analysis / Data cleaning** | Nettoyage et analyse exploratoire | **R** |
| **Wx inputs extraction / Model tuning** | Extraction features + tuning | **WNI in-house** (système propriétaire) |
| **Model deployment / Customer delivery** | Déploiement + livraison client | **WNI in-house** (système propriétaire) |
| **Reporting / Continuous improvements / Retraining** | Reporting et retraining | **R**, **Shiny** |

### Situation actuelle

- **Sources météo :** Combinaison météo **in-house** (mélange propriétaire WNI)
- **Features :** Données business passées (**H-1, D-1**...), calendrier, température, vitesse du vent
- **Algorithme :** **MARS** (Multivariate Adaptive Regression Splines)

### Challenges identifiés

| Challenge | Détail |
|-----------|--------|
| **Accuracy** | Principalement sur les prévisions **day-ahead** |
| **Migration** | Vers un **nouveau système** |

### Observations clés (analyse Renaud)

- **MARS uniquement** = algorithme unique des années 1990 (Friedman, 1991). Solide mais ancien. Pas d'ensemble, pas de boosting → **gros potentiel d'amélioration** en passant à des approches modernes
- Le système est **entièrement propriétaire WNI** pour le tuning et le déploiement → dette technique probable, moins flexible qu'AWS
- Les features **H-1, D-1** (dernière heure, veille) = autorégressif. Classique pour la demande mais attention au data leakage dans les horizons plus longs
- **Migration en cours** → ils savent que le système est obsolète. C'est une opportunité directe pour Renaud : proposer l'architecture cible
- La combinaison météo **in-house** est un atout différenciant de WNI (multi-source, pondérée) — il faut comprendre comment elle fonctionne

---

## 4. Specific Use Cases

### Pipeline et stack technologique

| Étape | Activité | Technologies identifiées |
|-------|----------|------------------------|
| **Data analysis / Data cleaning** | Nettoyage et analyse exploratoire | **R** |
| **Wx inputs extraction / Model tuning** | Extraction features + tuning | **R**, **Shiny** |
| **Model deployment / Customer delivery** | Déploiement + livraison client | **R**, **Docker/serveur** (icône serveur visible) |
| **Reporting / Continuous improvements / Retraining** | Reporting et retraining | **R**, **Shiny** |

### Services mentionnés

- **Power loss forecasts** — prévision de pertes de production
- **"AI-Weather forecasts"** — prévisions météo augmentées IA
- **DLR** (Dynamic Line Rating) — capacité dynamique des lignes électriques selon la météo

### Situation actuelle

- **Sources météo :** ECMWF, ICON-EU, AROME + **sources probabilistes**
- **Features :** température, autres données business
- **Algorithmes :** "state-of-the-art" (non précisé)

### Challenges identifiés

| Challenge | Détail |
|-----------|--------|
| **Productivity** | Process trop manuels / lents |
| **Migration** | Vers un environnement **plus robuste** |

### Observations clés (analyse Renaud)

- **Stack 100% R** (y compris pour le déploiement !) → fragile pour de la production. R n'est pas fait pour du déploiement scalable
- **Shiny pour le model tuning** = interface interactive mais pas industrielle
- Le DLR est un **produit à fort potentiel** (la transition énergétique pousse les gestionnaires de réseau à optimiser les lignes existantes plutôt qu'en construire de nouvelles)
- Les **sources probabilistes** sont un signe de maturité — ils ne se contentent pas de la prévision déterministe
- L'aveu "migration to more robust environment" confirme qu'ils savent que le stack R-only n'est pas viable en production

---

## 5. Next Steps (le challenge)

### Ce qui est demandé

1. **Q&A session** — questions-réponses (fait pendant le call du 01/04)
2. **Challenge proposal :**
   - **Identifier des améliorations incrémentales** (where, how, if possible)
   - **Méthodologie d'implémentation**
   - **Plan de déploiement**
3. **Format de la proposition : OPEN** (libre choix)

### Ce que ça signifie concrètement

Renaud doit livrer une proposition qui couvre :
- Quelles améliorations concrètes sur chaque pipeline (Wind/Solar, Gas/Power, Use Cases)
- Comment les implémenter (par quoi on commence, quels outils, quelle méthode)
- Comment les déployer (architecture, timeline, risques)
- Le format est libre → opportunité de se démarquer avec un livrable structuré et pro

---

## Synthèse : Cartographie complète des technologies

### Par domaine

| Domaine | Analyse | ML / Tuning | Déploiement | Reporting |
|---------|---------|-------------|-------------|-----------|
| **Wind & Solar** | R | AWS, scikit-learn, AutoGluon, Python | AWS, scikit-learn, AutoGluon, Python | R, Shiny |
| **Gas & Power Demand** | R | WNI in-house | WNI in-house | R, Shiny |
| **Specific Use Cases** | R | R, Shiny | R, Docker/serveur | R, Shiny |

### Stack complète identifiée

| Technologie | Rôle | Où |
|------------|------|-----|
| **R** | Analyse de données, nettoyage, reporting, et même déploiement (use cases) | Partout |
| **Shiny** | Dashboards interactifs, reporting | Partout (reporting) + tuning (use cases) |
| **Python** | ML, scripting | Wind & Solar |
| **scikit-learn** | Modèles ML classiques | Wind & Solar |
| **AutoGluon** | AutoML (benchmark multi-modèles) | Wind & Solar |
| **catboost** | Gradient boosting (Yandex) | Wind & Solar |
| **xgboost** | Gradient boosting | Wind & Solar |
| **AWS** | Cloud compute + déploiement | Wind & Solar |
| **MARS** | Regression splines | Gas & Power Demand |
| **WNI in-house** | Système propriétaire (Wx combo + déploiement) | Gas & Power Demand |

### Sources météo

| Source | Type | Où utilisée |
|--------|------|-------------|
| **ECMWF** | Modèle global (European Centre) | Wind/Solar + Use Cases |
| **ICON-EU** | Modèle régional (DWD, Allemagne) | Wind/Solar + Use Cases |
| **AROME** | Modèle haute résolution (Météo-France) | Wind/Solar + Use Cases |
| **Sources probabilistes** | Ensembles / probabilités | Use Cases |
| **WNI in-house combination** | Mélange propriétaire multi-sources | Gas & Power Demand |

### Algorithmes

| Algorithme | Type | Domaine |
|------------|------|---------|
| **catboost** | Gradient boosting (gère bien les catégorielles) | Wind & Solar |
| **xgboost** | Gradient boosting (référence ML tabulaire) | Wind & Solar |
| **Bagging / Stacking** | Ensemble methods | Wind & Solar |
| **MARS** | Regression splines (Friedman 1991) | Gas & Power Demand |
| **"State-of-the-art"** | Non précisé | Use Cases |

---

## Contexte oral du call (infos hors slides)

### La tension fondamentale chez WN

Deux forces opposées à réconcilier :

**1. Standardiser pour libérer du temps cerveau**
- En phase de trial (compétition client), WN doit proposer le modèle le plus accurate dans un temps limité
- Les gains d'accuracy les plus significatifs viennent de la **compréhension des spécificités client/données** — du jus de cervelle qui réfléchit au problème > 80 trials Optuna en Bayesian optimization
- Objectif : automatiser tout ce qui est mécanique pour **libérer du temps à l'ingénieur** pour qu'il comprenne les spécificités du projet

**2. Ne pas empêcher l'innovation**
- WN est jugé sur la qualité → il faut pouvoir rapidement implémenter de nouvelles technologies, de nouveaux modèles pour les tester
- Le ML/IA avance vite — l'infra doit permettre d'itérer rapidement

### Spécificités Wind & Solar — Phases de trial

- Chaque client a **ses propres critères d'évaluation**
- Exemple concret : un client veut être bon quand le **prix spot est élevé** (c'est là que l'erreur coûte le plus cher)
- Conséquences opérationnelles :
  - Besoin d'implémenter rapidement de **nouveaux modèles** (ex : modèle de prix spot en complément)
  - Besoin d'implémenter rapidement de **nouvelles métriques** (ex : accuracy sur le D+1 à 11h spécifiquement)
  - Le **fine-tuning doit pouvoir intégrer les spécificités** de chaque demande client
- → Besoin de **flexibilité rapide** dans le pipeline
- Solaire moins bon que éolien — chaque secteur a ses spécificités propres

### Spécificités Gas & Power Demand

- C'est de la **prévision de demande** (pas de production)
- WN a accès à **beaucoup d'observations clients** :
  - Données au **point de livraison**
  - Données des **transporteurs**, liées à une **zone géographique**
- Ce sont des **spécificités fortes** par client/zone → même logique : comprendre le contexte > tuner aveuglément

### Spécificités Use Cases (DLR, Power Loss...)

- Demandes clients **très spécifiques** (Dynamic Line Rating, Power Loss...)
- Il faut **ajouter des données probabilistes à des données physiques**
- Enjeu de **robustesse en production** :
  - On utilise une nouvelle API pour une feature → que se passe-t-il quand elle plante ?
  - Pattern : test sur **petite échelle** (1 parc) puis **extend rapidement** à tous les parcs / sites de production
- → Besoin de scalabilité + résilience

### Modèles météo IA (bonus, mentionné mais probablement hors scope)

- Les modèles de prévision météo IA (Nvidia FourCastNet, Google GraphCast, ECMWF AIFS...) ont été mentionnés
- Il faudrait une infra qui permette de les intégrer
- **Impression Renaud : c'est un bonus à mentionner, pas l'axe principal du challenge**

### Contraintes d'implémentation

- **Plan incrémental** — pas de big bang, équipe de ~20 personnes avec ses habitudes
- **R est une habitude forte** — transition progressive, pas de rupture
- **Le Japon (siège WNI) recommande vivement AWS** — avantage politique si la solution est sur AWS
- **Mais Michel n'a pas forcément l'énergie de suivre le Japon** → à doser dans la proposition
- **Technologies mentionnées :** AWS SageMaker (workflows ML), fichiers Parquet (standard data actuel), Step Functions, Lambda Functions

---

## Analyse transversale : pain points et enjeux

### Pain points (slides)

1. **Productivité** — mentionné dans 3 domaines sur 3. Process trop manuels, data cleaning chronophage, retraining non automatisé
2. **Accuracy** — solaire (nuages) + day-ahead gas/power
3. **Migration technique** — Gas/Power sur système WNI legacy, Use Cases sur R-only en production
4. **R-dépendance** — R utilisé partout, y compris là où ce n'est pas adapté (déploiement, production)

### Enjeux transversaux (oral)

1. **Standardiser pour libérer du temps cerveau** — la chaîne doit être fluide pour que l'ingénieur se concentre sur ce qui fait la différence
2. **Flexibilité rapide** — pouvoir ajouter un nouveau modèle, une nouvelle métrique, une nouvelle source de données rapidement
3. **Scalabilité** — passer d'un test sur 1 parc à une mise en production sur tous les sites
4. **Robustesse** — gérer les pannes d'API, les sources de données manquantes, la résilience
5. **Incrémentalité** — respecter les habitudes de l'équipe (R), accompagner la transition
6. **Innovation** — ne pas fermer la porte aux nouvelles technologies (modèles météo IA, nouvelles libs ML)
7. **Alignement politique** — AWS poussé par le Japon, à prendre en compte sans forcer

### Opportunités d'amélioration identifiées (premières pistes, à affiner)

| Axe | Amélioration possible | Impact |
|-----|----------------------|--------|
| **Gas/Power : MARS → ensemble models** | Remplacer MARS par xgboost/catboost/LightGBM + stacking | Fort (accuracy day-ahead) |
| **Gas/Power : migration système** | Architecture AWS/Python comme Wind/Solar | Fort (productivité + maintenabilité) |
| **Wind/Solar : accuracy solaire** | Ajouter des features nuages (satellite, nowcasting), deep learning (LSTM/Transformer) | Moyen-Fort |
| **Use Cases : sortir de R** | Migrer le déploiement vers Python/Docker/AWS | Fort (robustesse) |
| **Transversal : pipeline automatisé** | CI/CD pour le retraining, monitoring de drift, alertes | Fort (productivité) |
| **Transversal : reporting moderne** | Remplacer Shiny par des dashboards modernes (Streamlit, Grafana, ou outil BI) | Moyen |
| **Transversal : feature store** | Centraliser les features météo pour tous les domaines | Moyen (réutilisabilité) |
