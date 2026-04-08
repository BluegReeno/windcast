---
type: "entretien"
date: "2026-04-01"
opportunite: "[[Senior Data Scientist Energy — WeatherNews]]"
interlocuteurs:
  - "Yoel Chetboun (Resp. Énergie)"
  - "Michel Kolasinski (Head of Ops)"
type_entretien: "Challenge technique — Briefing"
categorie: "Compte-rendu"
feeling: "⏳"
next_step: "Présentation analyse + améliorations itératives la semaine prochaine, en anglais, Craig présent en visio"
---

## Notes clés

### Format de la session
- **Visio 01/04 11h-12h** — Yoel + Michel présents
- L'objectif était de présenter à Renaud leurs **contraintes et challenges** pour observer comment il réfléchit, aborde le problème et structure sa démarche
- Yoel a envoyé la présentation après le call → **"Onshore energy services — Current landscape and challenges"** (PDF, 9 slides)
- Analyse détaillée : [[Analyse Présentation WeatherNews — Challenge Technique — 01-04-2026]]

### Ce qui a été convenu
- Renaud propose de présenter **la semaine prochaine** (visio) son analyse :
  - Sa compréhension du workflow et des problèmes
  - Les **améliorations itératives** qu'il peut proposer pour résoudre leurs problèmes
- **Présentation en anglais** (Craig sera présent en visio)
- Format : RDV visio (date exacte à confirmer)

### Participants à la présentation finale
- **Craig West** (en visio)
- Yoel + Michel (probablement sur place)

## Contenu de la présentation (9 slides)

### 3 domaines présentés avec leurs pipelines

**Wind & Solar Generation Forecasts :**
- Stack : R (analyse) → AWS + scikit-learn + AutoGluon + Python (ML/deploy) → R + Shiny (reporting)
- Sources météo : ECMWF, ICON-EU, AROME
- Algorithmes : catboost, xgboost, bagging/stacking
- **Challenges : accuracy solaire + productivité** (data cleaning, retraining, reporting)

**Gas & Power Demand Forecasts :**
- Stack : R (analyse) → WNI in-house (ML/deploy) → R + Shiny (reporting)
- Sources météo : combinaison in-house WNI
- Algorithme : **MARS uniquement** (Multivariate Adaptive Regression Splines)
- Features : données business passées (H-1, D-1), calendrier, température, vent
- **Challenges : accuracy day-ahead + migration vers nouveau système**

**Specific Use Cases** (Power loss, AI-Weather, DLR) :
- Stack : **100% R** (y compris déploiement !)
- Sources météo : ECMWF, ICON-EU, AROME + sources probabilistes
- **Challenges : productivité + migration vers environnement plus robuste**

### Le challenge demandé (slide 8)
- Identifier des **améliorations incrémentales** (where, how, if possible)
- Proposer une **méthodologie d'implémentation**
- Proposer un **plan de déploiement**
- **Format : OPEN** (libre choix)

## Notes détaillées du call (contexte oral, pas dans les slides)

### Philosophie générale — ce que WN attend

WN est jugé sur la **qualité de ses prévisions**. Lors des phases de trial (compétition pour gagner un client), WN doit proposer le modèle le plus accurate possible dans un **temps limité**. Deux tensions fondamentales :

**Tension 1 : Standardisation vs. Compréhension métier**
- Il faut **optimiser / standardiser la chaîne de traitement** pour dégager du temps à l'ingénieur
- Les améliorations les plus impactantes viennent de la **compréhension des spécificités des données et du contexte client** — du "jus de cervelle" qui réfléchit au problème est souvent plus pertinent que 80 trials Optuna en Bayesian optimization
- L'objectif de la standardisation = **libérer du temps cerveau** pour l'analyse, pas remplacer l'expertise humaine

**Tension 2 : Cadre standardisé vs. Innovation**
- Le cadre ne doit **pas empêcher l'innovation**. WN est jugé sur la qualité → il faut pouvoir **rapidement implémenter de nouvelles technologies, de nouveaux modèles** pour les tester
- Le rythme du ML/IA est rapide — l'infrastructure doit permettre d'itérer et tester vite

### Spécificités par domaine

#### Wind & Solar — Phases de trial
- Chaque client a **ses propres critères d'évaluation** pendant les trials
- Exemple : certains clients veulent être bons **quand le prix spot est élevé** (c'est là que l'erreur fait le plus mal financièrement)
- Conséquence : il faut pouvoir **rapidement implémenter** :
  - De **nouveaux modèles** (ex : un modèle de prix spot en complément)
  - De **nouvelles métriques** (ex : accuracy sur le D+1 à 11h spécifiquement)
  - Pour que le **fine-tuning prenne en compte les spécificités** de la demande client
- → Besoin de **flexibilité rapide** dans le pipeline
- La qualité est **moins bonne en solaire qu'en éolien** — le solaire a ses propres spécificités (nébulosité, etc.)

#### Gas & Power Demand
- C'est de la **prévision de demande** (pas de production)
- WN a accès à **beaucoup d'observations clients** : données au point de livraison, données des transporteurs liées à une zone géographique
- Ce sont des **spécificités fortes** par client/zone → même logique : il faut comprendre le contexte pour bien modéliser

#### Specific Use Cases
- Un client peut demander un truc **vraiment spécifique** : Dynamic Line Rating, Power Loss...
- Il faut **ajouter des données probabilistes à des données physiques**
- **Robustesse de la production** = enjeu clé : si on utilise une nouvelle API pour une feature, que se passe-t-il quand elle plante ?
- Pattern classique : **test sur petite échelle** (un parc) puis **extend rapidement** à tous les parcs / tous les sites de production
- → Besoin de scalabilité et de résilience

### Modèles météo IA (bonus, hors scope principal)
- Mentionné pendant le call : les modèles de prévision météo IA (Nvidia FourCastNet, Google GraphCast, ECMWF AIFS...)
- Il faudrait une infrastructure qui permette de les intégrer rapidement
- **Impression de Renaud : c'est un bonus, probablement hors scope du challenge, mais ça a été mentionné** → montrer qu'on est au courant, sans en faire l'axe principal

### Contraintes d'implémentation

- **Plan incrémental** — pas de big bang. L'équipe fait ~20 personnes avec ses habitudes
- **R est une habitude forte** — on ne peut pas demander à tout le monde de switcher du jour au lendemain
- **Le Japon (siège WNI) recommande vivement AWS** — si la solution proposée est sur AWS, c'est peut-être un avantage politique
- **Mais :** Michel n'a pas forcément l'énergie/l'envie de suivre les recommandations du Japon à la lettre → à doser
- **Technologies mentionnées/à explorer :** AWS SageMaker (workflows ML), fichiers Parquet (standard data), Step Functions, Lambda Functions

## Questions posées
> ⏳ À COMPLÉTER si Renaud a des notes spécifiques

## Impressions / Feeling
> ⏳ À COMPLÉTER par Renaud

## Synthèse des enjeux transversaux (sans proposer de solutions)

1. **Standardiser pour libérer du temps cerveau** — la chaîne doit être fluide pour que l'ingénieur puisse se concentrer sur ce qui fait la différence : comprendre le client
2. **Flexibilité rapide** — pouvoir ajouter un nouveau modèle, une nouvelle métrique, une nouvelle source de données en quelques heures, pas en quelques jours
3. **Scalabilité** — passer d'un test sur 1 parc à une mise en production sur tous les sites rapidement
4. **Robustesse** — gérer les pannes d'API, les sources de données manquantes, la résilience en production
5. **Incrémentalité** — respecter les habitudes de l'équipe (R), pas de révolution, accompagner la transition
6. **Innovation** — ne pas fermer la porte aux nouvelles technologies (modèles météo IA, nouvelles libs ML)
7. **Alignement politique** — AWS est poussé par le Japon, à prendre en compte sans forcer

## Next steps
1. ~~Recevoir la présentation de Yoel~~ ✅ Reçue
2. **Analyser en détail** et préparer la proposition (quelques jours)
3. **Présenter la semaine prochaine** en visio, en anglais, Craig présent
4. Format libre → opportunité de se démarquer avec un livrable structuré et pro
5. Continuer le pipe JS en parallèle — pas de CDI signé = pas d'exclusivité
