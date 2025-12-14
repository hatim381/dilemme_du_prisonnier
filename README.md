# 🎮 Dilemme du Prisonnier : IA vs Stratégies Codées

**Analyse narrative complète** des comportements émergents quand agents génératifs (LLM) rencontrent stratégies déterministes dans le classique **Prisoner's Dilemma**.

> Exploration empirique de la coopération, de l'apprentissage implicite, et des équilibres observés chez les modèles de langage vs. stratégies optimales codées.

---

## 📚 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Démarrage rapide](#démarrage-rapide)
3. [Architecture du projet](#architecture-du-projet)
4. [Installation & Configuration](#installation--configuration)
5. [Exécution](#exécution)
6. [Structure des données](#structure-des-données)
7. [Dashboard Streamlit](#dashboard-streamlit)
8. [Résultats clés](#résultats-clés)
9. [Customisation](#customisation)

---

## 👁️ Vue d'ensemble

### 🎯 Objectif

Comparer et analyser les **stratégies émergentes** dans le dilemme du prisonnier répété :

| Aspect | IA (Qwen, Gemma) | Codé |
|--------|---|---|
| **Comportement** | Adaptatif, sensible contexte | Déterministe, pré-défini |
| **Variabilité** | Température, context-aware | Aucune variation |
| **Apprentissage** | Implicite dans les tokens | Pas d'apprentissage |

### ❓ Questions de recherche

- ✓ La coopération émerge-t-elle naturellement par répétition ? (hypothèse Axelrod)
- ✓ L'IA suit-elle les rôles assignés ou émerge-t-elle des stratégies propres ?
- ✓ Impact de la température et du contexte sur les comportements IA ?
- ✓ Comment les équilibres observés se comparent-ils à Nash/Pareto ?

### 📊 Dataset en bref

- **283,200 rounds** à travers **1,416 matchs**
- **17 agents distincts** (combinaisons famille × rôle)
- Distribution : ~62% Qwen, ~21% Gemma, ~17% Codé
- **29 colonnes** : coopération, scores, température, contexte, conformité, etc.
- Taille : ~180 MB (enriched_games_full.parquet)
---

## ⚡ Démarrage rapide

```bash
# 1. Cloner et installer
git clone https://github.com/hatim381/dilemme_du_prisonnier.git
cd dilemme_du_prisonnier
pip install -r requirements.txt

# 2. Lancer le dashboard
streamlit run streamlit_report.py

# 3. Ouvrir dans le navigateur
# http://localhost:8501
```


```
dilemme_du_prisonnier/
│
├── 🎯 POINT D'ENTRÉE
│   └── streamlit_report.py              ⭐ Dashboard analytique (7 pages narratives)
│
├── 🎮 MOTEUR & EXPÉRIENCES
│   ├── game_engine.py                   Logique jeu (payoffs, rounds, matchs)
│   ├── run_experiment.py                Lancer expériences uniques
│   └── run_batch_parallel_turbo.py      Exécution parallèle multiprocessing
│
├── 📓 NOTEBOOKS
│   └── transform.ipynb                  Transformation & exploration données
│
├── 💾 DATA
│   └── enriched_data/
│       └── enriched_games_full.parquet  283,200 rows × 29 colonnes
│
├── 📊 RÉSULTATS
│   └── results/                         Outputs bruts (.parquet)
│
└── 📖 DOCUMENTATION
    ├── README.md                        ← Vous êtes ici
    ├── README_parallel.md               Guide parallélisation
    └── Lien_sujet_notion.txt            Sujet complet (Notion)
```

**Fichiers essentiels :**
- ⭐ `streamlit_report.py` — **À lancer en priorité**
- 💾 `enriched_data/enriched_games_full.parquet` — Source données principale
- 📓 `transform.ipynb` — Transformation & analyses données

---

## 🚀 Installation & Configuration

### Prérequis

- **Python 3.9+**
- pip ou conda
- ~2GB disque libre (données)

### Étapes d'installation

```bash
# 1️⃣ Cloner le repo
git clone https://github.com/hatim381/dilemme_du_prisonnier.git
cd dilemme_du_prisonnier

# 2️⃣ Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate              # Linux/Mac
# ou
.venv\Scripts\activate                 # Windows

# 3️⃣ Installer dépendances
pip install -r requirements.txt

# 4️⃣ Vérifier les données
python -c "import pandas as pd; df = pd.read_parquet('enriched_data/enriched_games_full.parquet'); print(f'✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols')"
```

---

## ⚙️ Exécution

### 1️⃣ Dashboard Streamlit (Recommandé)

```bash
streamlit run streamlit_report.py
```

👉 **Ouvrir** : http://localhost:8501

#### 📖 Les 7 Pages du Dashboard

| # | Page | Contenu |
|---|------|---------|
| 1 | 🌍 **Vue Globale** | Architecture générale, leaderboard agents, KPIs |
| 2 | 🤝 **Coopération & Motifs** | Taux par type, température, contexte, agent (4 tabs) |
| 3 | 🏆 **Performance & Efficacité** | Scores, corrélations, variabilité, effets |
| 4 | 🌡️ **Facteurs IA** | Température, modèle, contexte (heatmaps) |
| 5 | ⚡ **Dynamique Temporelle** | Évolution coopération par round, phases émergentes |
| 6 | 📈 **Théorie & Équilibres** | Matrice CC/CD/DC/DD, Nash vs. observé |
| 7 | 🎯 **Synthèse Finale** | Comparaison IA vs. Codé, insights clés, recommandations |

---

### 2️⃣ Notebook Transformation & Analyses


```bash
jupyter notebook transform.ipynb
```

**Contient :**
- Chargement & inspection des données (schéma, types)
- Transformations & features dérivées
- Analyses descriptives (coop rates, scores, variance)
- Clustering comportemental & profils d'agents
- Analyses IA : température, contexte, modèle
- Dynamiques temporelles et stabilité

---

### 3️⃣ Ré-exécuter les expériences

```bash
# Expérience simple (1 seed)
python run_experiment.py

# Batch parallèle optimisé (multiprocessing)
python run_batch_parallel_turbo.py
```

👉 Consulter `README_parallel.md` pour l'optimisation avancée

---

## 📊 Structure des données

### Aperçu

| Métrique | Valeur |
|----------|--------|
| Nombre de lignes | 283,200 rounds |
| Nombre de colonnes | 29 features |
| Nombre de matchs | 1,416 |
| Nombre d'agents distincts | 17 |
| Taille du fichier | ~180 MB (parquet) |
| Période de collecte | Complète |

### Colonnes principales (29 total)

**Identifiants & Structure**
- `match_id` : Identifiant unique du match
- `round_id` : Numéro du round dans le match

**Agent 1 (& idem pour Agent 2)**
- `agent1_name` : Nom (famille + rôle, ex: "Qwen_Cooperator")
- `agent1_family` : Famille ("qwen", "gemma", "coded")
- `agent1_is_cooperation` : Mouvement (1=Coopération, 0=Défection)
- `agent1_match_score` : Score du round (0-1000)
- `agent1_temperature_bucket` : Température ("low", "medium", "high", "coded")
- `agent1_context_used_flag` : Contexte fourni (1=oui, 0=non)
- `agent1_conformity_score` : Alignement rôle (0-1)

**Features dérivées (dans Streamlit)**
```python
df["agent1_move"] = df["agent1_is_cooperation"].map({1: "C", 0: "D"})
df["agent1_is_ia"] = df["agent1_family"].str.contains("qwen|gemma")
df["outcome"] = df["agent1_move"] + df["agent2_move"]  # CC, CD, DC, DD
df["total_score"] = df["agent1_match_score"] + df["agent2_match_score"]
```

---

## 📈 Dashboard Streamlit – Détails

### Page 1 : 🌍 Vue Globale

**Métriques clés (KPIs) :**
- Total mouvements IA / Codés
- Taux coopération global par type
- Nombre de rounds/matchs

**Visualisations :**
- 📊 Distribution rounds par famille (bar chart)
- 🏆 Top 15 agents par score moyen (horizontal bar chart)

---

### Page 2 : 🤝 Coopération & Motifs

**4 onglets interactifs :**
1. **Par type d'agent** : Taux coop Qwen vs. Gemma vs. Codé
2. **Par température** : Impact low/medium/high sur coopération (IA uniquement)
3. **Par contexte** : Avec/sans prompting fourni
4. **Par agent** : Détail individuel (17 agents)

**Visualisation** : Pie charts + barplots avec tendances

---

### Page 3 : 🏆 Performance & Efficacité

**3 onglets :**
1. **Score moyen** : Bar chart avec std deviation
2. **Score vs Coopération** : Scatter plot (révèle non-linéarité)
3. **Variabilité** : Boxplot distribution scores par type

**Insight clé** : Pas de corrélation linéaire score ↔ coopération

---

### Page 4 : 🌡️ Facteurs IA

**Exploration paramètres IA :**
- **Heatmap** : Température × Conformité
- **Comparaison** : Qwen vs. Gemma (coopération, score, stabilité)
- **Contexte** : Impact contexte → Score et Coopération (2 bar charts)

---

### Page 5 : ⚡ Dynamique Temporelle

**Évolution par round :**
- 📈 Ligne agent 1 & agent 2 coopération + moving average
- 🔍 Identification phases : Amorce → Exploration → Stabilisation → Équilibre
- 📊 Patterns d'émergence

---

### Page 6 : 📈 Théorie & Équilibres

**Axelrod revisité :**
- **Heatmap** : Fréquences outcomes CC/CD/DC/DD
- **Scatter** : Coopération vs. Score + trend line
- **Analyse** : Équilibre observé vs. Nash vs. Pareto optimum

---

### Page 7 : 🎯 Synthèse Finale

**Tableau comparatif + Insights narratifs**
- 📊 Codé vs. IA : transparent/stable vs. opaque/variable
- 🔑 **4 insights structurés** :
  1. Rôle du modèle LLM
  2. Impact température & contexte
  3. Dynamique d'apprentissage implicite
  4. Implications théoriques
- 💡 Recommandations pratiques

---

## 📁 Fichiers principaux

### `streamlit_report.py` (1000+ lignes)
Dashboard production avec 7 pages narratives, soft color palette (#6B9BD1, #A8B39F), custom CSS.

### `game_engine.py`
Moteur du jeu : logique interaction agents, calcul payoffs, gestion rounds/matchs.

### `run_experiment.py`
Launcher simple pour expériences uniques. Arguments CLI pour configurer agents/rounds.

### `run_batch_parallel_turbo.py`
Exécution parallèle optimisée avec `multiprocessing.Pool`. Gère batches agents automatiquement.

### `transform.ipynb`
Notebook de transformation & exploration :
- Chargement & inspection données (schéma, types)
- Transformations features (dérivées, agrégations)
- Analyses descriptives (coop rates, scores, variance)
- Profils comportementaux (clustering)
- Analyses spécifiques IA (température, contexte)
- Dynamiques temporelles & stabilité

---

## 🎯 Résultats clés

### Coopération par type d'agent

| Type | Taux Coopération | Contexte |
|------|---|---|
| **Qwen** | ~56% | ✅ Modèle stable, coopératif |
| **Gemma** | ~34% | ⚠️ Modèle moins coopératif |
| **Codé** | ~55% | ✅ Déterministe, proche Qwen |

**💡 Insight** : Le modèle LLM influe **plus** que le rôle spécifié. Qwen émerge naturellement coopératif malgré contexte neutre.

---

### Performance & Efficacité

**Observation clé :** ❌ **Pas de corrélation linéaire** entre coopération et score

- Agents optimaux : **coopération modérée + réactivité stratégique**
- Distribution outcomes observée : 
  - ~40% CC (mutual cooperation)
  - ~47% DD (mutual defection)
  - ~13% CD + DC (mixed)
- **87% d'états "purs"** (CC ou DD) → peu de patterns mixtes stables

---

### Dynamique temporelle

- ✅ Coopération **stable** (variation < 5% entre début/fin)
- 🔄 **Phases distinctes** :
  1. Amorce (rounds 1-5)
  2. Exploration (rounds 6-15)
  3. Stabilisation (rounds 16-25)
  4. Équilibre (rounds 26+)
- ❌ **Pas d'émergence progressive Axelrod-like** (pas d'apprentissage visible)

---

### Facteurs IA (Température & Contexte)

| Facteur | Impact | Observation |
|---------|--------|---|
| **Température** | Variabilité ↑ | Crée variance, réduit conformité au rôle |
| **Contexte fourni** | Coopération ↑ | +5 à 10% quand prompting explicite |
| **Modèle (Qwen vs Gemma)** | Stabilité ↑ | Qwen > Gemma en adaptabilité |

**Implication** : L'IA a une "préférence induite" pour la coopération, réactivité au contexte mais pas d'apprentissage explicite par round.

---

## 🛠️ Customisation & Développement

### Modifier les couleurs du dashboard

Éditer le dictionnaire `COLORS` dans `streamlit_report.py` (ligne ~15) :

```python
COLORS = {
    "blue": "#6B9BD1",      # Bleu principal
    "green": "#A8B39F",     # Vert foncé
    "red": "#D5636D",       # Rouge
    "yellow": "#E3B448",    # Jaune
    # ... ajouter/modifier couleurs
}
```

### Ajouter une nouvelle page d'analyse

```python
# Dans streamlit_report.py

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir page", ["Page 1", "Page N"])

if page == "Page N":
    # 1. Charger données enrichies
    df = load_and_prepare_data()
    
    # 2. Agrégations/transformations
    data_agg = df.groupby("agent1_family")["agent1_is_cooperation"].mean()
    
    # 3. Visualisation Plotly
    fig = px.bar(data_agg, title="Mon analyse")
    st.plotly_chart(fig, use_container_width=True)
```

### Changer la source de données

Modifier le chemin du fichier parquet :

```python
# Dans load_and_prepare_data() fonction
def load_and_prepare_data():
    df = pd.read_parquet("enriched_data/enriched_games_full.parquet")
    # ↓ remplacer par votre chemin ↓
    # df = pd.read_parquet("path/to/your/data.parquet")
    return df
```

### Exécution parallèle avancée

Consulter `README_parallel.md` pour :
- Batch parallelization
- Multiprocessing Pool
- Gestion ressources & timeouts
- Benchmarks performance

---

## 📚 Références théoriques

### Concepts clés

**Axelrod's Prisoner's Dilemma** (Axelrod, 1984)
- Étude du comportement coopératif en répétition
- Stratégie "Tit-for-Tat" : imiter l'adversaire à chaque round
- Émergence naturelle de coopération sans communication
- 📖 *The Evolution of Cooperation*

**Équilibres de Nash**
- Situation où aucun agent ne peut améliorer **seul** son gain
- Dilemme du prisonnier : Nash = (DD, DD) → sous-optimal collectivement
- CC serait Pareto optimal mais instable

**Température (LLM)**
- Contrôle la "créativité" du modèle (0 = déterministe, 1+ = stochastique)
- Impact observé : variabilité comportementale

**Conformité au rôle**
- Score 0-1 mesurant alignement entre rôle assigné et actions réelles
- Indicateur de "suivi des instructions"

---

## 📞 Support & FAQ

### ❓ Questions fréquentes

**Q: Comment relancer les expériences ?**
```bash
python run_batch_parallel_turbo.py  # Multiprocessing (recommandé)
```
Voir `README_parallel.md` pour détails.

---

**Q: Quels fichiers dois-je modifier pour adapter le code ?**
- `streamlit_report.py` → Dashboard
- `game_engine.py` → Logique jeu
- `run_experiment.py` → Paramètres expérience

---

**Q: Où sont les résultats bruts des expériences ?**
- `results/` → Fichiers .parquet bruts
- `enriched_data/enriched_games_full.parquet` → Dataset enrichi prêt pour analyse

---

**Q: Comment tracer un debug ?**
1. Consulter `transform.ipynb` pour exploration données
2. Vérifier `game_engine.py` pour logique jeu
3. Ajouter `st.write()` dans `streamlit_report.py` pour inspection

---

### 🐛 Troubleshooting

| Problème | Solution |
|----------|----------|
| `FileNotFoundError: enriched_games_full.parquet` | Vérifier chemin données (relativement au pwd) |
| Streamlit ne démarre pas | Vérifier port 8501 libre, relancer avec `--server.port 8502` |
| Erreur import `duckdb` | `pip install duckdb` |
| Memory error (283K rows) | Augmenter limite RAM ou filtrer données |

---

## 📄 Licence & Auteur

### Licence

Ce projet est **open source**. Libre d'utilisation, modification et distribution avec attribution.

### Auteur

**Bill H.** — Master Data Lakes & Data Integrations, EFREI Paris (2024-2025)

**Contact** : [GitHub](https://github.com/hatim381/)

---

## 🔄 Historique & Statut

| Date | Statut | Notes |
|------|--------|-------|
| Décembre 2024 | ✅ **Production Ready** | Dashboard complet, données enrichies validées |
| Décembre 2025 | ✅ **Maintenu** | Documentation à jour, tous fichiers présents |

---

**Version**: 1.0  
**Dernière mise à jour**: Décembre 2025  
**Statut**: ✅ Stable & Fonctionnel
