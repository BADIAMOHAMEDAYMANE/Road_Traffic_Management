# 🚦 Traffic Urbain — Q-Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/Licence-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Statut-Phase%201-F59E0B?style=for-the-badge)
![RL](https://img.shields.io/badge/RL-Q--Learning-8B5CF6?style=for-the-badge)

> Système de gestion intelligente des feux de circulation par apprentissage par renforcement.  
> **Phase 1** : un carrefour · un agent · MDP discret.

---

## 📋 Table des matières

- [Contexte](#-contexte)
- [Modélisation MDP](#-modélisation-mdp)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Interface Streamlit](#-interface-streamlit)

---

## 🎯 Contexte

Ce projet s'inscrit dans le domaine de l'**intelligence artificielle distribuée** et des **systèmes multi-agents**. L'objectif est de résoudre un problème réel : la gestion du trafic urbain à travers des feux de circulation intelligents.

**Phase 1 — Simplification :**
- 2 directions : Nord/Sud (NS) et Est/Ouest (EO)
- Files d'attente discrètes : 0 à 3 voitures maximum
- 2 actions : GARDER ou CHANGER la phase du feu

---

## 🧠 Modélisation MDP

| Composant | Définition | Justification |
|-----------|-----------|---------------|
| **État** | `(file_NS, file_EO, phase)` | 4×4×2 = 32 états — espace tractable, convergence rapide |
| **Actions** | `0 = GARDER` · `1 = CHANGER` | Binaire, sans ambiguïté |
| **Reward** | `-(file_NS + file_EO)` | Signal dense, pénalise les files à chaque step |
| **γ (gamma)** | `0.95` | Valorise le futur proche, garantit la convergence |
| **Arrivées** | Loi de Poisson (λ_NS=0.5, λ_EO=0.4) | Modèle réaliste de flux aléatoire |

### Hyperparamètres Q-Learning

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `α` (learning rate) | `0.1` | Stable sur environnements stochastiques |
| `γ` (discount) | `0.95` | Horizon ~20 steps, convergence garantie |
| `ε_start` | `1.0` | Exploration totale au départ |
| `ε_end` | `0.01` | 1% d'exploration résiduelle |
| `ε_decay` | `0.995` | ε ≈ 0.01 après ~920 épisodes |

---

## 📁 Structure du projet

```
Traffic_Urbain/
│
├── simulation.py       # Environnement MDP (carrefour, Poisson, transitions)
├── agent.py            # Agent Q-Learning (table Q, ε-greedy, mise à jour)
├── utils.py            # Baselines (Fixed-Time, Greedy) + métriques
├── train.py            # Boucle d'entraînement (1000 épisodes)
├── evaluate.py         # Évaluation post-entraînement + analyse politique
├── run_all.py          # Script maître (entraîne + évalue)
├── app.py              # Interface Streamlit (simulation + comparaison)
├── requirements.txt    # Dépendances Python
├── .gitignore
├── README.md
└── results/            # Généré automatiquement
    ├── q_table.npy
    ├── training_data.json
    └── eval_data.json
```

---

## ⚙️ Installation

```bash
# Cloner le projet
git clone https://github.com/votre-username/Traffic_Urbain.git
cd Traffic_Urbain

# Installer les dépendances
pip3 install -r requirements.txt
```

**Dépendances :**

![NumPy](https://img.shields.io/badge/numpy-%3E%3D1.24-013243?style=flat-square&logo=numpy)
![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.32-FF4B4B?style=flat-square&logo=streamlit)
![Pandas](https://img.shields.io/badge/pandas-%3E%3D2.0-150458?style=flat-square&logo=pandas)

---

## 🚀 Utilisation

### 1. Entraîner l'agent

```bash
python3 run_all.py
```

Lance automatiquement :
- L'entraînement Q-Learning (1000 épisodes)
- L'évaluation comparative vs baselines
- La sauvegarde de la table Q dans `results/`

### 2. Lancer l'interface

```bash
python3 -m streamlit run app.py
```

### 3. Exécuter les modules séparément

```bash
# Entraînement seul
python3 train.py

# Évaluation seule (nécessite un entraînement préalable)
python3 evaluate.py
```

---

## 📊 Résultats

Résultats obtenus après 1000 épisodes (100 steps/épisode) :

| Stratégie | Récompense moy | Voitures/step | Gain |
|-----------|---------------|---------------|------|
| **Q-Learning** | **−187** | **1.87** | — |
| Greedy Baseline | −224 | 2.24 | −16.5% |
| Fixed-Time (5 steps) | −277 | 2.77 | −32.4% |

![Q-Learning](https://img.shields.io/badge/Q--Learning%20vs%20Fixed--Time-%2B32.4%25-22C55E?style=for-the-badge)
![Q-Learning](https://img.shields.io/badge/Q--Learning%20vs%20Greedy-%2B16.5%25-22C55E?style=for-the-badge)

---

## 🖥️ Interface Streamlit

L'interface permet de comparer les 3 stratégies en temps réel :

- ✅ Sélection des stratégies à comparer (Q-Learning, Fixed-Time, Greedy)
- ✅ Courbe des files d'attente superposées
- ✅ Courbe de récompense cumulée
- ✅ Tableau de statistiques comparatif
- ✅ Gain Q-Learning affiché automatiquement
- ✅ Journal détaillé par stratégie + export CSV
- ✅ Seed configurable pour reproductibilité

---

## ⚠️ Erreurs classiques évitées

| Risque | Solution appliquée |
|--------|-------------------|
| Espace d'états trop grand | Discrétisation 0–3 voitures (32 états) |
| Mauvaise fonction de reward | `-(NS + EO)` — signal dense et clair |
| Mauvaise architecture | `simulation.py` et `agent.py` strictement séparés |
| α trop grand | `α = 0.1` — stable sur Poisson |
| Pas d'analyse | Courbes + comparaison + table Q commentée |
| Usage d'une lib RL | 100% from scratch — Q-Learning codé manuellement |

---

## 📌 Prochaines phases

![Phase 2](https://img.shields.io/badge/Phase%202-Multi--carrefours-6366F1?style=flat-square)
![Phase 3](https://img.shields.io/badge/Phase%203-Deep%20Q--Network-EC4899?style=flat-square)
![Phase 4](https://img.shields.io/badge/Phase%204-Coordination%20Multi--agents-F97316?style=flat-square)

---

<p align="center">
  Fait avec ❤️ dans le cadre d'un projet IA distribuée
</p>