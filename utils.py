"""
utils.py
========
Stratégies baseline et fonctions utilitaires.

BASELINE — "Fixed-Time" :
  → Changer le feu toutes les N secondes (ici steps), indépendamment des files.
  → C'est la stratégie la plus répandue dans les vraies villes (avant IA).
  → Sert de référence minimale : Q-learning DOIT faire mieux.
  → Si Q-learning ≈ baseline → hyperparamètres ou reward mal réglés.

BASELINE — "Greedy" :
  → Toujours donner le vert à la file la plus longue.
  → Plus intelligent que fixed-time, mais myope (0 vision future).
  → Plafond heuristique à dépasser.
"""

import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Stratégies Baseline
# ─────────────────────────────────────────────────────────────────────────────

class FixedTimeBaseline:
    """
    Stratégie à temps fixe : change de phase toutes les `interval` steps.

    Parameters
    ----------
    interval : int  nombre de steps entre chaque changement
    """

    def __init__(self, interval=5):
        self.interval = interval
        self._step_count = 0

    def choose_action(self, state):
        self._step_count += 1
        if self._step_count % self.interval == 0:
            return 1  # changer
        return 0      # garder

    def reset(self):
        self._step_count = 0

    def __repr__(self):
        return f"FixedTimeBaseline(interval={self.interval})"


class GreedyBaseline:
    """
    Stratégie gloutonne : donne le vert à la file la plus longue.

    - Si NS > EO et phase == 1 → changer (mettre NS au vert)
    - Si EO > NS et phase == 0 → changer (mettre EO au vert)
    - Sinon → garder
    """

    def choose_action(self, state):
        queue_ns, queue_eo, phase = state
        if queue_ns > queue_eo and phase == 1:
            return 1  # passer NS au vert
        if queue_eo > queue_ns and phase == 0:
            return 1  # passer EO au vert
        return 0      # garder

    def reset(self):
        pass

    def __repr__(self):
        return "GreedyBaseline"


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de métriques
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(data, window=50):
    """Moyenne glissante sur une série 1D."""
    if len(data) < window:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def episode_stats(rewards):
    """Retourne mean, std, min, max sur une liste de récompenses."""
    arr = np.array(rewards)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "total":  float(arr.sum()),
    }


def compute_throughput(history):
    """
    Calcule le débit moyen : 1 - (waiting / max_possible).
    history : liste de dicts {"total_waiting": int, "max_queue": int}
    """
    if not history:
        return 0.0
    max_possible = sum(h["max_possible"] for h in history)
    total_waiting = sum(h["total_waiting"] for h in history)
    return 1.0 - total_waiting / max(max_possible, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires d'affichage console
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(label, rewards_per_episode, n_last=100):
    arr = np.array(rewards_per_episode[-n_last:])
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Épisodes total     : {len(rewards_per_episode)}")
    print(f"  Récompense moyenne : {arr.mean():.2f}")
    print(f"  Récompense std     : {arr.std():.2f}")
    print(f"  Récompense min     : {arr.min():.2f}")
    print(f"  Récompense max     : {arr.max():.2f}")
    print(f"{'─'*50}")