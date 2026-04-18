"""
agent.py
========
Agent Q-Learning pour la gestion de feux de circulation.

Paramètres choisis — Justifications :
──────────────────────────────────────
α (learning rate) = 0.1 :
  → Petit pas → mises à jour stables.
  → α=0.5 diverge sur environnements stochastiques (arrivées Poisson).
  → α=0.01 trop lent à converger.

γ (discount) = 0.95 :
  → Voir simulation.py pour justification.

ε-greedy avec decay :
  → ε_start=1.0 : exploration totale au départ (aucune connaissance a priori).
  → ε_end=0.01  : 1 % d'exploration résiduelle (robustesse aux non-stationnarités).
  → ε_decay=0.995 : diminution exponentielle — ε ≈ 0.01 après ~920 épisodes.
  → Trop rapide (0.99) → exploitation prématurée avant convergence.
  → Trop lent (0.999) → gaspillage d'épisodes en exploration inutile.

Table Q initialisée à 0 :
  → Optimiste neutre. Permet à l'agent de tester toutes les actions au départ.
  → Alternative : init aléatoire, mais 0 suffit ici (petit espace).
"""

import numpy as np


class QLearningAgent:
    """
    Agent Q-Learning tabulaire pour un carrefour 2-phases.

    La table Q est indexée par (file_NS, file_EO, phase, action).

    Parameters
    ----------
    state_space_size : tuple  (max_q+1, max_q+1, 2)
    n_actions        : int    nombre d'actions (2)
    alpha            : float  taux d'apprentissage
    gamma            : float  facteur de discount
    epsilon_start    : float  exploration initiale
    epsilon_end      : float  exploration minimale
    epsilon_decay    : float  décroissance multiplicative par épisode
    """

    def __init__(
            self,
            state_space_size,
            n_actions=2,
            alpha=0.1,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

        # Table Q initialisée à 0
        self.Q = np.zeros((*state_space_size, n_actions))

        # Statistiques
        self.n_updates = 0

    # ── Politique ε-greedy ────────────────────────────────────────────────────
    def choose_action(self, state, greedy=False):
        """
        Choisit une action selon la politique ε-greedy.

        Parameters
        ----------
        state   : tuple (file_NS, file_EO, phase)
        greedy  : bool  si True → exploitation pure (évaluation)

        Returns
        -------
        action : int (0 ou 1)
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)   # exploration
        return int(np.argmax(self.Q[state]))            # exploitation

    # ── Mise à jour Q ─────────────────────────────────────────────────────────
    def update(self, state, action, reward, next_state):
        """
        Applique la règle de mise à jour Q-Learning :

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

        Parameters
        ----------
        state      : tuple
        action     : int
        reward     : float
        next_state : tuple
        """
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        self.n_updates += 1
        return td_error  # utile pour le monitoring

    # ── Décroissance de ε ─────────────────────────────────────────────────────
    def decay_epsilon(self):
        """Réduit ε après chaque épisode (appelé par train.py)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Politique greedy courante ─────────────────────────────────────────────
    def get_policy(self):
        """
        Retourne la politique greedy pour tous les états.

        Returns
        -------
        dict : state → action optimale
        """
        policy = {}
        ns_size, eo_size, phase_size = self.Q.shape[:3]
        for ns in range(ns_size):
            for eo in range(eo_size):
                for ph in range(phase_size):
                    state = (ns, eo, ph)
                    policy[state] = int(np.argmax(self.Q[state]))
        return policy

    # ── Sauvegarde / chargement ───────────────────────────────────────────────
    def save(self, path):
        np.save(path, self.Q)
        print(f"[Agent] Table Q sauvegardée → {path}")

    def load(self, path):
        self.Q = np.load(path)
        print(f"[Agent] Table Q chargée ← {path}")

    def __repr__(self):
        return (
            f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, updates={self.n_updates})"
        )