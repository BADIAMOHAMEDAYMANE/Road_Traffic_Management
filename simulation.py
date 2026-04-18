"""
simulation.py
=============
Simulation du carrefour à feux de circulation (un seul carrefour, 2 directions).

MODÉLISATION MDP — Justifications :
────────────────────────────────────
ÉTATS (file_NS, file_EO, phase) :
  - file_NS ∈ {0,1,2,3} : voitures en attente direction Nord-Sud
  - file_EO ∈ {0,1,2,3} : voitures en attente direction Est-Ouest
  - phase  ∈ {0, 1}     : 0 = feu vert NS (rouge EO), 1 = feu vert EO (rouge NS)
  → 4 × 4 × 2 = 32 états — petit espace, table Q tractable, convergence rapide.

ACTIONS :
  - 0 = GARDER la phase actuelle
  - 1 = CHANGER de phase (basculer le feu)
  → Binaire, direct, sans ambiguïté.

REWARD :
  reward = -(file_NS + file_EO)
  → Pénalise linéairement les voitures en attente.
  → Signal dense (chaque step), gradient clair pour l'apprentissage.
  → Alternative classique : -(max queue) pénalise la pire file, mais moins stable.

GAMMA = 0.95 :
  → Favorise les récompenses futures proches (horizon ~20 steps)
  → < 1 garantit la convergence de la somme des récompenses actualisées.
  → 0.9 converge plus vite mais sous-valorise le futur ; 0.99 trop lent ici.

ARRIVÉES (loi de Poisson) :
  → Modélise le flux aléatoire de voitures indépendamment sur chaque route.
  → λ (taux d'arrivée) configurable par direction.
  → Plus réaliste qu'un random uniforme.
"""

import numpy as np


class TrafficIntersection:
    """
    Carrefour à 2 phases de feu :
      phase 0 → NS vert, EO rouge
      phase 1 → EO vert, NS rouge

    Paramètres
    ----------
    lambda_ns : float  taux d'arrivée Poisson direction NS  (voitures/step)
    lambda_eo : float  taux d'arrivée Poisson direction EO  (voitures/step)
    max_queue : int    capacité maximale de chaque file (discrétisation)
    green_flow : int   voitures qui passent par step quand feu vert
    """

    def __init__(self, lambda_ns=0.5, lambda_eo=0.4, max_queue=3, green_flow=1):
        self.lambda_ns = lambda_ns
        self.lambda_eo = lambda_eo
        self.max_queue = max_queue
        self.green_flow = green_flow
        self.reset()

    # ── Initialisation ────────────────────────────────────────────────────────
    def reset(self):
        """Remet le carrefour à zéro, retourne l'état initial."""
        self.queue_ns = 0   # voitures en attente NS
        self.queue_eo = 0   # voitures en attente EO
        self.phase = 0       # phase courante du feu
        return self._get_state()

    # ── État courant ──────────────────────────────────────────────────────────
    def _get_state(self):
        """Retourne un tuple (file_NS, file_EO, phase)."""
        return (self.queue_ns, self.queue_eo, self.phase)

    # ── Arrivées (loi de Poisson) ─────────────────────────────────────────────
    def _arrivals(self):
        """Tire le nombre de nouvelles voitures selon Poisson, plafonné à max_queue."""
        arr_ns = np.random.poisson(self.lambda_ns)
        arr_eo = np.random.poisson(self.lambda_eo)
        self.queue_ns = min(self.queue_ns + arr_ns, self.max_queue)
        self.queue_eo = min(self.queue_eo + arr_eo, self.max_queue)

    # ── Passage (débit selon feu vert) ────────────────────────────────────────
    def _departures(self):
        """Les voitures du côté vert passent (green_flow par step, min 0)."""
        if self.phase == 0:   # NS vert
            self.queue_ns = max(0, self.queue_ns - self.green_flow)
        else:                  # EO vert
            self.queue_eo = max(0, self.queue_eo - self.green_flow)

    # ── Transition ────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Applique une action et avance d'un pas de temps.

        Ordre :
          1. Changer de phase si action == 1
          2. Les voitures passent (côté vert)
          3. Nouvelles arrivées Poisson

        Parameters
        ----------
        action : int  0 = garder, 1 = changer

        Returns
        -------
        next_state : tuple
        reward     : float  -(nb voitures totales en attente) après transitions
        """
        # 1. Application de l'action
        if action == 1:
            self.phase = 1 - self.phase  # bascule 0↔1

        # 2. Départ des voitures (feu vert actif)
        self._departures()

        # 3. Arrivées aléatoires
        self._arrivals()

        # 4. Reward et nouvel état
        reward = -(self.queue_ns + self.queue_eo)
        next_state = self._get_state()
        return next_state, reward

    # ── Informations ─────────────────────────────────────────────────────────
    def get_info(self):
        """Retourne un dict lisible de l'état interne."""
        return {
            "queue_ns": self.queue_ns,
            "queue_eo": self.queue_eo,
            "phase": "NS_VERT" if self.phase == 0 else "EO_VERT",
            "total_waiting": self.queue_ns + self.queue_eo,
        }

    @property
    def state_space_size(self):
        return (self.max_queue + 1, self.max_queue + 1, 2)

    @property
    def action_space_size(self):
        return 2