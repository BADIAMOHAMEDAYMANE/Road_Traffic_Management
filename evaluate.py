"""
evaluate.py
===========
Évaluation approfondie post-entraînement :
  - Comparaison Q-Learning vs Baselines
  - Analyse de la politique apprise
  - Statistiques détaillées
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from simulation import TrafficIntersection
from agent     import QLearningAgent
from utils     import FixedTimeBaseline, GreedyBaseline, episode_stats, moving_average


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_EVAL_EP   = 200   # épisodes d'évaluation
STEPS_PER_EP = 100


def evaluate_agent(env, agent_or_baseline, n_episodes=N_EVAL_EP, label=""):
    """Évalue une stratégie sur n_episodes épisodes en mode greedy."""
    all_rewards = []
    all_waiting = []

    for _ in range(n_episodes):
        state = env.reset()
        if hasattr(agent_or_baseline, "reset"):
            agent_or_baseline.reset()

        ep_rewards = []
        ep_waiting = []

        for _ in range(STEPS_PER_EP):
            if hasattr(agent_or_baseline, "update"):
                action = agent_or_baseline.choose_action(state, greedy=True)
            else:
                action = agent_or_baseline.choose_action(state)

            next_state, reward = env.step(action)
            ep_rewards.append(reward)
            ep_waiting.append(next_state[0] + next_state[1])  # total en attente
            state = next_state

        all_rewards.append(sum(ep_rewards))
        all_waiting.append(np.mean(ep_waiting))

    stats = episode_stats(all_rewards)
    avg_wait = np.mean(all_waiting)
    print(f"\n  [{label}]")
    print(f"    Récompense moy : {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"    Voitures moy   : {avg_wait:.3f} voitures/step en attente")
    return all_rewards, all_waiting, stats


def analyze_policy(agent, max_queue=3):
    """Affiche la politique apprise pour tous les états."""
    print("\n  Politique Q-Learning apprise :")
    print(f"  {'État (NS,EO,Phase)':<25} {'Action':<10} {'Q(garder)':<12} {'Q(changer)':<12}")
    print("  " + "-" * 60)

    phase_names = {0: "NS_vert", 1: "EO_vert"}
    action_names = {0: "GARDER", 1: "CHANGER"}

    for ph in range(2):
        for ns in range(max_queue + 1):
            for eo in range(max_queue + 1):
                state = (ns, eo, ph)
                q_vals = agent.Q[state]
                best_a = int(np.argmax(q_vals))
                print(
                    f"  ({ns},{eo},{phase_names[ph]}){'':<5}"
                    f" {action_names[best_a]:<10}"
                    f" {q_vals[0]:>10.3f}"
                    f" {q_vals[1]:>10.3f}"
                )


def run_evaluation():
    """Point d'entrée principal de l'évaluation."""

    # ── Chargement ────────────────────────────────────────────────────────────
    data_path = os.path.join(RESULTS_DIR, "training_data.json")
    q_path    = os.path.join(RESULTS_DIR, "q_table.npy")

    if not os.path.exists(data_path):
        print("[!] Pas de données d'entraînement. Lancez train.py d'abord.")
        return None

    with open(data_path) as f:
        data = json.load(f)

    cfg = data["config"]

    env = TrafficIntersection(
        lambda_ns=cfg["lambda_ns"],
        lambda_eo=cfg["lambda_eo"],
        max_queue=cfg["max_queue"],
        green_flow=cfg["green_flow"],
    )

    agent = QLearningAgent(
        state_space_size=env.state_space_size,
        n_actions=env.action_space_size,
    )
    agent.load(q_path)
    agent.epsilon = 0.0  # mode pur greedy

    fixed  = FixedTimeBaseline(interval=cfg["fixed_interval"])
    greedy = GreedyBaseline()

    # ── Évaluation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ÉVALUATION — Comparaison des stratégies")
    print("=" * 65)

    np.random.seed(123)
    ql_r, ql_w, ql_stats     = evaluate_agent(env, agent,  label="Q-Learning")
    ft_r, ft_w, ft_stats     = evaluate_agent(env, fixed,  label="Fixed-Time Baseline")
    gr_r, gr_w, gr_stats     = evaluate_agent(env, greedy, label="Greedy Baseline")

    # ── Amélioration relative ─────────────────────────────────────────────────
    improve_vs_fixed  = (ql_stats["mean"] - ft_stats["mean"]) / abs(ft_stats["mean"]) * 100
    improve_vs_greedy = (ql_stats["mean"] - gr_stats["mean"]) / abs(gr_stats["mean"]) * 100

    print("\n" + "=" * 65)
    print("  RÉSULTATS COMPARATIFS")
    print("=" * 65)
    print(f"  Q-Learning vs Fixed-Time : {improve_vs_fixed:+.1f}%")
    print(f"  Q-Learning vs Greedy     : {improve_vs_greedy:+.1f}%")
    print(f"\n  Voitures moy/step :")
    print(f"    Q-Learning    : {np.mean(ql_w):.3f}")
    print(f"    Fixed-Time    : {np.mean(ft_w):.3f}")
    print(f"    Greedy        : {np.mean(gr_w):.3f}")
    print("=" * 65)

    # ── Politique ────────────────────────────────────────────────────────────
    analyze_policy(agent, max_queue=cfg["max_queue"])

    # ── Sauvegarde résultats évaluation ──────────────────────────────────────
    eval_data = {
        "ql_eval":    {"rewards": ql_r, "waiting": ql_w, "stats": ql_stats},
        "fixed_eval": {"rewards": ft_r, "waiting": ft_w, "stats": ft_stats},
        "greedy_eval":{"rewards": gr_r, "waiting": gr_w, "stats": gr_stats},
        "improve_vs_fixed":  improve_vs_fixed,
        "improve_vs_greedy": improve_vs_greedy,
    }

    with open(os.path.join(RESULTS_DIR, "eval_data.json"), "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\n[✓] Résultats évaluation → results/eval_data.json")

    return eval_data


if __name__ == "__main__":
    run_evaluation()