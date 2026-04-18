"""
train.py
========
Boucle d'entraînement Q-Learning + évaluation des baselines.
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from simulation import TrafficIntersection
from agent     import QLearningAgent
from utils     import FixedTimeBaseline, GreedyBaseline, print_summary, moving_average


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparamètres
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Simulation
    "lambda_ns":    0.5,
    "lambda_eo":    0.4,
    "max_queue":    3,
    "green_flow":   1,
    # Entraînement
    "n_episodes":   1000,
    "steps_per_ep": 100,
    # Agent
    "alpha":          0.1,
    "gamma":          0.95,
    "epsilon_start":  1.0,
    "epsilon_end":    0.01,
    "epsilon_decay":  0.995,
    # Baseline fixed-time
    "fixed_interval": 5,
}


def run_episode(env, agent_or_baseline, greedy=False):
    """
    Exécute un épisode complet.

    Returns
    -------
    total_reward : float
    rewards      : list[float]
    """
    state = env.reset()
    if hasattr(agent_or_baseline, "reset"):
        agent_or_baseline.reset()

    rewards = []
    for _ in range(CONFIG["steps_per_ep"]):
        # Choix d'action
        if hasattr(agent_or_baseline, "update"):
            action = agent_or_baseline.choose_action(state, greedy=greedy)
        else:
            action = agent_or_baseline.choose_action(state)

        next_state, reward = env.step(action)

        # Mise à jour Q (seulement si c'est l'agent et pas en mode greedy pur)
        if hasattr(agent_or_baseline, "update") and not greedy:
            agent_or_baseline.update(state, action, reward, next_state)

        state = next_state
        rewards.append(reward)

    return sum(rewards), rewards


def train():
    """Entraîne l'agent Q-learning et évalue les baselines."""

    # ── Environnement ─────────────────────────────────────────────────────────
    env = TrafficIntersection(
        lambda_ns=CONFIG["lambda_ns"],
        lambda_eo=CONFIG["lambda_eo"],
        max_queue=CONFIG["max_queue"],
        green_flow=CONFIG["green_flow"],
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = QLearningAgent(
        state_space_size=env.state_space_size,
        n_actions=env.action_space_size,
        alpha=CONFIG["alpha"],
        gamma=CONFIG["gamma"],
        epsilon_start=CONFIG["epsilon_start"],
        epsilon_end=CONFIG["epsilon_end"],
        epsilon_decay=CONFIG["epsilon_decay"],
    )

    # ── Baselines ─────────────────────────────────────────────────────────────
    fixed_baseline  = FixedTimeBaseline(interval=CONFIG["fixed_interval"])
    greedy_baseline = GreedyBaseline()

    # ── Stockage des résultats ────────────────────────────────────────────────
    ql_rewards      = []   # récompense totale par épisode (training)
    ql_eval_rewards = []   # récompense totale par épisode (greedy eval)
    fixed_rewards   = []
    greedy_rewards  = []
    epsilon_trace   = []

    print("=" * 60)
    print("  ENTRAÎNEMENT Q-LEARNING — Carrefour Intelligent")
    print("=" * 60)
    print(f"  Épisodes      : {CONFIG['n_episodes']}")
    print(f"  Steps/épisode : {CONFIG['steps_per_ep']}")
    print(f"  α={CONFIG['alpha']}  γ={CONFIG['gamma']}  ε: {CONFIG['epsilon_start']}→{CONFIG['epsilon_end']}")
    print("=" * 60)

    # ── Boucle principale ─────────────────────────────────────────────────────
    np.random.seed(42)
    for ep in range(CONFIG["n_episodes"]):

        # — Q-Learning (train) —
        total_r, _ = run_episode(env, agent, greedy=False)
        ql_rewards.append(total_r)

        # — Q-Learning (eval greedy, tous les 10 épisodes) —
        if ep % 10 == 0:
            eval_r, _ = run_episode(env, agent, greedy=True)
            ql_eval_rewards.append(eval_r)

        # — Fixed-Time Baseline —
        f_r, _ = run_episode(env, fixed_baseline)
        fixed_rewards.append(f_r)

        # — Greedy Baseline —
        g_r, _ = run_episode(env, greedy_baseline)
        greedy_rewards.append(g_r)

        # — Décroissance ε —
        agent.decay_epsilon()
        epsilon_trace.append(agent.epsilon)

        # — Log tous les 100 épisodes —
        if (ep + 1) % 100 == 0:
            last_ql = np.mean(ql_rewards[-100:])
            last_ft = np.mean(fixed_rewards[-100:])
            last_gr = np.mean(greedy_rewards[-100:])
            print(
                f"  Ép {ep+1:4d} | ε={agent.epsilon:.4f} | "
                f"QL={last_ql:7.1f} | Fixed={last_ft:7.1f} | Greedy={last_gr:7.1f}"
            )

    # ── Résumés ───────────────────────────────────────────────────────────────
    print_summary("Q-Learning (training)",  ql_rewards)
    print_summary("Fixed-Time Baseline",    fixed_rewards)
    print_summary("Greedy Baseline",        greedy_rewards)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    results_path = os.path.join(os.path.dirname(__file__), "results")

    agent.save(os.path.join(results_path, "q_table.npy"))

    data = {
        "config":            CONFIG,
        "ql_rewards":        ql_rewards,
        "ql_eval_rewards":   ql_eval_rewards,
        "fixed_rewards":     fixed_rewards,
        "greedy_rewards":    greedy_rewards,
        "epsilon_trace":     epsilon_trace,
    }
    with open(os.path.join(results_path, "training_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[✓] Données sauvegardées dans results/")

    return agent, data


if __name__ == "__main__":
    train()