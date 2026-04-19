"""
app.py
======
Interface Streamlit — Simulation pas-à-pas + comparaison des 3 stratégies.
Compatible Streamlit Cloud : entraîne automatiquement si q_table.npy absent.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from simulation import TrafficIntersection
from agent     import QLearningAgent
from utils     import FixedTimeBaseline, GreedyBaseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

st.set_page_config(
    page_title="Simulation carrefour",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Simulation pas-à-pas du carrefour")
st.caption("Comparez Q-Learning, Fixed-Time et Greedy en temps réel.")

PHASE_LABEL  = {0: "🟢 NS  /  🔴 EO", 1: "🔴 NS  /  🟢 EO"}
ACTION_LABEL = {0: "GARDER", 1: "CHANGER"}

# ─────────────────────────────────────────────────────────────────────────────
# Auto-entraînement si q_table.npy absent (Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────
def auto_train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    if os.path.exists(q_path):
        return

    with st.spinner("Première utilisation — entraînement de l'agent en cours... (≈ 10s)"):
        import json
        env   = TrafficIntersection(lambda_ns=0.5, lambda_eo=0.4)
        agent = QLearningAgent(
            state_space_size=env.state_space_size,
            alpha=0.1, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        )
        fixed  = FixedTimeBaseline(interval=5)
        greedy = GreedyBaseline()

        ql_r, ft_r, gr_r, eps_trace = [], [], [], []
        np.random.seed(42)

        for ep in range(1000):
            # Q-Learning
            state = env.reset()
            r_ql  = 0
            for _ in range(100):
                a = agent.choose_action(state)
                ns, r = env.step(a)
                agent.update(state, a, r, ns)
                state = ns
                r_ql += r
            agent.decay_epsilon()

            # Fixed
            state = env.reset(); fixed.reset()
            r_ft = 0
            for _ in range(100):
                a = fixed.choose_action(state)
                state, r = env.step(a)
                r_ft += r

            # Greedy
            state = env.reset()
            r_gr = 0
            for _ in range(100):
                a = greedy.choose_action(state)
                state, r = env.step(a)
                r_gr += r

            ql_r.append(r_ql); ft_r.append(r_ft); gr_r.append(r_gr)
            eps_trace.append(agent.epsilon)

        np.save(q_path, agent.Q)
        data = {
            "config": {"lambda_ns": 0.5, "lambda_eo": 0.4,
                       "max_queue": 3, "green_flow": 1, "fixed_interval": 5},
            "ql_rewards": ql_r, "fixed_rewards": ft_r,
            "greedy_rewards": gr_r, "epsilon_trace": eps_trace,
            "ql_eval_rewards": [],
        }
        with open(os.path.join(RESULTS_DIR, "training_data.json"), "w") as f:
            json.dump(data, f)

    st.success("✅ Agent entraîné et prêt !")
    st.rerun()

auto_train()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.subheader("Stratégies à simuler")
    use_ql = st.checkbox("Q-Learning",  value=True)
    use_ft = st.checkbox("Fixed-Time",  value=True)
    use_gr = st.checkbox("Greedy",      value=True)

    st.divider()
    st.subheader("Environnement")
    lambda_ns = st.slider("λ NS (arrivées/step)", 0.1, 1.5, 0.5, 0.1)
    lambda_eo = st.slider("λ EO (arrivées/step)", 0.1, 1.5, 0.4, 0.1)
    fixed_int = st.slider("Intervalle fixed-time", 2, 20, 5)

    st.divider()
    n_steps = st.slider("Nombre de steps", 10, 300, 80)
    seed    = st.number_input("Seed", value=42, step=1)

    run_btn = st.button("▶ Lancer la comparaison", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_strategy(name, policy_fn, seed):
    np.random.seed(seed)
    env   = TrafficIntersection(lambda_ns=lambda_ns, lambda_eo=lambda_eo)
    state = env.reset()
    rows  = []
    total = 0.0
    for step in range(1, n_steps + 1):
        action          = policy_fn(state)
        next_state, rew = env.step(action)
        total          += rew
        rows.append({
            "Step":          step,
            "NS en attente": next_state[0],
            "EO en attente": next_state[1],
            "Total attente": next_state[0] + next_state[1],
            "Action":        ACTION_LABEL[action],
            "Phase":         PHASE_LABEL[next_state[2]],
            "Récompense":    rew,
            "Cumul":         round(total, 0),
        })
        state = next_state
    return pd.DataFrame(rows)


def load_ql_policy():
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    env    = TrafficIntersection(lambda_ns=lambda_ns, lambda_eo=lambda_eo)
    agent  = QLearningAgent(state_space_size=env.state_space_size)
    agent.load(q_path)
    agent.epsilon = 0.0
    return lambda s: agent.choose_action(s, greedy=True)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not (use_ql or use_ft or use_gr):
        st.warning("Sélectionne au moins une stratégie.")
        st.stop()

    results = {}

    if use_ql:
        results["Q-Learning"] = run_strategy("Q-Learning", load_ql_policy(), seed)
    if use_ft:
        bl = FixedTimeBaseline(interval=fixed_int)
        results["Fixed-Time"] = run_strategy("Fixed-Time", bl.choose_action, seed)
    if use_gr:
        bl = GreedyBaseline()
        results["Greedy"] = run_strategy("Greedy", bl.choose_action, seed)

    COLORS = {
        "Q-Learning": "#1D9E75",
        "Fixed-Time": "#D85A30",
        "Greedy":     "#185FA5",
    }

    st.subheader("Résumé comparatif")
    cols = st.columns(len(results))
    for col, (name, df) in zip(cols, results.items()):
        col.metric(f"{name}", f"{df['Cumul'].iloc[-1]:.0f}", help="Récompense cumulée")
        col.caption(f"Attente moy : **{df['Total attente'].mean():.2f}** · Changements : **{df[df['Action']=='CHANGER'].shape[0]}**")

    st.divider()
    st.subheader("Files d'attente totales (NS + EO)")
    df_total = pd.DataFrame({"Step": range(1, n_steps + 1)})
    for name, df in results.items():
        df_total[name] = df["Total attente"].values
    st.line_chart(df_total.set_index("Step"),
                  color=[COLORS[n] for n in results.keys()])

    st.subheader("Récompense cumulée")
    df_cumul = pd.DataFrame({"Step": range(1, n_steps + 1)})
    for name, df in results.items():
        df_cumul[name] = df["Cumul"].values
    st.line_chart(df_cumul.set_index("Step"),
                  color=[COLORS[n] for n in results.keys()])

    st.divider()
    st.subheader("Statistiques détaillées")
    stat_rows = []
    for name, df in results.items():
        stat_rows.append({
            "Stratégie":          name,
            "Récompense totale":  f"{df['Cumul'].iloc[-1]:.0f}",
            "Attente moy/step":   f"{df['Total attente'].mean():.2f}",
            "Attente max":        f"{df['Total attente'].max()}",
            "Changements phase":  f"{df[df['Action']=='CHANGER'].shape[0]}",
            "Steps NS=0 et EO=0": f"{((df['NS en attente']==0) & (df['EO en attente']==0)).sum()}",
        })
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    others = {n: d for n, d in results.items() if n != "Q-Learning"}
    if "Q-Learning" in results and others:
        st.divider()
        st.subheader("Gain Q-Learning")
        ql_r = results["Q-Learning"]["Cumul"].iloc[-1]
        gain_cols = st.columns(len(others))
        for i, (name, df) in enumerate(others.items()):
            other_r = df["Cumul"].iloc[-1]
            gain    = (ql_r - other_r) / abs(other_r) * 100
            gain_cols[i].metric(f"vs {name}", f"{gain:+.1f}%",
                                delta=f"{ql_r - other_r:.0f} pts")

    st.divider()
    st.subheader("Journaux détaillés")
    tabs = st.tabs(list(results.keys()))
    for tab, (name, df) in zip(tabs, results.items()):
        with tab:
            st.dataframe(
                df[["Step","Phase","Action","NS en attente","EO en attente","Récompense","Cumul"]],
                hide_index=True, use_container_width=True, height=300,
            )
            st.download_button(
                f"⬇ Télécharger {name} (CSV)",
                data=df.to_csv(index=False).encode(),
                file_name=f"sim_{name.lower().replace('-','_').replace(' ','_')}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )

else:
    st.info("Sélectionne les stratégies dans la barre latérale, puis clique sur **▶ Lancer la comparaison**.")
    with st.expander("Comment lire les résultats ?"):
        st.markdown("""
| Indicateur | Signification |
|---|---|
| **Récompense cumulée** | Somme des `-(NS+EO)` — plus c'est proche de 0, mieux c'est |
| **Attente moy/step** | Nombre moyen de voitures en attente à chaque instant |
| **Changements de phase** | Nombre de fois où le feu a basculé |
| **Steps NS=0 et EO=0** | Steps où les deux files étaient vides (idéal) |

> Toutes les stratégies sont simulées avec la **même seed** pour une comparaison équitable.
        """)