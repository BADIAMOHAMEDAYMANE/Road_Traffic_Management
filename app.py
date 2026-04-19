"""
app.py
======
Interface Streamlit — Simulation pas-à-pas du carrefour.

Lancement : streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import time

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
st.caption("Observez le comportement du feu step par step selon la stratégie choisie.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    strat = st.radio(
        "Stratégie",
        ["Q-Learning", "Fixed-Time", "Greedy"],
        help="Q-Learning nécessite un entraînement préalable (run_all.py).",
    )

    st.divider()
    st.subheader("Environnement")
    lambda_ns = st.slider("λ NS (arrivées/step)", 0.1, 1.5, 0.5, 0.1)
    lambda_eo = st.slider("λ EO (arrivées/step)", 0.1, 1.5, 0.4, 0.1)
    fixed_int = st.slider("Intervalle fixed-time (steps)", 2, 20, 5,
                          disabled=(strat != "Fixed-Time"))

    st.divider()
    n_steps = st.slider("Nombre de steps", 10, 300, 80)
    speed   = st.slider("Vitesse (s/step)", 0.0, 0.5, 0.05, 0.01,
                        help="0 = instantané")

    run_btn = st.button("▶ Lancer la simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
PHASE_LABEL  = {0: "🟢 NS  /  🔴 EO", 1: "🔴 NS  /  🟢 EO"}
ACTION_LABEL = {0: "GARDER", 1: "CHANGER"}

def load_agent():
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    if not os.path.exists(q_path):
        return None
    env   = TrafficIntersection(lambda_ns=lambda_ns, lambda_eo=lambda_eo)
    agent = QLearningAgent(state_space_size=env.state_space_size)
    agent.load(q_path)
    agent.epsilon = 0.0
    return agent

def get_strategy():
    if strat == "Q-Learning":
        agent = load_agent()
        if agent is None:
            st.error("Table Q introuvable. Lancez d'abord `python run_all.py`.")
            st.stop()
        return lambda s: agent.choose_action(s, greedy=True)
    elif strat == "Fixed-Time":
        bl = FixedTimeBaseline(interval=fixed_int)
        return bl.choose_action
    else:
        bl = GreedyBaseline()
        return bl.choose_action

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    env    = TrafficIntersection(lambda_ns=lambda_ns, lambda_eo=lambda_eo)
    policy = get_strategy()
    state  = env.reset()

    st.subheader("État du carrefour en temps réel")
    col_phase, col_ns, col_eo, col_total, col_reward = st.columns(5)
    ph_box  = col_phase.empty()
    ns_box  = col_ns.empty()
    eo_box  = col_eo.empty()
    tot_box = col_total.empty()
    rew_box = col_reward.empty()

    st.divider()
    vis_ph = st.empty()

    st.subheader("Files d'attente au fil du temps")
    chart_ph = st.empty()

    st.subheader("Journal des steps")
    log_ph = st.empty()

    history   = []
    total_rew = 0.0

    for step in range(1, n_steps + 1):
        action          = policy(state)
        next_state, rew = env.step(action)
        ns, eo, ph      = next_state
        total_rew      += rew

        history.append({
            "Step":          step,
            "Phase":         PHASE_LABEL[ph],
            "Action":        ACTION_LABEL[action],
            "NS en attente": ns,
            "EO en attente": eo,
            "Récompense":    rew,
            "Cumul":         round(total_rew, 0),
        })

        ph_box.metric("Phase actuelle", PHASE_LABEL[ph])
        ns_box.metric("File NS", ns, delta=int(ns - state[0]) or None)
        eo_box.metric("File EO", eo, delta=int(eo - state[1]) or None)
        tot_box.metric("Total attente", ns + eo)
        rew_box.metric("Récompense cumul", f"{total_rew:.0f}")

        ns_bar = "█" * ns + "░" * (3 - ns)
        eo_bar = "█" * eo + "░" * (3 - eo)
        ns_col = "🟢" if ph == 0 else "🔴"
        eo_col = "🟢" if ph == 1 else "🔴"

        vis_ph.markdown(f"""
<div style="font-family:monospace;font-size:15px;line-height:2.2;
            padding:14px 22px;border:1px solid #ccc;
            border-radius:8px;display:inline-block">
  <b>Step {step}/{n_steps}</b> &nbsp;·&nbsp; Action : <b>{ACTION_LABEL[action]}</b><br>
  {ns_col} NS &nbsp; [{ns_bar}] {ns} voiture(s)<br>
  {eo_col} EO &nbsp; [{eo_bar}] {eo} voiture(s)
</div>
""", unsafe_allow_html=True)

        df_hist = pd.DataFrame(history)
        chart_ph.line_chart(
            df_hist.set_index("Step")[["NS en attente", "EO en attente"]],
            color=["#D85A30", "#185FA5"],
        )
        log_ph.dataframe(
            df_hist.tail(20)[["Step","Phase","Action","NS en attente","EO en attente","Récompense","Cumul"]],
            hide_index=True, use_container_width=True,
        )

        state = next_state
        if speed > 0:
            time.sleep(speed)

    st.divider()
    st.subheader("Résumé")
    df_final = pd.DataFrame(history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Récompense totale",    f"{total_rew:.0f}")
    c2.metric("Attente moy / step",   f"{df_final['NS en attente'].add(df_final['EO en attente']).mean():.2f}")
    c3.metric("Changements de phase", f"{df_final[df_final['Action']=='CHANGER'].shape[0]}")
    c4.metric("Steps simulés",        n_steps)

    st.download_button(
        "⬇ Télécharger le journal (CSV)",
        data=df_final.to_csv(index=False).encode(),
        file_name=f"simulation_{strat.lower().replace('-','_')}.csv",
        mime="text/csv",
    )

else:
    st.info("Configure les paramètres dans la barre latérale, puis clique sur **▶ Lancer la simulation**.")

    with st.expander("Comment ça marche ?"):
        st.markdown("""
**3 stratégies disponibles :**

| Stratégie | Description |
|---|---|
| **Q-Learning** | Agent entraîné — choisit l'action qui maximise la récompense future estimée |
| **Fixed-Time** | Change le feu toutes les N steps, sans regarder les files |
| **Greedy** | Donne toujours le vert à la file la plus longue (myope) |

**Lecture du carrefour :**
- 🟢 = feu vert (voitures passent)
- 🔴 = feu rouge (voitures attendent)
- `████` = 3 voitures (maximum)
- `░░░░` = file vide

**Reward :** `-(NS + EO)` — plus c'est proche de 0, mieux c'est.
        """)