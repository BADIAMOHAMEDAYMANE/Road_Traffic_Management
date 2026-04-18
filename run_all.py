"""
run_all.py
==========
Script maître : entraîne, évalue et génère toutes les données.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from train    import train
from evaluate import run_evaluation

if __name__ == "__main__":
    print("\n🚦  SYSTÈME MULTI-AGENTS — Gestion du Trafic Urbain")
    print("     Phase 1 : Un carrefour, un agent Q-Learning\n")

    agent, training_data = train()
    eval_data            = run_evaluation()

    print("\n✅  Pipeline complet. Ouvrez le dashboard pour visualiser.")