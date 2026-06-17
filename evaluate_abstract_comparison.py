"""
evaluate_abstract_comparison.py
================================
Evaluasi perbandingan kondisi Paper GEEvo vs TA pada dataset abstract.
Semua hyperparameter disamakan dengan paper asli (Rupp et al., IEEE CEC 2024).

Tiga kondisi yang dibandingkan:
  A : No agent + obj3  →  replikasi paper (baseline)
  B : With agent + obj3  →  isolasi dampak agent
  C : With agent + obj4  →  setup TA penuh

Output: CSV di output_results/abstract_comparison_<timestamp>.csv
Partial save : output_results/abstract_comparison_PARTIAL.csv (diupdate tiap kondisi×alpha selesai)
Checkpoint   : output_results/abstract_comparison_checkpoint.csv (untuk resume jika crash)
"""

import os
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime

from geevo import nodes as n
from geevo.graph import Graph
from geevo.evolution.generator import EvolutionaryGraphGeneration
from geevo.evolution.balancer import Balancer
from geevo.agent_simulation import Agent

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Hyperparameter (identik dengan paper) ────────────────────────────────────
POP_SIZE    = 20
N_SIM       = 10
MAX_STEPS   = 500
ALPHAS      = [0.05, 0.01, 0.0]
N_GRAPHS    = 194  # sama dengan paper (Rupp et al. n=194)

# Konfigurasi graf abstract (sama seperti evaluate_metrics.ipynb)
ABSTRACT_CONF = {
    n.Source:     3,
    n.RandomGate: 2,
    n.Pool:       4,
    n.Converter:  1,
}

PARTIAL_PATH     = "output_results/abstract_comparison_PARTIAL.csv"
CHECKPOINT_PATH  = "output_results/abstract_comparison_checkpoint.csv"


# ── Helper ────────────────────────────────────────────────────────────────────
def generate_abstract_graph():
    egg = EvolutionaryGraphGeneration(ABSTRACT_CONF)
    egg.run(steps=50)
    g = Graph(ABSTRACT_CONF, egg.get_best())
    for c in g.get_nodes_of(n.Converter):
        c.is_auto = False
    return g


def load_checkpoint():
    """Baca hasil yang sudah selesai dari checkpoint. Return list of dicts."""
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_csv(CHECKPOINT_PATH)
        print(f"  [RESUME] Ditemukan checkpoint: {len(df)} baris sudah selesai.")
        return df.to_dict("records")
    return []


def is_done(results, label, alpha):
    """Cek apakah kombinasi kondisi × alpha sudah ada di results."""
    for r in results:
        if r["Kondisi"] == label and float(r["Alpha (α)"]) == float(alpha):
            return True
    return False


def save_partial(results):
    """Simpan semua hasil sementara ke PARTIAL dan CHECKPOINT."""
    os.makedirs("output_results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(PARTIAL_PATH, index=False)
    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"  [SAVED] Partial tersimpan ({len(results)} baris) → {PARTIAL_PATH}")


def run_condition(graphs, alpha, fitness_func, agents, label):
    """
    Jalankan evaluasi satu kondisi (satu alpha, satu fitness_func, satu set agent).
    Return dict metrik.
    """
    total         = len(graphs)
    balanced      = 0
    init_balanced = 0
    improved      = 0
    exec_times    = []
    generations   = []
    final_fits    = []

    for idx, g in enumerate(graphs):
        print(f"  [{label}] α={alpha} | graph {idx+1}/{total}...", end="\r")

        pools = g.get_nodes_of(n.Pool)
        if not pools:
            continue

        pool_id = random.randint(0, len(pools) - 1)
        t       = random.randint(10, 30)
        x       = random.randint(20, 100)

        b = Balancer(
            config           = g.config,
            edge_list        = g.edge_list,
            balance_pool_ids = [pool_id],
            n_sim_steps      = t,
            balance_value    = x,
            alpha            = alpha,
            fitness_func     = fitness_func,
            agent            = agents,
            pop_size         = POP_SIZE,
            n_sim            = N_SIM,
        )

        init_fit = b.get_ind_fitness(b.init_ind())
        if init_fit >= b.threshold:
            init_balanced += 1

        t0 = time.time()
        final_fit, iters = b.run(steps=MAX_STEPS)
        elapsed = time.time() - t0

        if final_fit is None:
            final_fit = b.get_ind_fitness(b.population[0])

        exec_times.append(elapsed)
        generations.append(iters)
        final_fits.append(final_fit)

        if final_fit >= b.threshold:
            balanced += 1
        if final_fit > init_fit:
            improved += 1

    total_runs = len(exec_times)
    print(f"  [{label}] α={alpha} | selesai ({total_runs} runs)          ")

    return {
        "Kondisi"             : label,
        "Alpha (α)"           : alpha,
        "Balanced %"          : f"{round(balanced / total_runs * 100, 2)}%",
        "Initial Balanced %"  : f"{round(init_balanced / total_runs * 100, 2)}%",
        "Improved %"          : f"{round(improved / total_runs * 100, 2)}%",
        "Median Generation"   : float(np.median(generations)),
        "Median Exec Time (s)": round(np.median(exec_times), 3),
        "Fitness Std Dev"     : round(float(np.std(final_fits)), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  evaluate_abstract_comparison.py")
    print("  Abstract-only, 3 kondisi vs Paper GEEvo")
    print("=" * 60)

    # ── Load checkpoint jika ada ──────────────────────────────────────────────
    results = load_checkpoint()

    # 1. Generate dataset (sama untuk semua kondisi)
    print(f"\nMembangun {N_GRAPHS} abstract graf (seed={SEED})...")
    graphs = []
    for i in range(N_GRAPHS):
        print(f"  Generating graph {i+1}/{N_GRAPHS}...", end="\r")
        graphs.append(generate_abstract_graph())
    print(f"  {N_GRAPHS} graf siap.                    ")

    # Simpan seed state setelah generate graf
    # agar tiap kondisi dimulai dari state yang sama
    rng_state    = random.getstate()
    np_rng_state = np.random.get_state()

    agents_ta = [Agent(behavior='aggressive'),
                 Agent(behavior='passive'),
                 Agent(behavior='random')]

    conditions = [
        # (label,          fitness_func, agents)
        ("A: No Agent+obj3",    "obj3", None),
        ("B: Agent+obj3",       "obj3", agents_ta),
        ("C: Agent+obj4 (TA)",  "obj4", agents_ta),
    ]

    for label, fitness_func, agents in conditions:
        for alpha in ALPHAS:
            # Skip jika sudah selesai (resume dari checkpoint)
            if is_done(results, label, alpha):
                print(f"  [SKIP] {label} α={alpha} sudah ada di checkpoint.")
                continue

            print(f"\n{'─'*60}")
            print(f"  Kondisi {label} | α={alpha}")
            print(f"{'─'*60}")

            # Reset seed sebelum tiap kondisi × alpha agar fair
            random.setstate(rng_state)
            np.random.set_state(np_rng_state)

            row = run_condition(graphs, alpha, fitness_func, agents, label)
            results.append(row)

            # Simpan partial setelah setiap kombinasi selesai
            save_partial(results)

    # 2. Buat DataFrame final
    df = pd.DataFrame(results)

    # 3. Tampilkan ringkasan
    print("\n" + "=" * 60)
    print("  HASIL PERBANDINGAN")
    print("=" * 60)

    for alpha in ALPHAS:
        print(f"\n── α = {alpha} ──")
        sub = df[df["Alpha (α)"] == alpha][
            ["Kondisi", "Balanced %", "Initial Balanced %",
             "Improved %", "Median Generation", "Median Exec Time (s)", "Fitness Std Dev"]
        ]
        print(sub.to_string(index=False))

    # Referensi paper
    print("\n── Referensi Paper GEEvo (n=194, no agent, obj3) ──")
    paper_ref = pd.DataFrame([
        {"Kondisi": "Paper GEEvo", "Alpha (α)": 0.05, "Balanced %": "93.3%",
         "Median Generation": 1.0, "Median Exec Time (s)": 18.4},
        {"Kondisi": "Paper GEEvo", "Alpha (α)": 0.01, "Balanced %": "83.0%",
         "Median Generation": 7.0, "Median Exec Time (s)": 66.0},
        {"Kondisi": "Paper GEEvo", "Alpha (α)": 0.00, "Balanced %": "58.8%",
         "Median Generation": 196.0, "Median Exec Time (s)": 703.2},
    ])
    print(paper_ref.to_string(index=False))

    # 4. Simpan ke CSV final
    os.makedirs("output_results", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"output_results/abstract_comparison_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\nHasil disimpan ke: {out}")
    print(f"(Partial/checkpoint juga tersedia di {PARTIAL_PATH})")


if __name__ == "__main__":
    main()
