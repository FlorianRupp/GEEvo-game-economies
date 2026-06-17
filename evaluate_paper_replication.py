"""
evaluate_paper_replication.py
==============================
Replikasi metodologi paper GEEvo (Rupp et al., IEEE CEC 2024) dengan perbaikan:

  Perbedaan dari evaluate_abstract_comparison.py:
  1. egg.run(steps=50000) — sama dengan max steps paper (median aktual paper: 641 steps)
  2. Konfigurasi graf diacak per graf (5–20 node total) — sama seperti paper

  Catatan keterbatasan (tidak bisa 100% identik dengan paper):
  - Seed asli paper tidak diketahui → graf yang dihasilkan berbeda
  - Distribusi exact node types tidak dirinci di paper → pakai distribusi uniform
  - Hardware berbeda → exec time tidak bisa dibandingkan langsung

  Tiga kondisi tetap sama:
    A : No agent + obj3   → replikasi paper (baseline)
    B : With agent + obj3 → isolasi dampak agent
    C : With agent + obj4 → setup TA penuh

Output    : output_replication/paper_replication_<timestamp>.csv
Partial   : output_replication/paper_replication_PARTIAL.csv
Checkpoint: output_replication/paper_replication_checkpoint.csv
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
POP_SIZE   = 20
N_SIM      = 10
MAX_STEPS  = 500
ALPHAS     = [0.05, 0.01, 0.0]
N_GRAPHS   = 194
EGG_STEPS  = 50000  # paper: max 50k steps (median aktual 641) — sama dengan paper

OUTPUT_DIR       = "output_replication"
PARTIAL_PATH     = f"{OUTPUT_DIR}/paper_replication_PARTIAL.csv"
CHECKPOINT_PATH  = f"{OUTPUT_DIR}/paper_replication_checkpoint.csv"


# ── Random config (5–20 node, seperti paper) ─────────────────────────────────
def random_graph_config():
    """
    Generate konfigurasi graf acak dengan total 5–20 node.
    Minimal 1 Source dan 1 Pool agar graf valid untuk balancing.
    """
    n_source = random.randint(1, 4)
    n_pool   = random.randint(1, 6)
    n_rgate  = random.randint(0, 4)
    n_conv   = random.randint(0, 4)
    total    = n_source + n_pool + n_rgate + n_conv

    # Clamp ke minimum 5 node
    if total < 5:
        n_pool += 5 - total

    # Clamp ke maksimum 20 node — kurangi dari yang opsional dulu
    elif total > 20:
        excess = total - 20
        cut = min(n_rgate, excess); n_rgate -= cut; excess -= cut
        if excess > 0:
            cut = min(n_conv, excess); n_conv -= cut; excess -= cut
        if excess > 0:
            n_pool = max(1, n_pool - excess)

    return {
        n.Source:     n_source,
        n.RandomGate: n_rgate,
        n.Pool:       n_pool,
        n.Converter:  n_conv,
    }


def generate_graph(conf):
    """Return graph jika valid (fitness=0), None jika tidak valid setelah EGG_STEPS."""
    egg = EvolutionaryGraphGeneration(conf)
    egg.run(steps=EGG_STEPS)
    if egg.fitness and egg.fitness[-1] != 0:
        return None  # graf tidak valid — constraints tidak terpenuhi
    g = Graph(conf, egg.get_best())
    for c in g.get_nodes_of(n.Converter):
        c.is_auto = False
    return g


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_csv(CHECKPOINT_PATH)
        print(f"  [RESUME] Checkpoint ditemukan: {len(df)} baris sudah selesai.")
        return df.to_dict("records")
    return []


def is_done(results, label, alpha):
    for r in results:
        if r["Kondisi"] == label and float(r["Alpha (α)"]) == float(alpha):
            return True
    return False


def save_partial(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(PARTIAL_PATH, index=False)
    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"  [SAVED] Partial tersimpan ({len(results)} baris) → {PARTIAL_PATH}")


# ── Evaluasi satu kondisi ─────────────────────────────────────────────────────
def run_condition(graphs, alpha, fitness_func, agents, label):
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
    print("  evaluate_paper_replication.py")
    print(f"  N_GRAPHS={N_GRAPHS} | EGG_STEPS={EGG_STEPS} | SEED={SEED}")
    print("  Config: random per graf (5–20 node)")
    print("=" * 60)

    results = load_checkpoint()

    # 1. Generate dataset dengan random config per graf
    # Paper: attempt 200 graf, 194 valid (97%) — replikasi strategi yang sama
    print(f"\nMembangun {N_GRAPHS} valid abstract graf (random config, seed={SEED})...")
    print(f"  Strategy: attempt hingga {N_GRAPHS} valid ditemukan (paper: 194/200 = 97%)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    graphs = []
    configs_used = []
    attempt = 0
    while len(graphs) < N_GRAPHS:
        attempt += 1
        conf = random_graph_config()
        print(f"  Attempt {attempt} | valid={len(graphs)}/{N_GRAPHS} | config={sum(conf.values())} node...", end="\r")
        g = generate_graph(conf)
        if g is not None:
            graphs.append(g)
            configs_used.append({k.__name__: v for k, v in conf.items()})

    success_rate = len(graphs) / attempt * 100
    print(f"  {N_GRAPHS} graf valid dari {attempt} attempt ({success_rate:.1f}% success rate)  ")

    # Simpan distribusi config yang dipakai
    pd.DataFrame(configs_used).to_csv(f"{OUTPUT_DIR}/graph_configs_used.csv", index=False)
    print(f"  Distribusi config disimpan ke {OUTPUT_DIR}/graph_configs_used.csv")

    # Simpan seed state setelah generate — agar tiap kondisi fair
    rng_state    = random.getstate()
    np_rng_state = np.random.get_state()

    agents_ta = [
        Agent(behavior='aggressive'),
        Agent(behavior='passive'),
        Agent(behavior='random'),
    ]

    conditions = [
        ("A: No Agent+obj3",   "obj3", None),
        ("B: Agent+obj3",      "obj3", agents_ta),
        ("C: Agent+obj4 (TA)", "obj4", agents_ta),
    ]

    for label, fitness_func, agents in conditions:
        for alpha in ALPHAS:
            if is_done(results, label, alpha):
                print(f"  [SKIP] {label} α={alpha} sudah ada di checkpoint.")
                continue

            print(f"\n{'─'*60}")
            print(f"  Kondisi {label} | α={alpha}")
            print(f"{'─'*60}")

            random.setstate(rng_state)
            np.random.set_state(np_rng_state)

            row = run_condition(graphs, alpha, fitness_func, agents, label)
            results.append(row)
            save_partial(results)

    # 2. DataFrame final
    df = pd.DataFrame(results)

    # 3. Tampilkan hasil
    print("\n" + "=" * 60)
    print("  HASIL REPLIKASI")
    print("=" * 60)

    for alpha in ALPHAS:
        print(f"\n── α = {alpha} ──")
        sub = df[df["Alpha (α)"] == alpha][[
            "Kondisi", "Balanced %", "Initial Balanced %",
            "Improved %", "Median Generation", "Median Exec Time (s)", "Fitness Std Dev"
        ]]
        print(sub.to_string(index=False))

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

    # 4. Simpan CSV final
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{OUTPUT_DIR}/paper_replication_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\nHasil disimpan ke: {out}")


if __name__ == "__main__":
    main()
