"""
run_convergence_plot.py
========================
Generate convergence curve A vs B vs C untuk laporan TA.
Plot: average best fitness per generasi (mean ± 1 std dev)
Dataset: abstract fixed config, n=30, alpha=0.05
"""
import random, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from geevo import nodes as n
from geevo.graph import Graph
from geevo.evolution.generator import EvolutionaryGraphGeneration
from geevo.evolution.balancer import Balancer
from geevo.agent_simulation import Agent

# ── Config ────────────────────────────────────────────────────────────────────
SEED      = 42
N_GRAPHS  = 30
MAX_STEPS = 500
ALPHA     = 0.05
POP_SIZE  = 20
N_SIM     = 10

ABSTRACT_CONF = {
    n.Source:     3,
    n.RandomGate: 2,
    n.Pool:       4,
    n.Converter:  1,
}

random.seed(SEED)
np.random.seed(SEED)

CONDITIONS = [
    ("A: No Agent + obj3",    "obj3", None),
    ("B: Agent + obj3",       "obj3", [Agent("aggressive"), Agent("passive"), Agent("random")]),
    ("C: Agent + obj4 (TA)",  "obj4", [Agent("aggressive"), Agent("passive"), Agent("random")]),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "output__evaluation_result")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generate graphs ────────────────────────────────────────────────────────────
print(f"Generating {N_GRAPHS} abstract graphs...")
graphs = []
for i in range(N_GRAPHS):
    print(f"  {i+1}/{N_GRAPHS}", end="\r", flush=True)
    egg = EvolutionaryGraphGeneration(ABSTRACT_CONF)
    egg.run(steps=50)
    g = Graph(ABSTRACT_CONF, egg.get_best())
    for c in g.get_nodes_of(n.Converter):
        c.is_auto = False
    graphs.append(g)
print(f"  {N_GRAPHS} graphs ready.      ")

rng_state    = random.getstate()
np_rng_state = np.random.get_state()

# ── Run tiap kondisi, capture monitor per generasi ────────────────────────────
all_curves  = {}
all_means   = {}
conv_gens   = {}  # median generasi konvergen

for label, fitness_func, agents in CONDITIONS:
    print(f"\nRunning: {label} | alpha={ALPHA}")
    random.setstate(rng_state)
    np.random.set_state(np_rng_state)

    curves       = []
    conv_list    = []

    for idx, g in enumerate(graphs):
        print(f"  graph {idx+1}/{N_GRAPHS}", end="\r", flush=True)
        pools   = g.get_nodes_of(n.Pool)
        pool_id = random.randint(0, len(pools) - 1)
        t       = random.randint(10, 30)
        x       = random.randint(20, 100)

        b = Balancer(
            config           = g.config,
            edge_list        = g.edge_list,
            balance_pool_ids = [pool_id],
            n_sim_steps      = t,
            balance_value    = x,
            alpha            = ALPHA,
            fitness_func     = fitness_func,
            agent            = agents,
            pop_size         = POP_SIZE,
            n_sim            = N_SIM,
        )
        _, iters = b.run(steps=MAX_STEPS)
        conv_list.append(iters)

        curve = np.array(b.monitor["best"])
        if len(curve) < MAX_STEPS:
            curve = np.pad(curve, (0, MAX_STEPS - len(curve)), mode="edge")
        curves.append(curve[:MAX_STEPS])

    all_curves[label]  = np.array(curves)
    all_means[label]   = all_curves[label].mean(axis=0)
    conv_gens[label]   = int(np.median(conv_list))
    print(f"  Done. Median conv gen = {conv_gens[label]}      ")

# ── Plot ───────────────────────────────────────────────────────────────────────
print("\nPlotting...")

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10, 6))

threshold = 1.0 - ALPHA
gens      = np.arange(MAX_STEPS)

for (label, _, _), color in zip(CONDITIONS, COLORS):
    curves = all_curves[label]
    mean   = curves.mean(axis=0)
    std    = curves.std(axis=0)
    med_g  = conv_gens[label]
    ax.plot(gens, mean, color=color, linewidth=2.2,
            label=f"{label}  (median conv. gen = {med_g})")
    ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.13)

ax.axhline(threshold, color="#CC0000", linestyle="--", linewidth=1.3, alpha=0.75,
           label=f"Balanced threshold  (1 − α = {threshold})")

ax.set_xlabel("Generation", fontsize=13)
ax.set_ylabel("Average Best Fitness", fontsize=13)
ax.set_title(
    f"Convergence Curve — Kondisi A vs B vs C\n"
    f"(n={N_GRAPHS} abstract graphs, α={ALPHA}, fixed config, max_steps={MAX_STEPS})",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlim(0, MAX_STEPS - 1)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10.5, framealpha=0.9, loc="lower right")
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

fig.tight_layout()
outpath = os.path.join(OUT_DIR, "convergence_ABC_fixed.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {outpath}")

# ── Ringkasan terminal ─────────────────────────────────────────────────────────
print("\nRingkasan rata-rata best fitness akhir (gen 499):")
for (label, _, _) in CONDITIONS:
    final = all_means[label][-1]
    print(f"  {label}: {final:.4f}")
print("\nSelesai.")
