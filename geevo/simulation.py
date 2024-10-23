import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from geevo.nodes import Pool, Source, Converter, FixedPool, FixedPoolLimit


class Simulator:
    def __init__(self, graph):
        self.graph = graph
        self.monitoring = {p: [] for p in self.graph if
                           isinstance(p, Pool) or isinstance(p, FixedPool) or isinstance(p, FixedPoolLimit)}
        self.pools = [n for n in self.graph if isinstance(n, Pool)]
        self.sources = [n for n in self.graph if isinstance(n, Source)]
        self.converters = [n for n in self.graph if isinstance(n, Converter)]

    def run(self, steps=10):
        for _ in range(steps):
            for source in self.sources:
                source.step([])
            self.monitor()

            # reset converters
            for c in self.converters:
                c.called = False
            # for p in self.pools:
            #     p.called = 0
        # reset pool values after simulation
        [p.reset() for p in self.pools]

    def monitor(self):
        for node in self.graph:
            if isinstance(node, Pool):
                self.monitoring[node].append(node.pool)

    def plot_monitor(self, drains=True, figsize=(10, 7), labels=None, save=False, filename="plots/graph.png",
                     xticks=None):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        count = 0
        for k, v in self.monitoring.items():
            if drains is True:
                if labels is None:
                    ax.plot(list(range(len(v))), v, label=k.name)
                else:
                    ax.plot(list(range(1, len(v) + 1)), v, label=labels[count])
                    count += 1
            else:
                if "drain" not in k.name.lower():
                    ax.plot(list(range(len(v))), v, label=k.name)
        ax.set_xlabel("Time steps", fontsize=14)
        ax.set_ylabel("Amount", fontsize=14)
        if xticks is not None:
            plt.xticks([1, 2, 3, 4, 5], fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        else:
            plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.ylim(0, 17)
        plt.legend()
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        if save is True:
            plt.savefig(filename, dpi=300)
        plt.plot()

        if labels is not None:
            print(", ".join([f"{l}: {v[-1]}" for l, v in zip(labels, self.monitoring.values())]))
